/*
 * Copyright (c) Aperture Data Inc.
 * HNSW traversal adapted from Meta Platforms' faiss/impl/HNSW.cpp.
 * Licensed under the MIT license in the root LICENSE file.
 */
#include <faiss/transactional/HNSW.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/distances.h>

namespace faiss::transactional {
namespace {
using Candidate = std::pair<float, NodeId>;
constexpr int max_node_level = 32;

void validate(const Config& c) {
    if (!c.dimensions || c.dimensions > (1u << 20) || c.M < 2 || c.M > 4096 ||
        c.ef_construction < c.M || c.ef_construction > (1u << 24) ||
        (c.metric != METRIC_L2 && c.metric != METRIC_INNER_PRODUCT)) {
        throw std::invalid_argument("Invalid transactional HNSW configuration");
    }
}
void validate_vector(const float* x, size_t dim) {
    if (!x) {
        throw std::invalid_argument("Null HNSW vector");
    }
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(x[i])) {
            throw std::invalid_argument("HNSW vectors must be finite");
        }
    }
}

// SplitMix64 gives each stable ID a reproducible random level. The seed and
// allocation counter are transactional, so an abort cannot advance shared RNG.
int node_level(uint64_t id, const Config& c) {
    uint64_t z = id + c.seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    const double u =
            (static_cast<double>(z >> 11) + 1.0) / (9007199254740992.0 + 1.0);
    return std::min(
            max_node_level, static_cast<int>(-std::log(u) / std::log(c.M)));
}

// Only dirty records are retained. All reads see our earlier writes, including
// previous insertions in a batch. Flushing each dirty record once coalesces the
// reciprocal-link updates that otherwise repeatedly rewrite database pages.
struct Overlay {
    Storage& base;
    std::map<NodeId, Node> nodes;
    std::map<NodeId, std::vector<float>> vectors;
    Node node(NodeId id) const {
        const auto it = nodes.find(id);
        return it == nodes.end() ? base.node(id) : it->second;
    }
    std::vector<float> vector(NodeId id) const {
        const auto it = vectors.find(id);
        return it == vectors.end() ? base.vector(id) : it->second;
    }
    void flush(const Metadata& metadata) {
        try {
            for (const auto& [id, v] : vectors) {
                base.put_vector(id, v);
            }
            for (const auto& [id, n] : nodes) {
                base.put_node(id, n);
            }
            base.put_metadata(metadata);
        } catch (...) {
            base.invalidate();
            throw;
        }
    }
};

struct Distance : faiss::DistanceComputer {
    const Overlay& store;
    const Config& config;
    const float* query = nullptr;
    // Per-query/per-insertion owned copies: never retain an LMDB borrowed value
    // over a write, nor reuse a cache with a different transaction snapshot.
    std::unordered_map<NodeId, std::vector<float>> cache;
    Distance(const Overlay& s, const Config& c) : store(s), config(c) {}
    const float* get(idx_t raw_id) {
        if (raw_id < 0 || raw_id > std::numeric_limits<NodeId>::max()) {
            throw std::runtime_error("Invalid HNSW distance node ID");
        }
        const auto id = static_cast<NodeId>(raw_id);
        auto it = cache.find(id);
        if (it == cache.end()) {
            auto v = store.vector(id);
            if (v.size() != config.dimensions) {
                throw std::runtime_error("Corrupt HNSW vector dimension");
            }
            it = cache.emplace(id, std::move(v)).first;
        }
        return it->second.data();
    }
    float compare(const float* a, const float* b) const {
        const float result = config.metric == METRIC_L2
                ? faiss::fvec_L2sqr(a, b, config.dimensions)
                : -faiss::fvec_inner_product(a, b, config.dimensions);
        if (!std::isfinite(result)) {
            throw std::runtime_error("Non-finite HNSW distance");
        }
        return result;
    }
    void set_query(const float* x) override {
        query = x;
    }
    float operator()(idx_t i) override {
        return compare(query, get(i));
    }
    float symmetric_dis(idx_t a, idx_t b) override {
        const float* x = get(a);
        const float* y = get(b);
        return compare(x, y);
    }
};

Candidate greedy(const Overlay& s, Distance& d, Candidate best, int level) {
    bool changed;
    do {
        changed = false;
        const auto node = s.node(best.second);
        if (level >= static_cast<int>(node.links.size())) {
            throw std::runtime_error("Corrupt HNSW node level");
        }
        for (auto neighbor : node.links[level]) {
            Candidate candidate{d(neighbor), neighbor};
            if (candidate < best) {
                best = candidate;
                changed = true;
            }
        }
    } while (changed);
    return best;
}

std::vector<Candidate> search_layer(
        const Overlay& s,
        Distance& d,
        Candidate entry,
        size_t ef,
        int level,
        bool live_only = false,
        const IDSelector* selector = nullptr) {
    std::priority_queue<
            Candidate,
            std::vector<Candidate>,
            std::greater<Candidate>>
            frontier;
    std::priority_queue<Candidate> best;
    std::unordered_set<NodeId> visited;
    auto accepted = [&](NodeId id, const Node& node) {
        return (!live_only || !node.deleted) &&
                (!selector || selector->is_member(id));
    };
    frontier.push(entry);
    visited.insert(entry.second);
    if (accepted(entry.second, s.node(entry.second))) {
        best.push(entry);
    }
    while (!frontier.empty()) {
        const auto current = frontier.top();
        if (best.size() >= ef && current.first > best.top().first) {
            break;
        }
        frontier.pop();
        const auto node = s.node(current.second);
        if (level >= static_cast<int>(node.links.size())) {
            throw std::runtime_error("Corrupt HNSW adjacency level");
        }
        for (auto id : node.links[level]) {
            if (!visited.insert(id).second) {
                continue;
            }
            Candidate candidate{d(id), id};
            if (best.size() < ef || candidate < best.top()) {
                frontier.push(candidate);
                if (accepted(id, s.node(id))) {
                    best.push(candidate);
                    if (best.size() > ef) {
                        best.pop();
                    }
                }
            }
        }
    }
    std::vector<Candidate> result;
    while (!best.empty()) {
        result.push_back(best.top());
        best.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<NodeId> prune(
        Distance& d,
        const std::vector<Candidate>& candidates,
        size_t capacity) {
    using C = faiss::HNSW::C_distance;
    using Neighbor = faiss::HNSW::NodeDistFartherT<C>;
    std::priority_queue<Neighbor> input;
    for (const auto& [distance, id] : candidates) {
        input.emplace(distance, id);
    }
    std::vector<Neighbor> chosen;
    faiss::HNSW::shrink_neighbor_list<C>(d, input, chosen, capacity, false);
    std::vector<NodeId> result;
    for (const auto& neighbor : chosen) {
        result.push_back(neighbor.id);
    }
    return result;
}

void insert(Overlay& store, Metadata& meta, NodeId id, const float* vector) {
    const int level = node_level(id, meta.config);
    Node node;
    node.links.resize(level + 1);
    store.vectors[id] =
            std::vector<float>(vector, vector + meta.config.dimensions);
    store.nodes[id] = node;
    if (meta.entry_point == no_node) {
        meta.entry_point = id;
        meta.max_level = level;
        return;
    }
    Distance distance(store, meta.config);
    distance.set_query(vector);
    Candidate nearest{distance(meta.entry_point), meta.entry_point};
    for (int l = meta.max_level; l > level; --l) {
        nearest = greedy(store, distance, nearest, l);
    }
    for (int l = std::min(level, meta.max_level); l >= 0; --l) {
        const auto candidates = search_layer(
                store, distance, nearest, meta.config.ef_construction, l);
        const size_t capacity = l == 0 ? 2 * meta.config.M : meta.config.M;
        auto links = prune(distance, candidates, capacity);
        store.nodes[id].links[l] = links;
        for (auto neighbor : links) {
            auto other = store.node(neighbor);
            if (l >= static_cast<int>(other.links.size())) {
                throw std::runtime_error("Corrupt reciprocal HNSW level");
            }
            auto& back = other.links[l];
            back.push_back(id);
            if (back.size() > capacity) {
                std::vector<Candidate> choices;
                for (auto candidate : back) {
                    choices.emplace_back(
                            distance.symmetric_dis(neighbor, candidate),
                            candidate);
                }
                back = prune(distance, choices, capacity);
            }
            store.nodes[neighbor] = std::move(other);
        }
        if (!candidates.empty()) {
            nearest = candidates.front();
        }
    }
    if (level > meta.max_level) {
        meta.max_level = level;
        meta.entry_point = id;
    }
}
} // namespace

void HNSW::create(const Config& config) {
    validate(config);
    if (storage_.metadata()) {
        throw std::invalid_argument("HNSW namespace already exists");
    }
    Metadata meta;
    meta.config = config;
    Overlay{storage_, {}, {}}.flush(meta);
}

Metadata HNSW::metadata() const {
    auto meta = storage_.metadata();
    if (!meta) {
        throw std::invalid_argument("HNSW namespace does not exist");
    }
    validate(meta->config);
    if (meta->next_id >
                static_cast<uint64_t>(std::numeric_limits<NodeId>::max()) ||
        meta->live_count > meta->next_id ||
        (meta->next_id == 0 &&
         (meta->entry_point != no_node || meta->max_level != -1)) ||
        (meta->next_id > 0 &&
         (meta->entry_point < 0 ||
          static_cast<uint64_t>(meta->entry_point) >= meta->next_id ||
          meta->max_level < 0 || meta->max_level > max_node_level))) {
        throw std::runtime_error("Corrupt HNSW metadata");
    }
    return *meta;
}

std::vector<NodeId> HNSW::add(size_t n, const float* vectors) {
    auto meta = metadata();
    if (n > static_cast<uint64_t>(std::numeric_limits<NodeId>::max()) -
                        meta.next_id ||
        n > std::numeric_limits<size_t>::max() / meta.config.dimensions) {
        throw std::overflow_error("HNSW local ID or input size exhausted");
    }
    if (!n) {
        return {};
    }
    validate_vector(vectors, n * meta.config.dimensions);
    Overlay overlay{storage_, {}, {}};
    std::vector<NodeId> ids;
    ids.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const auto id = static_cast<NodeId>(meta.next_id++);
        insert(overlay, meta, id, vectors + i * meta.config.dimensions);
        ++meta.live_count;
        ids.push_back(id);
    }
    overlay.flush(meta);
    return ids;
}

bool HNSW::erase(NodeId id) {
    auto meta = metadata();
    if (id < 0 || static_cast<uint64_t>(id) >= meta.next_id) {
        return false;
    }
    auto node = storage_.node(id);
    if (node.deleted) {
        return false;
    }
    node.deleted = true;
    --meta.live_count;
    Overlay overlay{storage_, {}, {}};
    overlay.nodes.emplace(id, std::move(node));
    overlay.flush(meta);
    return true;
}

std::vector<Result> HNSW::search(
        const float* query,
        size_t k,
        size_t ef,
        const IDSelector* selector) const {
    const auto meta = metadata();
    if (!k || !meta.live_count) {
        return {};
    }
    validate_vector(query, meta.config.dimensions);
    Overlay view{storage_, {}, {}};
    Distance distance(view, meta.config);
    distance.set_query(query);
    Candidate nearest{distance(meta.entry_point), meta.entry_point};
    for (int level = meta.max_level; level > 0; --level) {
        nearest = greedy(view, distance, nearest, level);
    }
    auto found = search_layer(
            view, distance, nearest, std::max(k, ef), 0, true, selector);
    if (found.size() > k) {
        found.resize(k);
    }
    std::vector<Result> result;
    for (const auto& [value, id] : found) {
        result.push_back(
                {id, meta.config.metric == METRIC_L2 ? value : -value});
    }
    return result;
}
} // namespace faiss::transactional
