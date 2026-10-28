/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Aperture Data Inc.
 * HNSW traversal adapted from faiss/impl/HNSW.cpp.
 * Licensed under the MIT license in the root LICENSE file.
 */
#include <faiss/transactional/HNSW.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/distances.h>

namespace faiss::transactional {
namespace {
using Candidate = std::pair<float, NodeId>;
constexpr int max_node_level = 32;
constexpr uint64_t id_capacity =
        uint64_t(std::numeric_limits<NodeId>::max()) + 1;

void validate_config(const Config& c) {
    if (!c.dimensions || c.dimensions > (1u << 20) || c.M < 2 || c.M > 4096 ||
        c.ef_construction < c.M || c.ef_construction > (1u << 24) ||
        (c.metric != METRIC_L2 && c.metric != METRIC_INNER_PRODUCT)) {
        throw std::invalid_argument("Invalid transactional HNSW configuration");
    }
}
void cancelled(const std::function<bool()>& callback) {
    if (callback && callback())
        throw SearchCancelled();
}
void validate_vector(const float* x, size_t dim) {
    if (!x)
        throw std::invalid_argument("Null HNSW vector");
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(x[i]))
            throw std::invalid_argument("HNSW vectors must be finite");
    }
}
void validate_node(const Node& node, NodeId id, const Metadata& meta) {
    if (id < 0 || uint64_t(id) >= meta.next_id || node.links.empty() ||
        node.links.size() > 33) {
        throw std::runtime_error("Corrupt HNSW node identity or level");
    }
    std::vector<NodeId> sorted;
    for (size_t level = 0; level < node.links.size(); ++level) {
        const auto& links = node.links[level];
        const size_t capacity = level ? meta.config.M : 2 * meta.config.M;
        if (links.size() > capacity)
            throw std::runtime_error("Corrupt HNSW node degree");
        for (auto neighbor : links) {
            if (neighbor < 0 || uint64_t(neighbor) >= meta.next_id ||
                neighbor == id) {
                throw std::runtime_error("Corrupt HNSW adjacency ID");
            }
        }
        sorted.assign(links.begin(), links.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::runtime_error("Duplicate HNSW adjacency ID");
        }
    }
}
size_t node_bytes(const Node& node) {
    size_t bytes =
            sizeof(Node) + node.links.capacity() * sizeof(std::vector<NodeId>);
    for (const auto& links : node.links)
        bytes += links.capacity() * sizeof(NodeId);
    return bytes;
}

// Owned, operation-local LRU entries. Shared ownership keeps an evicted vector
// alive while a two-vector distance calculation is using it. Never cache across
// transactions. Budget includes payloads and conservative container overhead.
template <typename T>
class Cache {
    struct Entry {
        std::shared_ptr<const T> value;
        size_t charge;
        typename std::list<NodeId>::iterator position;
    };
    size_t limit_, used_ = 0;
    std::list<NodeId> order_;
    std::unordered_map<NodeId, Entry> entries_;

   public:
    explicit Cache(size_t limit) : limit_(limit) {}
    std::shared_ptr<const T> get(NodeId id) {
        auto it = entries_.find(id);
        if (it == entries_.end())
            return {};
        order_.splice(order_.begin(), order_, it->second.position);
        return it->second.value;
    }
    void put(NodeId id, std::shared_ptr<const T> value, size_t bytes) {
        constexpr size_t overhead = 192;
        if (bytes > limit_ || overhead > limit_ - bytes)
            return;
        const auto charge = bytes + overhead;
        while (used_ > limit_ - charge && !order_.empty()) {
            auto it = entries_.find(order_.back());
            used_ -= it->second.charge;
            entries_.erase(it);
            order_.pop_back();
        }
        order_.push_front(id);
        try {
            entries_.emplace(
                    id, Entry{std::move(value), charge, order_.begin()});
        } catch (...) {
            order_.pop_front();
            throw;
        }
        used_ += charge;
    }
};
struct NodeView {
    const Node* ptr;
    std::shared_ptr<const Node> owner;
    const Node& operator*() const {
        return *ptr;
    }
    const Node* operator->() const {
        return ptr;
    }
};
struct Overlay {
    Storage& base;
    WriteBatch batch;
    size_t workspace_limit;
    size_t workspace = 0;
    Cache<Node> nodes;
    SearchStats* stats;
    const std::function<bool()>& cancel;
    Overlay(Storage& base,
            Metadata meta,
            size_t limit,
            size_t node_cache_bytes,
            const std::function<bool()>& cancel,
            SearchStats* stats = nullptr)
            : base(base),
              batch{meta, {}, {}},
              workspace_limit(limit),
              nodes(node_cache_bytes),
              stats(stats),
              cancel(cancel) {}
    NodeView node(NodeId id) {
        cancelled(cancel);
        if (id < 0 || uint64_t(id) >= batch.metadata.next_id) {
            throw std::runtime_error("HNSW node ID outside snapshot");
        }
        auto dirty = batch.nodes.find(id);
        if (dirty != batch.nodes.end())
            return {&dirty->second, {}};
        auto cached = nodes.get(id);
        if (!cached) {
            if (stats)
                ++stats->node_reads;
            auto record = base.node(id);
            validate_node(record, id, batch.metadata);
            cached = std::make_shared<const Node>(std::move(record));
            nodes.put(id, cached, node_bytes(*cached));
        }
        auto* ptr = cached.get();
        return {ptr, std::move(cached)};
    }
    std::vector<float> vector(NodeId id) {
        cancelled(cancel);
        auto dirty = batch.vectors.find(id);
        if (dirty != batch.vectors.end())
            return dirty->second;
        if (stats)
            ++stats->vector_reads;
        return base.vector(id);
    }
    void reserve_change(size_t previous, size_t next) {
        if (previous > workspace || next > workspace_limit ||
            workspace - previous > workspace_limit - next) {
            throw std::length_error("HNSW mutation workspace limit exceeded");
        }
        workspace = workspace - previous + next;
    }
    void put_node(NodeId id, Node value) {
        validate_node(value, id, batch.metadata);
        auto it = batch.nodes.find(id);
        const size_t before =
                it == batch.nodes.end() ? 0 : node_bytes(it->second) + 128;
        reserve_change(before, node_bytes(value) + 128);
        batch.nodes.insert_or_assign(id, std::move(value));
    }
    void put_vector(NodeId id, const float* value) {
        const size_t bytes =
                size_t(batch.metadata.config.dimensions) * sizeof(float);
        reserve_change(0, bytes + sizeof(std::vector<float>) + 128);
        batch.vectors.emplace(
                id,
                std::vector<float>(
                        value, value + batch.metadata.config.dimensions));
    }
    void flush() {
        cancelled(cancel);
        base.apply(batch);
    }
};

struct VisitLimit {};
struct Distance : faiss::DistanceComputer {
    Overlay& store;
    const Config& config;
    Cache<std::vector<float>> cache;
    const float* query = nullptr;
    std::unordered_map<NodeId, float> measured;
    size_t visit_limit = std::numeric_limits<size_t>::max();
    size_t evaluations = 0;
    bool memoize = true;
    bool cache_vectors = true;
    SearchStats* stats = nullptr;
    std::function<void(NodeId, float)> observe;
    Distance(Overlay& s, size_t bytes)
            : store(s),
              config(s.batch.metadata.config),
              cache(bytes - bytes / 4) {}
    std::shared_ptr<const std::vector<float>> get(idx_t raw_id) {
        cancelled(store.cancel);
        if (raw_id < 0 || uint64_t(raw_id) >= store.batch.metadata.next_id) {
            throw std::runtime_error("Invalid HNSW distance node ID");
        }
        auto id = static_cast<NodeId>(raw_id);
        auto value = cache.get(id);
        if (!value) {
            auto vector = store.vector(id);
            if (vector.size() != config.dimensions)
                throw std::runtime_error("Corrupt HNSW vector dimension");
            value = std::make_shared<const std::vector<float>>(
                    std::move(vector));
            cache.put(
                    id,
                    value,
                    sizeof(std::vector<float>) +
                            value->capacity() * sizeof(float));
        }
        return value;
    }
    float compare(const float* a, const float* b) const {
        const float result = config.metric == METRIC_L2
                ? faiss::fvec_L2sqr(a, b, config.dimensions)
                : -faiss::fvec_inner_product(a, b, config.dimensions);
        if (!std::isfinite(result))
            throw std::runtime_error("Non-finite HNSW distance");
        return result;
    }
    void set_query(const float* x) override {
        query = x;
        measured.clear();
        evaluations = 0;
    }
    float operator()(idx_t raw_id) override {
        cancelled(store.cancel);
        if (raw_id < 0 || uint64_t(raw_id) >= store.batch.metadata.next_id)
            throw std::runtime_error("Invalid HNSW query node ID");
        const auto id = static_cast<NodeId>(raw_id);
        if (memoize) {
            auto it = measured.find(id);
            if (it != measured.end())
                return it->second;
        }
        if (evaluations >= visit_limit)
            throw VisitLimit{};
        float result;
        if (cache_vectors) {
            auto value = get(id);
            result = compare(query, value->data());
        } else {
            // Search computes each query distance only once. Retaining decoded
            // vectors adds allocations without providing reuse; construction
            // still caches them for symmetric neighbor diversification.
            auto value = store.vector(id);
            if (value.size() != config.dimensions)
                throw std::runtime_error("Corrupt HNSW vector dimension");
            result = compare(query, value.data());
        }
        if (memoize)
            measured.emplace(id, result);
        ++evaluations;
        if (stats)
            ++stats->visited;
        if (observe)
            observe(id, result);
        return result;
    }
    float symmetric_dis(idx_t a, idx_t b) override {
        auto x = get(a);
        auto y = get(b);
        return compare(x->data(), y->data());
    }
};
int node_level(uint64_t id, const Config& c) {
    uint64_t z = id + c.seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    const double u = (static_cast<double>(z >> 11) + 1.0) / 9007199254740992.0;
    return std::min(
            max_node_level, static_cast<int>(-std::log(u) / std::log(c.M)));
}
Candidate greedy(Overlay& s, Distance& d, Candidate best, int level) {
    bool changed;
    do {
        changed = false;
        auto node = s.node(best.second);
        if (level >= int(node->links.size()))
            throw std::runtime_error("Corrupt HNSW node level");
        if (s.stats)
            ++s.stats->expanded;
        for (auto id : node->links[level]) {
            Candidate next{d(id), id};
            if (next < best) {
                best = next;
                changed = true;
            }
        }
    } while (changed);
    return best;
}

// Search routing has an ef-sized heap independent of the result filter. A
// restrictive predicate must never force filling ef accepted results by
// scanning the entire graph. Results are collected by Distance::observe
// instead.
std::vector<Candidate> search_layer(
        Overlay& s,
        Distance& d,
        std::span<const Candidate> entries,
        size_t ef,
        int level,
        const SearchOptions* options = nullptr) {
    std::priority_queue<
            Candidate,
            std::vector<Candidate>,
            std::greater<Candidate>>
            frontier;
    std::priority_queue<Candidate> best;
    std::unordered_set<NodeId> visited;
    std::unordered_set<NodeId> expanded_bridges;
    auto offer = [&](Candidate c) {
        if (best.size() < ef || c < best.top()) {
            frontier.push(c);
            best.push(c);
            if (best.size() > ef)
                best.pop();
        }
    };
    for (const auto& entry : entries)
        if (visited.insert(entry.second).second)
            offer(entry);
    while (!frontier.empty()) {
        cancelled(s.cancel);
        const auto current = frontier.top();
        if (best.size() >= ef && current > best.top())
            break;
        frontier.pop();
        auto node = s.node(current.second);
        if (level >= int(node->links.size()))
            throw std::runtime_error("Corrupt HNSW adjacency level");
        if (s.stats)
            ++s.stats->expanded;
        auto consider = [&](NodeId id) {
            if (visited.insert(id).second)
                offer({d(id), id});
        };
        if (options && options->acorn && level == 0) {
            const bool all_live =
                    s.batch.metadata.live_count == s.batch.metadata.next_id;
            const auto accepted = [&](NodeId id) {
                // Evaluate a selective predicate before decoding a node just
                // to check its tombstone. In a tombstone-free snapshot no
                // node lookup is needed to accept a matching candidate.
                return options->selector->is_member(id) &&
                        (all_live || !s.node(id)->deleted);
            };
            for (auto id : node->links[0]) {
                if (accepted(id)) {
                    consider(id);
                } else if (expanded_bridges.insert(id).second) {
                    // Expand each rejected bridge once per level search.
                    // Repeating it cannot add candidates: its accepted
                    // neighbors have already been considered.
                    auto neighbor = s.node(id);
                    if (s.stats)
                        ++s.stats->expanded;
                    for (auto second : neighbor->links[0]) {
                        if (accepted(second))
                            consider(second);
                    }
                }
            }
        } else {
            for (auto id : node->links[level])
                consider(id);
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
    for (const auto& [distance, id] : candidates)
        input.emplace(distance, id);
    std::vector<Neighbor> chosen;
    faiss::HNSW::shrink_neighbor_list<C>(d, input, chosen, capacity, false);
    std::vector<NodeId> result;
    for (const auto& neighbor : chosen)
        result.push_back(neighbor.id);
    return result;
}
size_t budget(const Metadata& meta, size_t ef);
void insert(
        Overlay& store,
        NodeId id,
        const float* vector,
        size_t cache_bytes,
        size_t max_visits) {
    auto& meta = store.batch.metadata;
    const int level = node_level(id, meta.config);
    Node node;
    node.links.resize(level + 1);
    store.put_vector(id, vector);
    store.put_node(id, std::move(node));
    if (meta.entry_point == no_node) {
        meta.entry_point = id;
        meta.max_level = level;
        return;
    }
    Distance distance(store, cache_bytes);
    distance.set_query(vector);
    distance.visit_limit =
            max_visits ? max_visits : budget(meta, meta.config.ef_construction);
    Candidate nearest{distance(meta.entry_point), meta.entry_point};
    for (int l = meta.max_level; l > level; --l)
        nearest = greedy(store, distance, nearest, l);
    for (int l = std::min(level, meta.max_level); l >= 0; --l) {
        const auto candidates = search_layer(
                store,
                distance,
                std::span(&nearest, 1),
                meta.config.ef_construction,
                l);
        const size_t capacity = l == 0 ? 2 * meta.config.M : meta.config.M;
        auto links = prune(distance, candidates, capacity);
        auto self = *store.node(id);
        self.links[l] = links;
        store.put_node(id, std::move(self));
        for (auto neighbor : links) {
            auto other = *store.node(neighbor);
            if (l >= int(other.links.size()))
                throw std::runtime_error("Corrupt reciprocal HNSW level");
            auto& back = other.links[l];
            back.push_back(id);
            if (back.size() > capacity) {
                std::vector<Candidate> choices;
                for (auto candidate : back)
                    choices.emplace_back(
                            distance.symmetric_dis(neighbor, candidate),
                            candidate);
                back = prune(distance, choices, capacity);
            }
            store.put_node(neighbor, std::move(other));
        }
        if (!candidates.empty())
            nearest = candidates.front();
    }
    if (level > meta.max_level) {
        meta.max_level = level;
        meta.entry_point = id;
    }
}
size_t budget(const Metadata& meta, size_t ef) {
    // Saturating arithmetic: a large requested k/ef cannot wrap the work limit.
    uint64_t value = std::min<uint64_t>(ef, meta.next_id);
    for (uint64_t factor :
         {uint64_t(std::max(32u, 2 * meta.config.M)),
          uint64_t(meta.max_level + 1)}) {
        if (factor && value > meta.next_id / factor)
            return size_t(meta.next_id);
        value *= factor;
    }
    return size_t(std::min(value, meta.next_id));
}
void validate_batch_size(
        size_t n,
        const Metadata& m,
        const MutationOptions& options) {
    if (n > id_capacity - m.next_id ||
        n > std::numeric_limits<size_t>::max() / m.config.dimensions) {
        throw std::overflow_error("HNSW local ID or input size exhausted");
    }
    if (n > options.max_batch_size)
        throw std::length_error("HNSW mutation batch limit exceeded");
    const size_t bytes_per_vector =
            size_t(m.config.dimensions) * sizeof(float) + 128 +
            sizeof(std::vector<float>);
    if (n > options.max_workspace_bytes / bytes_per_vector)
        throw std::length_error("HNSW mutation workspace limit exceeded");
}
} // namespace

void Storage::apply(const WriteBatch& batch) {
    try {
        for (const auto& [id, v] : batch.vectors)
            put_vector(id, v);
        for (const auto& [id, n] : batch.nodes)
            put_node(id, n);
        put_metadata(batch.metadata);
    } catch (...) {
        invalidate();
        std::throw_with_nested(MutationError(
                "HNSW mutation failed; abort caller transaction", true));
    }
}
void HNSW::create(const Config& config) {
    validate_config(config);
    if (storage_.metadata())
        throw std::invalid_argument("HNSW namespace already exists");
    WriteBatch batch;
    batch.metadata.config = config;
    storage_.apply(batch);
}
Metadata HNSW::metadata() const {
    auto meta = storage_.metadata();
    if (!meta)
        throw std::invalid_argument("HNSW namespace does not exist");
    validate_config(meta->config);
    if (meta->next_id > id_capacity || meta->live_count > meta->next_id ||
        (!meta->next_id &&
         (meta->entry_point != no_node || meta->max_level != -1)) ||
        (meta->next_id &&
         (meta->entry_point < 0 ||
          uint64_t(meta->entry_point) >= meta->next_id || meta->max_level < 0 ||
          meta->max_level > max_node_level))) {
        throw std::runtime_error("Corrupt HNSW metadata");
    }
    return *meta;
}
std::vector<NodeId> HNSW::add(size_t n, const float* vectors) {
    return add(n, vectors, MutationOptions{});
}
std::vector<NodeId> HNSW::add(
        size_t n,
        const float* vectors,
        const MutationOptions& options) {
    cancelled(options.cancelled);
    auto meta = metadata();
    validate_batch_size(n, meta, options);
    if (!n)
        return {};
    for (size_t i = 0; i < n; ++i) {
        cancelled(options.cancelled);
        if (!vectors)
            throw std::invalid_argument("Null HNSW vector");
        validate_vector(
                vectors + i * meta.config.dimensions, meta.config.dimensions);
    }
    Overlay overlay(
            storage_,
            meta,
            options.max_workspace_bytes,
            options.cache_bytes / 4,
            options.cancelled);
    std::vector<NodeId> ids;
    ids.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const auto id = static_cast<NodeId>(overlay.batch.metadata.next_id++);
        try {
            insert(overlay,
                   id,
                   vectors + i * meta.config.dimensions,
                   options.cache_bytes,
                   options.max_visits_per_insertion);
        } catch (const VisitLimit&) {
            throw std::length_error("HNSW construction visit limit exceeded");
        }
        ++overlay.batch.metadata.live_count;
        ids.push_back(id);
    }
    overlay.flush();
    return ids;
}
NodeId HNSW::update(
        NodeId id,
        const float* vector,
        const MutationOptions& options) {
    cancelled(options.cancelled);
    auto meta = metadata();
    if (id < 0 || uint64_t(id) >= meta.next_id)
        throw std::invalid_argument("Unknown HNSW update ID");
    validate_batch_size(1, meta, options);
    validate_vector(vector, meta.config.dimensions);
    Overlay overlay(
            storage_,
            meta,
            options.max_workspace_bytes,
            options.cache_bytes / 4,
            options.cancelled);
    auto old = *overlay.node(id);
    if (old.deleted)
        throw std::invalid_argument("Cannot update a deleted HNSW node");
    if (!meta.live_count)
        throw std::runtime_error("Corrupt HNSW live count");
    old.deleted = true;
    overlay.put_node(id, std::move(old));
    const auto replacement =
            static_cast<NodeId>(overlay.batch.metadata.next_id++);
    try {
        insert(overlay,
               replacement,
               vector,
               options.cache_bytes,
               options.max_visits_per_insertion);
    } catch (const VisitLimit&) {
        throw std::length_error("HNSW construction visit limit exceeded");
    }
    overlay.flush();
    return replacement;
}
bool HNSW::erase(NodeId id) {
    auto meta = metadata();
    if (id < 0 || uint64_t(id) >= meta.next_id)
        return false;
    const std::function<bool()> no_cancel;
    Overlay overlay(
            storage_, meta, std::numeric_limits<size_t>::max(), 0, no_cancel);
    auto node = *overlay.node(id);
    if (node.deleted)
        return false;
    if (!meta.live_count)
        throw std::runtime_error("Corrupt HNSW live count");
    node.deleted = true;
    --overlay.batch.metadata.live_count;
    overlay.put_node(id, std::move(node));
    overlay.flush();
    return true;
}
std::vector<Result> HNSW::search(
        const float* query,
        size_t k,
        size_t ef,
        const IDSelector* selector) const {
    SearchOptions options;
    options.ef_search = ef;
    options.selector = selector;
    return search(query, k, options);
}
std::vector<Result> HNSW::search(
        const float* query,
        size_t k,
        const SearchOptions& options,
        SearchStats* stats) const {
    SearchStats local;
    if (!stats)
        stats = &local;
    *stats = {};
    cancelled(options.cancelled);
    const auto meta = metadata();
    if (options.seed_ids.size() > 64 || (options.acorn && !options.selector))
        throw std::invalid_argument(
                "Invalid HNSW search seeds or ACORN predicate");
    for (auto id : options.seed_ids)
        if (id < 0 || uint64_t(id) >= meta.next_id)
            throw std::invalid_argument("HNSW search seed outside snapshot");
    if (!k || !meta.live_count)
        return {};
    validate_vector(query, meta.config.dimensions);
    k = std::min<uint64_t>(k, meta.live_count);
    const size_t ef =
            std::min<uint64_t>(std::max(k, options.ef_search), meta.next_id);
    Overlay view(
            storage_, meta, 0, options.cache_bytes, options.cancelled, stats);
    Distance distance(view, 0);
    distance.cache_vectors = false;
    distance.memoize = !options.exact;
    distance.stats = stats;
    distance.visit_limit = options.max_visits
            ? options.max_visits
            : (options.exact ? size_t(meta.next_id) : budget(meta, ef));
    distance.set_query(query);
    std::priority_queue<Candidate> results;
    distance.observe = [&](NodeId id, float value) {
        if (options.selector && !options.selector->is_member(id))
            return;
        if (meta.live_count != meta.next_id && view.node(id)->deleted)
            return;
        Candidate candidate{value, id};
        if (results.size() < k || candidate < results.top()) {
            results.push(candidate);
            if (results.size() > k)
                results.pop();
        }
    };
    try {
        if (options.exact) {
            for (uint64_t id = 0; id < meta.next_id; ++id) {
                cancelled(options.cancelled);
                auto nid = static_cast<NodeId>(id);
                if ((!options.selector || options.selector->is_member(nid)) &&
                    (meta.live_count == meta.next_id ||
                     !view.node(nid)->deleted))
                    distance(nid);
            }
        } else {
            auto entry_node = view.node(meta.entry_point);
            if (entry_node->links.size() != size_t(meta.max_level + 1))
                throw std::runtime_error("Corrupt HNSW entry level");
            Candidate nearest{distance(meta.entry_point), meta.entry_point};
            for (int level = meta.max_level; level > 0; --level)
                nearest = greedy(view, distance, nearest, level);
            std::vector<Candidate> entries{nearest};
            for (auto id : options.seed_ids)
                entries.emplace_back(distance(id), id);
            search_layer(view, distance, entries, ef, 0, &options);
        }
    } catch (const VisitLimit&) {
        stats->visit_limit_reached = true;
    }
    std::vector<Result> result;
    while (!results.empty()) {
        const auto [value, id] = results.top();
        results.pop();
        result.push_back(
                {id, meta.config.metric == METRIC_L2 ? value : -value});
    }
    std::reverse(result.begin(), result.end());
    return result;
}
ValidationReport HNSW::validate(const std::function<bool()>& cancel) const {
    cancelled(cancel);
    const auto meta = metadata();
    Overlay view(storage_, meta, 0, 8u << 20, cancel);
    ValidationReport report;
    int highest = -1;
    for (uint64_t id = 0; id < meta.next_id; ++id) {
        cancelled(cancel);
        auto node = view.node(static_cast<NodeId>(id));
        ++report.nodes;
        report.live_nodes += !node->deleted;
        highest = std::max(highest, int(node->links.size()) - 1);
        auto vector = storage_.vector(static_cast<NodeId>(id));
        if (vector.size() != meta.config.dimensions)
            throw std::runtime_error("Corrupt HNSW vector dimension");
        validate_vector(vector.data(), vector.size());
        for (size_t level = 0; level < node->links.size(); ++level) {
            for (auto neighbor : node->links[level]) {
                if (view.node(neighbor)->links.size() <= level)
                    throw std::runtime_error("Corrupt HNSW cross-level edge");
                ++report.edges;
            }
        }
    }
    if (report.live_nodes != meta.live_count || highest != meta.max_level ||
        (meta.next_id &&
         view.node(meta.entry_point)->links.size() !=
                 size_t(meta.max_level + 1))) {
        throw std::runtime_error("Corrupt HNSW metadata counts or entry level");
    }
    return report;
}
} // namespace faiss::transactional
