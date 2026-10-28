/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/LmdbStorage.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace faiss::transactional {
namespace {
constexpr uint32_t max_dimensions = 1u << 20;
constexpr uint32_t max_neighbors = 4096;
constexpr uint32_t max_levels = 33;
constexpr size_t metadata_bytes = 52;
constexpr size_t maximum_node_bytes =
        8 + 4 * max_levels + 4 * max_neighbors * (max_levels + 1);
constexpr size_t maximum_vector_bytes = 4 + 4 * size_t(max_dimensions);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The transactional record format requires IEEE-754 binary32");
static_assert(std::endian::native == std::endian::little ||
                      std::endian::native == std::endian::big,
              "The transactional record format requires uniform byte order");

uint32_t from_big_endian(uint32_t value) {
    if constexpr (std::endian::native == std::endian::little) {
        // Compilers recognize this portable C++20 expression as a byte swap;
        // unlike a decoder call per element it also permits loop vectorization.
        return ((value & 0x000000ffu) << 24) |
                ((value & 0x0000ff00u) << 8) |
                ((value & 0x00ff0000u) >> 8) |
                ((value & 0xff000000u) >> 24);
    }
    return value;
}

bool valid_metadata(const Metadata& m) {
    const auto& c = m.config;
    return c.dimensions && c.dimensions <= max_dimensions && c.M >= 2 &&
            c.M <= max_neighbors && c.ef_construction >= c.M &&
            c.ef_construction <= (1u << 24) &&
            (c.metric == METRIC_L2 || c.metric == METRIC_INNER_PRODUCT) &&
            m.next_id <= uint64_t(std::numeric_limits<NodeId>::max()) + 1 &&
            m.live_count <= m.next_id &&
            (m.next_id == 0
                     ? m.entry_point == no_node && m.max_level == -1
                     : m.entry_point >= 0 &&
                               uint64_t(m.entry_point) < m.next_id &&
                               m.max_level >= 0 && m.max_level < int(max_levels));
}

void status(int rc) {
    if (rc != MDB_SUCCESS) {
        throw LmdbError(rc);
    }
}
void u32(std::string& s, uint32_t n) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        s.push_back(static_cast<char>(n >> shift));
    }
}
void u64(std::string& s, uint64_t n) {
    u32(s, static_cast<uint32_t>(n >> 32));
    u32(s, static_cast<uint32_t>(n));
}
struct Decoder {
    std::string_view bytes;
    uint32_t u32() {
        if (bytes.size() < 4) {
            throw std::runtime_error("Truncated HNSW record");
        }
        uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) {
            value = (value << 8) | static_cast<unsigned char>(bytes[i]);
        }
        bytes.remove_prefix(4);
        return value;
    }
    uint64_t u64() {
        const uint64_t high = u32();
        return (high << 32) | u32();
    }
    void end() const {
        if (!bytes.empty()) {
            throw std::runtime_error("Trailing bytes in HNSW record");
        }
    }
};
NodeId id_from(uint32_t raw) {
    if (raw == UINT32_MAX) {
        return no_node;
    }
    if (raw > static_cast<uint32_t>(std::numeric_limits<NodeId>::max())) {
        throw std::runtime_error("Invalid HNSW node ID");
    }
    return static_cast<NodeId>(raw);
}
} // namespace

LmdbStorage::LmdbStorage(
        MDB_txn* txn,
        MDB_dbi table,
        uint64_t index_id,
        uint64_t generation,
        Put put,
        WriteMode mode)
        : txn_(txn),
          table_(table),
          put_(std::move(put)),
          mode_(mode == WriteMode::Automatic
                        ? (put_ ? WriteMode::CallerTransaction
                                : WriteMode::Savepoint)
                        : mode) {
    if (!txn_) {
        throw std::invalid_argument("HNSW requires an active LMDB transaction");
    }
    if (mode_ != WriteMode::Savepoint &&
        mode_ != WriteMode::CallerTransaction) {
        throw std::invalid_argument("Invalid HNSW LMDB write mode");
    }
    if (put_ && mode_ == WriteMode::Savepoint) {
        throw std::invalid_argument(
                "HNSW LMDB savepoints cannot roll back changelog callbacks");
    }
    unsigned flags = 0;
    status(mdb_dbi_flags(txn_, table_, &flags));
    if (flags != 0) {
        throw std::invalid_argument(
                "HNSW requires an ordinary byte-key LMDB table");
    }
    std::memcpy(prefix_.data(), "TH01", 4);
    for (size_t i = 0; i < 8; ++i) {
        prefix_[4 + i] = static_cast<char>(index_id >> (56 - 8 * i));
        prefix_[12 + i] = static_cast<char>(generation >> (56 - 8 * i));
    }
}

void LmdbStorage::check() const {
    if (std::this_thread::get_id() != owner_) {
        throw std::logic_error("HNSW LMDB adapter used from a different thread");
    }
    if (callback_active_) {
        throw std::logic_error("HNSW changelog callback cannot reenter its adapter");
    }
    if (failed_.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
                "HNSW adapter invalidated; discard it and follow the original "
                "mutation failure's abort requirement");
    }
}
LmdbStorage::EncodedKey LmdbStorage::key(char kind, NodeId id) const {
    EncodedKey result;
    std::memcpy(result.bytes.data(), prefix_.data(), prefix_.size());
    result.bytes[20] = kind;
    result.size = 21;
    if (kind != 'm') {
        if (id < 0) {
            throw std::invalid_argument("Negative HNSW node ID");
        }
        const auto value = static_cast<uint32_t>(id);
        result.bytes[21] = static_cast<char>(value >> 24);
        result.bytes[22] = static_cast<char>(value >> 16);
        result.bytes[23] = static_cast<char>(value >> 8);
        result.bytes[24] = static_cast<char>(value);
        result.size = 25;
    }
    return result;
}
std::optional<std::string_view> LmdbStorage::get(
        std::string_view bytes, size_t minimum, size_t maximum) const {
    check();
    MDB_val key{bytes.size(), const_cast<char*>(bytes.data())}, value;
    const auto rc = mdb_get(txn_, table_, &key, &value);
    if (rc == MDB_NOTFOUND) {
        return std::nullopt;
    }
    status(rc);
    // Check the borrowed record BEFORE copying it: corrupt records must not
    // cause allocations proportional to an arbitrary database value's size.
    if (value.mv_size < minimum || value.mv_size > maximum) {
        throw std::runtime_error("Invalid HNSW record size");
    }
    // This private view is decoded immediately, without any intervening LMDB
    // operation or callback. Public metadata/node/vector results own all data;
    // no database pointer survives a mutation or escapes from the adapter.
    return std::string_view(
            static_cast<const char*>(value.mv_data), value.mv_size);
}
void LmdbStorage::put(std::string_view key_bytes, std::string_view bytes) {
    check();
    try {
        if (put_) {
            callback_active_ = true;
            try {
                put_(txn_, table_, key_bytes, bytes);
            } catch (...) {
                callback_active_ = false;
                throw;
            }
            callback_active_ = false;
        } else {
            MDB_val key{key_bytes.size(), const_cast<char*>(key_bytes.data())};
            MDB_val value{bytes.size(), const_cast<char*>(bytes.data())};
            status(mdb_put(txn_, table_, &key, &value, 0));
        }
    } catch (...) {
        invalidate();
        throw;
    }
}
void LmdbStorage::apply(const WriteBatch& batch) {
    check();
    if (mode_ == WriteMode::CallerTransaction) {
        Storage::apply(batch);
        return;
    }
    auto* parent = txn_;
    MDB_txn* child = nullptr;
    bool committing = false;
    try {
        unsigned flags = 0;
        status(mdb_env_get_flags(mdb_txn_env(parent), &flags));
        if (flags & MDB_WRITEMAP) {
            throw std::invalid_argument(
                    "HNSW LMDB savepoints require an environment without "
                    "MDB_WRITEMAP; explicitly use CallerTransaction mode");
        }
        status(mdb_txn_begin(mdb_txn_env(parent), parent, 0, &child));
        txn_ = child;
        // The callback-free child owns every batch mutation until merged into
        // the parent. The parent must not be used while its child is active.
        // Do not use Storage::apply here: its generic abort-required exception
        // describes the root transaction, whereas this child can be discarded.
        for (const auto& [id, vector] : batch.vectors) {
            put_vector(id, vector);
        }
        for (const auto& [id, node] : batch.nodes) {
            put_node(id, node);
        }
        put_metadata(batch.metadata);
        txn_ = parent;
        auto* committing_child = std::exchange(child, nullptr);
        committing = true;
        // LMDB consumes the child even on failure. A commit failure can mark
        // its parent failed, so it must not be advertised as recoverable.
        status(mdb_txn_commit(committing_child));
    } catch (...) {
        txn_ = parent;
        if (child) {
            mdb_txn_abort(child);
        }
        invalidate();
        std::throw_with_nested(MutationError(
                committing ? "HNSW savepoint commit failed: abort required"
                           : "HNSW savepoint rolled back; replace the adapter",
                committing));
    }
}

std::optional<Metadata> LmdbStorage::metadata() const {
    auto bytes = get(key('m'), metadata_bytes, metadata_bytes);
    if (!bytes) {
        return std::nullopt;
    }
    Decoder d{*bytes};
    if (d.u32() != 1) {
        throw std::runtime_error("Unsupported HNSW metadata version");
    }
    Metadata result;
    result.config.dimensions = d.u32();
    result.config.M = d.u32();
    result.config.ef_construction = d.u32();
    const auto metric = d.u32();
    if (metric != METRIC_L2 && metric != METRIC_INNER_PRODUCT) {
        throw std::runtime_error("Invalid persisted HNSW metric");
    }
    result.config.metric = static_cast<MetricType>(metric);
    result.config.seed = d.u64();
    result.next_id = d.u64();
    result.live_count = d.u64();
    result.entry_point = id_from(d.u32());
    result.max_level = id_from(d.u32());
    d.end();
    if (!valid_metadata(result)) {
        throw std::runtime_error("Invalid persisted HNSW metadata");
    }
    return result;
}
void LmdbStorage::put_metadata(const Metadata& m) {
    check();
    if (!valid_metadata(m)) {
        throw std::invalid_argument("Invalid HNSW metadata");
    }
    std::string s;
    s.reserve(metadata_bytes);
    u32(s, 1);
    u32(s, m.config.dimensions);
    u32(s, m.config.M);
    u32(s, m.config.ef_construction);
    u32(s, m.config.metric);
    u64(s, m.config.seed);
    u64(s, m.next_id);
    u64(s, m.live_count);
    u32(s, static_cast<uint32_t>(m.entry_point));
    u32(s, static_cast<uint32_t>(m.max_level));
    put(key('m'), s);
}
Node LmdbStorage::node(NodeId id) const {
    auto bytes = get(key('n', id), 12, maximum_node_bytes);
    if (!bytes) {
        throw std::runtime_error("Missing HNSW node");
    }
    Decoder d{*bytes};
    Node node;
    const auto deleted = d.u32();
    if (deleted > 1) {
        throw std::runtime_error("Invalid HNSW tombstone");
    }
    node.deleted = deleted;
    const auto levels = d.u32();
    if (!levels || levels > max_levels) {
        throw std::runtime_error("Invalid HNSW level count");
    }
    node.links.resize(levels);
    std::vector<NodeId> sorted;
    for (size_t level = 0; level < node.links.size(); ++level) {
        auto& links = node.links[level];
        const auto count = d.u32();
        const auto capacity = level ? max_neighbors : 2 * max_neighbors;
        if (count > capacity || count > d.bytes.size() / 4) {
            throw std::runtime_error("Invalid HNSW adjacency length");
        }
        links.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto neighbor = id_from(d.u32());
            if (neighbor < 0 || neighbor == id) {
                throw std::runtime_error("Invalid HNSW adjacency ID");
            }
            links.push_back(neighbor);
        }
        sorted.assign(links.begin(), links.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::runtime_error("Duplicate HNSW adjacency ID");
        }
    }
    d.end();
    return node;
}
void LmdbStorage::put_node(NodeId id, const Node& n) {
    check();
    if (n.links.empty() || n.links.size() > max_levels) {
        throw std::invalid_argument("Invalid HNSW level count");
    }
    std::string s;
    u32(s, n.deleted);
    u32(s, static_cast<uint32_t>(n.links.size()));
    std::vector<NodeId> sorted;
    for (size_t level = 0; level < n.links.size(); ++level) {
        const auto& links = n.links[level];
        if (links.size() > (level ? max_neighbors : 2 * max_neighbors)) {
            throw std::invalid_argument("Invalid HNSW adjacency length");
        }
        u32(s, static_cast<uint32_t>(links.size()));
        sorted.assign(links.begin(), links.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("Duplicate HNSW adjacency ID");
        }
        for (auto neighbor : links) {
            if (neighbor < 0 || neighbor == id) {
                throw std::invalid_argument("Invalid HNSW adjacency ID");
            }
            u32(s, neighbor);
        }
    }
    put(key('n', id), s);
}
std::vector<float> LmdbStorage::vector(NodeId id) const {
    auto bytes = get(key('v', id), 8, maximum_vector_bytes);
    if (!bytes) {
        throw std::runtime_error("Missing HNSW vector");
    }
    Decoder d{*bytes};
    auto count = d.u32();
    if (!count || count > max_dimensions || d.bytes.size() != 4ull * count) {
        throw std::runtime_error("Invalid HNSW vector length");
    }
    // The length was validated once above. Decode directly into a sized owned
    // result, using alignment-safe loads and a reduction over exponent bits.
    // No per-element decoder branches, push_back growth checks or potentially
    // trapping floating-point operations are needed to reject NaN/Inf.
    std::vector<float> result(count);
    const char* encoded = d.bytes.data();
    uint32_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, encoded + i * sizeof(bits), sizeof(bits));
        bits = from_big_endian(bits);
        nonfinite |= uint32_t((bits & 0x7f800000u) == 0x7f800000u);
        result[i] = std::bit_cast<float>(bits);
    }
    if (nonfinite) {
        throw std::runtime_error("Non-finite persisted HNSW vector");
    }
    return result;
}
void LmdbStorage::put_vector(NodeId id, const std::vector<float>& v) {
    check();
    if (v.empty() || v.size() > max_dimensions) {
        throw std::invalid_argument("Invalid HNSW vector length");
    }
    std::string s;
    s.reserve(4 + 4 * v.size());
    u32(s, static_cast<uint32_t>(v.size()));
    for (float value : v) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("HNSW vectors must be finite");
        }
        u32(s, std::bit_cast<uint32_t>(value));
    }
    put(key('v', id), s);
}
} // namespace faiss::transactional
