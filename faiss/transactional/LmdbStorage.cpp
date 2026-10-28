/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/LmdbStorage.h>

#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>

namespace faiss::transactional {
namespace {
void status(int rc) {
    if (rc != MDB_SUCCESS) {
        throw std::runtime_error(std::string("HNSW LMDB: ") + mdb_strerror(rc));
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
        Put put)
        : txn_(txn), table_(table), prefix_("TH01"), put_(std::move(put)) {
    if (!txn_) {
        throw std::invalid_argument("HNSW requires an active LMDB transaction");
    }
    unsigned flags = 0;
    status(mdb_dbi_flags(txn_, table_, &flags));
    if (flags != 0) {
        throw std::invalid_argument(
                "HNSW requires an ordinary byte-key LMDB table");
    }
    u64(prefix_, index_id);
    u64(prefix_, generation);
}

void LmdbStorage::check() const {
    if (failed_) {
        throw std::runtime_error("HNSW transaction failed: abort required");
    }
}
std::string LmdbStorage::key(char kind, NodeId id) const {
    auto result = prefix_;
    result.push_back(kind);
    if (kind != 'm') {
        if (id < 0) {
            throw std::invalid_argument("Negative HNSW node ID");
        }
        u32(result, static_cast<uint32_t>(id));
    }
    return result;
}
std::optional<std::string> LmdbStorage::get(std::string_view bytes) const {
    check();
    MDB_val key{bytes.size(), const_cast<char*>(bytes.data())}, value;
    const auto rc = mdb_get(txn_, table_, &key, &value);
    if (rc == MDB_NOTFOUND) {
        return std::nullopt;
    }
    status(rc);
    // Own the result; no borrowed LMDB pointers escape across mutations.
    return std::string(static_cast<const char*>(value.mv_data), value.mv_size);
}
void LmdbStorage::put(std::string_view key_bytes, std::string_view bytes) {
    check();
    try {
        if (put_) {
            put_(txn_, table_, key_bytes, bytes);
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
std::optional<Metadata> LmdbStorage::metadata() const {
    auto bytes = get(key('m'));
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
    return result;
}
void LmdbStorage::put_metadata(const Metadata& m) {
    std::string s;
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
    auto bytes = get(key('n', id));
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
    if (!levels || levels > 33) {
        throw std::runtime_error("Invalid HNSW level count");
    }
    node.links.resize(levels);
    for (auto& links : node.links) {
        const auto count = d.u32();
        if (count > 8192 || count > d.bytes.size() / 4) {
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
    }
    d.end();
    return node;
}
void LmdbStorage::put_node(NodeId id, const Node& n) {
    if (n.links.empty() || n.links.size() > 33) {
        throw std::invalid_argument("Invalid HNSW level count");
    }
    std::string s;
    u32(s, n.deleted);
    u32(s, static_cast<uint32_t>(n.links.size()));
    for (const auto& links : n.links) {
        if (links.size() > 8192) {
            throw std::invalid_argument("Invalid HNSW adjacency length");
        }
        u32(s, static_cast<uint32_t>(links.size()));
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
    auto bytes = get(key('v', id));
    if (!bytes) {
        throw std::runtime_error("Missing HNSW vector");
    }
    Decoder d{*bytes};
    auto count = d.u32();
    if (!count || count > (1u << 20) || d.bytes.size() != 4ull * count) {
        throw std::runtime_error("Invalid HNSW vector length");
    }
    std::vector<float> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(std::bit_cast<float>(d.u32()));
    }
    return result;
}
void LmdbStorage::put_vector(NodeId id, const std::vector<float>& v) {
    if (v.empty() || v.size() > (1u << 20)) {
        throw std::invalid_argument("Invalid HNSW vector length");
    }
    std::string s;
    s.reserve(4 + 4 * v.size());
    u32(s, static_cast<uint32_t>(v.size()));
    for (float value : v) {
        u32(s, std::bit_cast<uint32_t>(value));
    }
    put(key('v', id), s);
}
} // namespace faiss::transactional
