/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/FastSearchScratch.h>
#include <faiss/transactional/LmdbStorage.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>

namespace faiss::transactional {
namespace {
constexpr uint32_t max_dimensions = 1u << 20;
constexpr uint32_t max_neighbors = 4096;
constexpr uint32_t max_levels = 33;
constexpr size_t metadata_bytes = 52;
constexpr size_t packed_metadata_bytes = 56;
constexpr size_t block_header_bytes = 16;
constexpr size_t minimum_block_bytes = 4096;
constexpr size_t maximum_block_bytes = 64u << 20;
constexpr uint32_t maximum_block_width = 4096;
constexpr uint32_t packed_magic = 0x50563032; // PV02
constexpr uint32_t live_magic = 0x4c563032;   // LV02
constexpr uint32_t live_width = 4096;
constexpr size_t live_record_bytes = block_header_bytes + live_width / 8;
constexpr size_t maximum_node_bytes =
        8 + 4 * max_levels + 4 * max_neighbors * (max_levels + 1);
constexpr size_t maximum_vector_bytes = 4 + 4 * size_t(max_dimensions);
static_assert(
        sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
        "The transactional record format requires IEEE-754 binary32");
static_assert(
        std::endian::native == std::endian::little ||
                std::endian::native == std::endian::big,
        "The transactional record format requires uniform byte order");

uint32_t from_big_endian(uint32_t value) {
    if constexpr (std::endian::native == std::endian::little) {
        // Compilers recognize this portable C++20 expression as a byte swap;
        // unlike a decoder call per element it also permits loop vectorization.
        return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
                ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
    }
    return value;
}

bool valid_packed_layout(uint32_t dimensions, uint32_t width) {
    return dimensions && dimensions <= max_dimensions && width &&
            width <= maximum_block_width &&
            uint64_t(width) * dimensions * sizeof(float) <=
            maximum_block_bytes - block_header_bytes;
}
size_t packed_bytes(uint32_t dimensions, uint32_t count) {
    return std::max(
            minimum_block_bytes,
            block_header_bytes + size_t(dimensions) * count * sizeof(float));
}
uint32_t from_little_endian(uint32_t value) {
    if constexpr (std::endian::native == std::endian::big) {
        return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
                ((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
    }
    return value;
}
void decode_little(const char* bytes, float* output, size_t count) {
    uint32_t nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
        bits = from_little_endian(bits);
        nonfinite |= uint32_t((bits & 0x7f800000u) == 0x7f800000u);
        output[i] = std::bit_cast<float>(bits);
    }
    if (nonfinite)
        throw std::runtime_error("Non-finite persisted HNSW vector");
}
void encode_little(char* output, const std::vector<float>& vector) {
    for (float value : vector)
        if (!std::isfinite(value))
            throw std::invalid_argument("HNSW vectors must be finite");
    if constexpr (std::endian::native == std::endian::little) {
        std::memcpy(output, vector.data(), vector.size() * sizeof(float));
    } else {
        for (size_t i = 0; i < vector.size(); ++i) {
            const auto bits =
                    from_little_endian(std::bit_cast<uint32_t>(vector[i]));
            std::memcpy(output + i * sizeof(bits), &bits, sizeof(bits));
        }
    }
}

bool valid_metadata(const Metadata& m) {
    const auto& c = m.config;
    return c.dimensions && c.dimensions <= max_dimensions && c.M >= 2 &&
            c.M <= max_neighbors && c.ef_construction >= c.M &&
            c.ef_construction <= (1u << 24) &&
            (c.metric == METRIC_L2 || c.metric == METRIC_INNER_PRODUCT) &&
            m.next_id <= uint64_t(std::numeric_limits<NodeId>::max()) + 1 &&
            m.live_count <= m.next_id &&
            (m.next_id == 0 ? m.entry_point == no_node && m.max_level == -1
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
        WriteMode mode,
        PackedVectors packed)
        : txn_(txn),
          table_(table),
          put_(std::move(put)),
          mode_(mode == WriteMode::Automatic
                        ? (put_ ? WriteMode::CallerTransaction
                                : WriteMode::Savepoint)
                        : mode),
          packed_(packed) {
    if (!txn_) {
        throw std::invalid_argument("HNSW requires an active LMDB transaction");
    }
    if (mode_ != WriteMode::Savepoint &&
        mode_ != WriteMode::CallerTransaction) {
        throw std::invalid_argument("Invalid HNSW LMDB write mode");
    }
    if (packed_.vectors_per_block > maximum_block_width) {
        throw std::invalid_argument("Invalid HNSW packed vector block width");
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
        throw std::logic_error(
                "HNSW LMDB adapter used from a different thread");
    }
    if (callback_active_) {
        throw std::logic_error(
                "HNSW changelog callback cannot reenter its adapter");
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
        std::string_view bytes,
        size_t minimum,
        size_t maximum) const {
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
    // Public record readers decode immediately into owned values. The packed
    // query computer may retain these views only within one operation with
    // no writes or callbacks; its epoch guard rejects adapter mutations.
    return std::string_view(
            static_cast<const char*>(value.mv_data), value.mv_size);
}
void LmdbStorage::put(std::string_view key_bytes, std::string_view bytes) {
    check();
    ++epoch_;
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
    ++epoch_;
    if (mode_ == WriteMode::CallerTransaction) {
        try {
            const auto format = planned_layout(batch.metadata);
            write_batch(batch, format);
        } catch (...) {
            invalidate();
            std::throw_with_nested(MutationError(
                    "HNSW mutation failed; abort caller transaction", true));
        }
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
        const auto format = planned_layout(batch.metadata);
        write_batch(batch, format);
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
    auto bytes = get(key('m'), metadata_bytes, packed_metadata_bytes);
    if (!bytes) {
        layout_ = Layout{};
        return std::nullopt;
    }
    Decoder d{*bytes};
    const auto version = d.u32();
    if (version != 1 && version != 2) {
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
    const uint32_t width = version == 2 ? d.u32() : 0;
    d.end();
    if (!valid_metadata(result) ||
        (version == 2 &&
         !valid_packed_layout(result.config.dimensions, width))) {
        throw std::runtime_error("Invalid persisted HNSW metadata");
    }
    layout_ = Layout{result.config.dimensions, width};
    return result;
}
LmdbStorage::Layout LmdbStorage::layout() const {
    if (!layout_)
        metadata();
    return *layout_;
}
LmdbStorage::Layout LmdbStorage::planned_layout(const Metadata& m) const {
    if (!valid_metadata(m))
        throw std::invalid_argument("Invalid HNSW metadata");
    if (metadata()) {
        auto format = *layout_;
        if (format.width && format.dimensions != m.config.dimensions)
            throw std::invalid_argument("Cannot change packed HNSW dimensions");
        format.dimensions = m.config.dimensions;
        return format;
    }
    uint32_t width = 0;
    if (packed_.enabled) {
        width = packed_.vectors_per_block;
        if (!width)
            width = std::bit_floor(
                    std::clamp<uint32_t>(
                            ((256u << 10) - block_header_bytes) /
                                    (size_t(m.config.dimensions) *
                                     sizeof(float)),
                            1,
                            maximum_block_width));
        if (!valid_packed_layout(m.config.dimensions, width))
            throw std::invalid_argument(
                    "HNSW packed vector block exceeds 64 MiB");
    }
    return {m.config.dimensions, width};
}
void LmdbStorage::put_metadata(const Metadata& m) {
    check();
    write_metadata(m, planned_layout(m));
}
void LmdbStorage::write_metadata(const Metadata& m, const Layout& format) {
    std::string s;
    s.reserve(format.width ? packed_metadata_bytes : metadata_bytes);
    u32(s, format.width ? 2 : 1);
    u32(s, m.config.dimensions);
    u32(s, m.config.M);
    u32(s, m.config.ef_construction);
    u32(s, m.config.metric);
    u64(s, m.config.seed);
    u64(s, m.next_id);
    u64(s, m.live_count);
    u32(s, static_cast<uint32_t>(m.entry_point));
    u32(s, static_cast<uint32_t>(m.max_level));
    if (format.width)
        u32(s, format.width);
    put(key('m'), s);
    layout_ = format;
}
std::optional<LmdbStorage::BlockView> LmdbStorage::block(
        NodeId id,
        const Layout& format) const {
    if (!valid_packed_layout(format.dimensions, format.width))
        throw std::runtime_error("Invalid HNSW packed vector layout");
    auto bytes =
            get(key('b', id),
                minimum_block_bytes,
                packed_bytes(format.dimensions, format.width));
    if (!bytes)
        return std::nullopt;
    Decoder d{bytes->substr(0, block_header_bytes)};
    const auto magic = d.u32();
    const auto dimensions = d.u32();
    const auto count = d.u32();
    const auto width = d.u32();
    if (magic != packed_magic || dimensions != format.dimensions ||
        width != format.width || !count || count > width ||
        bytes->size() != packed_bytes(dimensions, count))
        throw std::runtime_error("Invalid HNSW packed vector block");
    return BlockView{bytes->data() + block_header_bytes, count};
}
void LmdbStorage::write_vectors(
        const std::map<NodeId, std::vector<float>>& vectors,
        const Layout& format) {
    if (!valid_packed_layout(format.dimensions, format.width))
        throw std::invalid_argument("Invalid HNSW packed vector layout");
    auto first = vectors.begin();
    while (first != vectors.end()) {
        if (first->first < 0)
            throw std::invalid_argument("Negative HNSW node ID");
        const NodeId block_id = uint32_t(first->first) / format.width;
        auto last = first;
        while (last != vectors.end() && last->first >= 0 &&
               uint32_t(last->first) / format.width == uint32_t(block_id))
            ++last;
        auto previous = block(block_id, format);
        uint32_t count = previous ? previous->count : 0;
        for (auto it = first; it != last; ++it) {
            if (it->second.size() != format.dimensions)
                throw std::invalid_argument(
                        "Invalid HNSW packed vector length");
            const auto slot = uint32_t(it->first) % format.width;
            if (slot > count)
                throw std::invalid_argument("Sparse HNSW packed vector block");
            if (slot == count)
                ++count;
        }
        // Own the entire replacement before any put can invalidate the old
        // mapped view. Each touched block is written once for the whole batch.
        std::string bytes;
        bytes.reserve(packed_bytes(format.dimensions, count));
        u32(bytes, packed_magic);
        u32(bytes, format.dimensions);
        u32(bytes, count);
        u32(bytes, format.width);
        bytes.resize(packed_bytes(format.dimensions, count), '\0');
        if (previous)
            std::memcpy(
                    bytes.data() + block_header_bytes,
                    previous->data,
                    size_t(previous->count) * format.dimensions *
                            sizeof(float));
        for (auto it = first; it != last; ++it)
            encode_little(
                    bytes.data() + block_header_bytes +
                            size_t(uint32_t(it->first) % format.width) *
                                    format.dimensions * sizeof(float),
                    it->second);
        put(key('b', block_id), bytes);
        first = last;
    }
}
std::optional<LmdbStorage::BlockView> LmdbStorage::live_block(NodeId id) const {
    auto bytes = get(key('l', id), live_record_bytes, live_record_bytes);
    if (!bytes)
        return std::nullopt;
    Decoder d{bytes->substr(0, block_header_bytes)};
    const auto magic = d.u32();
    const auto count = d.u32();
    const auto width = d.u32();
    const auto reserved = d.u32();
    if (magic != live_magic || !count || count > live_width ||
        width != live_width || reserved)
        throw std::runtime_error("Invalid HNSW live bitmap header");
    const char* data = bytes->data() + block_header_bytes;
    if (count % 8 &&
        (static_cast<unsigned char>(data[count / 8]) >> (count % 8)))
        throw std::runtime_error("HNSW live bitmap contains unallocated IDs");
    for (size_t i = (count + 7) / 8; i < live_width / 8; ++i)
        if (data[i])
            throw std::runtime_error(
                    "HNSW live bitmap contains unallocated IDs");
    return BlockView{data, count};
}
void LmdbStorage::write_live(const WriteBatch& batch) {
    const auto old = metadata();
    const uint64_t before = old ? old->next_id : 0;
    const uint64_t after = batch.metadata.next_id;
    if (after < before)
        throw std::invalid_argument("Cannot shrink packed HNSW ID range");
    // Every newly allocated ID needs a node and therefore a live bit. Existing
    // adjacency-only updates can leave bitmap records byte-for-byte unchanged.
    uint64_t next = before;
    if (next < after) {
        for (auto it = batch.nodes.lower_bound(static_cast<NodeId>(next));
             it != batch.nodes.end() && next < after;
             ++it) {
            if (it->first < 0 || uint64_t(it->first) != next)
                throw std::invalid_argument(
                        "Sparse HNSW live bitmap allocation");
            ++next;
        }
        if (next != after)
            throw std::invalid_argument("Missing newly allocated HNSW nodes");
    }
    auto first = batch.nodes.begin();
    while (first != batch.nodes.end()) {
        if (first->first < 0 || uint64_t(first->first) >= after)
            throw std::invalid_argument("HNSW live bitmap ID out of range");
        const NodeId chunk = uint32_t(first->first) / live_width;
        auto last = first;
        while (last != batch.nodes.end() && last->first >= 0 &&
               uint32_t(last->first) / live_width == uint32_t(chunk))
            ++last;
        const uint64_t base = uint64_t(chunk) * live_width;
        const uint32_t expected_before = before > base
                ? std::min<uint64_t>(live_width, before - base)
                : 0;
        const uint32_t count = std::min<uint64_t>(live_width, after - base);
        auto previous = live_block(chunk);
        if ((previous ? previous->count : 0) != expected_before)
            throw std::runtime_error("Missing or incomplete HNSW live bitmap");
        std::string bytes;
        bytes.reserve(live_record_bytes);
        u32(bytes, live_magic);
        u32(bytes, count);
        u32(bytes, live_width);
        u32(bytes, 0);
        bytes.resize(live_record_bytes, '\0');
        if (previous)
            std::memcpy(
                    bytes.data() + block_header_bytes,
                    previous->data,
                    live_width / 8);
        bool changed = !previous || count != previous->count;
        for (auto it = first; it != last; ++it) {
            if (uint64_t(it->first) >= after)
                throw std::invalid_argument("HNSW live bitmap ID out of range");
            const uint32_t slot = uint32_t(it->first) % live_width;
            auto& encoded = bytes[block_header_bytes + slot / 8];
            const uint8_t mask = uint8_t(1u << (slot % 8));
            const auto bits = static_cast<unsigned char>(encoded);
            const auto value = it->second.deleted ? bits | mask : bits & ~mask;
            changed |= value != bits;
            encoded = static_cast<char>(value);
        }
        if (changed)
            put(key('l', chunk), bytes);
        first = last;
    }
}
void LmdbStorage::write_batch(const WriteBatch& batch, const Layout& format) {
    if (format.width) {
        write_vectors(batch.vectors, format);
    } else {
        for (const auto& [id, value] : batch.vectors)
            put_vector(id, value);
    }
    for (const auto& [id, value] : batch.nodes)
        put_node(id, value);
    if (format.width)
        write_live(batch);
    write_metadata(batch.metadata, format);
}

class LmdbStorage::QueryDistanceComputer final : public faiss::DistanceComputer,
                                                 public QueryLiveAccess {
    const LmdbStorage& storage_;
    Layout format_;
    uint64_t next_id_;
    MetricType metric_;
    uint64_t epoch_;
    const float* query_ = nullptr;
    SearchStats* stats_;
    struct CacheEntry {
        uint32_t id = UINT32_MAX;
        BlockView view;
    };
    // A fixed direct-mapped directory is bounded exactly by the supplied
    // allowance. Collisions evict only pointers, never mapped vector data.
    std::vector<CacheEntry> cache_;
    mutable std::vector<CacheEntry> live_cache_;
    bool complete_directory_ = false;
    bool complete_live_directory_ = false;
    unsigned width_shift_ = 32;
    std::array<std::vector<float>, 4> scratch_;

    void check() const {
        storage_.check();
        if (storage_.epoch_ != epoch_)
            throw std::logic_error("HNSW query view used after a mutation");
    }
    const float* vector(faiss::idx_t id, size_t scratch_slot) {
        if (id < 0 || uint64_t(id) >= next_id_)
            throw std::out_of_range("HNSW query vector ID out of range");
        const auto block_id = width_shift_ < 32 ? uint32_t(id) >> width_shift_
                                                : uint32_t(id) / format_.width;
        BlockView view;
        CacheEntry* entry = cache_.empty()
                ? nullptr
                : &cache_[complete_directory_ ? block_id
                                              : block_id % cache_.size()];
        if (entry && entry->id == block_id) {
            view = entry->view;
        } else {
            if (stats_)
                ++stats_->vector_block_reads;
            auto found = storage_.block(block_id, format_);
            const auto expected = std::min<uint64_t>(
                    format_.width,
                    next_id_ - uint64_t(block_id) * format_.width);
            if (!found || found->count != expected)
                throw std::runtime_error(
                        "Missing or incomplete HNSW packed block");
            view = *found;
            if (entry)
                *entry = CacheEntry{block_id, view};
        }
        const auto slot = width_shift_ < 32 ? uint32_t(id) & (format_.width - 1)
                                            : uint32_t(id) % format_.width;
        const char* bytes =
                view.data + size_t(slot) * format_.dimensions * sizeof(float);
        if constexpr (std::endian::native == std::endian::little) {
            if (reinterpret_cast<uintptr_t>(bytes) % alignof(float) == 0)
                return reinterpret_cast<const float*>(bytes);
        }
        // Separate scratch slots preserve all four operands through batch-4,
        // including unaligned mappings and hosts with the opposite byte order.
        auto& owned = scratch_[scratch_slot];
        owned.resize(format_.dimensions);
        decode_little(bytes, owned.data(), owned.size());
        return owned.data();
    }
    static float finite(float distance) {
        if (!std::isfinite(distance))
            throw std::runtime_error("Non-finite HNSW distance");
        return distance;
    }
    float distance(const float* a, const float* b) const {
        return finite(
                metric_ == METRIC_L2
                        ? faiss::fvec_L2sqr(a, b, format_.dimensions)
                        : -faiss::fvec_inner_product(a, b, format_.dimensions));
    }

   public:
    QueryDistanceComputer(
            const LmdbStorage& storage,
            Layout format,
            const Metadata& metadata,
            size_t cache_bytes,
            SearchStats* stats)
            : storage_(storage),
              format_(format),
              next_id_(metadata.next_id),
              metric_(metadata.config.metric),
              epoch_(storage.epoch_),
              stats_(stats) {
        const auto blocks = (next_id_ + format.width - 1) / format.width;
        const auto live_blocks = (next_id_ + live_width - 1) / live_width;
        // Both directories together stay within the caller's allowance.
        // Small live directories fit entirely; at most one eighth is reserved
        // for them when both indexes exceed the budget.
        live_cache_.resize(
                std::min<uint64_t>(
                        live_blocks, cache_bytes / 8 / sizeof(CacheEntry)));
        cache_bytes -= live_cache_.size() * sizeof(CacheEntry);
        cache_.resize(
                std::min<uint64_t>(blocks, cache_bytes / sizeof(CacheEntry)));
        complete_directory_ = cache_.size() == blocks;
        complete_live_directory_ = live_cache_.size() == live_blocks;
        if (std::has_single_bit(format.width))
            width_shift_ = std::countr_zero(format.width);
    }
    bool deleted(NodeId id) const override {
        check();
        if (id < 0 || uint64_t(id) >= next_id_)
            throw std::out_of_range("HNSW live bitmap ID out of range");
        const auto chunk = uint32_t(id) / live_width;
        CacheEntry* entry = live_cache_.empty()
                ? nullptr
                : &live_cache_
                          [complete_live_directory_
                                   ? chunk
                                   : chunk % live_cache_.size()];
        BlockView view;
        if (entry && entry->id == chunk) {
            view = entry->view;
        } else {
            if (stats_)
                ++stats_->live_block_reads;
            auto found = storage_.live_block(chunk);
            const auto expected = std::min<uint64_t>(
                    live_width, next_id_ - uint64_t(chunk) * live_width);
            if (!found || found->count != expected)
                throw std::runtime_error(
                        "Missing or incomplete HNSW live bitmap");
            view = *found;
            if (entry)
                *entry = CacheEntry{chunk, view};
        }
        const auto slot = uint32_t(id) % live_width;
        return (static_cast<unsigned char>(view.data[slot / 8]) >> (slot % 8)) &
                1;
    }
    void set_query(const float* query) override {
        check();
        if (!query)
            throw std::invalid_argument("Null HNSW query");
        for (uint32_t i = 0; i < format_.dimensions; ++i)
            if (!std::isfinite(query[i]))
                throw std::invalid_argument("HNSW query must be finite");
        query_ = query;
    }
    float operator()(faiss::idx_t id) override {
        check();
        if (!query_)
            throw std::logic_error("HNSW query was not set");
        return distance(query_, vector(id, 0));
    }
    float symmetric_dis(faiss::idx_t a, faiss::idx_t b) override {
        check();
        const auto* first = vector(a, 0);
        const auto* second = vector(b, 1);
        return distance(first, second);
    }
    void distances_batch_4(
            faiss::idx_t i0,
            faiss::idx_t i1,
            faiss::idx_t i2,
            faiss::idx_t i3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        check();
        if (!query_)
            throw std::logic_error("HNSW query was not set");
        const auto* p0 = vector(i0, 0);
        const auto* p1 = vector(i1, 1);
        const auto* p2 = vector(i2, 2);
        const auto* p3 = vector(i3, 3);
        if (metric_ == METRIC_L2) {
            faiss::fvec_L2sqr_batch_4(
                    query_, p0, p1, p2, p3, format_.dimensions, d0, d1, d2, d3);
        } else {
            faiss::fvec_inner_product_batch_4(
                    query_, p0, p1, p2, p3, format_.dimensions, d0, d1, d2, d3);
            d0 = -d0;
            d1 = -d1;
            d2 = -d2;
            d3 = -d3;
        }
        finite(d0);
        finite(d1);
        finite(d2);
        finite(d3);
    }
};
std::unique_ptr<faiss::DistanceComputer> LmdbStorage::query_distance_computer(
        const Metadata& metadata,
        size_t cache_bytes,
        SearchStats* stats) const {
    check();
    const auto format = layout();
    if (!format.width)
        return nullptr;
    if (!valid_metadata(metadata) ||
        metadata.config.dimensions != format.dimensions)
        throw std::invalid_argument("Invalid HNSW query metadata");
    return std::make_unique<QueryDistanceComputer>(
            *this, format, metadata, cache_bytes, stats);
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
        if (!detail::unique_nonnegative_ids(links)) {
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
    for (size_t level = 0; level < n.links.size(); ++level) {
        const auto& links = n.links[level];
        if (links.size() > (level ? max_neighbors : 2 * max_neighbors)) {
            throw std::invalid_argument("Invalid HNSW adjacency length");
        }
        u32(s, static_cast<uint32_t>(links.size()));
        if (!detail::unique_nonnegative_ids(links)) {
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
    check();
    const auto format = layout();
    if (format.width) {
        if (id < 0)
            throw std::invalid_argument("Negative HNSW node ID");
        auto value = block(id / format.width, format);
        if (!value || uint32_t(id) % format.width >= value->count)
            throw std::runtime_error("Missing HNSW packed vector");
        std::vector<float> result(format.dimensions);
        decode_little(
                value->data +
                        size_t(uint32_t(id) % format.width) *
                                format.dimensions * sizeof(float),
                result.data(),
                result.size());
        return result;
    }
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
    const auto format = layout();
    if (!format.dimensions && packed_.enabled)
        throw std::logic_error(
                "Initialize packed HNSW metadata before individual vector writes");
    if (format.width) {
        write_vectors({{id, v}}, format);
        return;
    }
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
