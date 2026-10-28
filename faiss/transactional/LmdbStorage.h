/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#pragma once

#include <faiss/transactional/HNSW.h>
#include <lmdb.h>
#include <array>
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <thread>

namespace faiss::transactional {

// Preserved as a nested exception in MutationError, so callers can distinguish
// MDB_MAP_FULL from other LMDB failures without parsing diagnostic text.
class LmdbError : public std::runtime_error {
   public:
    explicit LmdbError(int code)
            : std::runtime_error(
                      std::string("HNSW LMDB: ") + mdb_strerror(code)),
              code_(code) {}
    int code() const noexcept {
        return code_;
    }

   private:
    int code_;
};

/** LMDB implementation using a normal (non-DUPSORT) named table in the caller's
 * environment. Index and generation IDs partition the table. All objects must
 * be destroyed BEFORE their transaction ends; use a fresh adapter per txn.
 * An adapter is confined to its construction thread (including MDB_NOTLS).
 * Thread checks cannot detect an externally ended or reset transaction: its
 * lifetime remains the caller's responsibility.
 * Does not own/open/commit/abort the caller's environment, table, or parent
 * transaction. Savepoint mode creates and owns temporary child transactions.
 */
class LmdbStorage final : public Storage {
   public:
    // Opt-in format v2 groups little-endian float vectors into copy-on-write
    // blocks. New namespaces must be initialized via create/apply/put_metadata
    // before individual put_vector calls. Existing formats are never migrated.
    // Width is 1..4096 and encoded blocks are capped at 64 MiB.
    struct PackedVectors {
        bool enabled;
        // Zero chooses a power-of-two width up to approximately 256 KiB. It is
        // persisted; reopen detects it independently of constructor options.
        uint32_t vectors_per_block;
        PackedVectors(bool enabled = false, uint32_t vectors_per_block = 0)
                : enabled(enabled), vectors_per_block(vectors_per_block) {}
    };
    enum class WriteMode {
        // Callback-free batches use child transactions; callbacks require
        // aborting the caller transaction on failure.
        Automatic,
        CallerTransaction,
        Savepoint,
    };
    // Optional integration with a database's changelog. Must apply the put to
    // the SAME transaction before returning, and record it for replication.
    // Key/value views are only valid for the duration of the callback.
    // External changelog buffers/sinks must be discarded on caller abort.
    // A callback is incompatible with Savepoint mode: side effects outside
    // LMDB cannot be rolled back by aborting a child transaction. The callback
    // must not reenter this adapter or run index operations on this transaction
    // while the batch is only partially applied.
    using Put = std::function<
            void(MDB_txn*, MDB_dbi, std::string_view, std::string_view)>;
    LmdbStorage(
            MDB_txn* txn,
            MDB_dbi table,
            uint64_t index_id,
            uint64_t generation = 0,
            Put put = {},
            WriteMode mode = WriteMode::Automatic,
            PackedVectors packed = {});
    LmdbStorage(const LmdbStorage&) = delete;
    LmdbStorage& operator=(const LmdbStorage&) = delete;

    std::optional<Metadata> metadata() const override;
    Node node(NodeId id) const override;
    std::vector<float> vector(NodeId id) const override;
    void put_metadata(const Metadata& value) override;
    void put_node(NodeId id, const Node& value) override;
    void put_vector(NodeId id, const std::vector<float>& value) override;
    // Atomic batches in Savepoint mode. Failed child writes leave the parent
    // unchanged; child commit errors conservatively require caller abort.
    // Any failure invalidates this adapter. MDB_WRITEMAP does not support
    // savepoints; explicitly select CallerTransaction for those environments.
    void apply(const WriteBatch& batch) override;
    // The returned computer borrows mapped block pointers for this operation.
    // The transaction and adapter must outlive it. No mutation (including
    // writes through another adapter or raw LMDB) may occur while it exists.
    // A byte-bounded directory caches pointers, not the mapped vector payloads.
    std::unique_ptr<faiss::DistanceComputer> query_distance_computer(
            const Metadata& metadata,
            size_t cache_bytes,
            SearchStats* stats) const override;
    void invalidate() noexcept override {
        failed_.store(true, std::memory_order_relaxed);
    }

   private:
    MDB_txn* txn_;
    MDB_dbi table_;
    std::array<char, 20> prefix_{};
    Put put_;
    WriteMode mode_;
    PackedVectors packed_;
    struct Layout {
        uint32_t dimensions = 0;
        uint32_t width = 0; // zero is the legacy per-vector big-endian format
    };
    struct BlockView {
        const char* data = nullptr;
        uint32_t count = 0;
    };
    mutable std::optional<Layout> layout_;
    uint64_t epoch_ = 0;
    bool callback_active_ = false;
    const std::thread::id owner_ = std::this_thread::get_id();
    std::atomic<bool> failed_{false};
    void check() const;
    struct EncodedKey {
        std::array<char, 25> bytes;
        size_t size;
        operator std::string_view() const noexcept {
            return {bytes.data(), size};
        }
    };
    EncodedKey key(char kind, NodeId id = no_node) const;
    // Public record readers synchronously decode into owned results. The
    // packed query computer may retain views until the operation ends, with
    // no writes or callbacks; an epoch detects mutations through this adapter.
    std::optional<std::string_view> get(
            std::string_view key,
            size_t minimum,
            size_t maximum) const;
    void put(std::string_view key, std::string_view value);
    Layout layout() const;
    Layout planned_layout(const Metadata& metadata) const;
    std::optional<BlockView> block(NodeId id, const Layout& layout) const;
    std::optional<BlockView> live_block(NodeId id) const;
    void write_live(const WriteBatch& batch);
    void write_metadata(const Metadata& metadata, const Layout& layout);
    void write_batch(const WriteBatch& batch, const Layout& layout);
    void write_vectors(
            const std::map<NodeId, std::vector<float>>& vectors,
            const Layout& layout);
    class QueryDistanceComputer;
};
} // namespace faiss::transactional
