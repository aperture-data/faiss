/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#pragma once

#include <faiss/transactional/HNSW.h>
#include <lmdb.h>
#include <functional>
#include <string>
#include <string_view>

namespace faiss::transactional {

/** LMDB implementation using a normal (non-DUPSORT) named table in the caller's
 * environment. Index and generation IDs partition the table. All objects must
 * be destroyed BEFORE their transaction ends; use a fresh adapter per txn.
 * Does not own/open/commit/abort the environment, table, or transaction.
 */
class LmdbStorage final : public Storage {
   public:
    // Optional integration with a database's changelog. Must apply the put to
    // the SAME transaction before returning, and record it for replication.
    // Key/value views are only valid for the duration of the callback.
    using Put = std::function<
            void(MDB_txn*, MDB_dbi, std::string_view, std::string_view)>;
    LmdbStorage(
            MDB_txn* txn,
            MDB_dbi table,
            uint64_t index_id,
            uint64_t generation = 0,
            Put put = {});
    LmdbStorage(const LmdbStorage&) = delete;
    LmdbStorage& operator=(const LmdbStorage&) = delete;

    std::optional<Metadata> metadata() const override;
    Node node(NodeId id) const override;
    std::vector<float> vector(NodeId id) const override;
    void put_metadata(const Metadata& value) override;
    void put_node(NodeId id, const Node& value) override;
    void put_vector(NodeId id, const std::vector<float>& value) override;
    void invalidate() noexcept override {
        failed_ = true;
    }

   private:
    MDB_txn* txn_;
    MDB_dbi table_;
    std::string prefix_;
    Put put_;
    bool failed_ = false;
    void check() const;
    std::string key(char kind, NodeId id = no_node) const;
    std::optional<std::string> get(std::string_view key) const;
    void put(std::string_view key, std::string_view value);
};
} // namespace faiss::transactional
