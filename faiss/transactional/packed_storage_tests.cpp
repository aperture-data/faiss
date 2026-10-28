/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/HNSW.h>
#include <faiss/transactional/LmdbStorage.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

namespace ft = faiss::transactional;
namespace {
void checked(int rc) {
    if (rc != MDB_SUCCESS)
        throw std::runtime_error(mdb_strerror(rc));
}
struct Transaction {
    MDB_txn* txn = nullptr;
    explicit Transaction(MDB_env* env, bool readonly = false) {
        checked(mdb_txn_begin(env, nullptr, readonly ? MDB_RDONLY : 0, &txn));
    }
    ~Transaction() {
        if (txn)
            mdb_txn_abort(txn);
    }
    void commit() {
        checked(mdb_txn_commit(std::exchange(txn, nullptr)));
    }
};
void set32(std::string& s, size_t at, uint32_t v) {
    for (size_t i = 0; i < 4; ++i)
        s.at(at + i) = static_cast<char>(v >> (24 - i * 8));
}
std::string key(char kind, uint32_t id = 0) {
    std::string s(21 + (kind == 'm' ? 0 : 4), '\0');
    s.replace(0, 4, "TH01");
    s[11] = 1;
    s[20] = kind;
    if (kind != 'm')
        set32(s, 21, id);
    return s;
}
void raw_put(MDB_txn* txn, MDB_dbi db, std::string_view k, std::string_view v) {
    MDB_val mk{k.size(), const_cast<char*>(k.data())};
    MDB_val mv{v.size(), const_cast<char*>(v.data())};
    checked(mdb_put(txn, db, &mk, &mv, 0));
}
std::string raw_get(MDB_txn* txn, MDB_dbi db, std::string k) {
    MDB_val mk{k.size(), k.data()}, mv{};
    checked(mdb_get(txn, db, &mk, &mv));
    return {static_cast<const char*>(mv.mv_data), mv.mv_size};
}
class PackedStorage : public ::testing::Test {
   protected:
    MDB_env* env = nullptr;
    MDB_dbi db{};
    std::string path;
    void SetUp() override {
        char pattern[] = "/tmp/faiss-packed-storage-XXXXXX";
        const char* result = mkdtemp(pattern);
        ASSERT_NE(result, nullptr);
        path = result;
        checked(mdb_env_create(&env));
        checked(mdb_env_set_maxdbs(env, 2));
        checked(mdb_env_set_mapsize(env, 32u << 20));
        checked(mdb_env_open(env, path.c_str(), MDB_NOTLS, 0600));
        Transaction tx(env);
        checked(mdb_dbi_open(tx.txn, "hnsw", MDB_CREATE, &db));
        tx.commit();
    }
    void TearDown() override {
        if (env)
            mdb_env_close(env);
        std::filesystem::remove_all(path);
    }
    static ft::WriteBatch batch(uint32_t n, uint32_t d = 3) {
        ft::WriteBatch b;
        b.metadata.config.dimensions = d;
        b.metadata.next_id = b.metadata.live_count = n;
        if (n) {
            b.metadata.entry_point = 0;
            b.metadata.max_level = 0;
        }
        for (uint32_t i = 0; i < n; ++i) {
            auto& v = b.vectors[i];
            for (uint32_t j = 0; j < d; ++j)
                v.push_back(float(i * 3 + j) / 8);
            b.nodes[i] = {false, {{}}};
        }
        return b;
    }
    static constexpr auto automatic = ft::LmdbStorage::WriteMode::Automatic;
};

TEST_F(PackedStorage, RoundTripAndReopenAutodetectBothFormats) {
    Transaction tx(env);
    auto b = batch(7);
    ft::LmdbStorage packed(tx.txn, db, 1, 0, {}, automatic, {true, 3});
    packed.apply(b);
    auto m = raw_get(tx.txn, db, key('m'));
    ASSERT_EQ(m.size(), 56);
    EXPECT_EQ(static_cast<unsigned char>(m[3]), 2);
    EXPECT_EQ(static_cast<unsigned char>(m[55]), 3);
    EXPECT_EQ(raw_get(tx.txn, db, key('b', 2)).size(), 4096);
    ft::LmdbStorage reopened(tx.txn, db, 1);
    for (const auto& [id, vector] : b.vectors)
        EXPECT_EQ(reopened.vector(id), vector);
    EXPECT_THROW(reopened.vector(7), std::runtime_error);
    EXPECT_THROW(reopened.vector(-1), std::invalid_argument);
    ft::LmdbStorage legacy(tx.txn, db, 2);
    legacy.apply(b);
    ft::LmdbStorage legacy_optin(tx.txn, db, 2, 0, {}, automatic, {true, 3});
    EXPECT_EQ(legacy_optin.vector(6), b.vectors[6]);
    EXPECT_EQ(
            legacy_optin.query_distance_computer(b.metadata, 4096, nullptr),
            nullptr);
}

TEST_F(PackedStorage, BatchFourAndScalarDistancesBothMetricsAndCacheBudgets) {
    Transaction tx(env);
    const float q[] = {1.5f, -1.25f, 0.75f};
    for (auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        auto b = batch(8);
        b.metadata.config.metric = metric;
        ft::LmdbStorage storage(
                tx.txn, db, metric + 10, 0, {}, automatic, {true, 2});
        storage.apply(b);
        for (size_t cache : {size_t(0), size_t(32), size_t(1 << 20)}) {
            ft::SearchStats stats;
            auto dc =
                    storage.query_distance_computer(b.metadata, cache, &stats);
            ASSERT_NE(dc, nullptr);
            EXPECT_THROW((*dc)(0), std::logic_error);
            dc->set_query(q);
            float d[4];
            dc->distances_batch_4(0, 2, 4, 7, d[0], d[1], d[2], d[3]);
            const int ids[] = {0, 2, 4, 7};
            for (int i = 0; i < 4; ++i) {
                float expected = 0;
                for (int j = 0; j < 3; ++j)
                    expected += metric == faiss::METRIC_L2
                            ? (q[j] - b.vectors[ids[i]][j]) *
                                    (q[j] - b.vectors[ids[i]][j])
                            : -q[j] * b.vectors[ids[i]][j];
                EXPECT_FLOAT_EQ(d[i], expected);
                EXPECT_FLOAT_EQ((*dc)(ids[i]), expected);
            }
            EXPECT_EQ(stats.vector_block_reads, cache >= 4096 ? 4 : 8);
            EXPECT_EQ(stats.vector_reads, 0); // front end owns logical counts
            EXPECT_THROW((*dc)(8), std::out_of_range);
            EXPECT_THROW((*dc)(-1), std::out_of_range);
            EXPECT_THROW(dc->set_query(nullptr), std::invalid_argument);
            EXPECT_TRUE(std::isfinite(dc->symmetric_dis(0, 7)));
        }
    }
}

TEST_F(PackedStorage, ActiveComputerRejectsMutationAndInvalidation) {
    Transaction tx(env);
    auto b = batch(4);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
    storage.apply(b);
    auto dc = storage.query_distance_computer(b.metadata, 4096, nullptr);
    dc->set_query(b.vectors[0].data());
    EXPECT_FLOAT_EQ((*dc)(0), 0);
    storage.put_vector(0, {9, 9, 9});
    EXPECT_THROW((*dc)(0), std::logic_error);
    dc = storage.query_distance_computer(b.metadata, 4096, nullptr);
    dc->set_query(b.vectors[0].data());
    storage.apply(b);
    EXPECT_THROW(dc->symmetric_dis(0, 1), std::logic_error);
    dc = storage.query_distance_computer(b.metadata, 4096, nullptr);
    storage.invalidate();
    EXPECT_THROW(dc->set_query(b.vectors[0].data()), std::runtime_error);
}

TEST_F(PackedStorage, OldSnapshotKeepsBorrowedBlocksAcrossWriterCommit) {
    auto b = batch(4);
    {
        Transaction tx(env);
        ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
        storage.apply(b);
        tx.commit();
    }
    Transaction reader(env, true);
    ft::LmdbStorage old(reader.txn, db, 1);
    auto dc = old.query_distance_computer(*old.metadata(), 4096, nullptr);
    dc->set_query(b.vectors[0].data());
    EXPECT_FLOAT_EQ((*dc)(0), 0);
    {
        Transaction writer(env);
        ft::LmdbStorage storage(writer.txn, db, 1);
        storage.put_vector(0, {100, 100, 100});
        writer.commit();
    }
    EXPECT_FLOAT_EQ((*dc)(0), 0);
    EXPECT_EQ(old.vector(0), b.vectors[0]);
    Transaction latest(env, true);
    ft::LmdbStorage current(latest.txn, db, 1);
    EXPECT_EQ(current.vector(0), (std::vector<float>{100, 100, 100}));
}

TEST_F(PackedStorage, CallbackWritesEachBlockOnceAndLogReplays) {
    auto b = batch(9);
    std::vector<std::pair<std::string, std::string>> log;
    {
        Transaction tx(env);
        ft::LmdbStorage storage(
                tx.txn,
                db,
                1,
                0,
                [&](MDB_txn* t,
                    MDB_dbi table,
                    std::string_view k,
                    std::string_view v) {
                    raw_put(t, table, k, v);
                    log.emplace_back(k, v);
                },
                automatic,
                {true, 4});
        storage.apply(b);
        size_t blocks = 0;
        for (const auto& [k, v] : log)
            blocks += k[20] == 'b';
        EXPECT_EQ(blocks, 3);
        EXPECT_EQ(log.back().first, key('m'));
        // Abort source and replay exact records into a fresh transaction.
    }
    Transaction replay(env);
    for (const auto& [k, v] : log)
        raw_put(replay.txn, db, k, v);
    ft::LmdbStorage target(replay.txn, db, 1);
    EXPECT_EQ(target.metadata()->next_id, 9);
    for (const auto& [id, v] : b.vectors)
        EXPECT_EQ(target.vector(id), v);
    auto dc = target.query_distance_computer(*target.metadata(), 4096, nullptr);
    auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
    ASSERT_NE(live, nullptr);
    for (const auto& [id, v] : b.vectors)
        EXPECT_FALSE(live->deleted(id));
}

TEST_F(PackedStorage, SavepointMapFullDuringNodeWritesPreservesParent) {
    checked(mdb_env_set_mapsize(env, 256u << 10));
    {
        Transaction tx(env);
        raw_put(tx.txn, db, "graph", "prior mutation");
        ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 4096});
        try {
            // Three small vector blocks fit comfortably in 256 KiB. The
            // 10,000 node keys cannot, so failure occurs after node writes
            // have begun inside the savepoint rather than during vectors.
            storage.apply(batch(10000, 1));
            FAIL() << "Expected map full";
        } catch (const ft::MutationError& error) {
            EXPECT_FALSE(error.abort_required());
        }
        EXPECT_EQ(raw_get(tx.txn, db, "graph"), "prior mutation");
        ft::LmdbStorage fresh(tx.txn, db, 1);
        EXPECT_FALSE(fresh.metadata());
        tx.commit();
    }
    Transaction reader(env, true);
    ft::LmdbStorage persisted(reader.txn, db, 1);
    EXPECT_FALSE(persisted.metadata());
    MDB_val k{key('b').size(), nullptr}, v{};
    auto block_key = key('b');
    k.mv_data = block_key.data();
    EXPECT_EQ(mdb_get(reader.txn, db, &k, &v), MDB_NOTFOUND);
}

TEST_F(PackedStorage, RejectsSparseWritesDimensionsAndOversizedBlocks) {
    Transaction tx(env);
    auto b = batch(2);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 4});
    EXPECT_THROW(storage.put_vector(0, {1, 2, 3}), std::logic_error);
    storage.apply(b);
    EXPECT_THROW(storage.put_vector(3, {1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(storage.put_vector(1, {1, 2}), std::invalid_argument);
    EXPECT_THROW(
            storage.put_vector(1, {1, 2, INFINITY}), std::invalid_argument);
    storage.put_vector(2, {1, 2, 3});
    EXPECT_EQ(storage.vector(0), b.vectors[0]);
    EXPECT_EQ(storage.vector(2), (std::vector<float>{1, 2, 3}));
    auto m = b.metadata;
    m.config.dimensions = 4;
    EXPECT_THROW(storage.put_metadata(m), std::invalid_argument);
    ft::LmdbStorage too_large(tx.txn, db, 2, 0, {}, automatic, {true, 4096});
    m = batch(0).metadata;
    m.config.dimensions = 1u << 20;
    EXPECT_THROW(too_large.put_metadata(m), std::invalid_argument);
}

TEST_F(PackedStorage, LittleEndianFloatPatternsAndCorruptionChecks) {
    Transaction tx(env);
    auto b = batch(1, 5);
    const uint32_t bits[] = {0, 0x80000000u, 0x3f800000u, 1u, 0x7f7fffffu};
    for (size_t i = 0; i < 5; ++i)
        b.vectors[0][i] = std::bit_cast<float>(bits[i]);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 4});
    storage.apply(b);
    auto original = raw_get(tx.txn, db, key('b'));
    auto decoded = storage.vector(0);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(std::bit_cast<uint32_t>(decoded[i]), bits[i]);
        for (size_t j = 0; j < 4; ++j)
            EXPECT_EQ(
                    static_cast<unsigned char>(original[16 + i * 4 + j]),
                    (bits[i] >> (j * 8)) & 255);
    }
    for (const auto& [offset, value] :
         {std::pair<size_t, uint32_t>{0, 0}, {4, 6}, {8, 0}, {8, 5}, {12, 3}}) {
        auto broken = original;
        set32(broken, offset, value);
        raw_put(tx.txn, db, key('b'), broken);
        EXPECT_THROW(storage.vector(0), std::runtime_error);
    }
    auto broken = original;
    broken.push_back('\0');
    raw_put(tx.txn, db, key('b'), broken);
    EXPECT_THROW(storage.vector(0), std::runtime_error);
    broken = original;
    broken[18] = static_cast<char>(0x80);
    broken[19] = static_cast<char>(0x7f); // little-endian +Inf
    raw_put(tx.txn, db, key('b'), broken);
    EXPECT_THROW(storage.vector(0), std::runtime_error);
    auto dc = storage.query_distance_computer(b.metadata, 4096, nullptr);
    const float query[5]{};
    dc->set_query(query);
    EXPECT_THROW((*dc)(0), std::runtime_error);
}

TEST_F(PackedStorage, HnswCreateInsertUpdateExactAndRollback) {
    {
        Transaction tx(env);
        ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 3});
        ft::HNSW index(storage);
        ft::Config config;
        config.dimensions = 3;
        index.create(config);
        const float vectors[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        index.add(4, vectors);
        ft::SearchOptions exact;
        exact.exact = true;
        auto result = index.search(vectors, 4, exact);
        ASSERT_EQ(result.size(), 4);
        EXPECT_EQ(result[0].id, 0);
        const float replacement[] = {0, 1, 2};
        EXPECT_EQ(index.update(3, replacement), 4);
        result = index.search(vectors, 5, exact);
        EXPECT_EQ(result.size(), 4);
        EXPECT_EQ(index.validate().live_nodes, 4);
        // All blocks and metadata are discarded with caller abort.
    }
    Transaction reader(env, true);
    ft::LmdbStorage storage(reader.txn, db, 1);
    EXPECT_FALSE(storage.metadata());
}

TEST_F(PackedStorage, LiveBitmapsPreserveSnapshotsAndDoNotRewriteVectors) {
    auto b = batch(4100, 1);
    {
        Transaction tx(env);
        {
            ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true});
            storage.apply(b);
        }
        tx.commit();
    }
    Transaction reader(env, true);
    ft::LmdbStorage old(reader.txn, db, 1);
    ft::SearchStats old_stats;
    auto old_dc =
            old.query_distance_computer(*old.metadata(), 4096, &old_stats);
    auto* old_live = dynamic_cast<ft::QueryLiveAccess*>(old_dc.get());
    ASSERT_NE(old_live, nullptr);
    EXPECT_FALSE(old_live->deleted(0));
    EXPECT_FALSE(old_live->deleted(4099));
    EXPECT_EQ(old_stats.live_block_reads, 2);
    {
        Transaction writer(env);
        std::vector<char> writes;
        {
            ft::LmdbStorage storage(
                    writer.txn,
                    db,
                    1,
                    0,
                    [&](MDB_txn* t,
                        MDB_dbi table,
                        std::string_view k,
                        std::string_view v) {
                        writes.push_back(k[20]);
                        raw_put(t, table, k, v);
                    });
            ft::WriteBatch deletion;
            deletion.metadata = b.metadata;
            deletion.metadata.live_count -= 2;
            deletion.nodes[0] = deletion.nodes[4099] = {true, {{}}};
            storage.apply(deletion);
            EXPECT_EQ(std::count(writes.begin(), writes.end(), 'l'), 2);
            EXPECT_EQ(std::count(writes.begin(), writes.end(), 'b'), 0);
            writes.clear();
            storage.apply(deletion);
            EXPECT_EQ(std::count(writes.begin(), writes.end(), 'l'), 0);
            EXPECT_EQ(raw_get(writer.txn, db, key('l')).size(), 528);
        }
        writer.commit();
    }
    EXPECT_FALSE(old_live->deleted(0));
    EXPECT_FALSE(old_live->deleted(4099));
    EXPECT_EQ(old_stats.live_block_reads, 2);
    Transaction latest(env, true);
    ft::LmdbStorage fresh(latest.txn, db, 1);
    for (size_t budget : {size_t(0), size_t(4096)}) {
        ft::SearchStats stats;
        auto dc = fresh.query_distance_computer(
                *fresh.metadata(), budget, &stats);
        auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
        ASSERT_NE(live, nullptr);
        EXPECT_TRUE(live->deleted(0));
        EXPECT_TRUE(live->deleted(4099));
        EXPECT_FALSE(live->deleted(4095));
        EXPECT_FALSE(live->deleted(4096));
        EXPECT_EQ(stats.live_block_reads, budget ? 2 : 4);
        EXPECT_THROW(live->deleted(-1), std::out_of_range);
        EXPECT_THROW(live->deleted(4100), std::out_of_range);
    }
}

TEST_F(PackedStorage, RejectsMissingMalformedAndIncompleteLiveBitmaps) {
    Transaction tx(env);
    auto b = batch(4);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
    storage.apply(b);
    const auto original = raw_get(tx.txn, db, key('l'));
    auto fails = [&] {
        auto dc = storage.query_distance_computer(b.metadata, 4096, nullptr);
        auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
        ASSERT_NE(live, nullptr);
        EXPECT_THROW(live->deleted(0), std::runtime_error);
    };
    for (const auto& [offset, value] :
         {std::pair<size_t, uint32_t>{0, 0},
          {4, 0},
          {4, 3},
          {4, 4097},
          {8, 0},
          {12, 1}}) {
        auto bytes = original;
        set32(bytes, offset, value);
        raw_put(tx.txn, db, key('l'), bytes);
        fails();
    }
    for (size_t offset : {size_t(16), size_t(527)}) {
        auto bytes = original;
        bytes[offset] = static_cast<char>(0x80);
        raw_put(tx.txn, db, key('l'), bytes);
        fails();
    }
    auto bytes = original + '\0';
    raw_put(tx.txn, db, key('l'), bytes);
    fails();
    auto k = key('l');
    MDB_val mk{k.size(), k.data()};
    checked(mdb_del(tx.txn, db, &mk, nullptr));
    fails();
    raw_put(tx.txn, db, key('l'), original);
    EXPECT_EQ(ft::HNSW(storage).validate().live_nodes, 4);
    bytes = original;
    bytes[16] = 1; // valid allocated bit, inconsistent with the node record
    raw_put(tx.txn, db, key('l'), bytes);
    EXPECT_THROW(ft::HNSW(storage).validate(), std::runtime_error);
}

TEST_F(PackedStorage, LiveBitmapUpdateRollbackAndEpochGuard) {
    {
        Transaction tx(env);
        {
            ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
            ft::HNSW index(storage);
            ft::Config config;
            config.dimensions = 3;
            index.create(config);
            const float values[] = {1, 2, 3, 4, 5, 6};
            index.add(2, values);
        }
        tx.commit();
    }
    {
        Transaction tx(env);
        ft::LmdbStorage storage(tx.txn, db, 1);
        auto dc = storage.query_distance_computer(
                *storage.metadata(), 4096, nullptr);
        auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
        ASSERT_NE(live, nullptr);
        EXPECT_FALSE(live->deleted(0));
        ft::HNSW index(storage);
        const float replacement[] = {7, 8, 9};
        EXPECT_EQ(index.update(0, replacement), 2);
        EXPECT_THROW(live->deleted(0), std::logic_error);
        dc = storage.query_distance_computer(
                *storage.metadata(), 4096, nullptr);
        live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
        EXPECT_TRUE(live->deleted(0));
        EXPECT_FALSE(live->deleted(2));
        EXPECT_EQ(index.validate().live_nodes, 2);
        // Caller abort must undo both replacement allocation and old tombstone.
    }
    Transaction reader(env, true);
    ft::LmdbStorage storage(reader.txn, db, 1);
    EXPECT_EQ(storage.metadata()->next_id, 2);
    auto dc = storage.query_distance_computer(*storage.metadata(), 0, nullptr);
    auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
    ASSERT_NE(live, nullptr);
    EXPECT_FALSE(live->deleted(0));
}

TEST_F(PackedStorage, MapFullAfterNodeAndBitmapWritesRequiresCallerAbort) {
    checked(mdb_env_set_mapsize(env, 512u << 10));
    auto b = batch(4);
    {
        Transaction tx(env);
        {
            ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
            storage.apply(b);
        }
        tx.commit();
    }
    {
        Transaction tx(env);
        bool bitmap_written = false;
        bool map_full = false;
        ft::LmdbStorage storage(
                tx.txn,
                db,
                1,
                0,
                [&](MDB_txn* t,
                    MDB_dbi table,
                    std::string_view k,
                    std::string_view v) {
                    raw_put(t, table, k, v);
                    if (k[20] == 'l') {
                        bitmap_written = true;
                        // Fault injection: exhaust real LMDB pages after both
                        // the node and bitmap writes have actually succeeded.
                        const std::string payload(4096, 'x');
                        for (size_t i = 0;; ++i) {
                            auto fill_key = "fault-fill-" + std::to_string(i);
                            MDB_val mk{fill_key.size(), fill_key.data()};
                            MDB_val mv{
                                    payload.size(),
                                    const_cast<char*>(payload.data())};
                            const auto rc = mdb_put(t, table, &mk, &mv, 0);
                            if (rc != MDB_SUCCESS) {
                                map_full = rc == MDB_MAP_FULL;
                                throw ft::LmdbError(rc);
                            }
                        }
                    }
                });
        ft::WriteBatch deletion;
        deletion.metadata = b.metadata;
        --deletion.metadata.live_count;
        deletion.nodes[0] = {true, {{}}};
        try {
            storage.apply(deletion);
            FAIL() << "Expected map full";
        } catch (const ft::MutationError& error) {
            EXPECT_TRUE(error.abort_required());
        }
        EXPECT_TRUE(bitmap_written);
        EXPECT_TRUE(map_full);
        EXPECT_THROW(storage.metadata(), std::runtime_error);
        // Aborting the caller discards both written node and bitmap records.
    }
    Transaction reader(env, true);
    ft::LmdbStorage storage(reader.txn, db, 1);
    EXPECT_EQ(storage.metadata()->live_count, 4);
    EXPECT_FALSE(storage.node(0).deleted);
    auto dc =
            storage.query_distance_computer(*storage.metadata(), 4096, nullptr);
    auto* live = dynamic_cast<ft::QueryLiveAccess*>(dc.get());
    ASSERT_NE(live, nullptr);
    EXPECT_FALSE(live->deleted(0));
}

TEST_F(PackedStorage, AdministrativeValidationChecksPhysicalBlockCounts) {
    Transaction tx(env);
    auto b = batch(4);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 8});
    storage.apply(b);
    ft::HNSW index(storage);
    EXPECT_EQ(index.validate().live_nodes, 4);
    const auto original = raw_get(tx.txn, db, key('b'));
    ASSERT_EQ(original.size(), 4096);
    for (uint32_t count : {3u, 5u}) {
        auto malformed = original;
        set32(malformed, 8, count);
        raw_put(tx.txn, db, key('b'), malformed);
        // Minimum-size padding means the record's byte length stays valid;
        // low-level vector reads still work for a populated slot.
        EXPECT_EQ(storage.vector(0), b.vectors[0]);
        EXPECT_THROW(index.validate(), std::runtime_error);
    }
    auto block_key = key('b');
    MDB_val mk{block_key.size(), block_key.data()};
    checked(mdb_del(tx.txn, db, &mk, nullptr));
    EXPECT_THROW(index.validate(), std::runtime_error);
    raw_put(tx.txn, db, key('b'), original);
    const auto original_live = raw_get(tx.txn, db, key('l'));
    auto malformed_live = original_live;
    set32(malformed_live, 4, 5);
    raw_put(tx.txn, db, key('l'), malformed_live);
    EXPECT_THROW(index.validate(), std::runtime_error);
    auto live_key = key('l');
    mk = {live_key.size(), live_key.data()};
    checked(mdb_del(tx.txn, db, &mk, nullptr));
    EXPECT_THROW(index.validate(), std::runtime_error);
    raw_put(tx.txn, db, key('l'), original_live);
    EXPECT_EQ(index.validate().live_nodes, 4);
}

TEST_F(PackedStorage, PhysicalBlockValidationHonorsCancellation) {
    Transaction tx(env);
    auto b = batch(9);
    ft::LmdbStorage storage(tx.txn, db, 1, 0, {}, automatic, {true, 2});
    storage.apply(b);
    int calls = 0;
    EXPECT_THROW(
            storage.validate_storage(b.metadata, [&] { return ++calls == 2; }),
            ft::SearchCancelled);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(ft::HNSW(storage).validate().live_nodes, 9);
}
} // namespace
