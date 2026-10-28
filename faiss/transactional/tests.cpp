/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/transactional/HNSW.h>
#include <faiss/transactional/LmdbStorage.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ft = faiss::transactional;
namespace {
void ok(int status) {
    if (status) {
        throw std::runtime_error(mdb_strerror(status));
    }
}
struct Tx {
    MDB_txn* value = nullptr;
    Tx(MDB_env* env, bool readonly) {
        ok(mdb_txn_begin(env, nullptr, readonly ? MDB_RDONLY : 0, &value));
    }
    ~Tx() {
        if (value)
            mdb_txn_abort(value);
    }
    void commit() {
        auto* txn = std::exchange(value, nullptr);
        ok(mdb_txn_commit(txn));
    }
};
class TransactionalHNSW : public ::testing::Test {
   protected:
    MDB_env* env = nullptr;
    MDB_dbi table{}, graph{};
    std::string path;
    void SetUp() override {
        char pattern[] = "/tmp/faiss-lmdb-test-XXXXXX";
        char* created = mkdtemp(pattern);
        ASSERT_NE(created, nullptr);
        path = created;
        open();
    }
    void open() {
        ok(mdb_env_create(&env));
        ok(mdb_env_set_maxdbs(env, 4));
        ok(mdb_env_set_mapsize(env, 128 * 1024 * 1024));
        ok(mdb_env_open(env, path.c_str(), MDB_NOTLS, 0600));
        Tx tx(env, false);
        ok(mdb_dbi_open(tx.value, "hnsw", MDB_CREATE, &table));
        ok(mdb_dbi_open(tx.value, "graph", MDB_CREATE, &graph));
        tx.commit();
    }
    void TearDown() override {
        if (env)
            mdb_env_close(env);
        std::filesystem::remove_all(path);
    }
    static ft::Config config(unsigned dim = 2) {
        ft::Config c;
        c.dimensions = dim;
        c.M = 8;
        c.ef_construction = 64;
        return c;
    }
    void initialize() {
        Tx tx(env, false);
        {
            ft::LmdbStorage storage(tx.value, table, 1);
            ft::HNSW h(storage);
            h.create(config());
            const float vectors[]{0, 0, 1, 0, 2, 0};
            EXPECT_EQ(h.add(3, vectors), (std::vector<ft::NodeId>{0, 1, 2}));
        }
        tx.commit();
    }
};

TEST_F(TransactionalHNSW, ReadYourWritesCommitAndReopen) {
    initialize();
    mdb_env_close(env);
    env = nullptr;
    open();
    Tx tx(env, true);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    EXPECT_EQ(h.metadata().live_count, 3);
    const float query[]{1, 0};
    auto results = h.search(query, 3);
    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].id, 1);
    EXPECT_FLOAT_EQ(results[0].distance, 0);
}

TEST_F(TransactionalHNSW, AbortRollsBackGraphAndIndexTogether) {
    initialize();
    {
        Tx tx(env, false);
        ft::LmdbStorage storage(tx.value, table, 1);
        ft::HNSW h(storage);
        const float vector[]{3, 0};
        EXPECT_EQ(h.add(1, vector)[0], 3);
        EXPECT_TRUE(h.erase(0));
        std::string key = "descriptor-3", bytes = "created";
        MDB_val k{key.size(), key.data()}, v{bytes.size(), bytes.data()};
        ok(mdb_put(tx.value, graph, &k, &v, 0));
        EXPECT_EQ(h.search(vector, 1)[0].id, 3);
    }
    Tx tx(env, false);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    EXPECT_EQ(h.metadata().next_id, 3);
    EXPECT_EQ(h.metadata().live_count, 3);
    EXPECT_FALSE(storage.node(0).deleted);
    std::string key = "descriptor-3";
    MDB_val k{key.size(), key.data()}, value;
    EXPECT_EQ(mdb_get(tx.value, graph, &k, &value), MDB_NOTFOUND);
    const float vector[]{3, 0};
    EXPECT_EQ(h.add(1, vector)[0], 3);
}

TEST_F(TransactionalHNSW, OldSnapshotSurvivesInsertAndDelete) {
    initialize();
    Tx old(env, true);
    ft::LmdbStorage before(old.value, table, 1);
    ft::HNSW old_index(before);
    {
        Tx writer(env, false);
        {
            ft::LmdbStorage storage(writer.value, table, 1);
            ft::HNSW h(storage);
            EXPECT_TRUE(h.erase(1));
            const float vector[]{1, 0};
            EXPECT_EQ(h.add(1, vector)[0], 3);
        }
        writer.commit();
    }
    const float query[]{1, 0};
    EXPECT_EQ(old_index.search(query, 1)[0].id, 1);
    EXPECT_EQ(old_index.metadata().next_id, 3);
    Tx current(env, true);
    ft::LmdbStorage after(current.value, table, 1);
    ft::HNSW new_index(after);
    EXPECT_EQ(new_index.search(query, 1)[0].id, 3);
    EXPECT_EQ(new_index.metadata().next_id, 4);
}

TEST_F(TransactionalHNSW, TombstonesFiltersAndEmptyResults) {
    initialize();
    Tx tx(env, false);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    const float query[]{0, 0};
    faiss::IDSelectorRange only_two(2, 3);
    auto found = h.search(query, 10, 64, &only_two);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 2);
    const auto entry = h.metadata().entry_point;
    EXPECT_TRUE(h.erase(entry));
    EXPECT_FALSE(h.erase(entry));
    EXPECT_EQ(h.search(query, 10).size(), 2);
    EXPECT_FALSE(h.erase(-1));
    EXPECT_FALSE(h.erase(100));
    for (int i = 0; i < 3; ++i)
        h.erase(i);
    EXPECT_TRUE(h.search(query, 10).empty());
    EXPECT_EQ(h.metadata().live_count, 0);
    const float vector[]{4, 0};
    EXPECT_EQ(h.add(1, vector)[0], 3);
    ASSERT_EQ(h.search(vector, 10).size(), 1);
    EXPECT_EQ(h.search(vector, 10)[0].id, 3);
}

TEST_F(TransactionalHNSW, IndexAndGenerationIsolation) {
    Tx tx(env, false);
    const float vector[]{1, 2};
    for (auto [index, generation] : {std::pair{1, 0}, {1, 1}, {2, 0}}) {
        ft::LmdbStorage storage(tx.value, table, index, generation);
        ft::HNSW h(storage);
        h.create(config());
        EXPECT_EQ(h.add(1, vector)[0], 0);
    }
    ft::LmdbStorage untouched(tx.value, table, 2, 1);
    EXPECT_FALSE(untouched.metadata());
}

TEST_F(TransactionalHNSW, InvalidInputsDoNotModifyState) {
    initialize();
    Tx tx(env, false);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    EXPECT_THROW(h.create(config()), std::invalid_argument);
    EXPECT_THROW(h.add(1, nullptr), std::invalid_argument);
    const float bad[]{0, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(h.add(1, bad), std::invalid_argument);
    EXPECT_THROW(h.search(bad, 1), std::invalid_argument);
    EXPECT_TRUE(h.add(0, nullptr).empty());
    EXPECT_TRUE(h.search(nullptr, 0).empty());
    EXPECT_THROW(
            h.add(std::numeric_limits<size_t>::max(), bad),
            std::overflow_error);
    EXPECT_EQ(h.metadata().next_id, 3);
    ft::LmdbStorage other(tx.value, table, 2);
    ft::HNSW empty(other);
    auto c = config();
    c.M = 1;
    EXPECT_THROW(empty.create(c), std::invalid_argument);
    EXPECT_FALSE(other.metadata());
}

TEST_F(TransactionalHNSW, MutationFailureRequiresAbortAndDoesNotPublish) {
    initialize();
    {
        Tx tx(env, false);
        int writes = 0;
        ft::LmdbStorage storage(
                tx.value,
                table,
                1,
                0,
                [&](MDB_txn* txn,
                    MDB_dbi db,
                    std::string_view key,
                    std::string_view value) {
                    if (++writes == 3)
                        throw std::runtime_error("injected failure");
                    MDB_val k{key.size(), const_cast<char*>(key.data())};
                    MDB_val v{value.size(), const_cast<char*>(value.data())};
                    ok(mdb_put(txn, db, &k, &v, 0));
                });
        ft::HNSW h(storage);
        const float vector[]{4, 0};
        EXPECT_THROW(h.add(1, vector), std::runtime_error);
        EXPECT_EQ(writes, 3);
        EXPECT_THROW(h.metadata(), std::runtime_error);
        EXPECT_THROW(h.search(vector, 1), std::runtime_error);
    }
    Tx tx(env, true);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    EXPECT_EQ(h.metadata().next_id, 3);
    EXPECT_THROW(storage.vector(3), std::runtime_error);
}

TEST_F(TransactionalHNSW, ChangelogReplayReproducesTopology) {
    initialize();
    std::vector<std::pair<std::string, std::string>> log;
    {
        Tx tx(env, false);
        {
            ft::LmdbStorage storage(
                    tx.value,
                    table,
                    1,
                    0,
                    [&](MDB_txn* txn,
                        MDB_dbi db,
                        std::string_view key,
                        std::string_view value) {
                        log.emplace_back(key, value);
                        MDB_val k{key.size(), const_cast<char*>(key.data())};
                        MDB_val v{
                                value.size(), const_cast<char*>(value.data())};
                        ok(mdb_put(txn, db, &k, &v, 0));
                    });
            ft::HNSW h(storage);
            const float vector[]{5, 0};
            h.add(1, vector);
        }
        // Deliberately abort, then replay the captured mutations.
    }
    ASSERT_FALSE(log.empty());
    {
        Tx tx(env, false);
        for (auto& [key, value] : log) {
            MDB_val k{key.size(), key.data()}, v{value.size(), value.data()};
            ok(mdb_put(tx.value, table, &k, &v, 0));
        }
        tx.commit();
    }
    Tx tx(env, true);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    const float query[]{5, 0};
    EXPECT_EQ(h.metadata().next_id, 4);
    EXPECT_EQ(h.search(query, 1)[0].id, 3);
}

TEST_F(TransactionalHNSW, ReadOnlyStorageRejectsMutation) {
    initialize();
    Tx tx(env, true);
    ft::LmdbStorage storage(tx.value, table, 1);
    ft::HNSW h(storage);
    EXPECT_THROW(h.erase(0), std::runtime_error);
    EXPECT_THROW(h.metadata(), std::runtime_error);
}

TEST_F(TransactionalHNSW, RejectsTruncatedMetadataAndUnsupportedTables) {
    Tx tx(env, false);
    MDB_dbi duplicates;
    ok(mdb_dbi_open(
            tx.value, "duplicates", MDB_CREATE | MDB_DUPSORT, &duplicates));
    EXPECT_THROW(
            ft::LmdbStorage(tx.value, duplicates, 1), std::invalid_argument);
    ft::LmdbStorage storage(
            tx.value,
            table,
            1,
            0,
            [&](MDB_txn* txn,
                MDB_dbi db,
                std::string_view key,
                std::string_view) {
                std::string bad = "bad";
                MDB_val k{key.size(), const_cast<char*>(key.data())},
                        v{bad.size(), bad.data()};
                ok(mdb_put(txn, db, &k, &v, 0));
            });
    ft::HNSW h(storage);
    h.create(config());
    EXPECT_THROW(h.metadata(), std::runtime_error);
}

TEST_F(TransactionalHNSW, DeterministicIncrementalConstruction) {
    Tx tx(env, false);
    ft::LmdbStorage a(tx.value, table, 1), b(tx.value, table, 2);
    ft::HNSW first(a), second(b);
    first.create(config());
    second.create(config());
    std::vector<float> vectors;
    for (int i = 0; i < 80; ++i) {
        vectors.push_back(std::sin(static_cast<float>(i)));
        vectors.push_back(std::cos(static_cast<float>(i)));
    }
    first.add(80, vectors.data());
    for (int i = 0; i < 80; ++i)
        second.add(1, vectors.data() + 2 * i);
    EXPECT_EQ(first.metadata().entry_point, second.metadata().entry_point);
    for (int i = 0; i < 80; ++i)
        EXPECT_EQ(a.node(i).links, b.node(i).links);
}

TEST_F(TransactionalHNSW, RecallAgainstExactSearchL2InnerProductAndCosine) {
    constexpr int n = 600, dim = 24, queries = 30, k = 10;
    std::mt19937 rng(719);
    std::normal_distribution<float> sample;
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> vectors(n * dim), query(queries * dim);
        for (auto& x : vectors)
            x = sample(rng);
        for (auto& x : query)
            x = sample(rng);
        if (mode == 2) {
            for (auto* data : {&vectors, &query}) {
                for (size_t offset = 0; offset < data->size(); offset += dim) {
                    float norm = 0;
                    for (int j = 0; j < dim; ++j)
                        norm += (*data)[offset + j] * (*data)[offset + j];
                    for (int j = 0; j < dim; ++j)
                        (*data)[offset + j] /= std::sqrt(norm);
                }
            }
        }
        const auto metric =
                mode == 0 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
        faiss::IndexFlat exact(dim, metric);
        exact.add(n, vectors.data());
        std::vector<faiss::idx_t> labels(queries * k);
        std::vector<float> distances(queries * k);
        exact.search(queries, query.data(), k, distances.data(), labels.data());
        Tx tx(env, false);
        ft::LmdbStorage storage(tx.value, table, mode + 1);
        ft::HNSW h(storage);
        auto c = config(dim);
        c.metric = metric;
        c.M = 16;
        c.ef_construction = 100;
        h.create(c);
        h.add(n, vectors.data());
        int hits = 0;
        for (int q = 0; q < queries; ++q) {
            auto results = h.search(query.data() + q * dim, k, 128);
            ASSERT_EQ(results.size(), k);
            std::unordered_set<faiss::idx_t> wanted(
                    labels.begin() + q * k, labels.begin() + (q + 1) * k);
            for (const auto& r : results) {
                hits += wanted.contains(r.id) ? 1 : 0;
                auto pos = std::find(
                        labels.begin() + q * k,
                        labels.begin() + (q + 1) * k,
                        r.id);
                if (pos != labels.begin() + (q + 1) * k) {
                    EXPECT_NEAR(
                            r.distance, distances[pos - labels.begin()], 1e-4);
                }
            }
        }
        EXPECT_GE(double(hits) / (queries * k), 0.95) << "metric mode " << mode;
        // Every persisted edge references an existing node at the right level.
        for (int i = 0; i < n; ++i) {
            auto node = storage.node(i);
            for (size_t level = 0; level < node.links.size(); ++level) {
                EXPECT_LE(node.links[level].size(), level ? c.M : 2 * c.M);
                std::unordered_set<ft::NodeId> unique;
                for (auto neighbor : node.links[level]) {
                    EXPECT_TRUE(unique.insert(neighbor).second);
                    EXPECT_NE(neighbor, i);
                    EXPECT_GT(storage.node(neighbor).links.size(), level);
                }
            }
        }
    }
}

TEST_F(TransactionalHNSW, MapFullRequiresAbort) {
    ok(mdb_env_set_mapsize(env, 64 * 1024));
    {
        Tx tx(env, false);
        ft::LmdbStorage storage(tx.value, table, 1);
        ft::HNSW h(storage);
        h.create(config(65536));
        std::vector<float> vector(65536, 0);
        EXPECT_THROW(h.add(1, vector.data()), std::runtime_error);
        EXPECT_THROW(h.metadata(), std::runtime_error);
    }
    Tx tx(env, true);
    ft::LmdbStorage storage(tx.value, table, 1);
    EXPECT_FALSE(storage.metadata());
}

TEST_F(TransactionalHNSW, ExistingFaissHnswRemainsAvailable) {
    faiss::IndexHNSWFlat index(2, 8);
    const float vectors[]{0, 0, 1, 0, 2, 0};
    index.add(3, vectors);
    faiss::idx_t label;
    float distance;
    index.search(1, vectors + 2, 1, &distance, &label);
    EXPECT_EQ(label, 1);
    EXPECT_EQ(distance, 0);
}
} // namespace
