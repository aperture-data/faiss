/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/HNSW.h>
#include <faiss/transactional/LmdbStorage.h>
#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <utility>

namespace ft = faiss::transactional;
namespace {
void checked(int rc) {
    if (rc != MDB_SUCCESS) {
        throw std::runtime_error(mdb_strerror(rc));
    }
}
bool contains_lmdb_error(const std::exception& error, int code) {
    if (const auto* lmdb = dynamic_cast<const ft::LmdbError*>(&error)) {
        return lmdb->code() == code;
    }
    try {
        std::rethrow_if_nested(error);
    } catch (const std::exception& nested) {
        return contains_lmdb_error(nested, code);
    } catch (...) {
    }
    return false;
}
struct Transaction {
    MDB_txn* txn = nullptr;
    explicit Transaction(MDB_env* env, bool readonly = false) {
        checked(mdb_txn_begin(env, nullptr, readonly ? MDB_RDONLY : 0, &txn));
    }
    ~Transaction() {
        if (txn) {
            mdb_txn_abort(txn);
        }
    }
    void commit() {
        checked(mdb_txn_commit(std::exchange(txn, nullptr)));
    }
};
void append32(std::string& bytes, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<char>(value >> shift));
    }
}
void set32(std::string& bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        bytes.at(offset + i) = static_cast<char>(value >> (24 - 8 * i));
    }
}
std::string record_key(char kind, ft::NodeId id = 0) {
    std::string result = "TH01";
    append32(result, 0);
    append32(result, 1); // index 1
    append32(result, 0);
    append32(result, 0); // generation 0
    result.push_back(kind);
    if (kind != 'm') {
        append32(result, id);
    }
    return result;
}
void raw_put(MDB_txn* txn, MDB_dbi table, std::string key, std::string bytes) {
    MDB_val k{key.size(), key.data()}, v{bytes.size(), bytes.data()};
    checked(mdb_put(txn, table, &k, &v, 0));
}
std::string raw_get(MDB_txn* txn, MDB_dbi table, std::string key) {
    MDB_val k{key.size(), key.data()}, v{};
    checked(mdb_get(txn, table, &k, &v));
    return {static_cast<const char*>(v.mv_data), v.mv_size};
}
class StorageEdges : public ::testing::Test {
 protected:
    MDB_env* env = nullptr;
    MDB_dbi table{};
    std::string path;
    void SetUp() override {
        char pattern[] = "/tmp/faiss-storage-edge-XXXXXX";
        const auto directory = mkdtemp(pattern);
        ASSERT_NE(directory, nullptr);
        path = directory;
        checked(mdb_env_create(&env));
        checked(mdb_env_set_maxdbs(env, 2));
        checked(mdb_env_set_mapsize(env, 32 * 1024 * 1024));
        checked(mdb_env_open(env, path.c_str(), MDB_NOTLS, 0600));
        Transaction tx(env);
        checked(mdb_dbi_open(tx.txn, "hnsw", MDB_CREATE, &table));
        tx.commit();
    }
    void TearDown() override {
        if (env) {
            mdb_env_close(env);
        }
        std::filesystem::remove_all(path);
    }
    static ft::Metadata initial() {
        ft::Metadata result;
        result.config.dimensions = 2;
        return result;
    }
};

TEST_F(StorageEdges, RejectsCrossThreadAccessBeforeCallingLmdbOrCallback) {
    Transaction tx(env);
    std::atomic<int> callbacks{0};
    ft::LmdbStorage storage(
            tx.txn, table, 1, 0,
            [&](MDB_txn*, MDB_dbi, std::string_view, std::string_view) {
                ++callbacks;
            });
    std::thread other([&] {
        EXPECT_THROW(storage.metadata(), std::logic_error);
        EXPECT_THROW(storage.node(0), std::logic_error);
        EXPECT_THROW(storage.vector(0), std::logic_error);
        EXPECT_THROW(storage.put_metadata(initial()), std::logic_error);
        EXPECT_THROW(storage.put_vector(0, {0, 1}), std::logic_error);
        EXPECT_THROW(storage.put_node(0, {false, {{}}}), std::logic_error);
    });
    other.join();
    EXPECT_EQ(callbacks, 0);
    EXPECT_FALSE(storage.metadata());
}

TEST_F(StorageEdges, InvalidMetadataNeverReachesStorage) {
    Transaction tx(env);
    ft::LmdbStorage storage(tx.txn, table, 1);
    auto original = initial();
    storage.put_metadata(original);
    auto bytes = raw_get(tx.txn, table, record_key('m'));
    EXPECT_EQ(bytes.size(), 52);
    for (int mode = 0; mode < 8; ++mode) {
        auto invalid = original;
        switch (mode) {
            case 0: invalid.config.dimensions = 0; break;
            case 1: invalid.config.M = 1; break;
            case 2: invalid.config.ef_construction = 1; break;
            case 3: invalid.config.metric = faiss::METRIC_L1; break;
            case 4: invalid.live_count = 1; break;
            case 5: invalid.entry_point = 0; break;
            case 6: invalid.max_level = 0; break;
            case 7: invalid.next_id = UINT64_MAX; break;
        }
        EXPECT_THROW(storage.put_metadata(invalid), std::invalid_argument);
        EXPECT_EQ(raw_get(tx.txn, table, record_key('m')), bytes);
    }
    auto full = initial();
    full.next_id = uint64_t(INT32_MAX) + 1;
    full.entry_point = INT32_MAX;
    full.max_level = 32;
    full.live_count = full.next_id;
    storage.put_metadata(full);
    EXPECT_EQ(storage.metadata()->next_id, full.next_id);
}

TEST_F(StorageEdges, RejectsMalformedMetadataAndUnsupportedVersions) {
    Transaction tx(env);
    ft::LmdbStorage storage(tx.txn, table, 1);
    storage.put_metadata(initial());
    const auto valid = raw_get(tx.txn, table, record_key('m'));
    for (size_t size : {size_t(0), size_t(51), size_t(53), size_t(10000)}) {
        raw_put(tx.txn, table, record_key('m'), std::string(size, '\0'));
        EXPECT_THROW(storage.metadata(), std::runtime_error) << size;
    }
    for (const auto [offset, value] : {
                 std::pair<size_t, uint32_t>{0, 2}, {4, 0}, {8, 1},
                 {12, 1}, {16, UINT32_MAX}, {28, UINT32_MAX},
                 {36, 1}, {44, 0}, {48, 0}}) {
        auto malformed = valid;
        set32(malformed, offset, value);
        raw_put(tx.txn, table, record_key('m'), malformed);
        EXPECT_THROW(storage.metadata(), std::runtime_error) << offset;
    }
    raw_put(tx.txn, table, record_key('m'), valid);
    EXPECT_EQ(storage.metadata()->config.dimensions, 2);
}

TEST_F(StorageEdges, ValidatesVectorWireFormatAndPreservesFiniteFloatBits) {
    Transaction tx(env);
    ft::LmdbStorage storage(tx.txn, table, 1);
    std::vector<float> original{
            -0.0f, std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::max(), -3.25f};
    storage.put_vector(0, original);
    const auto decoded = storage.vector(0);
    ASSERT_EQ(decoded.size(), original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(std::bit_cast<uint32_t>(decoded[i]),
                  std::bit_cast<uint32_t>(original[i]));
    }
    const auto valid = raw_get(tx.txn, table, record_key('v'));
    EXPECT_EQ(static_cast<unsigned char>(valid[3]), original.size());
    EXPECT_EQ(static_cast<unsigned char>(valid[4]), 0x80); // -0 in big endian
    for (float invalid : {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_THROW(storage.put_vector(0, {invalid}), std::invalid_argument);
        auto malformed = valid;
        set32(malformed, 4, std::bit_cast<uint32_t>(invalid));
        raw_put(tx.txn, table, record_key('v'), malformed);
        EXPECT_THROW(storage.vector(0), std::runtime_error);
    }
    for (uint32_t count : {0u, 3u, 1u << 21}) {
        auto malformed = valid;
        set32(malformed, 0, count);
        raw_put(tx.txn, table, record_key('v'), malformed);
        EXPECT_THROW(storage.vector(0), std::runtime_error);
    }
    raw_put(tx.txn, table, record_key('v'), std::string((4u << 20) + 5, '\0'));
    EXPECT_THROW(storage.vector(0), std::runtime_error);
}

TEST_F(StorageEdges, RejectsInvalidAdjacencyWithoutPublishingIt) {
    Transaction tx(env);
    ft::LmdbStorage storage(tx.txn, table, 1);
    const ft::Node original{false, {{1, 2}, {2}}};
    storage.put_node(0, original);
    const auto valid = raw_get(tx.txn, table, record_key('n'));
    for (const auto& node : std::vector<ft::Node>{
                 {false, {}}, {false, {{0}}}, {false, {{-1}}},
                 {false, {{1, 1}}},
                 {false, std::vector<std::vector<ft::NodeId>>(34)},
                 {false, {{}, std::vector<ft::NodeId>(4097, 1)}}}) {
        EXPECT_THROW(storage.put_node(0, node), std::invalid_argument);
        EXPECT_EQ(raw_get(tx.txn, table, record_key('n')), valid);
    }
    for (const auto [offset, value] : {
                 std::pair<size_t, uint32_t>{0, 2}, {4, 0}, {4, 34},
                 {8, 8193}, {12, 0}, {12, UINT32_MAX}, {16, 1}}) {
        auto malformed = valid;
        set32(malformed, offset, value);
        raw_put(tx.txn, table, record_key('n'), malformed);
        EXPECT_THROW(storage.node(0), std::runtime_error) << offset;
    }
    raw_put(tx.txn, table, record_key('n'), valid + "x");
    EXPECT_THROW(storage.node(0), std::runtime_error);
    raw_put(tx.txn, table, record_key('n'), std::string(600000, '\0'));
    EXPECT_THROW(storage.node(0), std::runtime_error);
}

TEST_F(StorageEdges, SavepointFailurePreservesEarlierParentWrites) {
    checked(mdb_env_set_mapsize(env, 64 * 1024));
    {
        Transaction tx(env);
        {
            ft::LmdbStorage storage(tx.txn, table, 1);
            auto metadata = initial();
            metadata.config.dimensions = 4096;
            storage.put_metadata(metadata);
        }
        tx.commit();
    }
    {
        Transaction tx(env);
        raw_put(tx.txn, table, "application-record", "pending-before-index");
        {
            ft::LmdbStorage storage(tx.txn, table, 1);
            ft::HNSW index(storage);
            const std::vector<float> batch(20 * 4096, 0);
            try {
                index.add(20, batch.data());
                FAIL() << "Expected MDB_MAP_FULL while writing the child";
            } catch (const ft::MutationError& error) {
                EXPECT_FALSE(error.abort_required());
                EXPECT_TRUE(contains_lmdb_error(error, MDB_MAP_FULL));
            }
            EXPECT_THROW(storage.metadata(), std::runtime_error);
        }
        // A failed adapter is discarded; the caller's parent transaction is
        // still usable and may commit its unrelated earlier mutations.
        {
            ft::LmdbStorage after(tx.txn, table, 1);
            EXPECT_EQ(after.metadata()->next_id, 0);
            EXPECT_THROW(after.vector(0), std::runtime_error);
            EXPECT_EQ(raw_get(tx.txn, table, "application-record"),
                      "pending-before-index");
        }
        tx.commit();
    }
    Transaction read(env, true);
    ft::LmdbStorage storage(read.txn, table, 1);
    EXPECT_EQ(storage.metadata()->next_id, 0);
    EXPECT_THROW(storage.node(0), std::runtime_error);
    EXPECT_THROW(storage.vector(0), std::runtime_error);
    EXPECT_EQ(raw_get(read.txn, table, "application-record"),
              "pending-before-index");
}

TEST_F(StorageEdges, CallbackFailureExplicitlyRequiresCallerAbort) {
    Transaction tx(env);
    size_t writes = 0;
    ft::LmdbStorage storage(
            tx.txn, table, 1, 0,
            [&](MDB_txn* txn, MDB_dbi db, std::string_view key,
                std::string_view value) {
                EXPECT_EQ(txn, tx.txn);
                if (++writes == 3) {
                    throw std::runtime_error("changelog failure");
                }
                raw_put(txn, db, std::string(key), std::string(value));
            });
    ft::HNSW index(storage);
    index.create(initial().config); // first callback
    const float vector[]{0, 1};
    try {
        index.add(1, vector); // vector succeeds, node callback throws
        FAIL() << "Expected callback failure";
    } catch (const ft::MutationError& error) {
        EXPECT_TRUE(error.abort_required());
    }
    EXPECT_EQ(writes, 3);
    EXPECT_THROW(storage.metadata(), std::runtime_error);
    // Transaction destructor deliberately aborts, including external caller
    // state. The caller must also discard any externally buffered changelog.
}

TEST_F(StorageEdges, SavepointModeRejectsCallbacksAndInvalidMode) {
    Transaction tx(env);
    const ft::LmdbStorage::Put callback =
            [](MDB_txn*, MDB_dbi, std::string_view, std::string_view) {};
    EXPECT_THROW(
            ft::LmdbStorage(tx.txn, table, 1, 0, callback,
                            ft::LmdbStorage::WriteMode::Savepoint),
            std::invalid_argument);
    EXPECT_THROW(
            ft::LmdbStorage(tx.txn, table, 1, 0, {},
                            static_cast<ft::LmdbStorage::WriteMode>(99)),
            std::invalid_argument);
}

TEST_F(StorageEdges, WriteMapRequiresExplicitCallerTransactionMode) {
    mdb_env_close(std::exchange(env, nullptr));
    checked(mdb_env_create(&env));
    checked(mdb_env_set_maxdbs(env, 2));
    checked(mdb_env_set_mapsize(env, 32 * 1024 * 1024));
    checked(mdb_env_open(env, path.c_str(), MDB_NOTLS | MDB_WRITEMAP, 0600));
    Transaction tx(env);
    checked(mdb_dbi_open(tx.txn, "hnsw", 0, &table));
    {
        ft::LmdbStorage automatic(tx.txn, table, 1);
        ft::HNSW index(automatic);
        try {
            index.create(initial().config);
            FAIL() << "Savepoints must reject MDB_WRITEMAP";
        } catch (const ft::MutationError& error) {
            EXPECT_FALSE(error.abort_required());
        }
    }
    ft::LmdbStorage explicit_mode(
            tx.txn, table, 1, 0, {},
            ft::LmdbStorage::WriteMode::CallerTransaction);
    ft::HNSW index(explicit_mode);
    index.create(initial().config);
    const float vector[]{2, 3};
    EXPECT_EQ(index.add(1, vector).front(), 0);
    EXPECT_EQ(index.search(vector, 1).front().id, 0);
}

TEST_F(StorageEdges, ChangelogCallbackCannotReenterPartiallyAppliedIndex) {
    Transaction tx(env);
    ft::LmdbStorage* same = nullptr;
    ft::LmdbStorage storage(
            tx.txn, table, 1, 0,
            [&](MDB_txn*, MDB_dbi, std::string_view, std::string_view) {
                same->metadata();
            });
    same = &storage;
    ft::HNSW index(storage);
    try {
        index.create(initial().config);
        FAIL() << "Reentrant callbacks must fail";
    } catch (const ft::MutationError& error) {
        EXPECT_TRUE(error.abort_required());
    }
    EXPECT_THROW(storage.metadata(), std::runtime_error);
}

TEST_F(StorageEdges, DecodedResultsOwnDataAcrossWritesAndTransactionAbort) {
    std::optional<ft::Metadata> metadata;
    ft::Node node;
    std::vector<float> vector;
    {
        Transaction tx(env);
        ft::LmdbStorage storage(tx.txn, table, 1);
        storage.put_metadata(initial());
        storage.put_node(0, {false, {{1, 2}}});
        storage.put_vector(0, {1, 2});
        metadata = storage.metadata();
        node = storage.node(0);
        vector = storage.vector(0);
        auto changed = initial();
        changed.config.seed = 99;
        storage.put_metadata(changed);
        storage.put_node(0, {true, {{3}}});
        storage.put_vector(0, {8, 9});
        // These owned results must survive both page invalidation on writes
        // and destruction/abort of their originating transaction below.
        EXPECT_EQ(vector, (std::vector<float>{1, 2}));
        EXPECT_EQ(node.links, (std::vector<std::vector<ft::NodeId>>{{1, 2}}));
    }
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->config.seed, initial().config.seed);
    EXPECT_FALSE(node.deleted);
    EXPECT_EQ(node.links, (std::vector<std::vector<ft::NodeId>>{{1, 2}}));
    EXPECT_EQ(vector, (std::vector<float>{1, 2}));
}

} // namespace
