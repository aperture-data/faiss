/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/HNSW.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace ft = faiss::transactional;
namespace {
// A transaction-local test store. Read counters make accidental full scans
// observable without making tests depend on machine speed or the LMDB cache.
struct CountingStorage final : ft::Storage {
    std::optional<ft::Metadata> meta;
    std::map<ft::NodeId, ft::Node> nodes;
    std::map<ft::NodeId, std::vector<float>> vectors;
    mutable size_t node_reads = 0, vector_reads = 0;
    mutable std::map<ft::NodeId, size_t> vector_reads_by_id;
    size_t writes = 0;
    bool invalidated = false;
    bool fast_queries = false;
    bool fast_live_queries = false;
    std::map<ft::NodeId, bool> fast_live_overrides;
    mutable size_t fast_live_reads = 0;
    mutable std::map<ft::NodeId, size_t> fast_live_reads_by_id;
    ft::NodeId nonfinite_fast_id = ft::no_node;
    mutable size_t fast_factory_calls = 0, fast_scalar_calls = 0,
                   fast_batch_calls = 0;
    mutable std::map<ft::NodeId, size_t> fast_evaluations;
    std::function<void()> after_fast_batch;

    std::unique_ptr<faiss::DistanceComputer> query_distance_computer(
            const ft::Metadata& metadata,
            size_t,
            ft::SearchStats*) const override {
        if (!fast_queries)
            return nullptr;
        ++fast_factory_calls;
        struct Computer : faiss::DistanceComputer {
            const CountingStorage& storage;
            ft::Config config;
            const float* query = nullptr;
            Computer(const CountingStorage& storage, const ft::Config& config)
                    : storage(storage), config(config) {}
            void set_query(const float* value) override {
                query = value;
            }
            float evaluate(faiss::idx_t raw) {
                storage.check();
                const auto id = static_cast<ft::NodeId>(raw);
                ++storage.fast_evaluations[id];
                if (id == storage.nonfinite_fast_id)
                    return std::numeric_limits<float>::quiet_NaN();
                const auto& vector = storage.vectors.at(id);
                float result = 0;
                for (size_t dim = 0; dim < config.dimensions; ++dim) {
                    if (config.metric == faiss::METRIC_L2) {
                        const float delta = query[dim] - vector[dim];
                        result += delta * delta;
                    } else {
                        result -= query[dim] * vector[dim];
                    }
                }
                return result;
            }
            float operator()(faiss::idx_t id) override {
                ++storage.fast_scalar_calls;
                return evaluate(id);
            }
            void distances_batch_4(
                    faiss::idx_t a,
                    faiss::idx_t b,
                    faiss::idx_t c,
                    faiss::idx_t d,
                    float& da,
                    float& db,
                    float& dc,
                    float& dd) override {
                ++storage.fast_batch_calls;
                da = evaluate(a);
                db = evaluate(b);
                dc = evaluate(c);
                dd = evaluate(d);
                if (storage.after_fast_batch)
                    storage.after_fast_batch();
            }
            float symmetric_dis(faiss::idx_t, faiss::idx_t) override {
                throw std::logic_error("query backend used for construction");
            }
        };
        struct LiveComputer final : Computer, ft::QueryLiveAccess {
            using Computer::Computer;
            bool deleted(ft::NodeId id) const override {
                this->storage.check();
                if (id < 0 || uint64_t(id) >= this->storage.meta->next_id)
                    throw std::runtime_error("invalid test live lookup ID");
                ++this->storage.fast_live_reads;
                ++this->storage.fast_live_reads_by_id[id];
                const auto overridden =
                        this->storage.fast_live_overrides.find(id);
                if (overridden != this->storage.fast_live_overrides.end())
                    return overridden->second;
                return this->storage.nodes.at(id).deleted;
            }
        };
        if (fast_live_queries)
            return std::make_unique<LiveComputer>(*this, metadata.config);
        return std::make_unique<Computer>(*this, metadata.config);
    }

    void check() const {
        if (invalidated)
            throw std::runtime_error("invalidated test transaction");
    }
    std::optional<ft::Metadata> metadata() const override {
        check();
        return meta;
    }
    ft::Node node(ft::NodeId id) const override {
        check();
        ++node_reads;
        return nodes.at(id);
    }
    std::vector<float> vector(ft::NodeId id) const override {
        check();
        ++vector_reads;
        ++vector_reads_by_id[id];
        return vectors.at(id);
    }
    void put_metadata(const ft::Metadata& value) override {
        check();
        ++writes;
        meta = value;
    }
    void put_node(ft::NodeId id, const ft::Node& value) override {
        check();
        ++writes;
        nodes[id] = value;
    }
    void put_vector(ft::NodeId id, const std::vector<float>& value) override {
        check();
        ++writes;
        vectors[id] = value;
    }
    void invalidate() noexcept override {
        invalidated = true;
    }
};

ft::Config configuration(unsigned dim = 2) {
    ft::Config config;
    config.dimensions = dim;
    config.M = 8;
    config.ef_construction = 64;
    return config;
}

// A deliberately connected graph makes work bounds independent of how HNSW
// construction happens to prune links on a particular random input dataset.
CountingStorage chain(int n) {
    CountingStorage storage;
    ft::Metadata meta;
    meta.config = configuration(1);
    meta.next_id = meta.live_count = n;
    meta.entry_point = 0;
    meta.max_level = 0;
    storage.meta = meta;
    for (int id = 0; id < n; ++id) {
        ft::Node node;
        node.links.resize(1);
        if (id)
            node.links[0].push_back(id - 1);
        if (id + 1 < n)
            node.links[0].push_back(id + 1);
        storage.nodes[id] = std::move(node);
        storage.vectors[id] = {static_cast<float>(id)};
    }
    return storage;
}

CountingStorage star() {
    auto storage = chain(9);
    storage.nodes[0].links[0].clear();
    for (ft::NodeId id = 1; id < 9; ++id) {
        storage.nodes[0].links[0].push_back(id);
        storage.nodes[id].links[0] = {0};
    }
    storage.fast_queries = true;
    return storage;
}

TEST(TransactionalHNSWEdge, EmptyPredicateDoesNotScanWholeIndex) {
    auto storage = chain(4096);
    ft::HNSW index(storage);
    faiss::IDSelectorRange none(4096, 4097);
    const float query = 0;
    EXPECT_TRUE(index.search(&query, 1, 8, &none).empty());
    EXPECT_LT(storage.vector_reads, 128);
    EXPECT_LT(storage.node_reads, 128);
    EXPECT_EQ(storage.writes, 0);
}

TEST(TransactionalHNSWEdge, SparsePredicateDoesNotScanWholeIndex) {
    auto storage = chain(4096);
    ft::HNSW index(storage);
    faiss::IDSelectorRange sparse(4095, 4096);
    const float query = 0;
    // ANN search may return fewer than k matches. It must not silently turn
    // every selective predicate into an unbounded exact scan.
    for (const auto& hit : index.search(&query, 1, 8, &sparse))
        EXPECT_EQ(hit.id, 4095);
    EXPECT_LT(storage.vector_reads, 128);
    EXPECT_LT(storage.node_reads, 128);
}

TEST(TransactionalHNSWEdge, TombstonesDoNotScanWholeIndex) {
    auto storage = chain(4096);
    for (int id = 0; id < 4095; ++id)
        storage.nodes[id].deleted = true;
    storage.meta->live_count = 1;
    ft::HNSW index(storage);
    const float query = 0;
    for (const auto& hit : index.search(&query, 1, 8))
        EXPECT_EQ(hit.id, 4095);
    EXPECT_LT(storage.vector_reads, 128);
    EXPECT_LT(storage.node_reads, 128);
}

TEST(TransactionalHNSWEdge, FilteredAndDeletedNodesRemainTraversalBridges) {
    auto storage = chain(6);
    storage.nodes[0].deleted = true;
    storage.nodes[1].deleted = true;
    storage.meta->live_count = 4;
    ft::HNSW index(storage);
    faiss::IDSelectorRange last(4, 6);
    const float query = 0;
    const auto found = index.search(&query, 2, 16, &last);
    ASSERT_EQ(found.size(), 2);
    EXPECT_EQ(found[0].id, 4);
    EXPECT_EQ(found[1].id, 5);
    EXPECT_FLOAT_EQ(found[0].distance, 16);
    EXPECT_FLOAT_EQ(found[1].distance, 25);
}

TEST(TransactionalHNSWEdge, EqualDistancesReturnUniqueOrderedResults) {
    auto storage = chain(64);
    for (auto& [id, vector] : storage.vectors)
        vector[0] = 0;
    ft::HNSW index(storage);
    const float query = 0;
    const auto found = index.search(&query, 8, 16);
    ASSERT_EQ(found.size(), 8);
    std::unordered_set<ft::NodeId> unique;
    for (size_t i = 0; i < found.size(); ++i) {
        EXPECT_TRUE(unique.insert(found[i].id).second);
        EXPECT_FLOAT_EQ(found[i].distance, 0);
        if (i)
            EXPECT_LT(found[i - 1].id, found[i].id);
    }
    EXPECT_LT(storage.vector_reads, 64);
}

TEST(TransactionalHNSWEdge, OverflowDuringBatchDoesNotFlushEarlierInsertions) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float origin[]{0, 0};
    index.add(1, origin);
    const auto writes = storage.writes;
    const auto before = storage.nodes;
    const float large = std::numeric_limits<float>::max() / 2;
    const float batch[]{1, 1, large, large};
    EXPECT_THROW(index.add(2, batch), std::runtime_error);
    // All arithmetic and topology preparation precedes the first persistent
    // write. A failed second insertion cannot publish the first one.
    EXPECT_EQ(storage.writes, writes);
    ASSERT_TRUE(storage.meta.has_value());
    EXPECT_EQ(storage.meta->next_id, 1);
    EXPECT_EQ(storage.meta->live_count, 1);
    EXPECT_EQ(storage.vectors.size(), 1);
    ASSERT_EQ(storage.nodes.size(), before.size());
    EXPECT_EQ(storage.nodes.at(0).links, before.at(0).links);
}

TEST(TransactionalHNSWEdge, FiniteExtremeVectorRejectsOverflowingSearch) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration(1));
    const float large = std::numeric_limits<float>::max() / 2;
    index.add(1, &large);
    const float opposite = -large;
    EXPECT_THROW(index.search(&opposite, 1), std::runtime_error);
    const auto found = index.search(&large, 1);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 0);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
}

TEST(TransactionalHNSWEdge, IdExhaustionRejectsBeforeReadingInput) {
    auto storage = chain(1);
    ft::HNSW index(storage);
    storage.meta->next_id =
            static_cast<uint64_t>(std::numeric_limits<ft::NodeId>::max()) - 1;
    EXPECT_THROW(index.add(3, nullptr), std::overflow_error);
    EXPECT_EQ(storage.writes, 0);
    EXPECT_EQ(storage.vector_reads, 0);
    EXPECT_EQ(storage.node_reads, 0);
}

TEST(TransactionalHNSWEdge, RejectsInvalidMetadata) {
    auto storage = chain(1);
    ft::HNSW index(storage);
    storage.meta->live_count = 2;
    EXPECT_THROW(index.metadata(), std::runtime_error);
    storage.meta->live_count = 1;
    storage.meta->entry_point = 1;
    EXPECT_THROW(index.metadata(), std::runtime_error);
    storage.meta->entry_point = 0;
    storage.meta->max_level = 33;
    EXPECT_THROW(index.metadata(), std::runtime_error);
}

TEST(TransactionalHNSWEdge, RejectsBrokenTraversalRecords) {
    const float query = 0;
    {
        auto storage = chain(1);
        storage.vectors[0].push_back(1);
        ft::HNSW index(storage);
        EXPECT_THROW(index.search(&query, 1), std::runtime_error);
    }
    {
        auto storage = chain(1);
        storage.nodes[0].links.clear();
        ft::HNSW index(storage);
        EXPECT_THROW(index.search(&query, 1), std::runtime_error);
    }
    {
        auto storage = chain(1);
        storage.nodes[0].links[0].push_back(42);
        ft::HNSW index(storage);
        EXPECT_THROW(index.search(&query, 1), std::exception);
    }
}

TEST(TransactionalHNSWEdge, DeleteAndReinsertPreservesIdsAndReadYourWrites) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float vectors[]{0, 0, 2, 0, 4, 0};
    EXPECT_EQ(index.add(3, vectors), (std::vector<ft::NodeId>{0, 1, 2}));
    EXPECT_TRUE(index.erase(1));
    EXPECT_FALSE(index.erase(1));
    EXPECT_EQ(index.add(1, vectors + 2), (std::vector<ft::NodeId>{3}));
    const auto found = index.search(vectors + 2, 8, 16);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 3);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    for (const auto& hit : found)
        EXPECT_NE(hit.id, 1);
    EXPECT_TRUE(storage.nodes.at(1).deleted);
    EXPECT_EQ(index.metadata().next_id, 4);
    EXPECT_EQ(index.metadata().live_count, 3);
}

TEST(TransactionalHNSWEdge, ExplicitVisitBudgetReportsPartialSearch) {
    auto storage = chain(4096);
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.ef_search = 64;
    options.max_visits = 5;
    ft::SearchStats stats;
    const float query = 4096;
    const auto found = index.search(&query, 10, options, &stats);
    EXPECT_LE(found.size(), 5);
    EXPECT_EQ(stats.visited, 5);
    EXPECT_TRUE(stats.visit_limit_reached);
    EXPECT_LE(storage.vector_reads, 5);
    for (size_t i = 1; i < found.size(); ++i)
        EXPECT_LE(found[i - 1].distance, found[i].distance);
}

TEST(TransactionalHNSWEdge, SearchCancellationPrecedesVectorReads) {
    auto storage = chain(4096);
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.cancelled = [] { return true; };
    const float query = 0;
    EXPECT_THROW(index.search(&query, 10, options), ft::SearchCancelled);
    EXPECT_EQ(storage.vector_reads, 0);
    EXPECT_EQ(storage.node_reads, 0);
    EXPECT_EQ(storage.writes, 0);
    EXPECT_FALSE(storage.invalidated);
}

TEST(TransactionalHNSWEdge, ExactModeFindsDisconnectedDuplicateVectors) {
    auto storage = chain(64);
    for (auto& [id, vector] : storage.vectors)
        vector[0] = 0;
    for (auto& [id, node] : storage.nodes)
        node.links[0].clear();
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.exact = true;
    options.cache_bytes = 1;
    ft::SearchStats stats;
    const float query = 0;
    const auto found = index.search(&query, 64, options, &stats);
    ASSERT_EQ(found.size(), 64);
    for (size_t i = 0; i < found.size(); ++i) {
        EXPECT_EQ(found[i].id, i);
        EXPECT_FLOAT_EQ(found[i].distance, 0);
    }
    EXPECT_FALSE(stats.visit_limit_reached);
    EXPECT_EQ(stats.visited, 64);
}

TEST(TransactionalHNSWEdge, ExactModeHonorsPredicateTombstonesAndBudget) {
    auto storage = chain(64);
    for (int id = 0; id < 64; id += 2)
        storage.nodes[id].deleted = true;
    storage.meta->live_count = 32;
    ft::HNSW index(storage);
    faiss::IDSelectorRange selected(40, 64);
    ft::SearchOptions options;
    options.exact = true;
    options.selector = &selected;
    const float query = 0;
    const auto found = index.search(&query, 5, options);
    ASSERT_EQ(found.size(), 5);
    for (size_t i = 0; i < found.size(); ++i)
        EXPECT_EQ(found[i].id, 41 + 2 * i);

    options.selector = nullptr;
    options.max_visits = 5;
    ft::SearchStats stats;
    index.search(&query, 64, options, &stats);
    EXPECT_LE(stats.visited, 5);
    EXPECT_TRUE(stats.visit_limit_reached);
}

TEST(TransactionalHNSWEdge, QueryCachesDoNotCrossOperationsOrChangeResults) {
    auto storage = chain(64);
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.ef_search = 16;
    const float query = 20;
    const auto cached = index.search(&query, 8, options);
    options.cache_bytes = 1;
    const auto uncached = index.search(&query, 8, options);
    ASSERT_EQ(uncached.size(), cached.size());
    for (size_t i = 0; i < cached.size(); ++i) {
        EXPECT_EQ(uncached[i].id, cached[i].id);
        EXPECT_FLOAT_EQ(uncached[i].distance, cached[i].distance);
    }
    storage.nodes[20].deleted = true;
    --storage.meta->live_count;
    for (const auto& hit : index.search(&query, 8, options))
        EXPECT_NE(hit.id, 20);
}

TEST(TransactionalHNSWEdge, AtomicUpdateAllocatesNewIdAndHidesOldVersion) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float vectors[]{0, 0, 2, 0, 4, 0};
    index.add(3, vectors);
    const float changed[]{3, 1};
    EXPECT_EQ(index.update(1, changed), 3);
    EXPECT_TRUE(storage.nodes.at(1).deleted);
    EXPECT_FALSE(storage.nodes.at(3).deleted);
    EXPECT_EQ(index.metadata().next_id, 4);
    EXPECT_EQ(index.metadata().live_count, 3);
    const auto found = index.search(changed, 8, 16);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 3);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    for (const auto& hit : found)
        EXPECT_NE(hit.id, 1);
}

TEST(TransactionalHNSWEdge, InvalidUpdateDoesNotErasePreviousVersion) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float initial[]{0, 0};
    index.add(1, initial);
    const auto writes = storage.writes;
    const float bad[]{0, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(index.update(0, bad), std::invalid_argument);
    EXPECT_EQ(storage.writes, writes);
    EXPECT_FALSE(storage.nodes.at(0).deleted);
    EXPECT_EQ(index.metadata().next_id, 1);
    EXPECT_EQ(index.metadata().live_count, 1);
}

TEST(TransactionalHNSWEdge, MutationBudgetsRejectWithoutPersistentWrites) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float vectors[]{0, 0, 2, 0, 4, 0};
    const auto writes = storage.writes;
    ft::MutationOptions options;
    options.max_batch_size = 2;
    EXPECT_THROW(index.add(3, vectors, options), std::length_error);
    EXPECT_EQ(storage.writes, writes);
    EXPECT_EQ(index.metadata().next_id, 0);

    options.max_batch_size = 3;
    options.max_workspace_bytes = 1;
    EXPECT_THROW(index.add(3, vectors, options), std::length_error);
    EXPECT_EQ(storage.writes, writes);
    EXPECT_EQ(index.metadata().next_id, 0);
}

TEST(TransactionalHNSWEdge, CancelledMutationDoesNotPublish) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const float vector[]{0, 0};
    const auto writes = storage.writes;
    ft::MutationOptions options;
    options.cancelled = [] { return true; };
    EXPECT_THROW(index.add(1, vector, options), std::exception);
    EXPECT_EQ(storage.writes, writes);
    EXPECT_EQ(index.metadata().next_id, 0);
}

TEST(TransactionalHNSWEdge, PredicateSeedsReachDisconnectedDistantRegion) {
    auto storage = chain(4096);
    storage.nodes[0].links[0].clear();
    ft::HNSW index(storage);
    faiss::IDSelectorRange selected(4090, 4096);
    const ft::NodeId seed = 4095;
    ft::SearchOptions options;
    options.selector = &selected;
    options.seed_ids = std::span(&seed, 1);
    options.max_visits = 3;
    ft::SearchStats stats;
    const float query = 4095;
    const auto found = index.search(&query, 1, options, &stats);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 4095);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    EXPECT_LE(stats.visited, 3);
}

TEST(TransactionalHNSWEdge, AcornCrossesRejectedBridgesWithoutVectorReads) {
    struct EvenIds final : faiss::IDSelector {
        bool is_member(faiss::idx_t id) const override {
            return id % 2 == 0;
        }
    } selected;
    auto storage = chain(8);
    ft::HNSW index(storage);
    const ft::NodeId seed = 0;
    ft::SearchOptions options;
    options.acorn = true;
    options.selector = &selected;
    options.seed_ids = std::span(&seed, 1);
    options.max_visits = 3;
    ft::SearchStats stats;
    const float query = 4;
    const auto found = index.search(&query, 1, options, &stats);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 4);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    EXPECT_LE(stats.visited, 3);
    for (int id = 1; id < 8; id += 2)
        EXPECT_EQ(storage.vector_reads_by_id[id], 0);
}

TEST(TransactionalHNSWEdge, RejectsInvalidSeedIdsAndAcornWithoutPredicate) {
    auto storage = chain(8);
    ft::HNSW index(storage);
    const float query = 0;
    ft::SearchOptions options;
    options.acorn = true;
    EXPECT_THROW(index.search(&query, 1, options), std::invalid_argument);
    options.acorn = false;
    for (const ft::NodeId seed : {-1, 8}) {
        options.seed_ids = std::span(&seed, 1);
        EXPECT_THROW(index.search(&query, 1, options), std::invalid_argument);
    }
    const std::vector<ft::NodeId> too_many(65, 0);
    options.seed_ids = too_many;
    EXPECT_THROW(index.search(&query, 1, options), std::invalid_argument);
}

TEST(TransactionalHNSWEdge, FullValidationChecksCountsEdgesAndTombstones) {
    auto storage = chain(6);
    ft::HNSW index(storage);
    auto report = index.validate();
    EXPECT_EQ(report.nodes, 6);
    EXPECT_EQ(report.live_nodes, 6);
    EXPECT_EQ(report.edges, 10);
    EXPECT_TRUE(index.erase(1));
    EXPECT_TRUE(index.erase(3));
    report = index.validate();
    EXPECT_EQ(report.nodes, 6);
    EXPECT_EQ(report.live_nodes, 4);
    EXPECT_EQ(report.edges, 10);
    storage.meta->live_count = 5;
    EXPECT_THROW(index.validate(), std::runtime_error);
}

TEST(TransactionalHNSWEdge,
     FullValidationRejectsDuplicateAndOversizedAdjacency) {
    auto storage = chain(32);
    ft::HNSW index(storage);
    storage.nodes[0].links[0] = {1, 1};
    EXPECT_THROW(index.validate(), std::runtime_error);
    storage.nodes[0].links[0].clear();
    for (int neighbor = 1; neighbor <= 17; ++neighbor)
        storage.nodes[0].links[0].push_back(neighbor);
    EXPECT_THROW(index.validate(), std::runtime_error);
}

TEST(TransactionalHNSWEdge, FullValidationRejectsNonFinitePersistedVectors) {
    auto storage = chain(6);
    ft::HNSW index(storage);
    storage.vectors[5][0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(index.validate(), std::exception);
}

TEST(TransactionalHNSWEdge, VisitCapIncludesUpperLevelRouting) {
    auto storage = chain(2);
    storage.meta->max_level = 1;
    storage.nodes[0].links.push_back({1});
    storage.nodes[1].links.push_back({0});
    storage.vectors[0][0] = 100;
    storage.vectors[1][0] = 0;
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.max_visits = 1;
    ft::SearchStats stats;
    const float query = 0;
    const auto found = index.search(&query, 2, options, &stats);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 0);
    EXPECT_EQ(stats.visited, 1);
    EXPECT_TRUE(stats.visit_limit_reached);
    EXPECT_EQ(storage.vector_reads, 1);
}

TEST(TransactionalHNSWEdge, LastSignedNodeIdCanBeAllocatedExactlyOnce) {
    auto storage = chain(1);
    const auto last = std::numeric_limits<ft::NodeId>::max();
    // Sparse fake storage simulates a nearly exhausted generation without
    // allocating billions of nodes. Traversal only reaches the existing root.
    storage.meta->next_id = static_cast<uint64_t>(last) - 1;
    ft::HNSW index(storage);
    const float vectors[]{1, 2};
    const auto ids = index.add(2, vectors);
    EXPECT_EQ(ids, (std::vector<ft::NodeId>{last - 1, last}));
    EXPECT_EQ(index.metadata().next_id, static_cast<uint64_t>(last) + 1);
    const auto writes = storage.writes;
    EXPECT_THROW(index.add(1, vectors), std::overflow_error);
    EXPECT_EQ(storage.writes, writes);
}

TEST(TransactionalHNSWEdge, CancellationDuringBatchPreparationDoesNotPublish) {
    CountingStorage storage;
    ft::HNSW index(storage);
    index.create(configuration());
    const auto writes = storage.writes;
    std::vector<float> vectors(40, 0);
    size_t polls = 0;
    ft::MutationOptions options;
    // Validation polls once per input vector. Wait until insertion is already
    // preparing adjacency before cancelling to exercise operation rollback.
    options.cancelled = [&] { return ++polls > 30; };
    EXPECT_THROW(index.add(20, vectors.data(), options), ft::SearchCancelled);
    EXPECT_EQ(storage.writes, writes);
    EXPECT_EQ(index.metadata().next_id, 0);
    EXPECT_EQ(index.metadata().live_count, 0);
    EXPECT_TRUE(storage.nodes.empty());
    EXPECT_TRUE(storage.vectors.empty());
}

TEST(TransactionalHNSWEdge,
     ConstructionVisitLimitNeverPublishesPartialTopology) {
    auto storage = chain(4);
    ft::HNSW index(storage);
    ft::MutationOptions options;
    options.max_visits_per_insertion = 1;
    const float vector = 5;
    EXPECT_THROW(index.add(1, &vector, options), std::length_error);
    EXPECT_EQ(storage.writes, 0);
    EXPECT_EQ(index.metadata().next_id, 4);
    EXPECT_EQ(index.metadata().live_count, 4);
    EXPECT_FALSE(storage.nodes.contains(4));
    EXPECT_FALSE(storage.vectors.contains(4));
    EXPECT_THROW(index.update(1, &vector, options), std::length_error);
    EXPECT_EQ(storage.writes, 0);
    EXPECT_FALSE(storage.nodes.at(1).deleted);
    EXPECT_EQ(index.metadata().next_id, 4);
    EXPECT_EQ(index.metadata().live_count, 4);
}

TEST(TransactionalHNSWEdge, AcornExpandsEachDeletedBridgeOnce) {
    auto storage = chain(5);
    storage.nodes[1].deleted = true;
    storage.nodes[3].deleted = true;
    storage.meta->live_count = 3;
    ft::HNSW index(storage);
    faiss::IDSelectorRange selected(0, 5);
    ft::SearchOptions options;
    options.acorn = true;
    options.selector = &selected;
    options.max_visits = 3;
    options.cache_bytes =
            1; // Exercise owned views even without retained nodes.
    ft::SearchStats stats;
    const float query = 4;
    const auto found = index.search(&query, 3, options, &stats);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 4);
    EXPECT_EQ(found[1].id, 2);
    EXPECT_EQ(found[2].id, 0);
    EXPECT_EQ(stats.visited, 3);
    // Three live candidates and two deleted bridges, each expanded once even
    // though each bridge is reachable from two different live candidates.
    EXPECT_EQ(stats.expanded, 5);
    EXPECT_EQ(storage.vector_reads_by_id[1], 0);
    EXPECT_EQ(storage.vector_reads_by_id[3], 0);
}

TEST(TransactionalHNSWEdge, FastBackendBatchesAndMemoizesAcrossLevels) {
    auto storage = star();
    storage.meta->max_level = 1;
    for (auto& [id, node] : storage.nodes)
        node.links.push_back(node.links[0]);
    ft::HNSW index(storage);
    const float query = -1;
    ft::SearchStats stats;
    const auto found = index.search(&query, 3, ft::SearchOptions{}, &stats);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 0);
    EXPECT_EQ(found[1].id, 1);
    EXPECT_EQ(found[2].id, 2);
    EXPECT_EQ(storage.fast_factory_calls, 1);
    EXPECT_EQ(storage.fast_scalar_calls, 1);
    EXPECT_EQ(storage.fast_batch_calls, 2);
    EXPECT_EQ(storage.vector_reads, 0);
    EXPECT_EQ(stats.visited, 9);
    EXPECT_EQ(stats.vector_reads, 9);
    for (ft::NodeId id = 0; id < 9; ++id)
        EXPECT_EQ(storage.fast_evaluations[id], 1);
}

TEST(TransactionalHNSWEdge, FastBackendBudgetPreservesPartialBatchResults) {
    for (size_t budget : {1, 3, 5}) {
        auto storage = star();
        ft::HNSW index(storage);
        ft::SearchOptions options;
        options.max_visits = budget;
        ft::SearchStats stats;
        const float query = 8;
        const auto found = index.search(&query, 9, options, &stats);
        ASSERT_EQ(found.size(), budget);
        EXPECT_EQ(stats.visited, budget);
        EXPECT_EQ(stats.vector_reads, budget);
        EXPECT_TRUE(stats.visit_limit_reached);
        for (size_t i = 0; i < budget; ++i)
            EXPECT_EQ(found[i].id, budget - 1 - i);
        EXPECT_EQ(storage.fast_batch_calls, budget == 5 ? 1 : 0);
        EXPECT_EQ(storage.vector_reads, 0);
    }
}

TEST(TransactionalHNSWEdge,
     FastBackendRejectsNonFiniteScalarAndBatchDistances) {
    for (ft::NodeId bad_id : {0, 3}) {
        auto storage = star();
        storage.nonfinite_fast_id = bad_id;
        ft::HNSW index(storage);
        const float query = 0;
        EXPECT_THROW(index.search(&query, 3), std::runtime_error);
        EXPECT_EQ(storage.writes, 0);
        EXPECT_FALSE(storage.invalidated);
        if (bad_id)
            EXPECT_EQ(storage.fast_batch_calls, 1);
    }
}

TEST(TransactionalHNSWEdge, FastBackendInnerProductPreservesPublicScoreSign) {
    auto storage = star();
    storage.meta->config.metric = faiss::METRIC_INNER_PRODUCT;
    ft::HNSW index(storage);
    const float query = 2;
    const auto accelerated = index.search(&query, 3);
    storage.fast_queries = false;
    const auto fallback = index.search(&query, 3);
    ASSERT_EQ(accelerated.size(), 3);
    ASSERT_EQ(fallback.size(), 3);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(accelerated[i].id, 8 - i);
        EXPECT_FLOAT_EQ(accelerated[i].distance, 2 * (8 - i));
        EXPECT_EQ(accelerated[i].id, fallback[i].id);
        EXPECT_FLOAT_EQ(accelerated[i].distance, fallback[i].distance);
    }
    EXPECT_GE(storage.fast_batch_calls, 1);
}

TEST(TransactionalHNSWEdge, FastBackendCancellationDoesNotPublish) {
    auto storage = star();
    bool stop = false;
    storage.after_fast_batch = [&] { stop = true; };
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.cancelled = [&] { return stop; };
    const float query = 0;
    EXPECT_THROW(index.search(&query, 3, options), ft::SearchCancelled);
    EXPECT_EQ(storage.fast_batch_calls, 1);
    EXPECT_EQ(storage.writes, 0);
    EXPECT_FALSE(storage.invalidated);
}

TEST(TransactionalHNSWEdge, FastBackendIsQueryOnlyAndSeesPendingUpdates) {
    CountingStorage storage;
    storage.fast_queries = true;
    ft::HNSW index(storage);
    index.create(configuration(1));
    const float vectors[]{0, 1, 2, 3, 4, 5, 6, 7, 8};
    index.add(9, vectors);
    EXPECT_EQ(storage.fast_factory_calls, 0);
    const float changed = 100;
    EXPECT_EQ(index.update(1, &changed), 9);
    EXPECT_EQ(storage.fast_factory_calls, 0);
    const auto found = index.search(&changed, 3);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 9);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    for (const auto& hit : found)
        EXPECT_NE(hit.id, 1);
    EXPECT_EQ(storage.fast_factory_calls, 1);
}

TEST(TransactionalHNSWEdge, FastLiveBackendCombinesTombstonesAndPredicate) {
    auto storage = star();
    storage.fast_live_queries = true;
    storage.nodes[1].deleted = true;
    storage.nodes[5].deleted = true;
    storage.meta->live_count = 7;
    ft::HNSW index(storage);
    faiss::IDSelectorRange selected(2, 8);
    const float query = 5;
    const auto accelerated = index.search(&query, 8, 64, &selected);
    ASSERT_EQ(accelerated.size(), 5);
    EXPECT_EQ(storage.fast_live_reads_by_id[1], 0);
    EXPECT_EQ(storage.fast_live_reads_by_id[8], 0);
    EXPECT_GT(storage.fast_live_reads_by_id[5], 0);
    storage.fast_live_queries = false;
    const auto fallback = index.search(&query, 8, 64, &selected);
    ASSERT_EQ(fallback.size(), accelerated.size());
    for (size_t i = 0; i < accelerated.size(); ++i) {
        EXPECT_EQ(accelerated[i].id, fallback[i].id);
        EXPECT_FLOAT_EQ(accelerated[i].distance, fallback[i].distance);
        EXPECT_GE(accelerated[i].id, 2);
        EXPECT_LT(accelerated[i].id, 8);
        EXPECT_NE(accelerated[i].id, 5);
    }
}

TEST(TransactionalHNSWEdge, FastLiveBackendPreservesPartialVisitBudget) {
    auto storage = star();
    storage.fast_live_queries = true;
    storage.nodes[1].deleted = true;
    --storage.meta->live_count;
    ft::HNSW index(storage);
    ft::SearchOptions options;
    options.max_visits = 5;
    ft::SearchStats stats;
    const float query = 8;
    const auto found = index.search(&query, 8, options, &stats);
    ASSERT_EQ(found.size(), 4);
    EXPECT_EQ(found[0].id, 4);
    EXPECT_EQ(found[1].id, 3);
    EXPECT_EQ(found[2].id, 2);
    EXPECT_EQ(found[3].id, 0);
    EXPECT_EQ(stats.visited, 5);
    EXPECT_TRUE(stats.visit_limit_reached);
    EXPECT_GT(storage.fast_live_reads_by_id[1], 0);
}

TEST(TransactionalHNSWEdge, FastLiveBackendRetainsAcornTombstoneBridges) {
    auto storage = chain(5);
    storage.fast_queries = true;
    storage.fast_live_queries = true;
    storage.nodes[1].deleted = true;
    storage.nodes[3].deleted = true;
    storage.meta->live_count = 3;
    ft::HNSW index(storage);
    faiss::IDSelectorRange selected(0, 5);
    ft::SearchOptions options;
    options.acorn = true;
    options.selector = &selected;
    options.max_visits = 3;
    ft::SearchStats stats;
    const float query = 4;
    const auto found = index.search(&query, 3, options, &stats);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 4);
    EXPECT_EQ(found[1].id, 2);
    EXPECT_EQ(found[2].id, 0);
    EXPECT_EQ(stats.visited, 3);
    EXPECT_EQ(stats.expanded, 5);
    EXPECT_EQ(storage.fast_evaluations[1], 0);
    EXPECT_EQ(storage.fast_evaluations[3], 0);
    EXPECT_GT(storage.fast_live_reads_by_id[1], 0);
    EXPECT_GT(storage.fast_live_reads_by_id[3], 0);
}

TEST(TransactionalHNSWEdge,
     ValidationRejectsBothLiveBackendMismatchDirections) {
    for (const bool node_deleted : {false, true}) {
        auto storage = chain(3);
        storage.fast_queries = true;
        storage.fast_live_queries = true;
        storage.nodes[1].deleted = node_deleted;
        storage.meta->live_count -= node_deleted;
        ft::HNSW index(storage);
        auto report = index.validate();
        EXPECT_EQ(report.nodes, 3);
        EXPECT_EQ(report.live_nodes, 3 - node_deleted);
        storage.fast_live_overrides[1] = !node_deleted;
        // This also tests the all-live metadata case: full validation must
        // inspect live bits even when ordinary search can skip tombstones.
        EXPECT_THROW(index.validate(), std::runtime_error);
    }
}

TEST(TransactionalHNSWEdge, FastLiveBackendDoesNotRetainStateAcrossOperations) {
    auto storage = star();
    storage.fast_live_queries = true;
    ft::HNSW index(storage);
    const float query = 1;
    ASSERT_EQ(index.search(&query, 1)[0].id, 1);
    EXPECT_TRUE(index.erase(1));
    ASSERT_EQ(index.search(&query, 1)[0].id, 0);
    EXPECT_EQ(index.update(0, &query), 9);
    const auto found = index.search(&query, 3);
    ASSERT_EQ(found.size(), 3);
    EXPECT_EQ(found[0].id, 9);
    EXPECT_FLOAT_EQ(found[0].distance, 0);
    for (const auto& hit : found) {
        EXPECT_NE(hit.id, 0);
        EXPECT_NE(hit.id, 1);
    }
    EXPECT_EQ(storage.fast_factory_calls, 3);
}

TEST(TransactionalHNSWEdge,
     CollectorAvoidsPredicateAndLiveReadsForWorseCandidates) {
    struct CountPredicate final : faiss::IDSelector {
        mutable size_t calls = 0;
        bool is_member(faiss::idx_t) const override {
            ++calls;
            return true;
        }
    } selected;
    auto storage = star();
    storage.fast_live_queries = true;
    storage.nodes[8].deleted = true;
    --storage.meta->live_count;
    ft::HNSW index(storage);
    const float query = 0;
    const auto found = index.search(&query, 1, 64, &selected);
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(found[0].id, 0);
    EXPECT_EQ(selected.calls, 1);
    EXPECT_EQ(storage.fast_live_reads, 1);
    EXPECT_EQ(storage.fast_live_reads_by_id[0], 1);
}
} // namespace
