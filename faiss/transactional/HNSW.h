/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Aperture Data Inc.
 * HNSW traversal adapted from faiss/impl/HNSW.cpp.
 * Licensed under the MIT license in the root LICENSE file.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

namespace faiss::transactional {
using NodeId = int32_t;
constexpr NodeId no_node = -1;

struct Config {
    uint32_t dimensions = 0;
    uint32_t M = 32;
    uint32_t ef_construction = 200;
    MetricType metric = METRIC_L2;
    uint64_t seed = 12345;
};
struct Metadata {
    Config config;
    uint64_t next_id = 0;
    uint64_t live_count = 0;
    NodeId entry_point = no_node;
    int32_t max_level = -1;
};
struct Node {
    bool deleted = false;
    std::vector<std::vector<NodeId>> links;
};
struct WriteBatch {
    Metadata metadata;
    std::map<NodeId, Node> nodes;
    std::map<NodeId, std::vector<float>> vectors;
};

// A storage failure can require aborting the caller's entire transaction.
// Inspect abort_required(), not the error text. The original exception is
// nested.
class MutationError : public std::runtime_error {
   public:
    MutationError(std::string message, bool abort_required)
            : std::runtime_error(std::move(message)),
              abort_required_(abort_required) {}
    bool abort_required() const noexcept {
        return abort_required_;
    }

   private:
    bool abort_required_;
};
class SearchCancelled : public std::runtime_error {
   public:
    SearchCancelled() : std::runtime_error("HNSW operation cancelled") {}
};

struct SearchStats;

// Optional fast tombstone access implemented by a query distance computer.
// It shares the computer's snapshot and operation-scoped lifetime. Invalid or
// missing IDs must throw; the answer must agree with Storage::node(id).
struct QueryLiveAccess {
    virtual ~QueryLiveAccess() = default;
    virtual bool deleted(NodeId id) const = 0;
};

/** One caller-owned transaction including pending writes. Implementations must
 * reject further access after invalidate(). Instances and all their views must
 * be destroyed before the caller ends the transaction. No concurrent access.
 */
struct Storage {
    virtual ~Storage() = default;
    virtual std::optional<Metadata> metadata() const = 0;
    virtual Node node(NodeId id) const = 0;
    virtual std::vector<float> vector(NodeId id) const = 0;
    // Optional operation-scoped fast query access. Distances are ascending:
    // squared L2 or NEGATIVE inner product. No writes, callback reentry, or
    // transaction end may occur while the computer exists. A null result uses
    // the owned vector() fallback. Borrowed backend data never outlives it.
    virtual std::unique_ptr<faiss::DistanceComputer> query_distance_computer(
            const Metadata&,
            size_t /*cache_bytes*/,
            SearchStats*) const {
        return nullptr;
    }
    virtual void put_metadata(const Metadata& value) = 0;
    virtual void put_node(NodeId id, const Node& value) = 0;
    virtual void put_vector(NodeId id, const std::vector<float>& value) = 0;
    virtual void invalidate() noexcept = 0;
    // Default implementation writes records, then metadata. On any failure it
    // invalidates the adapter and throws MutationError(abort_required=true).
    // Backends with savepoints can provide strong operation-level atomicity.
    virtual void apply(const WriteBatch& batch);
};

struct Result {
    NodeId id;
    // Squared L2 distance or inner product; smallest/largest first
    // respectively.
    float distance;
};
struct SearchStats {
    size_t visited = 0;  // unique vectors whose query distance was computed
    size_t expanded = 0; // adjacency lists expanded (including upper levels)
    size_t vector_reads = 0;
    size_t node_reads = 0;
    size_t vector_block_reads = 0; // packed backend cache misses
    size_t live_block_reads = 0;   // packed tombstone cache misses
    bool visit_limit_reached = false;
};
struct SearchOptions {
    size_t ef_search = 64;
    // Zero selects a finite ANN budget derived from ef, M and graph height.
    // In exact mode zero scans all IDs. Nonzero is a hard limit on unique query
    // distance evaluations, including upper-level routing and supplied seeds.
    size_t max_visits = 0;
    // Cache allowance: 3/4 for owned nodes, 1/4 for optional backend views.
    // Search does not retain decoded vectors. Distance/visited bookkeeping
    // is bounded separately by the visit budget or a fixed small-index cap.
    size_t cache_bytes = 8u << 20;
    // Predicates and cancellation callbacks must not mutate the transaction
    // or reenter the index. Their referenced state must outlive this call.
    const IDSelector* selector = nullptr;
    std::function<bool()> cancelled;
    // A completed exact scan guarantees top-k within floating point arithmetic,
    // including disconnected components. An explicit max_visits can truncate
    // even this mode: inspect SearchStats::visit_limit_reached. Approximate
    // search can return <k without exhausting its visit budget.
    bool exact = false;
    // Optional predicate-matching entry points, e.g. chosen from a graph query.
    // Bounded to 64. They share the same snapshot and work budget as routing.
    std::span<const NodeId> seed_ids;
    // Explore accepted one/two-hop neighbors through rejected bridge nodes.
    // Requires a selector. Use seeds for selective predicates in distant
    // regions.
    bool acorn = false;
};
struct MutationOptions {
    size_t max_batch_size = 10000;
    // Bounds buffered dirty records. Read caches are separately bounded below.
    size_t max_workspace_bytes = 256u << 20;
    // Zero derives a finite ef/M/height bound. Exceeding it aborts this
    // operation before applying any records; traversal bookkeeping is O(this
    // limit). Counts query-to-candidate evaluations; symmetric neighbor
    // diversification performs additional pairwise distances.
    size_t max_visits_per_insertion = 0;
    size_t cache_bytes = 8u << 20;
    // Must not mutate the transaction or reenter the index.
    std::function<bool()> cancelled;
};
struct ValidationReport {
    uint64_t nodes = 0;
    uint64_t live_nodes = 0;
    uint64_t edges = 0;
};

/** HNSW over a transaction snapshot. No mutable global graph or query caches.
 * The caller owns commit/abort. Existing FAISS indexes remain independent.
 */
class HNSW {
   public:
    explicit HNSW(Storage& storage) : storage_(storage) {}
    void create(const Config& config);
    Metadata metadata() const;
    std::vector<NodeId> add(size_t n, const float* vectors);
    std::vector<NodeId> add(
            size_t n,
            const float* vectors,
            const MutationOptions& options);
    // One atomic mutation: tombstone old ID, insert replacement; no ID reuse.
    NodeId update(
            NodeId id,
            const float* vector,
            const MutationOptions& options = {});
    bool erase(NodeId id);
    std::vector<Result> search(
            const float* query,
            size_t k,
            size_t ef_search = 64,
            const IDSelector* selector = nullptr) const;
    std::vector<Result> search(
            const float* query,
            size_t k,
            const SearchOptions& options,
            SearchStats* stats = nullptr) const;
    // Administrative full scan. Checks vectors, counts, degree/ID/level
    // invariants and the entry point. Does not assert ANN graph connectivity.
    ValidationReport validate(
            const std::function<bool()>& cancelled = {}) const;

   private:
    Storage& storage_;
};
} // namespace faiss::transactional
