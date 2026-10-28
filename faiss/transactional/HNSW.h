/*
 * Copyright (c) Aperture Data Inc.
 * HNSW traversal adapted from Meta Platforms' faiss/impl/HNSW.cpp.
 * Licensed under the MIT license in the root LICENSE file.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <faiss/MetricType.h>
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
    // One packed adjacency list per level; level zero has capacity 2*M.
    std::vector<std::vector<NodeId>> links;
};

/** All methods operate in ONE caller-owned transaction, including reads of
 * pending writes. An instance must not outlive that transaction or be shared
 * concurrently. On a mutation/flush failure the caller MUST abort the entire
 * transaction. Neither this interface nor HNSW commits or starts transactions.
 * Implementations must reject further use after invalidate().
 */
struct Storage {
    virtual ~Storage() = default;
    virtual std::optional<Metadata> metadata() const = 0;
    virtual Node node(NodeId id) const = 0;
    virtual std::vector<float> vector(NodeId id) const = 0;
    virtual void put_metadata(const Metadata& value) = 0;
    virtual void put_node(NodeId id, const Node& value) = 0;
    virtual void put_vector(NodeId id, const std::vector<float>& value) = 0;
    virtual void invalidate() noexcept = 0;
};

struct Result {
    NodeId id;
    // Squared L2 distance or inner product, as in the ordinary FAISS API.
    float distance;
};

/** A transaction-bound HNSW view. The storage is authoritative: no mutable
 * topology, count, or entry-point state is retained across API operations.
 * Uses FAISS distance kernels and neighbor diversification. This is an opt-in
 * engine, not a replacement for IndexHNSWFlat and not a FAISS file format.
 */
class HNSW {
   public:
    explicit HNSW(Storage& storage) : storage_(storage) {}
    void create(const Config& config);
    Metadata metadata() const;
    std::vector<NodeId> add(size_t n, const float* vectors);
    // Tombstones remain traversable. IDs are never reused in this namespace.
    bool erase(NodeId id);
    std::vector<Result> search(
            const float* query,
            size_t k,
            size_t ef_search = 64,
            const IDSelector* selector = nullptr) const;

   private:
    Storage& storage_;
};

} // namespace faiss::transactional
