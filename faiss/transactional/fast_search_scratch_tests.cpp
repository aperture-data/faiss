/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/FastSearchScratch.h>
#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ft = faiss::transactional;
namespace scratch = faiss::transactional::detail;
namespace {
TEST(TransactionalHNSWScratch, AllocatesByVisitsRatherThanIdMagnitude) {
    scratch::IdSet ids;
    scratch::IdDistanceMap distances;
    EXPECT_EQ(ids.memory_bytes(), 0);
    EXPECT_EQ(distances.memory_bytes(), 0);
    const auto id = std::numeric_limits<ft::NodeId>::max();
    EXPECT_TRUE(ids.insert(id));
    EXPECT_TRUE(ids.contains(id));
    EXPECT_TRUE(distances.insert(id, 3.5f));
    ASSERT_NE(distances.find(id), nullptr);
    EXPECT_FLOAT_EQ(*distances.find(id), 3.5f);
    EXPECT_EQ(ids.memory_bytes(), 8 * sizeof(ft::NodeId));
    EXPECT_EQ(
            distances.memory_bytes(), 8 * (sizeof(ft::NodeId) + sizeof(float)));
}

TEST(TransactionalHNSWScratch, DuplicatesPreserveFirstDistanceAtCapacity) {
    scratch::IdSet ids(1);
    scratch::IdDistanceMap distances(1);
    EXPECT_TRUE(ids.insert(0));
    EXPECT_FALSE(ids.insert(0));
    EXPECT_TRUE(distances.insert(0, -17.5f));
    EXPECT_FALSE(distances.insert(0, 9));
    EXPECT_FLOAT_EQ(*distances.find(0), -17.5f);
    EXPECT_THROW(ids.insert(1), std::length_error);
    EXPECT_THROW(distances.insert(1, 4), std::length_error);
    EXPECT_TRUE(ids.contains(0));
    EXPECT_FALSE(ids.contains(1));
    EXPECT_EQ(distances.find(1), nullptr);
    EXPECT_EQ(ids.size(), 1);
    EXPECT_EQ(distances.size(), 1);
}

TEST(TransactionalHNSWScratch, EmptyLimitAndNegativeIdsDoNotAllocate) {
    scratch::IdSet ids(0);
    scratch::IdDistanceMap distances(0);
    EXPECT_FALSE(ids.contains(0));
    EXPECT_EQ(distances.find(0), nullptr);
    EXPECT_THROW(ids.insert(0), std::length_error);
    EXPECT_THROW(distances.insert(0, 1), std::length_error);
    EXPECT_THROW(ids.insert(-1), std::invalid_argument);
    EXPECT_THROW(ids.contains(-2), std::invalid_argument);
    EXPECT_THROW(distances.insert(-1, 1), std::invalid_argument);
    EXPECT_THROW(distances.find(-2), std::invalid_argument);
    EXPECT_EQ(ids.memory_bytes(), 0);
    EXPECT_EQ(distances.memory_bytes(), 0);
}

TEST(TransactionalHNSWScratch, GrowthRetainsAllValuesAndHalfEmptyInvariant) {
    scratch::IdSet ids;
    scratch::IdDistanceMap distances;
    for (ft::NodeId id = 0; id < 4096; ++id) {
        ASSERT_TRUE(ids.insert(id));
        ASSERT_TRUE(distances.insert(id, id * -0.25f));
        EXPECT_LE(2 * ids.size(), ids.capacity());
        EXPECT_LE(2 * distances.size(), distances.capacity());
        EXPECT_LE(ids.capacity(), std::max<size_t>(8, 4 * ids.size()));
        EXPECT_LE(
                distances.capacity(),
                std::max<size_t>(8, 4 * distances.size()));
    }
    for (ft::NodeId id = 0; id < 4096; ++id) {
        EXPECT_TRUE(ids.contains(id));
        ASSERT_NE(distances.find(id), nullptr);
        EXPECT_FLOAT_EQ(*distances.find(id), id * -0.25f);
    }
    EXPECT_FALSE(ids.contains(4096));
    EXPECT_EQ(distances.find(4096), nullptr);
}

TEST(TransactionalHNSWScratch, ClearDiscardsMembershipAndDistances) {
    scratch::IdSet ids(64);
    scratch::IdDistanceMap distances(64);
    for (int id = 0; id < 64; ++id) {
        ids.insert(id);
        distances.insert(id, id + 0.5f);
    }
    const auto capacity = ids.capacity();
    ids.clear();
    distances.clear();
    EXPECT_EQ(ids.size(), 0);
    EXPECT_EQ(distances.size(), 0);
    EXPECT_EQ(ids.capacity(), capacity);
    for (int id = 0; id < 64; ++id) {
        EXPECT_FALSE(ids.contains(id));
        EXPECT_EQ(distances.find(id), nullptr);
        EXPECT_TRUE(ids.insert(id));
        EXPECT_TRUE(distances.insert(id, -1));
        EXPECT_FLOAT_EQ(*distances.find(id), -1);
    }
}

TEST(TransactionalHNSWScratch, DifferentialRandomAndRegularlySpacedIds) {
    scratch::IdSet ids;
    scratch::IdDistanceMap distances;
    std::unordered_set<ft::NodeId> reference_set;
    std::unordered_map<ft::NodeId, float> reference_map;
    std::mt19937 rng(19429);
    for (size_t i = 0; i < 20000; ++i) {
        const ft::NodeId id = i % 3 == 0
                ? static_cast<ft::NodeId>((i % 1000) << 16)
                : static_cast<ft::NodeId>(rng() & 0x7fffffffU);
        const float value = static_cast<float>(i);
        EXPECT_EQ(ids.insert(id), reference_set.insert(id).second);
        EXPECT_EQ(
                distances.insert(id, value),
                reference_map.emplace(id, value).second);
        EXPECT_TRUE(ids.contains(id));
        ASSERT_NE(distances.find(id), nullptr);
        EXPECT_FLOAT_EQ(*distances.find(id), reference_map.at(id));
    }
    EXPECT_EQ(ids.size(), reference_set.size());
    EXPECT_EQ(distances.size(), reference_map.size());
    for (const auto& [id, value] : reference_map) {
        EXPECT_TRUE(ids.contains(id));
        ASSERT_NE(distances.find(id), nullptr);
        EXPECT_FLOAT_EQ(*distances.find(id), value);
    }
}

TEST(TransactionalHNSWScratch,
     AdjacencyValidationHandlesThresholdAndCorruption) {
    EXPECT_TRUE(scratch::unique_nonnegative_ids({}));
    for (const size_t count : {1, 64, 128, 129, 8192}) {
        std::vector<ft::NodeId> ids(count);
        for (size_t i = 0; i < count; ++i)
            ids[i] = static_cast<ft::NodeId>(i);
        ids.back() = std::numeric_limits<ft::NodeId>::max();
        EXPECT_TRUE(scratch::unique_nonnegative_ids(ids));
        if (count > 1) {
            ids.back() = ids.front();
            EXPECT_FALSE(scratch::unique_nonnegative_ids(ids));
        }
        ids.back() = -1;
        EXPECT_FALSE(scratch::unique_nonnegative_ids(ids));
        ids.back() = std::numeric_limits<ft::NodeId>::min();
        EXPECT_FALSE(scratch::unique_nonnegative_ids(ids));
    }
}

TEST(TransactionalHNSWScratch,
     AdjacencyValidationResolvesStackTableCollisions) {
    // Force every key to the same initial bucket. The half-empty table must
    // still terminate, preserve uniqueness, and detect the repeated last key.
    std::vector<ft::NodeId> ids;
    for (ft::NodeId id = 0; ids.size() < 128; ++id) {
        if ((scratch::scratch_detail::hash_id(id) & 255) == 0)
            ids.push_back(id);
    }
    EXPECT_TRUE(scratch::unique_nonnegative_ids(ids));
    ids.back() = ids.front();
    EXPECT_FALSE(scratch::unique_nonnegative_ids(ids));
}

TEST(TransactionalHNSWScratch, AdjacencyValidationMatchesReferenceAcrossSizes) {
    std::mt19937 rng(128129);
    for (size_t count = 0; count < 260; ++count) {
        std::vector<ft::NodeId> ids(count);
        std::unordered_set<ft::NodeId> reference;
        bool expected = true;
        for (auto& id : ids) {
            id = static_cast<ft::NodeId>(rng() % 4096);
            expected &= reference.insert(id).second;
        }
        EXPECT_EQ(scratch::unique_nonnegative_ids(ids), expected)
                << "size " << count;
    }
}
} // namespace
