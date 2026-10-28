/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <faiss/transactional/HNSW.h>

namespace faiss::transactional::detail {
namespace scratch_detail {

// A compact table for append-only, operation-local query bookkeeping. Unlike
// a dense visited array, allocation depends on visited IDs, not index size.
// There is no erased state: clear() resets the whole table between operations.
struct IdSlot {
    NodeId id = no_node;
};
struct DistanceSlot {
    NodeId id = no_node;
    float value = 0;
};
static_assert(sizeof(IdSlot) == sizeof(NodeId));
static_assert(sizeof(DistanceSlot) == sizeof(NodeId) + sizeof(float));

inline size_t hash_id(NodeId id) noexcept {
    // Mix high bits too: local IDs near INT32_MAX and regularly spaced IDs
    // should not collapse into clusters in small power-of-two tables.
    uint32_t value = static_cast<uint32_t>(id);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

template <typename Slot>
class IdTable {
    std::vector<Slot> slots_;
    size_t size_ = 0;
    size_t max_entries_;

    static void validate_id(NodeId id) {
        if (id < 0)
            throw std::invalid_argument("Negative HNSW scratch ID");
    }
    static size_t position(const std::vector<Slot>& slots, NodeId id) noexcept {
        const size_t mask = slots.size() - 1;
        size_t pos = hash_id(id) & mask;
        while (slots[pos].id != no_node && slots[pos].id != id)
            pos = (pos + 1) & mask;
        return pos;
    }
    void grow() {
        const size_t before = slots_.size();
        if (before > std::numeric_limits<size_t>::max() / 2)
            throw std::length_error("HNSW scratch capacity exhausted");
        const size_t capacity = before ? before * 2 : 8;
        // Allocate and rehash before changing state. Allocation failure leaves
        // membership and recorded distances intact for the caller's cleanup.
        std::vector<Slot> expanded(capacity);
        for (const auto& slot : slots_) {
            if (slot.id != no_node)
                expanded[position(expanded, slot.id)] = slot;
        }
        slots_.swap(expanded);
    }

   public:
    explicit IdTable(size_t max_entries) : max_entries_(max_entries) {}

    const Slot* find(NodeId id) const {
        validate_id(id);
        if (slots_.empty())
            return nullptr;
        const auto& slot = slots_[position(slots_, id)];
        return slot.id == no_node ? nullptr : &slot;
    }
    bool insert(Slot value) {
        validate_id(value.id);
        size_t pos = 0;
        if (!slots_.empty()) {
            pos = position(slots_, value.id);
            if (slots_[pos].id != no_node)
                return false;
        }
        if (size_ >= max_entries_)
            throw std::length_error("HNSW scratch entry limit exceeded");
        // At most half the slots are occupied. Thus every lookup terminates,
        // and new insertion costs do not explode near a full table.
        if (slots_.empty() || size_ >= slots_.size() / 2) {
            grow();
            pos = position(slots_, value.id);
        }
        slots_[pos] = value;
        ++size_;
        return true;
    }
    void clear() noexcept {
        std::fill(slots_.begin(), slots_.end(), Slot{});
        size_ = 0;
    }
    size_t size() const noexcept {
        return size_;
    }
    size_t capacity() const noexcept {
        return slots_.size();
    }
    size_t memory_bytes() const noexcept {
        return slots_.size() * sizeof(Slot);
    }
};
} // namespace scratch_detail

class IdSet {
    scratch_detail::IdTable<scratch_detail::IdSlot> table_;

   public:
    explicit IdSet(size_t max_entries = std::numeric_limits<size_t>::max())
            : table_(max_entries) {}
    bool insert(NodeId id) {
        return table_.insert({id});
    }
    bool contains(NodeId id) const {
        return table_.find(id) != nullptr;
    }
    void clear() noexcept {
        table_.clear();
    }
    size_t size() const noexcept {
        return table_.size();
    }
    size_t capacity() const noexcept {
        return table_.capacity();
    }
    size_t memory_bytes() const noexcept {
        return table_.memory_bytes();
    }
};

class IdDistanceMap {
    scratch_detail::IdTable<scratch_detail::DistanceSlot> table_;

   public:
    explicit IdDistanceMap(
            size_t max_entries = std::numeric_limits<size_t>::max())
            : table_(max_entries) {}
    // Preserves the first value. Returned pointers remain valid only until
    // the next insertion, clear(), or destruction of this map.
    bool insert(NodeId id, float value) {
        return table_.insert({id, value});
    }
    const float* find(NodeId id) const {
        const auto* slot = table_.find(id);
        return slot ? &slot->value : nullptr;
    }
    void clear() noexcept {
        table_.clear();
    }
    size_t size() const noexcept {
        return table_.size();
    }
    size_t capacity() const noexcept {
        return table_.capacity();
    }
    size_t memory_bytes() const noexcept {
        return table_.memory_bytes();
    }
};
// Validate the common small adjacency list without allocation or sorting.
// Self-links and snapshot ID bounds remain the caller's responsibility.
inline bool unique_nonnegative_ids(std::span<const NodeId> ids) {
    if (ids.empty())
        return true;
    if (ids.size() <= 128) {
        std::array<NodeId, 256> occupied;
        occupied.fill(no_node);
        constexpr size_t mask = 255;
        for (const auto id : ids) {
            if (id < 0)
                return false;
            size_t slot = scratch_detail::hash_id(id) & mask;
            while (occupied[slot] != no_node) {
                if (occupied[slot] == id)
                    return false;
                slot = (slot + 1) & mask;
            }
            occupied[slot] = id;
        }
        return true;
    }
    IdSet unique(ids.size());
    for (const auto id : ids) {
        if (id < 0 || !unique.insert(id))
            return false;
    }
    return true;
}
} // namespace faiss::transactional::detail
