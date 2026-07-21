#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mini_infer {

/**
 * BumpAllocator — linear, non-freeing device memory pool.
 *
 * Week 1: a single large cudaMalloc'd buffer; allocate() carves out
 * 256-byte-aligned sub-buffers in monotonically increasing order.
 * reset() frees everything at once by resetting an offset cursor.
 *
 * Week 5+: this becomes the foundation of the block pool used by
 * PagedAttention (each block_size tokens' worth of KV Cache is one
 * allocation unit).
 */
class BumpAllocator {
public:
    explicit BumpAllocator(std::size_t capacity_bytes,
                           int device_index = 0);
    ~BumpAllocator();

    BumpAllocator(const BumpAllocator&) = delete;
    BumpAllocator& operator=(const BumpAllocator&) = delete;

    // Returns a 256-byte-aligned pointer inside the pool, or nullptr
    // if the request would exceed capacity.
    void* allocate(std::size_t bytes);

    // Reset the bump cursor back to zero. Memory stays mapped; only
    // the offset is reset.
    void reset();

    std::size_t capacity() const { return capacity_; }
    std::size_t used()     const { return used_; }
    std::size_t peak()     const { return peak_; }
    int device_index()     const { return device_index_; }
    void* base()           const { return base_; }

private:
    void* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
    std::size_t peak_ = 0;
    int device_index_ = 0;
    static constexpr std::size_t kAlignment = 256;
};

}  // namespace mini_infer