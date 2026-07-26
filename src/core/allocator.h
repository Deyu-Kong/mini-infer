#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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

/**
 * BlockAllocator — Week 5 PagedAttention block pool.
 *
 * Manages a fixed pool of physical blocks, each holding `block_size`
 * consecutive tokens of K (or V). Alloc / free are O(1) via a free-list
 * stack; per-block reference counts support future prefix-cache sharing.
 *
 * Storage layout (per layer; we keep K and V in separate tensors):
 *
 *   K / V : [num_layers, num_blocks, num_kv_heads, block_size, head_dim]   FP16
 *
 * Block b in layer L starts at byte offset:
 *
 *   (L * num_blocks + b) * num_kv_heads * block_size * head_dim * 2
 *
 * Within a block, the layout is [num_kv_heads, block_size, head_dim].
 * This keeps the per-(kv_head, token) slice contiguous, which is what
 * the PagedAttention kernel needs.
 *
 * Block size is fixed at 16 (industry standard; small enough to keep
 * internal fragmentation low for short requests, large enough to make
 * the block_table lookup cheap).
 */
class BlockAllocator {
public:
    static constexpr int kBlockSize = 16;

    BlockAllocator(int num_blocks,
                   int num_layers,
                   int num_kv_heads,
                   int head_dim,
                   int device_index);

    ~BlockAllocator();

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;
    BlockAllocator(BlockAllocator&& other) noexcept;
    BlockAllocator& operator=(BlockAllocator&& other) noexcept;

    // Returns a fresh block_id in [0, num_blocks), or -1 if the pool
    // is exhausted (OOM).
    int alloc();

    // Returns a block to the pool. Decrements its ref count; the block
    // is only returned to the free list when ref count hits zero.
    void free(int block_id);

    // Increment / decrement ref count without going through alloc/free
    // (used when sharing a block across sequences for prefix cache).
    void ref(int block_id);
    void unref(int block_id);

    // K storage pointer for (layer, block).
    void* k_block_ptr(int layer, int block_id) const;
    // V storage pointer for (layer, block).
    void* v_block_ptr(int layer, int block_id) const;

    // Number of bytes each block occupies (per K or per V).
    std::size_t block_bytes() const { return block_bytes_; }

    int num_blocks()      const { return num_blocks_; }
    int num_free_blocks() const { return static_cast<int>(free_list_.size()); }
    int ref_count(int block_id) const {
        return (block_id >= 0 && block_id < num_blocks_) ? ref_count_[block_id] : 0;
    }

    // Fragmentation metrics (test-only).
    //   in_use_blocks = blocks with ref_count > 0
    //   wasted_tokens = sum over in_use blocks of (BLOCK_SIZE - tokens_used)
    //   wasted_ratio  = wasted_tokens / (in_use_blocks * BLOCK_SIZE)
    // Caller is responsible for tracking tokens_used per block externally
    // (via PagedKVCache); here we just report the block-pool state.
    int in_use_blocks() const;

private:
    int  num_blocks_      = 0;
    int  num_layers_      = 0;
    int  num_kv_heads_    = 0;
    int  head_dim_        = 0;
    int  device_index_    = 0;
    std::size_t block_bytes_ = 0;   // bytes per block (K or V)
    std::size_t total_bytes_ = 0;   // bytes for K (== bytes for V)

    // Two flat device buffers (one for K, one for V). Each holds
    //   num_layers * num_blocks * num_kv_heads * block_size * head_dim  FP16
    void* k_storage_ = nullptr;
    void* v_storage_ = nullptr;

    std::vector<int> free_list_;
    std::vector<int> ref_count_;
};

}  // namespace mini_infer