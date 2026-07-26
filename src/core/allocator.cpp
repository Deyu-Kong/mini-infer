#include "core/allocator.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace mini_infer {

namespace {
inline void cuda_check_(cudaError_t err, const char* expr, const char* file,
                        int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error ") + expr + " at " + file + ":" +
            std::to_string(line) + " : " + cudaGetErrorString(err));
    }
}
#define MI_CHECK_CUDA(expr) cuda_check_((expr), #expr, __FILE__, __LINE__)
}  // namespace

BumpAllocator::BumpAllocator(std::size_t capacity_bytes, int device_index)
    : capacity_(capacity_bytes), device_index_(device_index) {
    if (capacity_ == 0) return;
    MI_CHECK_CUDA(cudaSetDevice(device_index_));
    MI_CHECK_CUDA(cudaMalloc(&base_, capacity_));
}

BumpAllocator::~BumpAllocator() {
    if (base_) {
        cudaFree(base_);
        base_ = nullptr;
    }
}

void* BumpAllocator::allocate(std::size_t bytes) {
    if (bytes == 0) return static_cast<char*>(base_) + used_;  // returns current cursor
    // Align the next cursor up to kAlignment.
    const std::size_t aligned_used =
        (used_ + (kAlignment - 1)) & ~(kAlignment - 1);
    if (aligned_used + bytes > capacity_) return nullptr;
    void* p = static_cast<char*>(base_) + aligned_used;
    used_ = aligned_used + bytes;
    if (used_ > peak_) peak_ = used_;
    return p;
}

void BumpAllocator::reset() {
    used_ = 0;
    // peak_ intentionally preserved for diagnostics.
}

// =========================================================================
// BlockAllocator (Week 5, PagedAttention)
// =========================================================================

BlockAllocator::BlockAllocator(int num_blocks,
                               int num_layers,
                               int num_kv_heads,
                               int head_dim,
                               int device_index)
    : num_blocks_(num_blocks),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      device_index_(device_index) {
    if (num_blocks_ <= 0 || num_layers_ <= 0 || num_kv_heads_ <= 0
        || head_dim_ <= 0) {
        throw std::runtime_error("BlockAllocator: invalid dimensions");
    }
    // Per-block bytes = kv_heads * BLOCK_SIZE * head_dim * sizeof(__half).
    block_bytes_ = static_cast<std::size_t>(num_kv_heads_)
                 * static_cast<std::size_t>(BlockAllocator::kBlockSize)
                 * static_cast<std::size_t>(head_dim_)
                 * sizeof(__half);
    total_bytes_ = block_bytes_ * static_cast<std::size_t>(num_layers_)
                 * static_cast<std::size_t>(num_blocks_);

    MI_CHECK_CUDA(cudaSetDevice(device_index_));
    MI_CHECK_CUDA(cudaMalloc(&k_storage_, total_bytes_));
    MI_CHECK_CUDA(cudaMalloc(&v_storage_, total_bytes_));

    free_list_.reserve(num_blocks_);
    for (int i = num_blocks_ - 1; i >= 0; --i) {
        free_list_.push_back(i);   // LIFO so recent frees are reused first
    }
    ref_count_.assign(num_blocks_, 0);
}

BlockAllocator::~BlockAllocator() {
    if (k_storage_) { cudaFree(k_storage_); k_storage_ = nullptr; }
    if (v_storage_) { cudaFree(v_storage_); v_storage_ = nullptr; }
}

BlockAllocator::BlockAllocator(BlockAllocator&& other) noexcept
    : num_blocks_(other.num_blocks_),
      num_layers_(other.num_layers_),
      num_kv_heads_(other.num_kv_heads_),
      head_dim_(other.head_dim_),
      device_index_(other.device_index_),
      block_bytes_(other.block_bytes_),
      total_bytes_(other.total_bytes_),
      k_storage_(other.k_storage_),
      v_storage_(other.v_storage_),
      free_list_(std::move(other.free_list_)),
      ref_count_(std::move(other.ref_count_)) {
    other.k_storage_ = nullptr;
    other.v_storage_ = nullptr;
    other.num_blocks_ = 0;
}

BlockAllocator& BlockAllocator::operator=(BlockAllocator&& other) noexcept {
    if (this != &other) {
        if (k_storage_) cudaFree(k_storage_);
        if (v_storage_) cudaFree(v_storage_);
        num_blocks_   = other.num_blocks_;
        num_layers_   = other.num_layers_;
        num_kv_heads_ = other.num_kv_heads_;
        head_dim_     = other.head_dim_;
        device_index_ = other.device_index_;
        block_bytes_  = other.block_bytes_;
        total_bytes_  = other.total_bytes_;
        k_storage_    = other.k_storage_;
        v_storage_    = other.v_storage_;
        free_list_    = std::move(other.free_list_);
        ref_count_    = std::move(other.ref_count_);
        other.k_storage_ = nullptr;
        other.v_storage_ = nullptr;
        other.num_blocks_ = 0;
    }
    return *this;
}

int BlockAllocator::alloc() {
    if (free_list_.empty()) return -1;
    const int b = free_list_.back();
    free_list_.pop_back();
    ref_count_[b] = 1;
    return b;
}

void BlockAllocator::free(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("BlockAllocator::free: bad block_id");
    }
    if (ref_count_[block_id] <= 0) {
        throw std::runtime_error("BlockAllocator::free: ref count underflow");
    }
    --ref_count_[block_id];
    if (ref_count_[block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

void BlockAllocator::ref(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("BlockAllocator::ref: bad block_id");
    }
    ++ref_count_[block_id];
}

void BlockAllocator::unref(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("BlockAllocator::unref: bad block_id");
    }
    if (ref_count_[block_id] <= 0) {
        throw std::runtime_error("BlockAllocator::unref: ref count underflow");
    }
    --ref_count_[block_id];
    if (ref_count_[block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

void* BlockAllocator::k_block_ptr(int layer, int block_id) const {
    if (layer < 0 || layer >= num_layers_
        || block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("BlockAllocator::k_block_ptr: bad indices");
    }
    char* base = static_cast<char*>(k_storage_);
    const std::size_t off =
        (static_cast<std::size_t>(layer) * num_blocks_ + block_id) * block_bytes_;
    return base + off;
}

void* BlockAllocator::v_block_ptr(int layer, int block_id) const {
    if (layer < 0 || layer >= num_layers_
        || block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("BlockAllocator::v_block_ptr: bad indices");
    }
    char* base = static_cast<char*>(v_storage_);
    const std::size_t off =
        (static_cast<std::size_t>(layer) * num_blocks_ + block_id) * block_bytes_;
    return base + off;
}

int BlockAllocator::in_use_blocks() const {
    int n = 0;
    for (int r : ref_count_) if (r > 0) ++n;
    return n;
}

}  // namespace mini_infer