#include "core/allocator.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

}  // namespace mini_infer