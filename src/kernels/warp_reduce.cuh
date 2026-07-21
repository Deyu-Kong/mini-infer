#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Warp-level sum reduction using shuffle.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// Warp-level max reduction.
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    }
    return v;
}

// Block-level sum reduction (assumes blockDim.x <= 1024 and power-of-2).
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kMaxWarps = 32;  // 1024 threads / 32

    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (tid < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// Block-level max reduction.
__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (tid < (blockDim.x + 31) / 32) ? smem[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

}  // namespace kernels
}  // namespace mini_infer