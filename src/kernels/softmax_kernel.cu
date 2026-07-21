/**
 * Softmax CUDA kernel — two-pass, numerically-stable variant.
 *
 * Storage: x [N, D] FP16 -> y [N, D] FP16, row-major.
 *
 * Per row:
 *   pass 1: row_max = max_i x[i]                       (FP32 reduce)
 *   pass 2: row_sum = sum_i exp(x[i] - row_max)        (FP32 reduce)
 *   pass 3: y[i] = exp(x[i] - row_max) / row_sum       (fused)
 *
 * "Online-style" — runs the row-max and row-sum reductions in independent
 * passes (instead of tracking running stats per thread). This is the
 * numerically-stable form; the single-pass online form will be used in
 * Week 5 inside FlashAttention.
 */
#include "kernels/softmax_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/warp_reduce.cuh"

namespace mini_infer {
namespace kernels {

template <int BLOCK_SIZE>
__global__ void softmax_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int N, int D) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* x_row = x + row * D;
    __half*       y_row = y + row * D;

    // ---- pass 1: max -------------------------------------------------------
    float local_max = -INFINITY;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __half2float(x_row[i]));
    }
    __shared__ float smem[BLOCK_SIZE / 32];
    float row_max = block_reduce_max(local_max, smem);

    // ---- pass 2: sum of exp(x - max) --------------------------------------
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        local_sum += expf(__half2float(x_row[i]) - row_max);
    }
    float row_sum = block_reduce_sum(local_sum, smem);
    const float inv_sum = 1.0f / row_sum;

    // ---- pass 3: write -----------------------------------------------------
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = expf(__half2float(x_row[i]) - row_max) * inv_sum;
        y_row[i] = __float2half(v);
    }
}

void launch_softmax(const __half* x, __half* y, int N, int D,
                    cudaStream_t stream) {
    if (D >= 4096) {
        softmax_kernel<512><<<N, 512, 0, stream>>>(x, y, N, D);
    } else if (D >= 1024) {
        softmax_kernel<256><<<N, 256, 0, stream>>>(x, y, N, D);
    } else if (D >= 256) {
        softmax_kernel<128><<<N, 128, 0, stream>>>(x, y, N, D);
    } else {
        softmax_kernel<64><<<N, 64, 0, stream>>>(x, y, N, D);
    }
}

}  // namespace kernels
}  // namespace mini_infer