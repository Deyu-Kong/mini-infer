/**
 * RMSNorm CUDA kernel — one block per row.
 *
 * Formula:  y = (x / sqrt(mean(x^2) + eps)) * weight
 *
 * Storage layout (row-major):
 *   x       : [N, D]   FP16
 *   weight  : [D]      FP16
 *   y       : [N, D]   FP16
 *
 * Implementation notes:
 *   - All reductions are in FP32 for numerical stability.
 *   - Warp shuffle reduction avoids shared-memory bank conflicts (each lane
 *     produces a partial sum, warp shuffle sums in O(log W) steps).
 *   - One block-reduce via shared memory finalizes the row.
 *   - The fused write multiplies by weight in the same kernel — no extra pass.
 */
#include "kernels/rmsnorm_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/warp_reduce.cuh"

namespace mini_infer {
namespace kernels {

template <int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ y,
    int N, int D, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __half* x_row = x + row * D;
    __half*       y_row = y + row * D;

    // ---- Pass 1: sum of squares --------------------------------------------
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    __shared__ float smem[BLOCK_SIZE / 32];
    sum_sq = block_reduce_sum(sum_sq, smem);

    // rsqrt fused into the affine transform — saves one division per element.
    const float rrms = rsqrtf(sum_sq / static_cast<float>(D) + eps);

    // ---- Pass 2: write normalized + weighted output ------------------------
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float xv = __half2float(x_row[i]);
        float wv = __half2float(weight[i]);
        y_row[i] = __float2half(xv * rrms * wv);
    }
}

void launch_rmsnorm(const __half* x, const __half* weight, __half* y,
                    int N, int D, float eps, cudaStream_t stream) {
    // Pick block size based on D. Qwen2.5 hidden_dim = 3584 (large) so 512
    // threads works well. Smaller dims use smaller blocks.
    if (D >= 4096) {
        rmsnorm_kernel<512><<<N, 512, 0, stream>>>(x, weight, y, N, D, eps);
    } else if (D >= 1024) {
        rmsnorm_kernel<256><<<N, 256, 0, stream>>>(x, weight, y, N, D, eps);
    } else if (D >= 256) {
        rmsnorm_kernel<128><<<N, 128, 0, stream>>>(x, weight, y, N, D, eps);
    } else {
        rmsnorm_kernel<64><<<N, 64, 0, stream>>>(x, weight, y, N, D, eps);
    }
}

}  // namespace kernels
}  // namespace mini_infer