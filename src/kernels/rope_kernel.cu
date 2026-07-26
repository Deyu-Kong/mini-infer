/**
 * RoPE (Rotary Position Embedding) CUDA kernel.
 *
 * Formula (per Qwen2.5 / LLaMA-style RoPE):
 *
 *   for each (b, s, h, i) with i < head_dim/2:
 *       cos_v = cos_table[b * S + s, i]
 *       sin_v = sin_table[b * S + s, i]
 *       y[b,s,h,i]         = x[b,s,h,i]         * cos_v - x[b,s,h,i+D/2] * sin_v
 *       y[b,s,h,i+D/2]     = x[b,s,h,i+D/2]     * cos_v + x[b,s,h,i]     * sin_v
 *
 * Precomputed cos_table and sin_table live on device with shape
 * [B * S, head_dim/2] (one row per token across the batch).
 *
 * The kernel uses one block per (sequence, head) pair and threads stride
 * across head_dim/2. Thread-per-element is simple and sufficient for
 * Week-2 validation; a vectorized __half2 version is a Week-5+ optimization.
 */
#include "kernels/rope_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

template <int BLOCK_SIZE>
__global__ void rope_kernel(
    const __half* __restrict__ x,        // [B, S, H, D]
    const __half* __restrict__ cos_t,    // [B*S, D/2]
    const __half* __restrict__ sin_t,    // [B*S, D/2]
    __half* __restrict__ y,              // [B, S, H, D]
    int S, int H, int D) {
    const int s   = blockIdx.x;
    const int h   = blockIdx.y;
    const int b   = blockIdx.z;
    const int tid = threadIdx.x;
    const int half = D >> 1;

    const __half* x_row = x + ((b * S + s) * H + h) * D;
    __half*       y_row = y + ((b * S + s) * H + h) * D;
    // cos/sin indexed by global token index = b*S + s (per-batch positions).
    const __half* cos_row = cos_t + (b * S + s) * half;
    const __half* sin_row = sin_t + (b * S + s) * half;

    for (int i = tid; i < half; i += BLOCK_SIZE) {
        float xi  = __half2float(x_row[i]);
        float xj  = __half2float(x_row[i + half]);
        float c   = __half2float(cos_row[i]);
        float sn  = __half2float(sin_row[i]);
        y_row[i]        = __float2half(xi * c - xj * sn);
        y_row[i + half] = __float2half(xj * c + xi * sn);
    }
}

void launch_rope(const __half* x, const __half* cos_t, const __half* sin_t,
                 __half* y, int B, int S, int H, int D, cudaStream_t stream) {
    dim3 grid(S, H, B);
    constexpr int BLOCK = 128;
    rope_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(x, cos_t, sin_t, y, S, H, D);
}

// Precompute cos/sin tables of shape [B*S, half] given flat positions [B*S].
// Per-token positions allow batched decode with different seq_lens per request.
__global__ void rope_precompute_kernel(
    const float* __restrict__ inv_freq,  // [half]
    const int64_t* __restrict__ positions, // [B*S]
    __half* __restrict__ cos_t,          // [B*S, half]
    __half* __restrict__ sin_t,          // [B*S, half]
    int N, int half) {
    const int n = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    const float angle = static_cast<float>(positions[n]) * inv_freq[i];
    cos_t[n * half + i] = __float2half(cosf(angle));
    sin_t[n * half + i] = __float2half(sinf(angle));
}

void launch_rope_precompute(const float* inv_freq_h, const int64_t* positions_h,
                            __half* cos_d, __half* sin_d,
                            int S, int half, cudaStream_t stream) {
    // The public API still takes positions of length S for backward compat.
    // Internally we treat N = S.
    constexpr int N_FACTOR = 1;
    (void)N_FACTOR;
    const int N = S;
    float* inv_freq_d = nullptr;
    int64_t* positions_d = nullptr;
    cudaMalloc(&inv_freq_d, half * sizeof(float));
    cudaMalloc(&positions_d, N * sizeof(int64_t));
    cudaMemcpyAsync(inv_freq_d, inv_freq_h, half * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(positions_d, positions_h, N * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(128);
    dim3 grid((half + 127) / 128, N);
    rope_precompute_kernel<<<grid, block, 0, stream>>>(
        inv_freq_d, positions_d, cos_d, sin_d, N, half);

    cudaFree(inv_freq_d);
    cudaFree(positions_d);
}

// Batched variant: positions of length B*S (flat), tables of shape
// [B*S, half].
void launch_rope_precompute_batched(const float* inv_freq_h,
                                    const int64_t* positions_h,
                                    __half* cos_d, __half* sin_d,
                                    int B, int S, int half,
                                    cudaStream_t stream) {
    const int N = B * S;
    float* inv_freq_d = nullptr;
    int64_t* positions_d = nullptr;
    cudaMalloc(&inv_freq_d, half * sizeof(float));
    cudaMalloc(&positions_d, N * sizeof(int64_t));
    cudaMemcpyAsync(inv_freq_d, inv_freq_h, half * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(positions_d, positions_h, N * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    dim3 block(128);
    dim3 grid((half + 127) / 128, N);
    rope_precompute_kernel<<<grid, block, 0, stream>>>(
        inv_freq_d, positions_d, cos_d, sin_d, N, half);

    cudaFree(inv_freq_d);
    cudaFree(positions_d);
}

}  // namespace kernels
}  // namespace mini_infer