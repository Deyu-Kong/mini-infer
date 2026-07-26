/**
 * Naive (non-Flash) attention CUDA kernel — Week 4 baseline.
 *
 * Computes scaled dot-product attention with optional causal masking:
 *
 *   For each (b, h_q, sq):
 *     h_kv = h_q / num_kv_groups                          (GQA)
 *     scores[j] = (Q[b,sq,h_q,:] . K[b,j,h_kv,:]) * scale
 *     if is_prefill: scores[j] = -inf for j > sq         (causal)
 *     probs  = softmax(scores)
 *     out[b,sq,h_q,:] = sum_j probs[j] * V[b,j,h_kv,:]
 *
 * Layout (row-major):
 *   Q   : [B, S_q, H_q, D]
 *   K   : [B, S_k, H_kv, D]
 *   V   : [B, S_k, H_kv, D]
 *   Out : [B, S_q, H_q, D]
 *
 * Implementation:
 *   - One CUDA block per (b, h_q, sq).
 *   - BLOCK_THREADS == D (head_dim). Each thread is responsible for
 *     one element of Q / Out, and participates in the score / softmax
 *     reductions cooperatively.
 *   - Scores live in dynamic shared memory (size = S_k * sizeof(float)).
 *
 * Limitations vs FlashAttention (W5):
 *   - Scores are materialized in full -> O(S_k) shared memory + a sync
 *     between phases. Decode is fine (S_k ≤ a few thousand); prefill on
 *     very long prompts (>4k) will run out of shared memory.
 *   - No vectorization (8-byte fp16) — every load is a 2-byte transaction.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

template <int D, int MAX_SK>
__global__ void naive_attn_kernel(
    const __half* __restrict__ Q,        // [B, S_q, H_q, D]
    const __half* __restrict__ K,        // [B, S_k, H_kv, D]
    const __half* __restrict__ V,        // [B, S_k, H_kv, D]
    __half* __restrict__ Out,            // [B, S_q, H_q, D]
    int S_q, int S_k, int H_q, int H_kv,
    int num_kv_groups,
    float scale,
    int is_prefill) {
    const int b   = blockIdx.z;
    const int h_q = blockIdx.y;
    const int sq  = blockIdx.x;
    const int tid = threadIdx.x;        // 0..D-1

    const int h_kv = h_q / num_kv_groups;

    // -------- shared memory layout ---------------------------------------
    __shared__ float q_smem[D];
    extern __shared__ float scores[];    // [S_k]

    // -------- load Q for this (b, h_q, sq) -------------------------------
    const int q_off = ((b * S_q + sq) * H_q + h_q) * D;
    q_smem[tid] = __half2float(Q[q_off + tid]);
    __syncthreads();

    // -------- compute scores for each key position -----------------------
    for (int j = 0; j < S_k; ++j) {
        if (is_prefill && j > sq) {
            scores[j] = -INFINITY;
            continue;
        }
        float s = 0.0f;
        const int k_off = ((b * S_k + j) * H_kv + h_kv) * D;
        for (int d = 0; d < D; ++d) {
            s += q_smem[d] * __half2float(K[k_off + d]);
        }
        scores[j] = s * scale;
    }
    __syncthreads();

    // -------- softmax (numerically stable, in shared memory) ------------
    // Find max via a block-level reduction.
    float thread_max = -INFINITY;
    for (int j = tid; j < S_k; j += D) {
        thread_max = fmaxf(thread_max, scores[j]);
    }
    
    // Block-level reduction for max
    __shared__ float smem_reduce[D / 32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max,
                           __shfl_xor_sync(0xFFFFFFFFu, thread_max, offset));
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        smem_reduce[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp participates)
    __shared__ float row_max;
    if (warp_id == 0) {
        float val = (lane_id < D / 32) ? smem_reduce[lane_id] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
        }
        if (lane_id == 0) row_max = val;
    }
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = tid; j < S_k; j += D) {
        scores[j] = __expf(scores[j] - row_max);
        thread_sum += scores[j];
    }
    
    // Block-level reduction for sum
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xFFFFFFFFu, thread_sum, offset);
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        smem_reduce[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp participates)
    __shared__ float row_sum;
    if (warp_id == 0) {
        float val = (lane_id < D / 32) ? smem_reduce[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
        }
        if (lane_id == 0) row_sum = val;
    }
    __syncthreads();
    const float inv_sum = 1.0f / row_sum;

    // -------- output = sum_j probs[j] * V[j, h_kv, tid] ----------------
    float out = 0.0f;
    for (int j = 0; j < S_k; ++j) {
        const float p = scores[j] * inv_sum;
        const int v_off = ((b * S_k + j) * H_kv + h_kv) * D;
        out += p * __half2float(V[v_off + tid]);
    }
    const int out_off = ((b * S_q + sq) * H_q + h_q) * D;
    Out[out_off + tid] = __float2half(out);
}

// Launcher. Caller must ensure D divides BLOCK and MAX_SK >= S_k.
void launch_naive_attn(const __half* Q, const __half* K, const __half* V,
                       __half* Out,
                       int B, int S_q, int S_k,
                       int H_q, int H_kv, int D,
                       int num_kv_groups, float scale,
                       int is_prefill, cudaStream_t stream);

void launch_naive_attn(const __half* Q, const __half* K, const __half* V,
                       __half* Out,
                       int B, int S_q, int S_k,
                       int H_q, int H_kv, int D,
                       int num_kv_groups, float scale,
                       int is_prefill, cudaStream_t stream) {
    constexpr int MAX_SK = 4096;  // ~64 KB shared memory at fp32
    dim3 grid(S_q, H_q, B);
    dim3 block(D);

    // Specialise on D; we expect 128 for Qwen2.5.
    if (D == 128) {
        size_t shmem = S_k * sizeof(float);
        naive_attn_kernel<128, MAX_SK><<<grid, block, shmem, stream>>>(
            Q, K, V, Out, S_q, S_k, H_q, H_kv, num_kv_groups, scale, is_prefill);
    } else if (D == 64) {
        size_t shmem = S_k * sizeof(float);
        naive_attn_kernel<64, MAX_SK><<<grid, block, shmem, stream>>>(
            Q, K, V, Out, S_q, S_k, H_q, H_kv, num_kv_groups, scale, is_prefill);
    } else {
        // Fallback: not supported in W4. Caller is expected to use 64 or 128.
    }
}

}  // namespace kernels
}  // namespace mini_infer