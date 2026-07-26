/**
 * PagedAttention CUDA kernel — Week 5.
 *
 * Decode-only attention that reads K/V from a paged (block-table indexed)
 * cache. Each block of threads handles one (request, query_head) pair,
 * iterates over the request's KV blocks one at a time, and accumulates
 * the output via online softmax.
 *
 * Layouts:
 *   Q             : [B, H_q, head_dim]                      FP16
 *   K_cache, V_cache : [num_layers, num_blocks, H_kv, BLOCK_SIZE, head_dim]   FP16
 *   block_table   : [B, max_blocks_per_seq]                int32
 *   num_blocks_used : [B]                                  int32
 *   seq_len       : [B]                                    int32
 *   output        : [B, H_q, head_dim]                     FP16
 *
 * Each request has H_q query heads, each grouped under num_kv_groups
 * consecutive query heads that share one kv_head. Inside the kernel we
 * compute h_kv = h_q / num_kv_groups for the current thread block.
 *
 * Shared memory budget per block:
 *   K_smem, V_smem : BLOCK_SIZE * head_dim * 2 bytes   (FP16)
 *   scores, reduce : BLOCK_SIZE + BLOCK_THREADS floats
 * For head_dim=128, BLOCK_SIZE=16:
 *   K_smem = 16 * 128 * 2 = 4 KB
 *   V_smem = 16 * 128 * 2 = 4 KB
 *   total  ~ 8.3 KB   (well under A6000's 100 KB/SM limit)
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/paged_attn_kernel.cuh"
#include "core/allocator.h"   // for BlockAllocator::kBlockSize

namespace mini_infer {
namespace kernels {

namespace {

// Warp-wide max-reduce. `val` must be a per-lane value.
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

// Warp-wide sum-reduce.
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

// Block-wide max-reduce over a per-thread `val`. Uses warp-shuffles +
// shared-memory cross-warp combine. BLOCK_THREADS must be a multiple of 32.
template <int BLOCK_THREADS>
__device__ __forceinline__ float block_max(float val, float* smem) {
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : -INFINITY;
        val = warp_max(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_sum(float val, float* smem) {
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

// ----------------------------------------------------------------------------
// Kernel template
// ----------------------------------------------------------------------------
//
// `HEAD_DIM`     : per-head dimension (64 or 128 in practice)
// `BLOCK_THREADS`: must equal HEAD_DIM and be a multiple of 32.
// `BLOCK_SIZE`   : tokens per paged block. Fixed at 16 here.
//
template <int HEAD_DIM, int BLOCK_THREADS, int BLOCK_SIZE>
__global__ void paged_attn_kernel(
    const __half* __restrict__ Q,             // [B, S_q, H_q, D]
    const __half* __restrict__ K_cache,       // [L, num_blocks, H_kv, BLOCK_SIZE, D]
    const __half* __restrict__ V_cache,
    const int*    __restrict__ block_table,   // [B, max_blocks_per_seq]
    const int*    __restrict__ num_blocks_used, // [B]
    const int*    __restrict__ seq_len,       // [B]
    int S_q,
    int H_q,
    int H_kv,
    int num_kv_groups,
    int max_blocks_per_seq,
    int layer,
    int num_blocks,
    float scale,
    int is_prefill,                            // 1 -> apply causal mask
    __half*       __restrict__ output) {      // [B, S_q, H_q, D]
    static_assert(BLOCK_THREADS == HEAD_DIM,
                  "BLOCK_THREADS must equal HEAD_DIM");
    constexpr int D = HEAD_DIM;
    constexpr int BS = BLOCK_SIZE;

    const int b   = blockIdx.x;
    const int h_q = blockIdx.y;
    const int sq  = blockIdx.z;
    const int h_kv = h_q / num_kv_groups;
    const int tid = threadIdx.x;

    // ---- shared memory layout -------------------------------------------
    __shared__ __half K_smem[BS * D];        // [BS, D]   contiguous
    __shared__ __half V_smem[BS * D];
    __shared__ float scores[BS];             // partial scores per token in block
    __shared__ float reduce[BLOCK_THREADS / 32]; // warp reduce scratch
    __shared__ float Q_smem[D];              // Q for this (b, sq, h_q)

    // ---- per-request state (in registers, one thread is enough but keep
    //      them in shared memory so all threads agree) -------------------
    __shared__ int  n_blocks_smem;
    __shared__ int  seq_len_smem;
    if (tid == 0) {
        n_blocks_smem = num_blocks_used[b];
        seq_len_smem  = seq_len[b];
    }
    __syncthreads();
    const int n_blocks = n_blocks_smem;
    const int cur_len = seq_len_smem;
    if (cur_len <= 0) {
        // Degenerate: empty sequence. Just zero the output.
        if (tid < D) {
            output[(((int64_t)b * S_q + sq) * H_q + h_q) * D + tid] = __float2half(0.0f);
        }
        return;
    }

    // ---- load Q into shared memory -------------------------------------
    // Q layout: [B, S_q, H_q, D]; the row for (b, sq, h_q) starts at
    //   (b * S_q + sq) * H_q * D + h_q * D
    if (tid < D) {
        const int64_t q_off =
            (((int64_t)b * S_q + sq) * H_q + h_q) * D + tid;
        Q_smem[tid] = Q[q_off];
    }
    __syncthreads();

    // ---- online-softmax accumulators (per thread, per output dim) ------
    float m_state = -INFINITY;     // running max of scores
    float l_state = 0.0f;          // running denominator
    float o_state = 0.0f;          // running output value for THIS thread's dim

    // Stride helpers.
    const int64_t layer_stride     = (int64_t)num_blocks * H_kv * BS * D;
    const int64_t block_in_layer   = (int64_t)H_kv * BS * D;
    const int64_t kv_head_in_block = (int64_t)BS * D;

    // The query position in the global token stream (only used for causal
    // masking during prefill).
    const int query_global_pos = sq;  // for now we use S_q==1 for decode
                                      // and S_q==prompt_len for prefill where
                                      // sq == position within the prompt.

    // ---- iterate over KV blocks ---------------------------------------
    for (int blk = 0; blk < n_blocks; ++blk) {
        const int phys_block = block_table[b * max_blocks_per_seq + blk];
        const int tokens_in_block = (blk == n_blocks - 1)
            ? (cur_len - blk * BS)
            : BS;
        const int blk_start = blk * BS;            // first token in this block
        const int64_t blk_base =
            layer * layer_stride
            + (int64_t)phys_block * block_in_layer
            + (int64_t)h_kv * kv_head_in_block;

        // Cooperatively load K and V slabs: total BS * D elements each.
        const int total = BS * D;
        #pragma unroll 1
        for (int idx = tid; idx < total; idx += BLOCK_THREADS) {
            K_smem[idx] = K_cache[blk_base + idx];
            V_smem[idx] = V_cache[blk_base + idx];
        }
        __syncthreads();

        // ---- compute scores[t] for t in [0, tokens_in_block) -----------
        constexpr int NUM_WARPS = BLOCK_THREADS / 32;
        constexpr int T_PER_WARP = BS / NUM_WARPS;     // 4 for BS=16,4 warps
        const int warp_id = tid >> 5;
        const int lane    = tid & 31;

        #pragma unroll
        for (int tw = 0; tw < T_PER_WARP; ++tw) {
            const int t_local = warp_id * T_PER_WARP + tw;
            if (t_local >= tokens_in_block) {
                if (lane == 0) scores[t_local] = -INFINITY;
                continue;
            }
            const int t_global = blk_start + t_local;
            // Causal mask during prefill: scores[t_global] = -inf if t_global > sq.
            // Also mask t_global >= cur_len (partial last block may contain
            // garbage in slots past seq_len).
            if (t_global >= cur_len) {
                if (lane == 0) scores[t_local] = -INFINITY;
                continue;
            }
            if (is_prefill && t_global > query_global_pos) {
                if (lane == 0) scores[t_local] = -INFINITY;
                continue;
            }
            float partial = 0.0f;
            #pragma unroll
            for (int d = lane; d < D; d += 32) {
                partial += __half2float(Q_smem[d])
                         * __half2float(K_smem[t_local * D + d]);
            }
            partial = warp_sum(partial);
            if (lane == 0) {
                scores[t_local] = partial * scale;
            }
        }
        __syncthreads();

        // ---- online-softmax update ------------------------------------
        float local_max = -INFINITY;
        for (int t = tid; t < tokens_in_block; t += BLOCK_THREADS) {
            local_max = fmaxf(local_max, scores[t]);
        }
        float block_max_v = block_max<BLOCK_THREADS>(local_max, reduce);
        const float m_new = fmaxf(m_state, block_max_v);
        const float alpha = __expf(m_state - m_new);

        float local_sum = 0.0f;
        for (int t = tid; t < tokens_in_block; t += BLOCK_THREADS) {
            const float e = __expf(scores[t] - m_new);
            scores[t] = e;
            local_sum += e;
        }
        float block_sum_v = block_sum<BLOCK_THREADS>(local_sum, reduce);

        // Update o_state (unnormalized) and l_state (denominator) with the
        // rescaling factor alpha. After all blocks we divide o_state by
        // l_state to get the final output.
        float contrib = 0.0f;
        for (int t = 0; t < tokens_in_block; ++t) {
            contrib += scores[t] * __half2float(V_smem[t * D + tid]);
        }
        o_state = o_state * alpha + contrib;
        l_state = l_state * alpha + block_sum_v;

        m_state = m_new;
        __syncthreads();
    }

    // ---- write output --------------------------------------------------
    if (tid < D) {
        const float final = (l_state > 0.0f) ? (o_state / l_state) : 0.0f;
        const int64_t out_off =
            (((int64_t)b * S_q + sq) * H_q + h_q) * D + tid;
        output[out_off] = __float2half(final);
    }
}

}  // namespace

void launch_paged_attn(const __half* Q,
                       const __half* K_cache,
                       const __half* V_cache,
                       const int* block_table,
                       const int* num_blocks_used,
                       const int* seq_len,
                       int B,
                       int S_q,
                       int H_q,
                       int H_kv,
                       int head_dim,
                       int num_kv_groups,
                       int max_blocks_per_seq,
                       int layer,
                       int num_blocks,
                       float scale,
                       int is_prefill,
                       __half* output,
                       cudaStream_t stream) {
    if (head_dim != 128 && head_dim != 64) {
        return;
    }
    dim3 grid(B, H_q, S_q);
    if (head_dim == 128) {
        dim3 block(128);
        paged_attn_kernel<128, 128, 16><<<grid, block, 0, stream>>>(
            Q, K_cache, V_cache, block_table, num_blocks_used, seq_len,
            S_q, H_q, H_kv, num_kv_groups, max_blocks_per_seq, layer,
            num_blocks, scale, is_prefill, output);
    } else {
        dim3 block(64);
        paged_attn_kernel<64, 64, 16><<<grid, block, 0, stream>>>(
            Q, K_cache, V_cache, block_table, num_blocks_used, seq_len,
            S_q, H_q, H_kv, num_kv_groups, max_blocks_per_seq, layer,
            num_blocks, scale, is_prefill, output);
    }
}

}  // namespace kernels
}  // namespace mini_infer