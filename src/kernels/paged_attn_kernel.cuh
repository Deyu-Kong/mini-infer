#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

/**
 * PagedAttention — attention over a paged KV cache.
 *
 * Computes, for each (sequence, query_position, query_head, kv_head_group):
 *
 *   out[b, sq, h_q, :] = softmax( Q[b, sq, h_q, :] @ K[0..S-1, h_kv, :].T * scale )
 *                              @ V[0..S-1, h_kv, :]
 *
 * K and V are looked up via a per-sequence block table; each block
 * holds BLOCK_SIZE (16) consecutive tokens.
 *
 * Inputs:
 *   Q            : [B, S_q, H_q, head_dim]     FP16, CUDA, contiguous
 *   K_cache      : [num_layers, num_blocks, H_kv, BLOCK_SIZE, head_dim] FP16
 *   V_cache      : same layout as K_cache
 *   block_table  : [B, max_blocks_per_seq]     int32, CUDA
 *                   for request b, logical block i -> block_table[b*max + i]
 *                   invalid slots are >= num_blocks_used[b]
 *   num_blocks_used : [B]                       int32, CUDA
 *                   how many entries of block_table are valid for each request
 *   seq_len      : [B]                         int32, CUDA
 *                   current sequence length (must be <= num_blocks_used * BLOCK_SIZE)
 *   start_pos    : [B]                         int32, CUDA
 *                   global position where the new tokens begin for each
 *                   sequence (typically seq_len[b] - S_q).
 *   S_q          : int                          # of query positions per request
 *                   =1 for decode, =prompt_len for prefill
 *   layer        : int                          which layer's K/V to read
 *   scale        : float                        1 / sqrt(head_dim)
 *   num_kv_groups: int                          H_q / H_kv  (GQA ratio)
 *   is_prefill   : int (0 or 1)                 1 -> apply causal mask
 *   sliding_window: int                         if > 0, additionally mask
 *                   tokens whose global position is more than
 *                   `sliding_window` steps behind the query (Gemma 2/3).
 *
 * Output:
 *   output       : [B, S_q, H_q, head_dim]     FP16, CUDA
 *
 * Implementation notes:
 *   - One CUDA block per (request, query_head, query_position).
 *   - BLOCK_THREADS == head_dim. Each thread owns one element of the
 *     output vector.
 *   - For each KV block: cooperatively load K and V slabs into shared
 *     memory (BLOCK_SIZE * head_dim FP16 each = 4 KB at head_dim=128),
 *     compute BLOCK_SIZE dot-product scores via warp reduction, then
 *     perform the online-softmax update.
 *
 * Limits:
 *   - head_dim must be 128 (Qwen2.5) or 64.
 *   - BLOCK_SIZE is fixed at 16 in the kernel.
 */
void launch_paged_attn(const __half* Q,
                       const __half* K_cache,
                       const __half* V_cache,
                       const int* block_table,
                       const int* num_blocks_used,
                       const int* seq_len,
                       const int* start_pos,            // [B] global start of new tokens
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
                       int sliding_window,               // 0 = disabled
                       __half* output,
                       cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer