#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

void launch_add_bias(__half* y, const __half* bias, int M, int N,
                     int batch_stride, cudaStream_t stream);

void launch_add_inplace(__half* y, const __half* x, int64_t n, cudaStream_t s);

void launch_embedding_gather(const int64_t* token_ids, const __half* embed_table,
                              __half* out, int B, int S, int H, cudaStream_t stream);

/**
 * In-place embedding scale: y[i] *= scale  (FP16, row-major, on CUDA).
 *
 * Gemma multiplies the embedding-table output by sqrt(hidden_size).
 * `scale` is passed in FP32 for precision (we cast to FP16 inside the kernel).
 */
void launch_scale_inplace(__half* y, float scale, int64_t n, cudaStream_t stream);

/**
 * In-place elementwise multiply: y[i] *= x[i]  (FP16, row-major, on CUDA).
 * Used by GeGLU after the gate is GELU-activated.
 */
void launch_mul_inplace(__half* y, const __half* x, int64_t n, cudaStream_t stream);

/**
 * Scatter a (K or V) tensor into a paged KV cache.
 *
 * For each (b, s, kv_head, d) of the source tensor [B, S, H_kv, D], the
 * destination is
 *
 *   cache[(layer * num_blocks + block_table[b, (start_pos[b] + s)/BLOCK_SIZE])
 *           * H_kv * BLOCK_SIZE * D
 *        + kv_head * BLOCK_SIZE * D
 *        + ((start_pos[b] + s) % BLOCK_SIZE) * D
 *        + d]
 *
 * `start_pos` is a per-sequence array (length B) giving the global
 * sequence position at which the new tokens begin for each sequence
 * (typically seq_len[b] - S, i.e. just before the new tokens were
 * appended). For single-sequence use, pass a length-1 array.
 *
 * block_table and seq_len are read from device memory.
 *
 * src must be in [B, S, H_kv, D] layout — i.e. reshape it from
 * [B, S, H_kv*D] before calling.
 */
void launch_paged_kv_scatter(const __half* src,           // [B, S, H_kv, D]
                             __half* cache,               // [L, num_blocks, H_kv, BS, D]
                             const int* block_table,      // [B, max_blocks_per_seq]
                             const int* seq_len,          // [B]
                             const int* start_pos,        // [B]
                             int B, int S,
                             int layer,
                             int num_blocks, int H_kv, int D,
                             int block_size,
                             int max_blocks_per_seq,
                             cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer