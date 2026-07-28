/**
 * Utility CUDA kernels for model forward pass.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// row-major y[M,N] += bias[N]  (broadcast over M).
__global__ void add_bias_kernel(__half* y, const __half* bias,
                                 int M, int N, int batch_stride) {
    // y: [B, M, N], bias: [N]
    int b = blockIdx.z;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    if (n >= N) return;
    int off = b * batch_stride + m * N + n;
    y[off] = __float2half(__half2float(y[off]) + __half2float(bias[n]));
}

void launch_add_bias(__half* y, const __half* bias, int M, int N,
                     int batch_stride, cudaStream_t stream) {
    if (M == 0) return;
    dim3 block(256);
    dim3 grid((N + 255) / 256, M, batch_stride / (M * N));
    add_bias_kernel<<<grid, block, 0, stream>>>(y, bias, M, N, batch_stride);
}

// CUDA element-wise add: y[i] += x[i] (in-place)
__global__ void add_inplace_kernel(__half* y, const __half* x, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(__half2float(y[i]) + __half2float(x[i]));
}

void launch_add_inplace(__half* y, const __half* x, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    add_inplace_kernel<<<grid, block, 0, s>>>(y, x, n);
}

// CUDA gather+fp16: out[b, s, d] = embed_table[token_ids[b, s], d]
__global__ void embedding_gather_kernel(
    const int64_t* __restrict__ token_ids,   // [B, S]
    const __half*  __restrict__ embed_table, // [V, H]
    __half*        __restrict__ out,         // [B, S, H]
    int B, int S, int H) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;
    int64_t tok = token_ids[b * S + s];
    out[(b * S + s) * H + h] = embed_table[tok * H + h];
}

void launch_embedding_gather(const int64_t* token_ids, const __half* embed_table,
                              __half* out, int B, int S, int H, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((H + 255) / 256, S, B);
    embedding_gather_kernel<<<grid, block, 0, stream>>>(
        token_ids, embed_table, out, B, S, H);
}

// y[i] *= scale  (FP16, in-place). Used by Gemma to scale embeddings by
// sqrt(hidden_size).
__global__ void scale_inplace_kernel(__half* y, float scale, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(__half2float(y[i]) * scale);
}

void launch_scale_inplace(__half* y, float scale, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    scale_inplace_kernel<<<grid, block, 0, s>>>(y, scale, n);
}

// ===========================================================================
// Paged KV scatter (Week 5)
// ===========================================================================
//
// Each thread copies one (b, s, kv_head, d) element from the projected
// K/V tensor into its slot in the paged cache, indirection via
// block_table[b, s / BLOCK_SIZE].
//
// Grid : (1, S, B * H_kv)
// Block: 128 threads (one warp handles 32 contiguous dims; 4 warps cover
//        D=128 in stride). Threads cooperate along D: each thread covers
//        one dim per row.
//
__global__ void paged_kv_scatter_kernel(
    const __half* __restrict__ src,        // [B, S, H_kv, D]
    __half*       __restrict__ cache,      // [L, num_blocks, H_kv, BS, D]
    const int*    __restrict__ block_table, // [B, max_blocks_per_seq]
    const int*    __restrict__ seq_len,    // [B]
    const int*    __restrict__ start_pos,  // [B]
    int S,
    int layer,
    int num_blocks,
    int H_kv,
    int D,
    int block_size,
    int max_blocks_per_seq) {
    const int tid = threadIdx.x;
    const int s   = blockIdx.y;          // local index in src
    const int bh  = blockIdx.z;
    const int b   = bh / H_kv;
    const int hkv = bh % H_kv;

    const int slen = seq_len[b];
    const int sp    = start_pos[b];
    const int global_pos = sp + s;
    if (global_pos >= slen) return;

    const int block_idx = global_pos / block_size;
    const int tok_off   = global_pos % block_size;
    const int phys_block = block_table[b * max_blocks_per_seq + block_idx];

    const int64_t base_src = (((int64_t)b * S + s) * H_kv + hkv) * D;
    const int64_t base_dst =
        ((int64_t)layer * num_blocks + phys_block) * (H_kv * block_size * D)
        + (int64_t)hkv * (block_size * D)
        + (int64_t)tok_off * D;
    for (int d = tid; d < D; d += blockDim.x) {
        cache[base_dst + d] = src[base_src + d];
    }
}

void launch_paged_kv_scatter(const __half* src,
                             __half* cache,
                             const int* block_table,
                             const int* seq_len,
                             const int* start_pos,
                             int B, int S,
                             int layer,
                             int num_blocks, int H_kv, int D,
                             int block_size,
                             int max_blocks_per_seq,
                             cudaStream_t stream) {
    if (B <= 0 || S <= 0) return;
    const int block_threads = 128;
    dim3 block(block_threads);
    dim3 grid(1, S, B * H_kv);
    paged_kv_scatter_kernel<<<grid, block, 0, stream>>>(
        src, cache, block_table, seq_len, start_pos, S, layer, num_blocks, H_kv, D,
        block_size, max_blocks_per_seq);
}

}  // namespace kernels
}  // namespace mini_infer