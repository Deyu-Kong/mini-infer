#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Naive (non-flash) scaled dot-product attention with optional causal mask
// and optional sliding-window mask.
//
// Sliding-window mask (when sliding_window > 0):
//   mask[j] = -inf  if  j < (sq + causal_offset) - sliding_window + 1
//   where causal_offset = S_k - S_q, so (sq + causal_offset) is the global
//   position of query sq. This matches PyTorch's
//     torch.tril(ones, diagonal=-sliding_window)
// convention: positions strictly more than `sliding_window` steps behind
// the query are masked out.
//
// See naive_attn_kernel.cu for layout and semantics.
void launch_naive_attn(const __half* Q, const __half* K, const __half* V,
                       __half* Out,
                       int B, int S_q, int S_k,
                       int H_q, int H_kv, int D,
                       int num_kv_groups, float scale,
                       int is_prefill,
                       int sliding_window,   // 0 = disabled
                       cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer