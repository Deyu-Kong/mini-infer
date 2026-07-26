#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Naive (non-flash) scaled dot-product attention with optional causal mask.
// See naive_attn_kernel.cu for layout and semantics.
void launch_naive_attn(const __half* Q, const __half* K, const __half* V,
                       __half* Out,
                       int B, int S_q, int S_k,
                       int H_q, int H_kv, int D,
                       int num_kv_groups, float scale,
                       int is_prefill, cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer