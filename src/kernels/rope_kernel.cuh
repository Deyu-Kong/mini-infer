#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// RoPE forward — see rope_kernel.cu for the formula.
void launch_rope(const __half* x, const __half* cos_t, const __half* sin_t,
                 __half* y, int B, int S, int H, int D, cudaStream_t stream);

// Precompute cos / sin tables of shape [S, head_dim/2] on the device.
// inv_freq (length half) and positions (length S) are host-side.
void launch_rope_precompute(const float* inv_freq_h,
                            const int64_t* positions_h,
                            __half* cos_d, __half* sin_d,
                            int S, int half, cudaStream_t stream);

// Precompute cos / sin tables of shape [B*S, head_dim/2] on the device.
// inv_freq (length half) and positions (length B*S, flat row-major
// positions[b*S + s]) are host-side.
void launch_rope_precompute_batched(const float* inv_freq_h,
                                    const int64_t* positions_h,
                                    __half* cos_d, __half* sin_d,
                                    int B, int S, int half,
                                    cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer