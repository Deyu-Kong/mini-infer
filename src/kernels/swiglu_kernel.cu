/**
 * SwiGLU activation: out = silu(gate) * up   element-wise.
 *
 *   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Fused: one read of gate, one read of up, one write of out per element.
 * FP16 storage, FP32 math.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace mini_infer {
namespace kernels {

__global__ void swiglu_kernel(const __half* __restrict__ gate,
                              const __half* __restrict__ up,
                              __half* __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu * u);
}

void launch_swiglu(const __half* gate, const __half* up,
                   __half* out, int n, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int blocks = (n + BLOCK - 1) / BLOCK;
    swiglu_kernel<<<blocks, BLOCK, 0, stream>>>(gate, up, out, n);
}

}  // namespace kernels
}  // namespace mini_infer