/**
 * GELU (tanh approximation) CUDA kernel — elementwise.
 *
 * Formula (matches PyTorch `F.gelu(x, approximate='tanh')`):
 *
 *     y = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
 *
 * FP16 in / FP16 out. All math in FP32 for precision.
 */
#include "kernels/gelu_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

namespace {
constexpr float kInvSqrt2OverPi = 0.7978845608028654f;   // sqrt(2 / pi)
constexpr float kGeluCoef       = 0.044715f;
}  // namespace

__global__ void gelu_tanh_kernel(const __half* __restrict__ x,
                                __half* __restrict__ y,
                                int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = __half2float(x[i]);
    const float inner = kInvSqrt2OverPi * (xv + kGeluCoef * xv * xv * xv);
    // tanhf approximates tanh for small arguments; the kernel only needs
    // ~3 ulp accuracy so this is fine.
    const float t = tanhf(inner);
    const float yv = 0.5f * xv * (1.0f + t);
    y[i] = __float2half(yv);
}

void launch_gelu_tanh(const __half* x, __half* y, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    const int grid  = static_cast<int>((n + block - 1) / block);
    gelu_tanh_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

}  // namespace kernels
}  // namespace mini_infer