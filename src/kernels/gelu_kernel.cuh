#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Launches GeLU with the tanh approximation (PyTorch `gelu_pytorch_tanh`):
//
//   y[i] = 0.5 * x[i] * (1 + tanh( sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3) ))
//
// Used by Gemma's GeGLU MLP. Both `x` and `y` are length-n FP16 row-major
// buffers on the device.
void launch_gelu_tanh(const __half* x, __half* y, int64_t n, cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer