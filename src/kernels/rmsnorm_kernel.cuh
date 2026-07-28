#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Launches RMSNorm:  y = (x / sqrt(mean(x^2) + eps)) * f(weight)
//
//   x       : [N, D]   FP16, row-major, on CUDA
//   weight  : [D]      FP16, on CUDA
//   y       : [N, D]   FP16, row-major, on CUDA
//   eps     : added inside the sqrt for numerical stability
//   add_one : if non-zero, f(weight) = (1 + weight)  (Gemma formulation);
//             otherwise f(weight) = weight            (Qwen/LLaMA default).
//
// One block per row. Block size auto-selected based on D.
void launch_rmsnorm(const __half* x, const __half* weight, __half* y,
                    int N, int D, float eps, int add_one, cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer