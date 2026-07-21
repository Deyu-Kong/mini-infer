#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Two-pass numerically-stable softmax along the last dim.
//   x : [N, D] FP16, row-major
//   y : [N, D] FP16, row-major
void launch_softmax(const __half* x, __half* y, int N, int D,
                    cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer