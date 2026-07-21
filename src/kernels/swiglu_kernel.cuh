#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

void launch_swiglu(const __half* gate, const __half* up,
                   __half* out, int n, cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer