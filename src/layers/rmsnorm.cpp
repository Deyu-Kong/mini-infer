#include "layers/rmsnorm.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "kernels/rmsnorm_kernel.cuh"

namespace mini_infer {

RMSNorm::RMSNorm(int64_t dim, float eps, int device_index)
    : dim_(dim), eps_(eps), device_index_(device_index),
      weight_(Tensor::empty({dim}, DType::FP16, Device::cuda(device_index))) {}

RMSNorm::~RMSNorm() = default;

void RMSNorm::set_weight(const Tensor& weight) {
    if (weight.dtype() != DType::FP16) {
        throw std::runtime_error("RMSNorm::set_weight expects FP16");
    }
    if (weight.numel() != dim_) {
        throw std::runtime_error("RMSNorm::set_weight: weight size mismatch");
    }
    // Move weight to the right device.
    weight_ = weight.to(Device::cuda(device_index_));
}

Tensor RMSNorm::forward(const Tensor& x) const {
    if (x.dtype() != DType::FP16) {
        throw std::runtime_error("RMSNorm::forward: only FP16 supported in W2");
    }
    if (x.device().index != device_index_) {
        throw std::runtime_error("RMSNorm::forward: device mismatch");
    }
    if (x.ndim() != 2) {
        throw std::runtime_error("RMSNorm::forward: only 2-D input supported in W2");
    }
    const int N = static_cast<int>(x.shape()[0]);
    const int D = static_cast<int>(x.shape()[1]);
    if (D != dim_) {
        throw std::runtime_error("RMSNorm::forward: hidden dim mismatch");
    }
    if (!x.is_contiguous()) {
        throw std::runtime_error("RMSNorm::forward: only contiguous input supported");
    }

    Tensor y = Tensor::empty({N, D}, DType::FP16, Device::cuda(device_index_));

    kernels::launch_rmsnorm(
        static_cast<const __half*>(x.data()),
        static_cast<const __half*>(weight_.data()),
        static_cast<__half*>(y.data()),
        N, D, eps_, /*stream=*/0);

    return y;
}

}  // namespace mini_infer