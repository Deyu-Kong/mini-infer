#include "layers/rope.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

#include "kernels/rope_kernel.cuh"

namespace mini_infer {

RoPE::RoPE(int head_dim, float theta_base, int device_index)
    : head_dim_(head_dim), theta_base_(theta_base), device_index_(device_index) {
    if (head_dim_ <= 0 || (head_dim_ & 1) != 0) {
        throw std::runtime_error("RoPE: head_dim must be positive even");
    }
    const int half = head_dim_ / 2;
    inv_freq_.resize(half);
    for (int i = 0; i < half; ++i) {
        // Qwen2.5-style: inv_freq[i] = 1 / theta_base^(2i/dim)
        const float exp = (2.0f * i) / static_cast<float>(head_dim_);
        inv_freq_[i] = 1.0f / std::pow(theta_base_, exp);
    }
}

Tensor RoPE::forward(const Tensor& x, const std::vector<int64_t>& positions) {
    if (x.dtype() != DType::FP16) {
        throw std::runtime_error("RoPE::forward: only FP16 supported");
    }
    if (x.ndim() != 4) {
        throw std::runtime_error("RoPE::forward: expect 4-D input [B,S,H,D]");
    }
    const int B = static_cast<int>(x.shape()[0]);
    const int S = static_cast<int>(x.shape()[1]);
    const int H = static_cast<int>(x.shape()[2]);
    const int D = static_cast<int>(x.shape()[3]);
    if (D != head_dim_) {
        throw std::runtime_error("RoPE::forward: head_dim mismatch");
    }
    if (!x.is_contiguous()) {
        throw std::runtime_error("RoPE::forward: only contiguous input supported");
    }
    if (static_cast<int>(positions.size()) != S) {
        throw std::runtime_error("RoPE::forward: positions size must equal S");
    }

    const int half = D / 2;

    // Precompute cos/sin tables on device.
    // For backward compatibility: positions has length S, but kernel expects [B*S, half].
    // Replicate positions B times to create [B*S] positions array.
    std::vector<int64_t> positions_replicated(B * S);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            positions_replicated[b * S + s] = positions[s];
        }
    }

    Tensor cos_t = Tensor::empty({B * S, half}, DType::FP16, Device::cuda(device_index_));
    Tensor sin_t = Tensor::empty({B * S, half}, DType::FP16, Device::cuda(device_index_));

    kernels::launch_rope_precompute_batched(
        inv_freq_.data(), positions_replicated.data(),
        static_cast<__half*>(cos_t.data()),
        static_cast<__half*>(sin_t.data()),
        B, S, half, /*stream=*/0);

    Tensor y = Tensor::empty({B, S, H, D}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_rope(
        static_cast<const __half*>(x.data()),
        static_cast<const __half*>(cos_t.data()),
        static_cast<const __half*>(sin_t.data()),
        static_cast<__half*>(y.data()),
        B, S, H, D, /*stream=*/0);

    return y;
}

Tensor RoPE::forward_batched(const Tensor& x,
                             const std::vector<int64_t>& positions) {
    if (x.dtype() != DType::FP16) {
        throw std::runtime_error("RoPE::forward_batched: only FP16 supported");
    }
    if (x.ndim() != 4) {
        throw std::runtime_error("RoPE::forward_batched: expect 4-D input [B,S,H,D]");
    }
    const int B = static_cast<int>(x.shape()[0]);
    const int S = static_cast<int>(x.shape()[1]);
    const int H = static_cast<int>(x.shape()[2]);
    const int D = static_cast<int>(x.shape()[3]);
    if (D != head_dim_) {
        throw std::runtime_error("RoPE::forward_batched: head_dim mismatch");
    }
    if (!x.is_contiguous()) {
        throw std::runtime_error("RoPE::forward_batched: only contiguous input supported");
    }
    if (static_cast<int>(positions.size()) != B * S) {
        throw std::runtime_error("RoPE::forward_batched: positions size must equal B*S");
    }

    const int half = D / 2;

    Tensor cos_t = Tensor::empty({B * S, half}, DType::FP16, Device::cuda(device_index_));
    Tensor sin_t = Tensor::empty({B * S, half}, DType::FP16, Device::cuda(device_index_));

    kernels::launch_rope_precompute_batched(
        inv_freq_.data(), positions.data(),
        static_cast<__half*>(cos_t.data()),
        static_cast<__half*>(sin_t.data()),
        B, S, half, /*stream=*/0);

    Tensor y = Tensor::empty({B, S, H, D}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_rope(
        static_cast<const __half*>(x.data()),
        static_cast<const __half*>(cos_t.data()),
        static_cast<const __half*>(sin_t.data()),
        static_cast<__half*>(y.data()),
        B, S, H, D, /*stream=*/0);

    return y;
}

}  // namespace mini_infer