#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.h"

namespace mini_infer {

/**
 * RoPE wrapper. Computes inv_freq from theta_base / head_dim, precomputes
 * cos/sin tables per call (small overhead in W2; W5 will cache).
 *
 * Input layout: x [B, S, num_heads, head_dim], positions [S] or [B, S].
 */
class RoPE {
public:
    RoPE(int head_dim, float theta_base = 1000000.0f, int device_index = 0);

    // Returns y [B, S, num_heads, head_dim] FP16.
    Tensor forward(const Tensor& x, const std::vector<int64_t>& positions);

    // Batched variant: positions is a flat length-B*S array, where each
    // token (b, s) has its own position. The cos/sin tables produced have
    // shape [B*S, head_dim/2]; the RoPE kernel indexes them per-token.
    Tensor forward_batched(const Tensor& x,
                           const std::vector<int64_t>& positions);

    int head_dim() const { return head_dim_; }
    int half_dim() const { return head_dim_ / 2; }
    float theta_base() const { return theta_base_; }
    const std::vector<float>& inv_freq() const { return inv_freq_; }

private:
    int head_dim_;
    float theta_base_;
    int device_index_;
    std::vector<float> inv_freq_;  // [head_dim/2]
};

}  // namespace mini_infer