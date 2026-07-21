#pragma once

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mini_infer {

/**
 * High-level RMSNorm wrapper around the CUDA kernel.
 * Owns its weight tensor; takes a hidden-state tensor and produces the
 * normalized version (in place is also supported).
 */
class RMSNorm {
public:
    // Default-constructs an "empty" instance; must be initialized via
    // set_weight() and a non-zero `dim_` set explicitly before use.
    RMSNorm() = default;
    RMSNorm(int64_t dim, float eps = 1e-6f, int device_index = 0);
    ~RMSNorm();

    // Move-only — copying would deep-copy the weight tensor unnecessarily.
    RMSNorm(const RMSNorm&) = delete;
    RMSNorm& operator=(const RMSNorm&) = delete;
    RMSNorm(RMSNorm&& other) noexcept;
    RMSNorm& operator=(RMSNorm&& other) noexcept;

    // Forward: y = rmsnorm(x, weight)
    // x: [N, D] FP16, on CUDA
    // returns y: [N, D] FP16, on CUDA
    Tensor forward(const Tensor& x) const;

    // For testing / loading from safetensors.
    void set_weight(const Tensor& weight);

    // Initializes an "empty" instance with a real weight buffer. Used when
    // a default-constructed RMSNorm is later given its dimensions.
    void set_dim_device(int64_t dim, int device_index);

private:
    int64_t dim_ = 0;
    float eps_ = 1e-6f;
    int device_index_ = 0;
    Tensor weight_;  // [D] FP16
public:
    // Exposed for the model loader's summarize() path. Treat as const.
    const Tensor& weight() const { return weight_; }
};

}  // namespace mini_infer