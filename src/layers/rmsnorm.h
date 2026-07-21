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
    RMSNorm(int64_t dim, float eps = 1e-6f, int device_index = 0);
    ~RMSNorm();

    // Forward: y = rmsnorm(x, weight)
    // x: [N, D] FP16, on CUDA
    // returns y: [N, D] FP16, on CUDA
    Tensor forward(const Tensor& x) const;

    // For testing / loading from safetensors.
    void set_weight(const Tensor& weight);

private:
    int64_t dim_ = 0;
    float eps_ = 1e-6f;
    int device_index_ = 0;
    Tensor weight_;  // [D] FP16
};

}  // namespace mini_infer