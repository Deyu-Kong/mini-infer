#pragma once

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mini_infer {

/**
 * High-level RMSNorm wrapper around the CUDA kernel.
 * Owns its weight tensor; takes a hidden-state tensor and produces the
 * normalized version.
 *
 * Two formulations are supported:
 *
 *   Standard (Qwen / LLaMA / Mistral / Yi / DeepSeek / etc.):
 *     y = (x * rsqrt(mean(x^2) + eps)) * weight
 *
 *   Gemma variant (add_one_ == true):
 *     y = (x * rsqrt(mean(x^2) + eps)) * (1 + weight)
 */
class RMSNorm {
public:
    // Default-constructs an "empty" instance; must be initialized via
    // set_weight() and a non-zero `dim_` set explicitly before use.
    RMSNorm() = default;
    RMSNorm(int64_t dim, float eps = 1e-6f, int device_index = 0,
            bool add_one = false);
    ~RMSNorm();

    // Move-only — copying would deep-copy the weight tensor unnecessarily.
    RMSNorm(const RMSNorm&) = delete;
    RMSNorm& operator=(const RMSNorm&) = delete;
    RMSNorm(RMSNorm&& other) noexcept;
    RMSNorm& operator=(RMSNorm&& other) noexcept;

    // Forward: y = rmsnorm(x, weight)        (Qwen / LLaMA)
    //       or y = rmsnorm(x, 1 + weight)    (Gemma when add_one_ is true)
    // x: [N, D] FP16, on CUDA
    // returns y: [N, D] FP16, on CUDA
    Tensor forward(const Tensor& x) const;

    // For testing / loading from safetensors.
    void set_weight(const Tensor& weight);

    // Initializes an "empty" instance with a real weight buffer. Used when
    // a default-constructed RMSNorm is later given its dimensions.
    void set_dim_device(int64_t dim, int device_index);

    // Switch to the Gemma formulation (1 + weight). Must be called before
    // forward().
    void set_add_one(bool v) { add_one_ = v; }
    bool add_one() const { return add_one_; }

private:
    int64_t dim_ = 0;
    float eps_ = 1e-6f;
    int device_index_ = 0;
    bool add_one_ = false;   // Gemma uses (1 + weight)
    Tensor weight_;  // [D] FP16
public:
    // Exposed for the model loader's summarize() path. Treat as const.
    const Tensor& weight() const { return weight_; }
};

}  // namespace mini_infer