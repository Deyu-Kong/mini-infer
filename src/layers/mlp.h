#pragma once

#include <cstdint>

#include "core/tensor.h"

namespace mini_infer {

/**
 * SwiGLU MLP layer for Qwen2.5 (LLaMA-style MLP).
 *
 *   gate = x @ W_gate^T           -> [B, I]
 *   up   = x @ W_up^T             -> [B, I]
 *   h    = silu(gate) * up        -> [B, I]
 *   y    = h @ W_down^T           -> [B, H]
 *
 * All GEMMs go through cuBLAS (FP16 in, FP32 accumulate).
 *
 * Weights are stored row-major matching HF safetensors:
 *   W_gate: [I, H]   W_up: [I, H]   W_down: [H, I]
 */
class MLP {
public:
    // Default-constructs an "empty" instance; configure with init() before use.
    MLP() = default;
    MLP(int64_t hidden, int64_t intermediate, int device_index = 0);
    ~MLP();

    // Move-only: copying would duplicate the cuBLAS handle.
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    MLP(MLP&& other) noexcept;
    MLP& operator=(MLP&& other) noexcept;

    Tensor forward(const Tensor& x);

    void set_weights(const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down);

    // Initialize a default-constructed instance.
    void init(int64_t hidden, int64_t intermediate, int device_index);

    int64_t hidden()       const { return hidden_; }
    int64_t intermediate() const { return intermediate_; }
    int device_index()     const { return device_index_; }

private:
    int64_t hidden_;
    int64_t intermediate_;
    int device_index_;

    Tensor w_gate_;   // [I, H] FP16
    Tensor w_up_;     // [I, H] FP16
    Tensor w_down_;   // [H, I] FP16

    // Persistent device-side buffer for the [B, I] intermediate.
    Tensor gate_buf_;
    Tensor up_buf_;
    Tensor silu_buf_;

    // cuBLAS handle (per device).
    void* cublas_handle_ = nullptr;
};

}  // namespace mini_infer