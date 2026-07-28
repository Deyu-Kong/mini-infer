#pragma once

#include <cstdint>

#include "core/tensor.h"
#include "model/model_config.h"

namespace mini_infer {

/**
 * Gated MLP layer.
 *
 * Two flavours:
 *
 *   - SwiGLU  (Qwen / LLaMA / Mistral / Yi / DeepSeek):
 *         gate = silu(x @ W_gate^T)
 *         up   = x @ W_up^T
 *         h    = gate * up
 *         y    = h @ W_down^T
 *
 *   - GeGLU   (Gemma 1/2/3):
 *         gate = gelu_tanh(x @ W_gate^T)
 *         up   = x @ W_up^T
 *         h    = gate * up
 *         y    = h @ W_down^T
 *
 * The activation is selected by passing `ActKind::Silu` or `ActKind::GeluTanh`
 * to the constructor (or via `init()`). It is fixed for the lifetime of the
 * MLP — different layers in the same model can use different activations if
 * the model requires, but this is rare (no mainstream decoder does it).
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
    MLP(int64_t hidden, int64_t intermediate, int device_index = 0,
        ActKind act = ActKind::Silu);
    ~MLP();

    // Move-only: copying would duplicate the cuBLAS handle.
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    MLP(MLP&& other) noexcept;
    MLP& operator=(MLP&& other) noexcept;

    Tensor forward(const Tensor& x);

    void set_weights(const Tensor& w_gate, const Tensor& w_up, const Tensor& w_down);

    // Initialize a default-constructed instance.
    void init(int64_t hidden, int64_t intermediate, int device_index,
              ActKind act = ActKind::Silu);

    int64_t hidden()       const { return hidden_; }
    int64_t intermediate() const { return intermediate_; }
    int     device_index() const { return device_index_; }
    ActKind act()          const { return act_; }

private:
    int64_t hidden_       = 0;
    int64_t intermediate_ = 0;
    int     device_index_ = 0;
    ActKind act_          = ActKind::Silu;

    Tensor w_gate_;   // [I, H] FP16
    Tensor w_up_;     // [I, H] FP16
    Tensor w_down_;   // [H, I] FP16

    // Persistent device-side buffer for the [B, I] intermediate.
    Tensor gate_buf_;
    Tensor up_buf_;
    Tensor act_buf_;

    // cuBLAS handle (per device).
    void* cublas_handle_ = nullptr;
};

}  // namespace mini_infer