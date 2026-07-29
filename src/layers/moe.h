#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.h"
#include "layers/mlp.h"
#include "model/model_config.h"

namespace mini_infer {

/**
 * MoELayer — Mixture of Experts layer for Mixtral / DeepSeek-MoE / Qwen2-MoE.
 *
 * Replaces a dense MLP with a sparse routing mechanism:
 *   1. Router gate: logits = x @ W_gate^T  -> [B, num_experts]
 *   2. Top-K selection: pick top num_experts_per_tok experts per token
 *   3. Weighted sum: output = sum(weight_i * expert_i(x)) for selected experts
 *
 * Each expert is a standard MLP (SwiGLU or GeGLU). The router uses softmax
 * over the selected expert logits to compute weights.
 *
 * Weight layout (HuggingFace Mixtral convention):
 *   W_gate: [num_experts, hidden_size]  (router gate)
 *   Expert i: w_gate[i], w_up[i], w_down[i]  (each is an MLP)
 */
class MoELayer {
public:
    MoELayer() = default;
    MoELayer(int64_t hidden, int64_t moe_intermediate, int64_t num_experts,
             int64_t num_experts_per_tok, int device_index = 0,
             ActKind act = ActKind::Silu);
    ~MoELayer();

    MoELayer(const MoELayer&) = delete;
    MoELayer& operator=(const MoELayer&) = delete;
    MoELayer(MoELayer&& other) noexcept;
    MoELayer& operator=(MoELayer&& other) noexcept;

    Tensor forward(const Tensor& x);

    void set_router_gate(const Tensor& w_gate);
    void set_expert_weights(int64_t expert_idx, const Tensor& w_gate,
                            const Tensor& w_up, const Tensor& w_down);

    void set_shared_expert_weights(const Tensor& w_gate, const Tensor& w_up,
                                    const Tensor& w_down);
    void set_shared_expert_gate(const Tensor& w_gate);

    void init(int64_t hidden, int64_t moe_intermediate, int64_t num_experts,
              int64_t num_experts_per_tok, int device_index,
              ActKind act = ActKind::Silu);

    void init_shared_expert(int64_t hidden, int64_t shared_intermediate,
                            int device_index, ActKind act = ActKind::Silu);

    int64_t hidden() const { return hidden_; }
    int64_t moe_intermediate() const { return moe_intermediate_; }
    int64_t num_experts() const { return num_experts_; }
    int64_t num_experts_per_tok() const { return num_experts_per_tok_; }
    int device_index() const { return device_index_; }

private:
    int64_t hidden_ = 0;
    int64_t moe_intermediate_ = 0;
    int64_t num_experts_ = 0;
    int64_t num_experts_per_tok_ = 0;
    int device_index_ = 0;
    ActKind act_ = ActKind::Silu;

    Tensor w_router_gate_;  // [num_experts, hidden] FP16
    std::vector<MLP> experts_;  // num_experts MLPs

    MLP shared_expert_;  // shared expert (Qwen2-MoE)
    Tensor w_shared_expert_gate_;  // [hidden, 1] FP16 (Qwen2-MoE)
    bool has_shared_expert_ = false;

    // Persistent buffers
    Tensor router_logits_buf_;  // [B, num_experts]
    Tensor expert_weights_buf_; // [B, num_experts_per_tok]
    Tensor expert_indices_buf_; // [B, num_experts_per_tok] int32

    void* cublas_handle_ = nullptr;
};

}  // namespace mini_infer
