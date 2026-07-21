#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/tensor.h"
#include "layers/mlp.h"
#include "layers/rmsnorm.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"

namespace mini_infer {

/**
 * Per-layer view of Qwen2 / LLaMA attention weights.
 *
 *   Q = x @ W_q^T + b_q       [B, S, H_q, D]
 *   K = x @ W_k^T + b_k       [B, S, H_kv, D]   (GQA: H_kv < H_q)
 *   V = x @ W_v^T + b_v       [B, S, H_kv, D]
 *   O = attn(Q,K,V) @ W_o^T   [B, S, H_q, D]
 *
 * Storage shape in HF safetensors:
 *   W_q : [H_q,  D,  H]      W_k : [H_kv, D, H]   W_v : [H_kv, D, H]
 *   W_o : [H,    H_q, D]     b_q/k/v : [H_q*D] / [H_kv*D]
 */
struct AttentionWeights {
    Tensor w_q, w_k, w_v, w_o;
    Tensor b_q, b_k, b_v;       // Qwen2 has biases; empty for LLaMA
};

struct LayerWeights {
    AttentionWeights attn;
    RMSNorm          input_layernorm;    // owns its own weight tensor
    RMSNorm          post_attn_layernorm;
    MLP              mlp;                // owns its three weights

    LayerWeights() = default;
    LayerWeights(const LayerWeights&) = delete;
    LayerWeights& operator=(const LayerWeights&) = delete;
    LayerWeights(LayerWeights&&) = default;
    LayerWeights& operator=(LayerWeights&&) = default;
};

/**
 * QwenModel — Qwen2.5 / LLaMA-style decoder.
 *
 * Week 3: load weights into per-layer components, build the linear graph
 * (one block = RMSNorm -> Attn -> +res -> RMSNorm -> MLP -> +res, repeated
 * `num_hidden_layers` times, plus the embed / final-norm / LM head).
 *
 * The attention forward is still a placeholder (W5); MLP and RMSNorm are
 * real (W2).
 */
class QwenModel {
public:
    QwenModel(const ModelConfig& cfg, int device_index = 0);
    ~QwenModel();

    // Populate weights from a multi-shard safetensors index. Copies (not
    // aliases) onto the device so later code can munmap the source file.
    void load_weights(const WeightIndex& index);

    // Build (or rebuild) the computation graph of one decoder block plus
    // the embedding + final norm + lm_head. This is a pure data-structure
    // operation — no tensor memory allocated.
    void build_graph();

    const Graph& graph() const { return graph_; }
    const ModelConfig& config() const { return cfg_; }
    int device_index() const { return device_index_; }

    // Lightweight accessors used by tests + later forward passes.
    const std::vector<LayerWeights>& layers() const { return layers_; }
    const Tensor& embed_tokens() const { return embed_; }
    const Tensor& lm_head() const { return lm_head_; }
    const RMSNorm& final_norm() const { return final_norm_; }

    // Print a per-tensor summary (shape, dtype, mean, std) for a sanity
    // check against scripts/inspect_weights.py.
    void summarize(std::ostream& os) const;

private:
    ModelConfig cfg_;
    int device_index_;
    std::vector<LayerWeights> layers_;   // size = num_hidden_layers
    RMSNorm final_norm_;                // model.norm.weight
    Tensor  embed_;                     // model.embed_tokens.weight [V, H]
    Tensor  lm_head_;                   // lm_head.weight          [V, H]
    Graph   graph_;
};

}  // namespace mini_infer