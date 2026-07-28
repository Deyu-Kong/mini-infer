#pragma once

#include <cstdint>
#include <string>

#include "core/tensor.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"

namespace mini_infer {

/**
 * One decoder block's worth of weight tensors, materialized on the
 * device. For Qwen / LLaMA family the QKV tensors come straight from the
 * safetensors shard. For Gemma the merged `qkv_proj.weight` is split into
 * three contiguous slices along the output (row) dimension.
 *
 *   Q chunk : [num_heads    * head_dim, hidden_size]
 *   K chunk : [num_kv_heads * head_dim, hidden_size]
 *   V chunk : [num_kv_heads * head_dim, hidden_size]
 *
 * All three live in [3*kv_dim, hidden_size] in the safetensors file when
 * arch == Gemma (Q rows first, then K, then V — this is the HuggingFace
 * Gemma convention).
 */
struct LayerWeightNames {
    // Attention projections.
    std::string input_layernorm;     // model.layers.{i}.input_layernorm.weight
    std::string post_attn_layernorm; // model.layers.{i}.post_attention_layernorm.weight
    std::string q_proj;              // self_attn.q_proj.weight   (Qwen/LLaMA or Gemma)
    std::string k_proj;              // self_attn.k_proj.weight   (Qwen/LLaMA or Gemma)
    std::string v_proj;              // self_attn.v_proj.weight   (Qwen/LLaMA or Gemma)
    std::string qkv_proj;            // self_attn.qkv_proj.weight (Gemma merged) OR empty
    std::string o_proj;              // self_attn.o_proj.weight   (both)
    std::string q_proj_bias;         // q_proj.bias (Qwen only)
    std::string k_proj_bias;         // k_proj.bias (Qwen only)
    std::string v_proj_bias;         // v_proj.bias (Qwen only)
    std::string q_norm;              // Gemma 3: q_norm.weight
    std::string k_norm;              // Gemma 3: k_norm.weight

    // MLP projections.
    std::string gate_proj;           // mlp.gate_proj.weight
    std::string up_proj;             // mlp.up_proj.weight
    std::string down_proj;           // mlp.down_proj.weight

    // Gemma 2/3 extra norms around the MLP.
    std::string pre_feedforward_layernorm;   // pre_feedforward_layernorm.weight
    std::string post_feedforward_layernorm;  // post_feedforward_layernorm.weight
};

/**
 * WeightNameMapper — translates HuggingFace weight names into the format
 * this engine expects. Per-arch differences live here so the rest of the
 * loader can stay generic.
 *
 * Currently handles two families:
 *
 *   - QwenLLaMA: Qwen2/2.5, LLaMA 2/3/3.1, Mistral, Yi, DeepSeek, etc.
 *                Same HF naming: q_proj, k_proj, v_proj, o_proj.
 *                Qwen has q/k/v_proj.bias; the LLaMA family does not.
 *
 *   - Gemma    : Gemma 1/2/3.  Merged `qkv_proj.weight` (no per-head
 *                projection split, no bias).
 */
class WeightNameMapper {
public:
    explicit WeightNameMapper(ModelArch arch) : arch_(arch) {}

    ModelArch arch() const { return arch_; }

    // Build the per-layer name template for a given layer index. The
    // returned struct contains fully-qualified (layer-prefixed) names.
    LayerWeightNames names_for_layer(int64_t layer_idx) const {
        LayerWeightNames n;
        const std::string p = "model.layers." + std::to_string(layer_idx) + ".";

        n.input_layernorm     = p + "input_layernorm.weight";
        n.post_attn_layernorm = p + "post_attention_layernorm.weight";
        n.gate_proj           = p + "mlp.gate_proj.weight";
        n.up_proj             = p + "mlp.up_proj.weight";
        n.down_proj           = p + "mlp.down_proj.weight";
        n.o_proj              = p + "self_attn.o_proj.weight";

        if (arch_ == ModelArch::Gemma) {
            n.qkv_proj = p + "self_attn.qkv_proj.weight";
            // Some Gemma versions use separate q/k/v_proj instead of the
            // merged form. Fill those in too so load_qkv can fall back.
            n.q_proj = p + "self_attn.q_proj.weight";
            n.k_proj = p + "self_attn.k_proj.weight";
            n.v_proj = p + "self_attn.v_proj.weight";
            // Gemma-specific attention sub-norms (Q/K RMSNorm pre-RoPE,
            // present in Gemma 3). Optional; the loader skips them.
            n.q_norm = p + "self_attn.q_norm.weight";
            n.k_norm = p + "self_attn.k_norm.weight";
            // Extra MLP-side norms in Gemma 2/3 (post-/pre-/post-FFN).
            n.pre_feedforward_layernorm  = p + "pre_feedforward_layernorm.weight";
            n.post_feedforward_layernorm = p + "post_feedforward_layernorm.weight";
        } else {
            n.q_proj = p + "self_attn.q_proj.weight";
            n.k_proj = p + "self_attn.k_proj.weight";
            n.v_proj = p + "self_attn.v_proj.weight";
            n.q_proj_bias = p + "self_attn.q_proj.bias";
            n.k_proj_bias = p + "self_attn.k_proj.bias";
            n.v_proj_bias = p + "self_attn.v_proj.bias";
        }
        return n;
    }

    // Top-level (non-layer) weight names.
    std::string embed_tokens()  const { return "model.embed_tokens.weight"; }
    std::string model_norm()    const { return "model.norm.weight"; }
    std::string lm_head()       const { return "lm_head.weight"; }

    // Convenience: read an attention QKV weight and return it as three
    // separate FP16 device tensors. For Qwen/LLaMA this is three independent
    // reads. For Gemma this is one read followed by three row-slices.
    struct QKVLoadResult {
        Tensor w_q, w_k, w_v;  // [num_*heads * head_dim, hidden_size] FP16 CUDA
        Tensor b_q, b_k, b_v;  // empty unless has_qkv_bias and bias tensors exist
    };
    QKVLoadResult load_qkv(const WeightIndex& idx,
                           const LayerWeightNames& names,
                           int64_t num_heads,
                           int64_t num_kv_heads,
                           int64_t head_dim,
                           int64_t hidden_size,
                           int device_index) const;

    // Read a weight tensor by HF name, converting to FP16 and uploading to
    // the given device. Throws if the tensor is missing.
    static Tensor load_weight_as_f16(const WeightIndex& idx,
                                     const std::string& name,
                                     int device_index);

private:
    ModelArch arch_;
};

}  // namespace mini_infer