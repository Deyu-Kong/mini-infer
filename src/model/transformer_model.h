#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/graph.h"
#include "core/tensor.h"
#include "layers/attention.h"
#include "layers/mlp.h"
#include "layers/moe.h"
#include "layers/rmsnorm.h"
#include "model/arch_registry.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"

namespace mini_infer {

/**
 * Per-layer decoder block.
 *
 *   - LLaMA / Qwen / Mistral / Yi / DeepSeek (2-norm block):
 *
 *         y = x + attn(input_norm(x))
 *         z = y + mlp(post_attn_norm(y))
 *
 *   - Gemma 1 (2-norm block, GeGLU MLP):
 *         Same as LLaMA, but MLP uses GeGLU.
 *
 *   - Gemma 2/3 (4-norm block, double-norm wrapping):
 *
 *         y = x + post_attn_norm(attn(input_norm(x)))
 *         z = y + post_ffn_norm(mlp(pre_ffn_norm(y)))
 *
 *     `pre_feedforward_layernorm` and `post_feedforward_layernorm` are
 *     only populated for Gemma 2/3.
 *
 * All components are constructed empty; `TransformerModel::load_weights` fills
 * them from a WeightIndex. Each component owns its own device buffers.
 */
struct LayerWeights {
    RMSNorm          input_layernorm;
    RMSNorm          post_attn_layernorm;
    RMSNorm          pre_feedforward_layernorm;   // Gemma 2/3 only
    RMSNorm          post_feedforward_layernorm;  // Gemma 2/3 only
    MLP              mlp;
    MoELayer         moe;                         // MoE models only
    Attention        attn;
    bool             is_sliding = false;          // Gemma 2/3 sliding layer
    bool             use_moe = false;             // true = use moe, false = use mlp

    LayerWeights() = default;
    LayerWeights(const LayerWeights&) = delete;
    LayerWeights& operator=(const LayerWeights&) = delete;
    LayerWeights(LayerWeights&&) = default;
    LayerWeights& operator=(LayerWeights&&) = default;
};

/**
 * TransformerModel — generic causal-decoder model covering Qwen2/2.5,
 * LLaMA 2/3/3.1, Mistral, Yi, DeepSeek, and Gemma 1/2/3.
 *
 * Loads weights from a HuggingFace safetensors index and owns the
 * per-layer components. `forward(...)` runs a prefill or decode step and
 * returns logits; `Engine::generate` wraps this in an autoregressive loop.
 *
 * Per-arch behaviour is driven by `ModelConfig::arch` (see model_config.h)
 * and the `ArchRegistry` (see arch_registry.h):
 *   - QwenLLaMA : RMSNorm (y = x * rrms * w), separate q/k/v_proj, optional
 *                 QKV bias (Qwen only). Embedding output is unscaled.
 *                 2-norm block. SwiGLU MLP.
 *   - Gemma 1   : 2-norm block. GeGLU MLP. (1+w) RMSNorm. Scaled embeddings.
 *   - Gemma 2   : 4-norm double-wrapped block. GeGLU MLP. Sliding window
 *                 on alternating layers.
 *   - Gemma 3   : Everything Gemma 2 has, plus Q/K RMSNorm pre-RoPE and
 *                 dual-band RoPE (global theta on global layers, local
 *                 theta on sliding layers).
 *   - GPT2      : (future) LayerNorm, learned positional embeddings, GELU MLP.
 *   - Bloom     : (future) LayerNorm, ALiBi position encoding, GeLU MLP.
 */
class TransformerModel {
public:
    TransformerModel(const ModelConfig& cfg, int device_index = 0);
    ~TransformerModel();

    // Populate weights from a multi-shard safetensors index. Copies (not
    // aliases) onto the device so later code can munmap the source file.
    void load_weights(const WeightIndex& index);

    const Graph& graph() const { return graph_; }
    const ModelConfig& config() const { return cfg_; }
    int device_index() const { return device_index_; }

    // Lightweight accessors used by tests + the engine.
    const std::vector<LayerWeights>& layers() const { return layers_; }
    const Tensor& embed_tokens() const { return embed_; }
    const Tensor& lm_head() const { return lm_head_; }
    const RMSNorm& final_norm() const { return final_norm_; }
    //   token_ids : [B, S] int64 CUDA
    //   positions : [S] int64 (host) — already-known positions for the tokens
    //   k_ptrs/v_ptrs : per-layer base pointers (KVCache::k_layer_ptr etc.)
    //   cur_len   : how many tokens are already in the cache
    //   is_prefill: apply causal mask
    //
    // Returns logits: [B, S, vocab_size] FP16 CUDA. Only the last
    // `S - cur_skip` positions' logits are "fresh"; callers usually
    // sample from the last row.
    Tensor forward(const Tensor& token_ids,
                   const std::vector<int64_t>& positions,
                   std::vector<__half*>& k_ptrs,
                   std::vector<__half*>& v_ptrs,
                   int64_t max_seq,
                   int64_t cur_len, bool is_prefill);

    // Paged forward (Week 5). Same semantics as `forward` but writes K/V
    // through the block table indirection and reads K/V via the
    // PagedAttention kernel. `paged_kv` must already have its block
    // table grown to include the `S` new tokens.
    Tensor forward_paged(const Tensor& token_ids,
                         const std::vector<int64_t>& positions,
                         class PagedKVCache& paged_kv,
                         int seq_id,
                         bool is_prefill);

    // Batched paged forward. Runs B sequences in parallel through the
    // decoder; K/V scatter, paged attention, O projection all batched.
    //
    //   token_ids : [B, S] int64 CUDA
    //   positions : length (B*S) flat row-major positions
    //   paged_kv  : shared PagedKVCache
    //   seq_ids   : vector<int> length B  -> cache index per sequence
    //   start_pos : vector<int> length B  -> global position where the
    //                new tokens begin for each sequence
    //   is_prefill : apply causal mask
    Tensor forward_paged_batched(const Tensor& token_ids,
                                 const std::vector<int64_t>& positions,
                                 class PagedKVCache& paged_kv,
                                 const std::vector<int>& seq_ids,
                                 const std::vector<int>& start_pos,
                                 bool is_prefill);

private:
    ModelConfig cfg_;
    int device_index_;
    std::vector<LayerWeights> layers_;
    RMSNorm final_norm_;
    Tensor  embed_;
    Tensor  lm_head_;
    Graph   graph_;

    Tensor residual_buf_;        // [B, S, H] reused across residual adds
    Tensor normed_buf_;          // [B, S, H]
};

}  // namespace mini_infer