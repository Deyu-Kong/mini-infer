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
#include "layers/rmsnorm.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"

namespace mini_infer {

/**
 * Per-layer decoder block — RMSNorm -> Attention -> +residual ->
 *                              RMSNorm -> MLP -> +residual
 *
 * All components are constructed empty; `QwenModel::load_weights` fills
 * them from a WeightIndex. Each component owns its own device buffers.
 */
struct LayerWeights {
    RMSNorm          input_layernorm;
    RMSNorm          post_attn_layernorm;
    MLP              mlp;
    Attention        attn;

    LayerWeights() = default;
    LayerWeights(const LayerWeights&) = delete;
    LayerWeights& operator=(const LayerWeights&) = delete;
    LayerWeights(LayerWeights&&) = default;
    LayerWeights& operator=(LayerWeights&&) = default;
};

/**
 * QwenModel — Qwen2.5 / LLaMA-style decoder.
 *
 * Loads weights from a HuggingFace safetensors index and owns the
 * per-layer components. `forward(...)` runs a prefill or decode step and
 * returns logits; `Engine::generate` wraps this in an autoregressive loop.
 */
class QwenModel {
public:
    QwenModel(const ModelConfig& cfg, int device_index = 0);
    ~QwenModel();

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