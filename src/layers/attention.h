#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.h"
#include "kernels/naive_attn_kernel.cuh"
#include "layers/rope.h"

namespace mini_infer {

/**
 * Single-layer attention (Qwen2 / LLaMA-style + Gemma variants).
 *
 * Storage layout for the projection weights matches HuggingFace safetensors:
 *   W_q : [num_heads * head_dim, hidden]
 *   W_k : [num_kv_heads * head_dim, hidden]    (GQA: num_kv_heads <= num_heads)
 *   W_v : [num_kv_heads * head_dim, hidden]
 *   W_o : [hidden, num_heads * head_dim]
 *   b_q/k/v : [num_heads * head_dim] / [num_kv_heads * head_dim]   (Qwen2 only)
 *   W_q_norm : [head_dim]   (Gemma 3 only — RMSNorm applied to Q before RoPE)
 *   W_k_norm : [head_dim]   (Gemma 3 only — RMSNorm applied to K before RoPE)
 *
 * Forward semantics:
 *
 *   Q = x @ W_q^T + b_q          [B, S, H_q*D]
 *   K = x @ W_k^T + b_k          [B, S, H_kv*D]
 *   V = x @ W_v^T + b_v          [B, S, H_kv*D]
 *   Q,K,V <- reshape to [B,S,H,D]
 *   Q <- Q_norm(Q)               (Gemma 3 only; requires use_qk_norm_)
 *   K <- K_norm(K)               (Gemma 3 only)
 *   Q,K <- RoPE at positions[0..S)
 *   K,V -> appended into kv_k_cache / kv_v_cache at slot [cur_len, cur_len+S)
 *   attn = SDPA(Q, K, V, causal = is_prefill, sliding_window = sliding_window_)
 *   out  = attn @ W_o^T                            [B, S, hidden]
 */
class Attention {
public:
    // Default constructor creates an uninitialized Attention.
    // Must call init() before use.
    Attention() = default;

    Attention(int64_t hidden, int64_t num_heads, int64_t num_kv_heads,
              int64_t head_dim, float rope_theta, int device_index);
    ~Attention();

    // Initialize with parameters.
    void init(int64_t hidden, int64_t num_heads, int64_t num_kv_heads,
              int64_t head_dim, float rope_theta, int device_index);

    // Enable Q/K RMSNorm pre-RoPE (Gemma 3).
    void set_qk_norm(int64_t head_dim, float eps, int device_index);

    // Enable sliding-window attention (Gemma 2/3). `window` is the number
    // of tokens each query can attend back to. 0 disables.
    void set_sliding_window(int64_t window) { sliding_window_ = window; }

    // Switch RoPE inv_freq source (Gemma 3 dual-band). When set, RoPE
    // uses `local_rope_theta` instead of the construction-time `rope_theta_`.
    void use_local_rope(bool v) { use_local_rope_ = v; }

    Attention(const Attention&) = delete;
    Attention& operator=(const Attention&) = delete;

    // Move constructor and assignment.
    Attention(Attention&& other) noexcept;
    Attention& operator=(Attention&& other) noexcept;

    // Weight pointers (set after the model loader materialises them on GPU).
    void set_weights(Tensor w_q, Tensor w_k, Tensor w_v, Tensor w_o,
                     Tensor b_q, Tensor b_k, Tensor b_v);

    // Q/K RMSNorm weights (Gemma 3). Optional; if numel==0 they're skipped.
    void set_qk_norm_weights(const Tensor& w_q_norm, const Tensor& w_k_norm);

    // Local vs global RoPE theta (Gemma 3 dual-band).
    void set_local_rope_theta(float theta) { local_rope_theta_ = theta; }
    float local_rope_theta() const { return local_rope_theta_; }

    // Forward pass.
    //   hidden_states : [B, S, hidden]   FP16 CUDA, contiguous
    //   positions     : length-S int64 vector (host)
    //   kv_k_ptr      : __half* to layer's K cache (size num_kv_heads*max_seq*head_dim)
    //   kv_v_ptr      : __half* to layer's V cache (size num_kv_heads*max_seq*head_dim)
    //   max_seq       : maximum sequence length for KV cache bounds checking
    //   cur_len       : how many tokens are already in the cache (0 prefill)
    //   is_prefill    : apply causal mask; otherwise single-token decode
    //
    // Returns hidden_states-shaped [B, S, hidden] FP16 on CUDA.
    Tensor forward(const Tensor& hidden_states,
                   const std::vector<int64_t>& positions,
                   __half* kv_k_ptr,
                   __half* kv_v_ptr,
                   int64_t max_seq,
                   int64_t cur_len,
                   bool is_prefill);

    // Paged forward (Week 5).
    //   hidden_states  : [B, S, hidden]                       FP16 CUDA
    //   positions      : length-S int64 vector (host)
    //   paged_kv       : the shared PagedKVCache
    //   seq_id         : which sequence in the cache
    //   layer_idx      : index of this layer in the cache
    //   is_prefill     : apply causal mask
    //
    // The caller is responsible for calling paged_kv.append_token() for
    // each token in positions BEFORE forward_paged (the attention reads
    // through the freshly extended block table).
    //
    // Returns hidden_states-shaped [B, S, hidden] FP16 on CUDA.
    Tensor forward_paged(const Tensor& hidden_states,
                         const std::vector<int64_t>& positions,
                         class PagedKVCache& paged_kv,
                         int seq_id,
                         int layer_idx,
                         bool is_prefill);

    // Batched paged forward (Week 5+). Same as forward_paged but accepts
    // B>1 active sequences sharing the PagedKVCache. Each sequence b
    // owns a contiguous range of rows in the input/output tensors and a
    // unique seq_id for cache lookup.
    //
    //   hidden_states  : [B, S, hidden] FP16 CUDA, contiguous
    //   positions      : length (B*S) int64 (host) — flat row-major
    //                    positions[b * S + s] for query token s of seq b
    //   paged_kv       : shared PagedKVCache
    //   seq_ids        : vector<int> of length B; seq_ids[b] identifies
    //                    the block table for sequence b
    //   start_pos      : vector<int> of length B; start_pos[b] = global
    //                    position where the new tokens begin for seq b
    //                    (typically seq_len[b] - S after append_token)
    //   layer_idx      : which layer to write/read in the cache
    //   is_prefill     : apply causal mask
    //
    // Returns hidden_states-shaped [B, S, hidden] FP16 on CUDA.
    Tensor forward_paged_batched(const Tensor& hidden_states,
                                 const std::vector<int64_t>& positions,
                                 class PagedKVCache& paged_kv,
                                 const std::vector<int>& seq_ids,
                                 const std::vector<int>& start_pos,
                                 int layer_idx,
                                 bool is_prefill);

    int64_t hidden()       const { return hidden_; }
    int64_t num_heads()    const { return num_heads_; }
    int64_t num_kv_heads() const { return num_kv_heads_; }
    int64_t head_dim()     const { return head_dim_; }
    int device_index()     const { return device_index_; }

private:
    int64_t hidden_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    float   rope_theta_;
    int     device_index_;

    Tensor w_q_, w_k_, w_v_, w_o_;
    Tensor b_q_, b_k_, b_v_;
    bool   has_bias_ = false;

    // Q/K RMSNorm (Gemma 3 only). If `use_qk_norm_` is true and the
    // weights are loaded, we apply RMSNorm to Q and K before RoPE.
    Tensor w_q_norm_, w_k_norm_;
    bool   use_qk_norm_ = false;
    float  qk_norm_eps_ = 1e-6f;

    // Sliding-window attention (Gemma 2/3). 0 means full attention.
    int64_t sliding_window_ = 0;

    // Local vs global RoPE (Gemma 3). When true, RoPE uses
    // local_rope_theta_ instead of rope_theta_ for inv_freq.
    bool   use_local_rope_ = false;
    float  local_rope_theta_ = 10000.0f;

    // Persistent buffers (allocated lazily based on B*S).
    Tensor q_buf_, k_buf_, v_buf_;     // [B, S, H_*D]  per-step projections
    Tensor attn_out_buf_;             // [B, S, H_q*D]
    Tensor k_scratch_, v_scratch_;    // [B, S, H_kv*D] before RoPE / cache copy

    // Paged-forward bookkeeping.
    Tensor block_table_dev_;          // [max_blocks_per_seq]      int32  CUDA
    Tensor num_blocks_used_dev_;      // [1]                       int32  CUDA
    Tensor seq_len_dev_;              // [1]                       int32  CUDA
    Tensor start_pos_dev_;            // [1]                       int32  CUDA
    Tensor q_4d_paged_, k_4d_paged_;  // [B, S, H_*, D] views for RoPE
    int   cached_max_blocks_ = 0;
    int   cached_num_kv_heads_ = 0;
    int   cached_head_dim_     = 0;

    // Batched paged-forward bookkeeping.
    Tensor block_tables_dev_;         // [B * max_blocks_per_seq]  int32  CUDA
    Tensor num_blocks_used_b_dev_;    // [B]                       int32  CUDA
    Tensor seq_len_b_dev_;            // [B]                       int32  CUDA
    Tensor start_pos_b_dev_;          // [B]                       int32  CUDA
    Tensor q_4d_b_, k_4d_b_;          // [B, S, H_*, D] views for RoPE
    Tensor attn_out_b_;               // [B, S, H_q*D]
    int   cached_max_B_ = 0;

    // RoPE for Q+K.
    std::unique_ptr<RoPE> rope_q_;
    std::unique_ptr<RoPE> rope_k_;

    // cuBLAS handle.
    void* cublas_handle_ = nullptr;
};

}  // namespace mini_infer