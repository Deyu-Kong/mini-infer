#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "core/tensor.h"

namespace mini_infer {

/**
 * KVCache — pre-allocated per-layer K/V cache for the naive-attention
 * decoder.
 *
 * Storage layout (per the W4 spec):
 *
 *   k_cache_, v_cache_ : [num_layers, num_kv_heads, max_seq, head_dim]   FP16
 *
 * Layer L's K slot starts at byte offset:
 *
 *   L * num_kv_heads * max_seq * head_dim * 2
 *
 * Within a layer, the layout is [num_kv_heads, max_seq, head_dim].
 * Week-4 baseline; replaced by PagedAttention in W5.
 */
class KVCache {
public:
    KVCache(int64_t num_layers, int64_t num_kv_heads, int64_t head_dim,
            int64_t max_seq, int device_index);

    // Pointer to layer L's K-cache base. Caller indexes into [H, S, D].
    __half* k_layer_ptr(int64_t layer_idx);
    __half* v_layer_ptr(int64_t layer_idx);

    // The full underlying tensors (useful for tests + diagnostics).
    const Tensor& k_cache() const { return k_cache_; }
    const Tensor& v_cache() const { return v_cache_; }

    int64_t num_layers()  const { return num_layers_; }
    int64_t num_kv_heads()const { return num_kv_heads_; }
    int64_t head_dim()    const { return head_dim_; }
    int64_t max_seq()     const { return max_seq_; }
    int64_t bytes_used()  const { return k_cache_.nbytes() + v_cache_.nbytes(); }

private:
    int64_t num_layers_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    int64_t max_seq_;
    int     device_index_;
    Tensor  k_cache_;
    Tensor  v_cache_;
};

}  // namespace mini_infer