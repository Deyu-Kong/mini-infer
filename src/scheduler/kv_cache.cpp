#include "scheduler/kv_cache.h"

#include <cuda_fp16.h>

#include <stdexcept>

namespace mini_infer {

KVCache::KVCache(int64_t num_layers, int64_t num_kv_heads, int64_t head_dim,
                 int64_t max_seq, int device_index)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads),
      head_dim_(head_dim), max_seq_(max_seq), device_index_(device_index) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
        throw std::runtime_error("KVCache: invalid dimensions");
    }
    const std::vector<int64_t> shape = {num_layers, num_kv_heads, max_seq, head_dim};
    k_cache_ = Tensor::empty(shape, DType::FP16, Device::cuda(device_index));
    v_cache_ = Tensor::empty(shape, DType::FP16, Device::cuda(device_index));
}

__half* KVCache::k_layer_ptr(int64_t layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("KVCache: layer_idx out of range");
    }
    const int64_t layer_stride = num_kv_heads_ * max_seq_ * head_dim_;
    return static_cast<__half*>(k_cache_.data()) + layer_idx * layer_stride;
}

__half* KVCache::v_layer_ptr(int64_t layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("KVCache: layer_idx out of range");
    }
    const int64_t layer_stride = num_kv_heads_ * max_seq_ * head_dim_;
    return static_cast<__half*>(v_cache_.data()) + layer_idx * layer_stride;
}

}  // namespace mini_infer