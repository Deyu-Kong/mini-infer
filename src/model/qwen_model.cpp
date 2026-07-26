#include "model/qwen_model.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "core/dtype_utils.h"
#include "layers/rmsnorm.h"
#include "kernels/model_utils_kernel.cuh"
#include "scheduler/paged_kv_cache.h"

namespace mini_infer {

namespace {

// Debug helper: print tensor statistics
static void debug_print_tensor(const char* name, const Tensor& t, int max_elements = 10) {
    if (t.dtype() != DType::FP16) {
        return;
    }
    
    // Copy to CPU
    Tensor cpu_t = t.to(Device::cpu());
    const __half* data = static_cast<const __half*>(cpu_t.data());
    int64_t numel = cpu_t.numel();
    
    // Compute mean and std
    double sum = 0.0, sum_sq = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
        float val = __half2float(data[i]);
        sum += val;
        sum_sq += val * val;
    }
    double mean = sum / numel;
    double variance = (sum_sq / numel) - (mean * mean);
    double std = std::sqrt(variance);
    
    for (size_t i = 0; i < t.shape().size(); ++i) {
        if (i > 0) std::printf(",");
        std::printf("%ld", t.shape()[i]);
    }
    std::printf("], mean=%.6f, std=%.6f", mean, std);
    
    // Print first few elements
    if (max_elements > 0 && numel > 0) {
        std::printf(", first_%d=[", std::min((int64_t)max_elements, numel));
        for (int64_t i = 0; i < std::min((int64_t)max_elements, numel); ++i) {
            if (i > 0) std::printf(",");
            std::printf("%.6f", __half2float(data[i]));
        }
        std::printf("]");
    }
    std::printf("\n");
}

// Convert a CPU tensor to FP16 (same device). Supports BF16 / FP32 inputs.
Tensor convert_cpu_to_f16(const Tensor& src) {
    if (src.dtype() == DType::FP16) return src;
    Tensor dst(src.shape(), DType::FP16, Device::cpu());
    const int64_t n = src.numel();
    if (src.dtype() == DType::BF16) {
        const auto* s = static_cast<const uint16_t*>(src.data());
        auto*       d = static_cast<uint16_t*>(dst.data());
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t f32_bits = static_cast<uint32_t>(s[i]) << 16;
            float f; std::memcpy(&f, &f32_bits, 4);
            d[i] = f32_to_f16_bits(f);
        }
        return dst;
    }
    if (src.dtype() == DType::FP32) {
        const auto* s = static_cast<const float*>(src.data());
        auto*       d = static_cast<uint16_t*>(dst.data());
        for (int64_t i = 0; i < n; ++i) d[i] = f32_to_f16_bits(s[i]);
        return dst;
    }
    throw std::runtime_error("convert_cpu_to_f16: unsupported dtype " +
                             std::string(dtype_name(src.dtype())));
}

void require_shape(const Tensor& t, const std::vector<int64_t>& want,
                   const std::string& name) {
    if (t.shape() != want) {
        std::ostringstream o;
        o << "QwenModel: bad shape for " << name << ": got [";
        for (size_t i = 0; i < t.shape().size(); ++i) {
            if (i) o << ",";
            o << t.shape()[i];
        }
        o << "] expected [";
        for (size_t i = 0; i < want.size(); ++i) {
            if (i) o << ",";
            o << want[i];
        }
        o << "]";
        throw std::runtime_error(o.str());
    }
}

Tensor upload_as_f16(const WeightIndex& idx, const std::string& name,
                     int device_index) {
    Tensor cpu = idx.read_to_cpu(name);
    Tensor cpu_f16 = convert_cpu_to_f16(cpu);
    return cpu_f16.to(Device::cuda(device_index));
}

// Small helper for summarize(): FP16 mean/std via host loop.
static void f16_stats(const Tensor& t, float& mean, float& stddev,
                      int64_t max_n = 4096) {
    if (t.dtype() != DType::FP16 || t.numel() == 0) { mean = 0; stddev = 0; return; }
    const int64_t n = std::min<int64_t>(t.numel(), max_n);
    const auto* p = static_cast<const uint16_t*>(t.data());
    double sum = 0.0, sum2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const uint16_t bits = p[i];
        uint32_t f32_bits;
        if (bits == 0) {
            f32_bits = 0;
        } else if ((bits & 0x7c00u) == 0x7c00u) {
            f32_bits = ((bits & 0x8000u) << 16) | 0x7f800000u |
                       ((bits & 0x3ffu) ? 0x200000u : 0u);
        } else {
            f32_bits = ((static_cast<uint32_t>((bits & 0x7c00u) >> 10) - 15 + 127) << 23) |
                       ((bits & 0x8000u) << 16) | ((bits & 0x3ffu) << 13);
        }
        float f; std::memcpy(&f, &f32_bits, 4);
        sum  += f;
        sum2 += static_cast<double>(f) * f;
    }
    mean   = static_cast<float>(sum / n);
    stddev = static_cast<float>(std::sqrt(sum2 / n - mean * mean));
}

// CUDA mat-mul: logits[S, V] = hidden[S, H] @ lm_head^T  (col-major trick).
// Implemented via cuBLAS. Reuses the same row-major -> col-major mapping as
// layers/mlp.cpp.

}  // namespace

QwenModel::QwenModel(const ModelConfig& cfg, int device_index)
    : cfg_(cfg),
      device_index_(device_index),
      final_norm_(cfg.hidden_size, cfg.rms_norm_eps, device_index) {
    if (cfg.num_hidden_layers <= 0) {
        throw std::runtime_error("QwenModel: invalid num_hidden_layers");
    }
    layers_.reserve(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        LayerWeights lw;
        lw.input_layernorm     = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device_index);
        lw.post_attn_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, device_index);
        lw.mlp.init(cfg.hidden_size, cfg.intermediate_size, device_index);
        lw.attn.init(cfg.hidden_size,
                     cfg.num_attention_heads,
                     cfg.num_key_value_heads,
                     cfg.head_dim(),
                     cfg.rope_theta,
                     device_index);
        layers_.push_back(std::move(lw));
    }
    embed_   = Tensor::empty({cfg.vocab_size, cfg.hidden_size}, DType::FP16,
                              Device::cuda(device_index));
    lm_head_ = Tensor::empty({cfg.vocab_size, cfg.hidden_size}, DType::FP16,
                              Device::cuda(device_index));
}

QwenModel::~QwenModel() = default;

void QwenModel::load_weights(const WeightIndex& idx) {
    
    const int64_t H   = cfg_.hidden_size;
    const int64_t I   = cfg_.intermediate_size;
    const int64_t Hq  = cfg_.num_attention_heads * cfg_.head_dim();
    const int64_t Hkv = cfg_.num_key_value_heads * cfg_.kv_head_dim();

    for (int64_t i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";

        
        layers_[i].input_layernorm.set_weight(
            upload_as_f16(idx, p + "input_layernorm.weight", device_index_));
        
        layers_[i].post_attn_layernorm.set_weight(
            upload_as_f16(idx, p + "post_attention_layernorm.weight", device_index_));

        
        Tensor wg = upload_as_f16(idx, p + "mlp.gate_proj.weight", device_index_);
        Tensor wu = upload_as_f16(idx, p + "mlp.up_proj.weight",   device_index_);
        Tensor wd = upload_as_f16(idx, p + "mlp.down_proj.weight", device_index_);
        
        std::fprintf(stderr, "Layer %ld MLP shapes: wg=[%ld,%ld] wu=[%ld,%ld] wd=[%ld,%ld] expected=[%ld,%ld]\n",
                     i, wg.shape()[0], wg.shape()[1], wu.shape()[0], wu.shape()[1],
                     wd.shape()[0], wd.shape()[1], cfg_.intermediate_size, cfg_.hidden_size);
        
        layers_[i].mlp.set_weights(wg, wu, wd);

        
        Tensor wq = upload_as_f16(idx, p + "self_attn.q_proj.weight", device_index_);
        Tensor wk = upload_as_f16(idx, p + "self_attn.k_proj.weight", device_index_);
        Tensor wv = upload_as_f16(idx, p + "self_attn.v_proj.weight", device_index_);
        Tensor wo = upload_as_f16(idx, p + "self_attn.o_proj.weight", device_index_);
        Tensor bq, bk, bv;
        const bool has_bias = (idx.find(p + "self_attn.q_proj.bias") != nullptr);
        if (has_bias) {
            bq = upload_as_f16(idx, p + "self_attn.q_proj.bias", device_index_);
            bk = upload_as_f16(idx, p + "self_attn.k_proj.bias", device_index_);
            bv = upload_as_f16(idx, p + "self_attn.v_proj.bias", device_index_);
        }
        layers_[i].attn.set_weights(wq, wk, wv, wo, bq, bk, bv);

        require_shape(wq, {Hq,  H},   p + "q_proj.weight");
        require_shape(wk, {Hkv, H},   p + "k_proj.weight");
        require_shape(wv, {Hkv, H},   p + "v_proj.weight");
        require_shape(wo, {H,   Hq},  p + "o_proj.weight");
        if (has_bias) {
            require_shape(bq, {Hq},  p + "q_proj.bias");
            require_shape(bk, {Hkv}, p + "k_proj.bias");
            require_shape(bv, {Hkv}, p + "v_proj.bias");
        }
    }

    embed_ = upload_as_f16(idx, "model.embed_tokens.weight", device_index_);
    require_shape(embed_, {cfg_.vocab_size, H}, "model.embed_tokens.weight");
    if (cfg_.tie_word_embeddings) {
        lm_head_ = embed_;
    } else {
        lm_head_ = upload_as_f16(idx, "lm_head.weight", device_index_);
        require_shape(lm_head_, {cfg_.vocab_size, H}, "lm_head.weight");
    }
    final_norm_.set_weight(upload_as_f16(idx, "model.norm.weight", device_index_));
}

// ---------------------------------------------------------------------------
// Forward pass — runs all `num_hidden_layers` blocks + final norm + LM head.
// ---------------------------------------------------------------------------
Tensor QwenModel::forward(const Tensor& token_ids,
                          const std::vector<int64_t>& positions,
                          std::vector<__half*>& k_ptrs,
                          std::vector<__half*>& v_ptrs,
                          int64_t max_seq,
                          int64_t cur_len, bool is_prefill) {
    if (token_ids.dtype() != DType::INT64) {
        throw std::runtime_error("QwenModel::forward: token_ids must be int64");
    }
    if (token_ids.ndim() != 2) {
        throw std::runtime_error("QwenModel::forward: token_ids must be 2-D");
    }
    const int B = static_cast<int>(token_ids.shape()[0]);
    const int S = static_cast<int>(token_ids.shape()[1]);
    const int H = static_cast<int>(cfg_.hidden_size);
    if (k_ptrs.size() != layers_.size() || v_ptrs.size() != layers_.size()) {
        throw std::runtime_error("QwenModel::forward: kv ptrs size mismatch");
    }

    // Embedding lookup on the GPU.
    Tensor hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_embedding_gather(
        static_cast<const int64_t*>(token_ids.data()),
        static_cast<const __half*>(embed_.data()),
        static_cast<__half*>(hidden.data()), B, S, H, /*stream=*/0);
    
    // debug_print_tensor("embeddings", hidden);

    residual_buf_ = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    normed_buf_   = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));

    for (size_t i = 0; i < layers_.size(); ++i) {
        // ---- attention block -----------------------------------------
        // Reshape hidden from [B, S, H] to [B*S, H] for RMSNorm
        Tensor hidden_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(hidden_2d.data(), hidden.data(), 
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        
        Tensor normed_2d = layers_[i].input_layernorm.forward(hidden_2d);
        
        if (i == 0) {
            // debug_print_tensor("after_input_layernorm_layer0", normed_2d);
        }
        
        // Reshape back to [B, S, H]
        normed_buf_ = Tensor({B, S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(normed_buf_.data(), normed_2d.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);

        Tensor attn_out = layers_[i].attn.forward(normed_buf_, positions,
                                                  k_ptrs[i], v_ptrs[i],
                                                  max_seq, cur_len, is_prefill);
        
        if (i == 0 || i == 3) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_attention_layer%zu", i);
            // debug_print_tensor(buf, attn_out);
        }

        // residual: hidden += attn_out
        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                           static_cast<const __half*>(attn_out.data()),
                           static_cast<int64_t>(B) * S * H, /*stream=*/0);
        
        if (i == 0) {
            // debug_print_tensor("after_residual1_layer0", hidden);
        }

        // ---- post-attention norm + MLP ------------------------------
        // Reshape hidden from [B, S, H] to [B*S, H] for RMSNorm
        cudaMemcpy(hidden_2d.data(), hidden.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        
        normed_2d = layers_[i].post_attn_layernorm.forward(hidden_2d);
        
        if (i == 0 || i == 3) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_post_attn_layernorm_layer%zu", i);
            // debug_print_tensor(buf, normed_2d);
        }
        
        // MLP expects [B*S, H] input, which is what we have in normed_2d
        Tensor mlp_out_2d = layers_[i].mlp.forward(normed_2d);
        
        if (i == 0 || i == 3) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_mlp_layer%zu", i);
            // debug_print_tensor(buf, mlp_out_2d);
        }
        
        // Reshape back to [B, S, H] and add residual
        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                           static_cast<const __half*>(mlp_out_2d.data()),
                           static_cast<int64_t>(B) * S * H, /*stream=*/0);
        
        if (i == 0 || i == 3 || i == 6 || i == 7 || i == 8 || i == 9 || i == 10 || i == 15 || i == 20 || i == 27) {
            char buf[64];
            snprintf(buf, sizeof(buf), "after_residual2_layer%zu", i);
            // debug_print_tensor(buf, hidden);
        }
        
        if (i == 27) {
            // debug_print_tensor("after_residual2_layer27", hidden);
        }
    }

    // Final RMSNorm + LM head.
    // Reshape hidden from [B, S, H] to [B*S, H] for final norm
    Tensor hidden_2d_final({B * S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(hidden_2d_final.data(), hidden.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);
    
    Tensor final_hidden_2d = final_norm_.forward(hidden_2d_final);
    
    // debug_print_tensor("final_hidden_before_norm", hidden_2d_final);
    // debug_print_tensor("final_hidden_after_norm", final_hidden_2d);
    
    // Reshape back to [B, S, H]
    Tensor final_hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(final_hidden.data(), final_hidden_2d.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    // logits = final_hidden @ lm_head^T  via cuBLAS, see layers/mlp.cpp.
    Tensor logits({B, S, static_cast<int>(cfg_.vocab_size)}, DType::FP16,
                  Device::cuda(device_index_));
    {
        // We can reuse the MLP's cuBLAS handle by going through a fresh
        // cublas call here. The MLP destructor will release it.
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float one = 1.0f, zero = 0.0f;
        const int Mflat = B * S;
        const int Nvoc = static_cast<int>(cfg_.vocab_size);
        // logits = final_hidden @ lm_head_^T
        // In cuBLAS (column-major):
        // - lm_head_ is stored as row-major [vocab_size, H], which is column-major [H, vocab_size]
        // - final_hidden is stored as row-major [B*S, H], which is column-major [H, B*S]
        // - logits is stored as row-major [B*S, vocab_size], which is column-major [vocab_size, B*S]
        // We want: logits^T = lm_head_ @ final_hidden^T
        // In column-major: C = A^T @ B, where A=lm_head_, B=final_hidden, C=logits
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nvoc, Mflat, H,
            &one,
            static_cast<const __half*>(lm_head_.data()), CUDA_R_16F, H,
            static_cast<const __half*>(final_hidden.data()), CUDA_R_16F, H,
            &zero,
            static_cast<__half*>(logits.data()), CUDA_R_16F, Nvoc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cublasDestroy(handle);
    }
    
    // debug_print_tensor("final_logits", logits, 20);

    return logits;
}

// ---------------------------------------------------------------------------
// Batched paged forward (Week 5+). Same as forward_paged but for B>1.
// ---------------------------------------------------------------------------
Tensor QwenModel::forward_paged_batched(const Tensor& token_ids,
                                        const std::vector<int64_t>& positions,
                                        PagedKVCache& paged_kv,
                                        const std::vector<int>& seq_ids,
                                        const std::vector<int>& start_pos,
                                        bool is_prefill) {
    if (token_ids.dtype() != DType::INT64) {
        throw std::runtime_error("forward_paged_batched: token_ids must be int64");
    }
    if (token_ids.ndim() != 2) {
        throw std::runtime_error("forward_paged_batched: token_ids must be 2-D");
    }
    const int B = static_cast<int>(token_ids.shape()[0]);
    const int S = static_cast<int>(token_ids.shape()[1]);
    const int H = static_cast<int>(cfg_.hidden_size);

    Tensor hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_embedding_gather(
        static_cast<const int64_t*>(token_ids.data()),
        static_cast<const __half*>(embed_.data()),
        static_cast<__half*>(hidden.data()), B, S, H, /*stream=*/0);

    residual_buf_ = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    normed_buf_   = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));

    for (size_t i = 0; i < layers_.size(); ++i) {
        Tensor hidden_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(hidden_2d.data(), hidden.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        Tensor normed_2d = layers_[i].input_layernorm.forward(hidden_2d);

        normed_buf_ = Tensor({B, S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(normed_buf_.data(), normed_2d.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);

        Tensor attn_out = layers_[i].attn.forward_paged_batched(
            normed_buf_, positions, paged_kv, seq_ids, start_pos,
            static_cast<int>(i), is_prefill);

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(attn_out.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);

        cudaMemcpy(hidden_2d.data(), hidden.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        normed_2d = layers_[i].post_attn_layernorm.forward(hidden_2d);
        Tensor mlp_out_2d = layers_[i].mlp.forward(normed_2d);

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(mlp_out_2d.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);
    }

    // Final RMSNorm.
    Tensor hidden_2d_final({B * S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(hidden_2d_final.data(), hidden.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);
    Tensor final_hidden_2d = final_norm_.forward(hidden_2d_final);
    Tensor final_hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(final_hidden.data(), final_hidden_2d.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    // LM head: [B*S, V] = [B*S, H] @ [V, H].T
    Tensor logits({B, S, static_cast<int>(cfg_.vocab_size)}, DType::FP16,
                  Device::cuda(device_index_));
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float one = 1.0f, zero = 0.0f;
        const int Mflat = B * S;
        const int Nvoc = static_cast<int>(cfg_.vocab_size);
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nvoc, Mflat, H,
            &one,
            static_cast<const __half*>(lm_head_.data()), CUDA_R_16F, H,
            static_cast<const __half*>(final_hidden.data()), CUDA_R_16F, H,
            &zero,
            static_cast<__half*>(logits.data()), CUDA_R_16F, Nvoc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cublasDestroy(handle);
    }
    return logits;
}

// ---------------------------------------------------------------------------
// Paged forward — Week 5. Mirrors `forward()` but uses
// Attention::forward_paged() for every block, so K/V storage is
// PagedKVCache-managed.
// ---------------------------------------------------------------------------
Tensor QwenModel::forward_paged(const Tensor& token_ids,
                                const std::vector<int64_t>& positions,
                                PagedKVCache& paged_kv,
                                int seq_id,
                                bool is_prefill) {
    if (token_ids.dtype() != DType::INT64) {
        throw std::runtime_error("QwenModel::forward_paged: token_ids must be int64");
    }
    if (token_ids.ndim() != 2) {
        throw std::runtime_error("QwenModel::forward_paged: token_ids must be 2-D");
    }
    const int B = static_cast<int>(token_ids.shape()[0]);
    const int S = static_cast<int>(token_ids.shape()[1]);
    const int H = static_cast<int>(cfg_.hidden_size);

    // Embedding lookup.
    Tensor hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_embedding_gather(
        static_cast<const int64_t*>(token_ids.data()),
        static_cast<const __half*>(embed_.data()),
        static_cast<__half*>(hidden.data()), B, S, H, /*stream=*/0);

    residual_buf_ = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));
    normed_buf_   = Tensor::empty({B, S, H}, DType::FP16, Device::cuda(device_index_));

    for (size_t i = 0; i < layers_.size(); ++i) {
        // ---- attention block -------------------------------------------
        Tensor hidden_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(hidden_2d.data(), hidden.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        Tensor normed_2d = layers_[i].input_layernorm.forward(hidden_2d);

        normed_buf_ = Tensor({B, S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(normed_buf_.data(), normed_2d.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);

        Tensor attn_out = layers_[i].attn.forward_paged(
            normed_buf_, positions, paged_kv, seq_id,
            static_cast<int>(i), is_prefill);

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(attn_out.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);

        // ---- post-attention norm + MLP -------------------------------
        cudaMemcpy(hidden_2d.data(), hidden.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);
        normed_2d = layers_[i].post_attn_layernorm.forward(hidden_2d);

        Tensor mlp_out_2d = layers_[i].mlp.forward(normed_2d);

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(mlp_out_2d.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);
    }

    // Final RMSNorm + LM head.
    Tensor hidden_2d_final({B * S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(hidden_2d_final.data(), hidden.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);
    Tensor final_hidden_2d = final_norm_.forward(hidden_2d_final);
    Tensor final_hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    cudaMemcpy(final_hidden.data(), final_hidden_2d.data(),
               static_cast<int64_t>(B) * S * H * sizeof(__half),
               cudaMemcpyDeviceToDevice);

    Tensor logits({B, S, static_cast<int>(cfg_.vocab_size)}, DType::FP16,
                  Device::cuda(device_index_));
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float one = 1.0f, zero = 0.0f;
        const int Mflat = B * S;
        const int Nvoc = static_cast<int>(cfg_.vocab_size);
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Nvoc, Mflat, H,
            &one,
            static_cast<const __half*>(lm_head_.data()), CUDA_R_16F, H,
            static_cast<const __half*>(final_hidden.data()), CUDA_R_16F, H,
            &zero,
            static_cast<__half*>(logits.data()), CUDA_R_16F, Nvoc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cublasDestroy(handle);
    }
    return logits;
}

}  // namespace mini_infer