#include "model/transformer_model.h"

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
#include "model/weight_mapper.h"
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
// Defined in weight_mapper.cpp; not needed here.

void require_shape(const Tensor& t, const std::vector<int64_t>& want,
                   const std::string& name) {
    if (t.shape() != want) {
        std::ostringstream o;
        o << "TransformerModel: bad shape for " << name << ": got [";
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

TransformerModel::TransformerModel(const ModelConfig& cfg, int device_index)
    : cfg_(cfg),
      device_index_(device_index),
      final_norm_(cfg.hidden_size, cfg.rms_norm_eps, device_index,
                  cfg.rmsnorm_add_one) {
    if (cfg.num_hidden_layers <= 0) {
        throw std::runtime_error("TransformerModel: invalid num_hidden_layers");
    }
    layers_.reserve(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        LayerWeights lw;
        lw.input_layernorm     = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                         device_index, cfg.rmsnorm_add_one);
        lw.post_attn_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                         device_index, cfg.rmsnorm_add_one);
        // Gemma 2/3 only — pre/post_feedforward_layernorm. They are
        // populated by load_weights when present in the checkpoint; the
        // forward path uses them only when arch == Gemma && double_norm_block.
        lw.pre_feedforward_layernorm  = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                device_index, cfg.rmsnorm_add_one);
        lw.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                device_index, cfg.rmsnorm_add_one);
        // MoE vs dense MLP
        if (cfg.is_moe()) {
            lw.use_moe = true;
            lw.moe.init(cfg.hidden_size, cfg.moe_intermediate_size,
                       cfg.num_experts, cfg.num_experts_per_tok,
                       device_index, cfg.mlp_act);
        } else {
            lw.use_moe = false;
            lw.mlp.init(cfg.hidden_size, cfg.intermediate_size, device_index,
                       cfg.mlp_act);
        }
        lw.attn.init(cfg.hidden_size,
                     cfg.num_attention_heads,
                     cfg.num_key_value_heads,
                     cfg.head_dim(),
                     cfg.rope_theta,
                     device_index);

        // Gemma 3: enable Q/K RMSNorm per layer.
        if (cfg.use_qk_norm) {
            lw.attn.set_qk_norm(cfg.head_dim(), cfg.rms_norm_eps, device_index);
        }
        // Gemma 2/3: sliding window attention on alternating layers.
        if (cfg.sliding_window > 0) {
            lw.attn.set_sliding_window(cfg.sliding_window);
            lw.is_sliding = cfg.is_layer_sliding(i);
            // Gemma 3 dual-band RoPE: sliding layers use local theta.
            if (cfg.dual_rope && lw.is_sliding) {
                lw.attn.use_local_rope(true);
                lw.attn.set_local_rope_theta(cfg.local_rope_theta);
            }
        }

        layers_.push_back(std::move(lw));
    }
    embed_   = Tensor::empty({cfg.vocab_size, cfg.hidden_size}, DType::FP16,
                              Device::cuda(device_index));
    lm_head_ = Tensor::empty({cfg.vocab_size, cfg.hidden_size}, DType::FP16,
                              Device::cuda(device_index));
}

TransformerModel::~TransformerModel() = default;

void TransformerModel::load_weights(const WeightIndex& idx) {
    const int64_t H   = cfg_.hidden_size;
    const int64_t I   = cfg_.intermediate_size;
    const int64_t Hq  = cfg_.num_attention_heads * cfg_.head_dim();
    const int64_t Hkv = cfg_.num_key_value_heads * cfg_.kv_head_dim();

    const WeightNameMapper mapper(cfg_.arch);

    for (int64_t i = 0; i < cfg_.num_hidden_layers; ++i) {
        const LayerWeightNames n = mapper.names_for_layer(i);

        layers_[i].input_layernorm.set_weight(
            WeightNameMapper::load_weight_as_f16(idx, n.input_layernorm,
                                                 device_index_));

        layers_[i].post_attn_layernorm.set_weight(
            WeightNameMapper::load_weight_as_f16(idx, n.post_attn_layernorm,
                                                 device_index_));

        // Gemma 2/3: extra pre/post_feedforward_layernorm. They are only
        // present in Gemma 2/3 safetensors; the forward path consults
        // cfg_.double_norm_block before applying them.
        if (cfg_.double_norm_block) {
            if (!n.pre_feedforward_layernorm.empty() &&
                idx.find(n.pre_feedforward_layernorm) != nullptr) {
                layers_[i].pre_feedforward_layernorm.set_weight(
                    WeightNameMapper::load_weight_as_f16(
                        idx, n.pre_feedforward_layernorm, device_index_));
            }
            if (!n.post_feedforward_layernorm.empty() &&
                idx.find(n.post_feedforward_layernorm) != nullptr) {
                layers_[i].post_feedforward_layernorm.set_weight(
                    WeightNameMapper::load_weight_as_f16(
                        idx, n.post_feedforward_layernorm, device_index_));
            }
        }

        // MLP or MoE: dense models use gate/up/down projections;
        // MoE models use router gate + per-expert MLPs.
        if (cfg_.is_moe()) {
            // MoE: load router gate and expert weights
            // Try Mixtral-style naming first, then DeepSeek-style
            std::string moe_prefix_mixtral = "model.layers." + std::to_string(i) +
                                              ".block_sparse_moe.";
            std::string moe_prefix_deepseek = "model.layers." + std::to_string(i) +
                                               ".mlp.";

            std::string gate_name;
            if (idx.find(moe_prefix_mixtral + "gate.weight") != nullptr) {
                gate_name = moe_prefix_mixtral + "gate.weight";
            } else {
                gate_name = moe_prefix_deepseek + "gate.weight";
            }

            Tensor router_gate = WeightNameMapper::load_weight_as_f16(
                idx, gate_name, device_index_);
            layers_[i].moe.set_router_gate(router_gate);

            // Determine expert naming convention
            bool use_mixtral_naming = (idx.find(moe_prefix_mixtral + "experts.0.w1.weight") != nullptr);

            for (int64_t e = 0; e < cfg_.num_experts; ++e) {
                std::string expert_prefix;
                std::string w1_name, w2_name, w3_name;

                if (use_mixtral_naming) {
                    expert_prefix = moe_prefix_mixtral + "experts." + std::to_string(e) + ".";
                    w1_name = expert_prefix + "w1.weight";
                    w2_name = expert_prefix + "w2.weight";
                    w3_name = expert_prefix + "w3.weight";
                } else {
                    expert_prefix = moe_prefix_deepseek + "experts." + std::to_string(e) + ".";
                    w1_name = expert_prefix + "gate_proj.weight";
                    w2_name = expert_prefix + "down_proj.weight";
                    w3_name = expert_prefix + "up_proj.weight";
                }

                Tensor wg = WeightNameMapper::load_weight_as_f16(idx, w1_name, device_index_);
                Tensor wu = WeightNameMapper::load_weight_as_f16(idx, w3_name, device_index_);
                Tensor wd = WeightNameMapper::load_weight_as_f16(idx, w2_name, device_index_);
                layers_[i].moe.set_expert_weights(e, wg, wu, wd);
            }
        } else {
            // Dense MLP: same shape for both archs, but we go through the
            // mapper for the names so future per-arch renames land in one place.
            Tensor wg = WeightNameMapper::load_weight_as_f16(idx, n.gate_proj,
                                                             device_index_);
            Tensor wu = WeightNameMapper::load_weight_as_f16(idx, n.up_proj,
                                                             device_index_);
            Tensor wd = WeightNameMapper::load_weight_as_f16(idx, n.down_proj,
                                                             device_index_);
            layers_[i].mlp.set_weights(wg, wu, wd);
        }

        // Attention projections. The mapper handles the per-arch split
        // (Gemma merges QKV into a single weight that we slice).
        auto qkv = mapper.load_qkv(idx, n,
                                    cfg_.num_attention_heads,
                                    cfg_.num_key_value_heads,
                                    cfg_.head_dim(), H, device_index_);

        Tensor wo = WeightNameMapper::load_weight_as_f16(idx, n.o_proj,
                                                         device_index_);
        require_shape(wo, {H, Hq}, n.o_proj);

        // Bias is only present for Qwen-style models (Qwen2/2.5); other
        // QwenLLaMA archs (LLaMA 2/3, Mistral, Yi, DeepSeek, ...) and Gemma
        // have no QKV bias.
        Tensor bq, bk, bv;
        bool has_bias = false;
        if (cfg_.arch == ModelArch::QwenLLaMA && cfg_.has_qkv_bias) {
            // Even within Qwen family, double-check the safetensors have it
            // (some checkpoints strip bias).
            if (idx.find(n.q_proj_bias) != nullptr) {
                has_bias = true;
                bq = WeightNameMapper::load_weight_as_f16(idx, n.q_proj_bias,
                                                          device_index_);
                bk = WeightNameMapper::load_weight_as_f16(idx, n.k_proj_bias,
                                                          device_index_);
                bv = WeightNameMapper::load_weight_as_f16(idx, n.v_proj_bias,
                                                          device_index_);
                require_shape(bq, {Hq},  n.q_proj_bias);
                require_shape(bk, {Hkv}, n.k_proj_bias);
                require_shape(bv, {Hkv}, n.v_proj_bias);
            }
        }

        layers_[i].attn.set_weights(qkv.w_q, qkv.w_k, qkv.w_v,
                                    wo, bq, bk, bv);

        // Gemma 3: Q/K RMSNorm pre-RoPE.
        if (cfg_.use_qk_norm) {
            if (!n.q_norm.empty() && !n.k_norm.empty() &&
                idx.find(n.q_norm) != nullptr && idx.find(n.k_norm) != nullptr) {
                Tensor wqn = WeightNameMapper::load_weight_as_f16(idx, n.q_norm,
                                                                  device_index_);
                Tensor wkn = WeightNameMapper::load_weight_as_f16(idx, n.k_norm,
                                                                  device_index_);
                layers_[i].attn.set_qk_norm_weights(wqn, wkn);
            }
        }
    }

    embed_ = WeightNameMapper::load_weight_as_f16(idx, mapper.embed_tokens(),
                                                  device_index_);
    require_shape(embed_, {cfg_.vocab_size, H}, mapper.embed_tokens());
    if (cfg_.tie_word_embeddings) {
        lm_head_ = embed_;
    } else {
        lm_head_ = WeightNameMapper::load_weight_as_f16(idx, mapper.lm_head(),
                                                       device_index_);
        require_shape(lm_head_, {cfg_.vocab_size, H}, mapper.lm_head());
    }
    final_norm_.set_weight(
        WeightNameMapper::load_weight_as_f16(idx, mapper.model_norm(),
                                             device_index_));
}

// ---------------------------------------------------------------------------
// Forward pass — runs all `num_hidden_layers` blocks + final norm + LM head.
// ---------------------------------------------------------------------------
Tensor TransformerModel::forward(const Tensor& token_ids,
                          const std::vector<int64_t>& positions,
                          std::vector<__half*>& k_ptrs,
                          std::vector<__half*>& v_ptrs,
                          int64_t max_seq,
                          int64_t cur_len, bool is_prefill) {
    if (token_ids.dtype() != DType::INT64) {
        throw std::runtime_error("TransformerModel::forward: token_ids must be int64");
    }
    if (token_ids.ndim() != 2) {
        throw std::runtime_error("TransformerModel::forward: token_ids must be 2-D");
    }
    const int B = static_cast<int>(token_ids.shape()[0]);
    const int S = static_cast<int>(token_ids.shape()[1]);
    const int H = static_cast<int>(cfg_.hidden_size);
    if (k_ptrs.size() != layers_.size() || v_ptrs.size() != layers_.size()) {
        throw std::runtime_error("TransformerModel::forward: kv ptrs size mismatch");
    }

    // Embedding lookup on the GPU.
    Tensor hidden({B, S, H}, DType::FP16, Device::cuda(device_index_));
    kernels::launch_embedding_gather(
        static_cast<const int64_t*>(token_ids.data()),
        static_cast<const __half*>(embed_.data()),
        static_cast<__half*>(hidden.data()), B, S, H, /*stream=*/0);

    // Gemma scales the embedding output by sqrt(hidden_size). For other
    // archs this is a no-op (scale = 1).
    if (cfg_.embed_scale) {
        const float scale = std::sqrt(static_cast<float>(H));
        kernels::launch_scale_inplace(static_cast<__half*>(hidden.data()),
                                      scale,
                                      static_cast<int64_t>(B) * S * H,
                                      /*stream=*/0);
    }
    
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

        // Reshape back to [B, S, H]
        normed_buf_ = Tensor({B, S, H}, DType::FP16, Device::cuda(device_index_));
        cudaMemcpy(normed_buf_.data(), normed_2d.data(),
                   static_cast<int64_t>(B) * S * H * sizeof(__half),
                   cudaMemcpyDeviceToDevice);

        Tensor attn_out = layers_[i].attn.forward(normed_buf_, positions,
                                                  k_ptrs[i], v_ptrs[i],
                                                  max_seq, cur_len, is_prefill);

        // Gemma 2/3 double-norm wrapping: post_attention_layernorm wraps
        // the attn output before the residual add.
        if (cfg_.double_norm_block) {
            Tensor attn_out_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
            cudaMemcpy(attn_out_2d.data(), attn_out.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor attn_out_normed_2d = layers_[i].post_attn_layernorm.forward(attn_out_2d);
            Tensor attn_out_normed({B, S, H}, DType::FP16,
                                   Device::cuda(device_index_));
            cudaMemcpy(attn_out_normed.data(), attn_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            attn_out = attn_out_normed;
        }

        // residual: hidden += attn_out
        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                           static_cast<const __half*>(attn_out.data()),
                           static_cast<int64_t>(B) * S * H, /*stream=*/0);

        // ---- MLP block ------------------------------------------------
        // For LLaMA / Qwen / Gemma 1: post_attention_layernorm goes before
        //   the MLP (and the residual is applied after the MLP).
        // For Gemma 2/3: pre_feedforward_layernorm wraps the MLP input;
        //   post_feedforward_layernorm wraps the MLP output, then residual.
        if (cfg_.double_norm_block) {
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor pre_normed_2d = layers_[i].pre_feedforward_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(pre_normed_2d)
                : layers_[i].mlp.forward(pre_normed_2d);
            Tensor mlp_out_normed_2d = layers_[i].post_feedforward_layernorm.forward(mlp_out_2d);
            Tensor mlp_out_normed({B, S, H}, DType::FP16,
                                  Device::cuda(device_index_));
            cudaMemcpy(mlp_out_normed.data(), mlp_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_normed.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
        } else {
            // Standard 2-norm block: hidden -> post_attn_norm -> MLP -> residual
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor normed_2d_mlp = layers_[i].post_attn_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(normed_2d_mlp)
                : layers_[i].mlp.forward(normed_2d_mlp);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_2d.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
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
Tensor TransformerModel::forward_paged_batched(const Tensor& token_ids,
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

    if (cfg_.embed_scale) {
        const float scale = std::sqrt(static_cast<float>(H));
        kernels::launch_scale_inplace(static_cast<__half*>(hidden.data()),
                                      scale,
                                      static_cast<int64_t>(B) * S * H,
                                      /*stream=*/0);
    }

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

        if (cfg_.double_norm_block) {
            Tensor attn_out_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
            cudaMemcpy(attn_out_2d.data(), attn_out.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor attn_out_normed_2d = layers_[i].post_attn_layernorm.forward(attn_out_2d);
            Tensor attn_out_normed({B, S, H}, DType::FP16,
                                   Device::cuda(device_index_));
            cudaMemcpy(attn_out_normed.data(), attn_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            attn_out = attn_out_normed;
        }

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(attn_out.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);

        if (cfg_.double_norm_block) {
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor pre_normed_2d = layers_[i].pre_feedforward_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(pre_normed_2d)
                : layers_[i].mlp.forward(pre_normed_2d);
            Tensor mlp_out_normed_2d = layers_[i].post_feedforward_layernorm.forward(mlp_out_2d);
            Tensor mlp_out_normed({B, S, H}, DType::FP16,
                                  Device::cuda(device_index_));
            cudaMemcpy(mlp_out_normed.data(), mlp_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_normed.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
        } else {
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            normed_2d = layers_[i].post_attn_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(normed_2d)
                : layers_[i].mlp.forward(normed_2d);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_2d.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
        }
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
Tensor TransformerModel::forward_paged(const Tensor& token_ids,
                                const std::vector<int64_t>& positions,
                                PagedKVCache& paged_kv,
                                int seq_id,
                                bool is_prefill) {
    if (token_ids.dtype() != DType::INT64) {
        throw std::runtime_error("TransformerModel::forward_paged: token_ids must be int64");
    }
    if (token_ids.ndim() != 2) {
        throw std::runtime_error("TransformerModel::forward_paged: token_ids must be 2-D");
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

    if (cfg_.embed_scale) {
        const float scale = std::sqrt(static_cast<float>(H));
        kernels::launch_scale_inplace(static_cast<__half*>(hidden.data()),
                                      scale,
                                      static_cast<int64_t>(B) * S * H,
                                      /*stream=*/0);
    }

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

        if (cfg_.double_norm_block) {
            Tensor attn_out_2d({B * S, H}, DType::FP16, Device::cuda(device_index_));
            cudaMemcpy(attn_out_2d.data(), attn_out.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor attn_out_normed_2d = layers_[i].post_attn_layernorm.forward(attn_out_2d);
            Tensor attn_out_normed({B, S, H}, DType::FP16,
                                   Device::cuda(device_index_));
            cudaMemcpy(attn_out_normed.data(), attn_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            attn_out = attn_out_normed;
        }

        kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
            static_cast<const __half*>(attn_out.data()),
            static_cast<int64_t>(B) * S * H, /*stream=*/0);

        if (cfg_.double_norm_block) {
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            Tensor pre_normed_2d = layers_[i].pre_feedforward_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(pre_normed_2d)
                : layers_[i].mlp.forward(pre_normed_2d);
            Tensor mlp_out_normed_2d = layers_[i].post_feedforward_layernorm.forward(mlp_out_2d);
            Tensor mlp_out_normed({B, S, H}, DType::FP16,
                                  Device::cuda(device_index_));
            cudaMemcpy(mlp_out_normed.data(), mlp_out_normed_2d.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_normed.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
        } else {
            // ---- post-attention norm + MLP -------------------------------
            cudaMemcpy(hidden_2d.data(), hidden.data(),
                       static_cast<int64_t>(B) * S * H * sizeof(__half),
                       cudaMemcpyDeviceToDevice);
            normed_2d = layers_[i].post_attn_layernorm.forward(hidden_2d);
            Tensor mlp_out_2d = layers_[i].use_moe
                ? layers_[i].moe.forward(normed_2d)
                : layers_[i].mlp.forward(normed_2d);
            kernels::launch_add_inplace(static_cast<__half*>(hidden.data()),
                static_cast<const __half*>(mlp_out_2d.data()),
                static_cast<int64_t>(B) * S * H, /*stream=*/0);
        }
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