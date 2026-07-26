#include "speculative/draft_engine.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "kernels/sampling_kernel.cuh"

namespace mini_infer {

namespace {
void cuda_check_d(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 file + ":" + std::to_string(line) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA_D(expr) cuda_check_d((expr), #expr, __FILE__, __LINE__)
}  // namespace

DraftEngine::DraftEngine(std::shared_ptr<QwenModel> model,
                         int64_t max_seq_len, int device_index)
    : model_(model), device_index_(device_index), max_seq_len_(max_seq_len) {
    if (!model_) throw std::runtime_error("DraftEngine: model is null");
    const auto& cfg = model_->config();
    kv_cache_ = std::make_unique<KVCache>(
        cfg.num_hidden_layers, cfg.num_key_value_heads,
        cfg.head_dim(), max_seq_len_, device_index_);
    k_ptrs_.reserve(cfg.num_hidden_layers);
    v_ptrs_.reserve(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        k_ptrs_.push_back(kv_cache_->k_layer_ptr(i));
        v_ptrs_.push_back(kv_cache_->v_layer_ptr(i));
    }
}

DraftEngine::~DraftEngine() = default;

void DraftEngine::set_sampling_greedy() { greedy_ = true; }

void DraftEngine::set_sampling_top_p(float top_p, float temperature,
                                      unsigned long long seed) {
    greedy_ = false;
    top_p_ = top_p;
    temperature_ = temperature;
    seed_ = seed;
}

Tensor DraftEngine::prefill(const std::vector<int64_t>& prompt_ids) {
    const int64_t prompt_len = prompt_ids.size();
    if (prompt_len > max_seq_len_) {
        throw std::runtime_error("DraftEngine::prefill: prompt too long");
    }

    token_ids_gpu_ = Tensor::empty({1, static_cast<int>(prompt_len)},
                                    DType::INT64, Device::cuda(device_index_));
    MI_CHECK_CUDA_D(cudaMemcpy(token_ids_gpu_.data(), prompt_ids.data(),
                                prompt_len * sizeof(int64_t),
                                cudaMemcpyHostToDevice));

    std::vector<int64_t> positions(prompt_len);
    for (int64_t i = 0; i < prompt_len; ++i) positions[i] = i;

    Tensor logits = model_->forward(token_ids_gpu_, positions,
                                     k_ptrs_, v_ptrs_,
                                     max_seq_len_,
                                     /*cur_len=*/0,
                                     /*is_prefill=*/true);
    cur_len_ = prompt_len;

    const int V = static_cast<int>(logits.shape()[2]);
    Tensor last_logits({1, 1, V}, DType::FP16, Device::cuda(device_index_));
    MI_CHECK_CUDA_D(cudaMemcpy(last_logits.data(),
        static_cast<const __half*>(logits.data()) +
            (prompt_len - 1) * V,
        V * sizeof(__half), cudaMemcpyDeviceToDevice));
    return last_logits;
}

DraftToken DraftEngine::sample_one_(const Tensor& logits) {
    const int V = static_cast<int>(logits.shape()[2]);

    std::vector<__half> logits_cpu(V);
    MI_CHECK_CUDA_D(cudaMemcpy(logits_cpu.data(),
                                static_cast<const __half*>(logits.data()),
                                V * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_val = -INFINITY;
    for (int i = 0; i < V; ++i) {
        float v = __half2float(logits_cpu[i]);
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < V; ++i) {
        sum_exp += std::exp(__half2float(logits_cpu[i]) - max_val);
    }

    const __half* logits_ptr = static_cast<const __half*>(logits.data());
    int* out_buf = nullptr;
    MI_CHECK_CUDA_D(cudaMalloc(&out_buf, sizeof(int)));

    int token = -1;
    float prob = 0.0f;

    if (greedy_) {
        kernels::launch_greedy_sample(logits_ptr, V, out_buf, /*stream=*/0);
        MI_CHECK_CUDA_D(cudaMemcpy(&token, out_buf, sizeof(int),
                                    cudaMemcpyDeviceToHost));
        prob = std::exp(__half2float(logits_cpu[token]) - max_val) / sum_exp;
    } else {
        kernels::launch_top_p_sample(logits_ptr, V, top_p_, temperature_,
                                     seed_++, out_buf, /*stream=*/0);
        MI_CHECK_CUDA_D(cudaMemcpy(&token, out_buf, sizeof(int),
                                    cudaMemcpyDeviceToHost));
        prob = std::exp(__half2float(logits_cpu[token]) - max_val) / sum_exp;
    }

    cudaFree(out_buf);
    return {static_cast<int64_t>(token), prob};
}

std::vector<DraftToken> DraftEngine::generate_draft(
    int gamma, const Tensor& initial_logits) {
    std::vector<DraftToken> draft_tokens;
    draft_tokens.reserve(gamma);

    DraftToken first = sample_one_(initial_logits);
    draft_tokens.push_back(first);

    for (int step = 0; step < gamma; ++step) {
        if (cur_len_ >= max_seq_len_) break;

        int64_t prev_token = draft_tokens.back().token_id;
        token_ids_gpu_ = Tensor::empty({1, 1}, DType::INT64,
                                        Device::cuda(device_index_));
        MI_CHECK_CUDA_D(cudaMemcpy(token_ids_gpu_.data(), &prev_token,
                                    sizeof(int64_t), cudaMemcpyHostToDevice));

        const std::vector<int64_t> pos = {cur_len_};
        Tensor step_logits = model_->forward(token_ids_gpu_, pos,
                                              k_ptrs_, v_ptrs_,
                                              max_seq_len_,
                                              /*cur_len=*/cur_len_,
                                              /*is_prefill=*/false);
        ++cur_len_;

        if (step < gamma - 1) {
            DraftToken dt = sample_one_(step_logits);
            draft_tokens.push_back(dt);
        }
    }

    return draft_tokens;
}

Tensor DraftEngine::forward_one(int64_t token, int64_t position) {
    token_ids_gpu_ = Tensor::empty({1, 1}, DType::INT64,
                                    Device::cuda(device_index_));
    MI_CHECK_CUDA_D(cudaMemcpy(token_ids_gpu_.data(), &token,
                                sizeof(int64_t), cudaMemcpyHostToDevice));

    const std::vector<int64_t> pos = {position};
    Tensor logits = model_->forward(token_ids_gpu_, pos,
                                     k_ptrs_, v_ptrs_,
                                     max_seq_len_,
                                     /*cur_len=*/cur_len_,
                                     /*is_prefill=*/false);
    cur_len_ = position + 1;
    return logits;
}

void DraftEngine::truncate(int64_t new_len) {
    cur_len_ = new_len;
}

DraftToken DraftEngine::sample_token(const Tensor& logits) {
    return sample_one_(logits);
}

void DraftEngine::reset() {
    cur_len_ = 0;
}

}  // namespace mini_infer
