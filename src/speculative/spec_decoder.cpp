#include "speculative/spec_decoder.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "kernels/sampling_kernel.cuh"

namespace mini_infer {

namespace {
void cuda_check_s(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 file + ":" + std::to_string(line) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA_S(expr) cuda_check_s((expr), #expr, __FILE__, __LINE__)

std::vector<float> softmax_from_logits(const __half* logits_h, int V) {
    float max_val = -INFINITY;
    for (int i = 0; i < V; ++i) {
        float v = __half2float(logits_h[i]);
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    std::vector<float> probs(V);
    for (int i = 0; i < V; ++i) {
        probs[i] = std::exp(__half2float(logits_h[i]) - max_val);
        sum_exp += probs[i];
    }
    for (int i = 0; i < V; ++i) probs[i] /= sum_exp;
    return probs;
}

int argmax(const std::vector<float>& v) {
    int idx = 0;
    float mx = v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > mx) { mx = v[i]; idx = i; }
    }
    return idx;
}

int sample_from_probs(const std::vector<float>& probs,
                      std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cum += probs[i];
        if (u <= cum) return i;
    }
    return static_cast<int>(probs.size()) - 1;
}

int sample_corrected(const std::vector<float>& p_target,
                     const std::vector<float>& p_draft,
                     std::mt19937& rng) {
    const int V = static_cast<int>(p_target.size());
    std::vector<float> corrected(V);
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) {
        corrected[i] = std::max(0.0f, p_target[i] - p_draft[i]);
        sum += corrected[i];
    }
    if (sum <= 0.0f) {
        return argmax(p_target);
    }
    for (int i = 0; i < V; ++i) corrected[i] /= sum;
    return sample_from_probs(corrected, rng);
}

Tensor extract_last_row(const Tensor& logits, int device_index) {
    const int S = static_cast<int>(logits.shape()[1]);
    const int V = static_cast<int>(logits.shape()[2]);
    Tensor last({1, 1, V}, DType::FP16, Device::cuda(device_index));
    MI_CHECK_CUDA_S(cudaMemcpy(last.data(),
        static_cast<const __half*>(logits.data()) + (S - 1) * V,
        V * sizeof(__half), cudaMemcpyDeviceToDevice));
    return last;
}

Tensor extract_row(const Tensor& logits, int row, int device_index) {
    const int V = static_cast<int>(logits.shape()[2]);
    Tensor out({1, 1, V}, DType::FP16, Device::cuda(device_index));
    MI_CHECK_CUDA_S(cudaMemcpy(out.data(),
        static_cast<const __half*>(logits.data()) + row * V,
        V * sizeof(__half), cudaMemcpyDeviceToDevice));
    return out;
}
}  // namespace

SpecDecoder::SpecDecoder(std::shared_ptr<TransformerModel> target_model,
                         std::shared_ptr<DraftEngine> draft_engine,
                         int64_t max_seq_len, int gamma, int device_index)
    : target_model_(target_model),
      draft_engine_(draft_engine),
      max_seq_len_(max_seq_len),
      gamma_(gamma),
      device_index_(device_index) {
    if (!target_model_) throw std::runtime_error("SpecDecoder: target model is null");
    const auto& cfg = target_model_->config();
    target_kv_cache_ = std::make_unique<KVCache>(
        cfg.num_hidden_layers, cfg.num_key_value_heads,
        cfg.head_dim(), max_seq_len_, device_index_);
    target_k_ptrs_.reserve(cfg.num_hidden_layers);
    target_v_ptrs_.reserve(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
        target_k_ptrs_.push_back(target_kv_cache_->k_layer_ptr(i));
        target_v_ptrs_.push_back(target_kv_cache_->v_layer_ptr(i));
    }
}

SpecDecoder::~SpecDecoder() = default;

void SpecDecoder::set_sampling_greedy() {
    greedy_ = true;
    draft_engine_->set_sampling_greedy();
}

void SpecDecoder::set_sampling_top_p(float top_p, float temperature,
                                      unsigned long long seed) {
    greedy_ = false;
    top_p_ = top_p;
    temperature_ = temperature;
    seed_ = seed;
    draft_engine_->set_sampling_top_p(top_p, temperature, seed);
}

Tensor SpecDecoder::target_prefill_(const std::vector<int64_t>& prompt_ids) {
    const int64_t prompt_len = prompt_ids.size();
    token_ids_gpu_ = Tensor::empty({1, static_cast<int>(prompt_len)},
                                    DType::INT64, Device::cuda(device_index_));
    MI_CHECK_CUDA_S(cudaMemcpy(token_ids_gpu_.data(), prompt_ids.data(),
                                prompt_len * sizeof(int64_t),
                                cudaMemcpyHostToDevice));

    std::vector<int64_t> positions(prompt_len);
    for (int64_t i = 0; i < prompt_len; ++i) positions[i] = i;

    Tensor logits = target_model_->forward(token_ids_gpu_, positions,
                                            target_k_ptrs_, target_v_ptrs_,
                                            max_seq_len_,
                                            /*cur_len=*/0,
                                            /*is_prefill=*/true);
    target_cur_len_ = prompt_len;
    return extract_last_row(logits, device_index_);
}

Tensor SpecDecoder::target_forward_tokens_(
    const std::vector<int64_t>& tokens,
    const std::vector<int64_t>& positions) {
    const int S = static_cast<int>(tokens.size());
    token_ids_gpu_ = Tensor::empty({1, S}, DType::INT64,
                                    Device::cuda(device_index_));
    MI_CHECK_CUDA_S(cudaMemcpy(token_ids_gpu_.data(), tokens.data(),
                                S * sizeof(int64_t), cudaMemcpyHostToDevice));

    Tensor logits = target_model_->forward(token_ids_gpu_, positions,
                                            target_k_ptrs_, target_v_ptrs_,
                                            max_seq_len_,
                                            /*cur_len=*/target_cur_len_,
                                            /*is_prefill=*/true);
    target_cur_len_ = positions.back() + 1;
    return logits;
}

Tensor SpecDecoder::target_forward_one_(int64_t token, int64_t position) {
    token_ids_gpu_ = Tensor::empty({1, 1}, DType::INT64,
                                    Device::cuda(device_index_));
    MI_CHECK_CUDA_S(cudaMemcpy(token_ids_gpu_.data(), &token,
                                sizeof(int64_t), cudaMemcpyHostToDevice));

    const std::vector<int64_t> pos = {position};
    Tensor logits = target_model_->forward(token_ids_gpu_, pos,
                                            target_k_ptrs_, target_v_ptrs_,
                                            max_seq_len_,
                                            /*cur_len=*/target_cur_len_,
                                            /*is_prefill=*/false);
    target_cur_len_ = position + 1;
    return logits;
}

int64_t SpecDecoder::sample_from_logits_(const Tensor& logits) {
    const int V = static_cast<int>(logits.shape()[2]);
    std::vector<__half> logits_cpu(V);
    MI_CHECK_CUDA_S(cudaMemcpy(logits_cpu.data(),
                                static_cast<const __half*>(logits.data()),
                                V * sizeof(__half), cudaMemcpyDeviceToHost));

    auto probs = softmax_from_logits(logits_cpu.data(), V);
    if (greedy_) {
        return argmax(probs);
    }
    std::mt19937 rng(static_cast<unsigned>(seed_++));
    return sample_from_probs(probs, rng);
}

int SpecDecoder::verify_and_accept_(
    const std::vector<DraftToken>& draft_tokens,
    const Tensor& target_last_logits,
    const Tensor& target_verify_logits,
    int vocab_size,
    std::vector<int64_t>& accepted_tokens) {

    const int gamma = static_cast<int>(draft_tokens.size());
    const int V = vocab_size;
    std::mt19937 rng(static_cast<unsigned>(seed_++));

    std::vector<__half> last_logits_cpu(V);
    MI_CHECK_CUDA_S(cudaMemcpy(last_logits_cpu.data(),
        static_cast<const __half*>(target_last_logits.data()),
        V * sizeof(__half), cudaMemcpyDeviceToHost));
    auto p_target_first = softmax_from_logits(last_logits_cpu.data(), V);

    std::vector<std::vector<__half>> verify_logits_cpu(gamma);
    for (int i = 0; i < gamma; ++i) {
        verify_logits_cpu[i].resize(V);
        MI_CHECK_CUDA_S(cudaMemcpy(verify_logits_cpu[i].data(),
            static_cast<const __half*>(target_verify_logits.data()) + i * V,
            V * sizeof(__half), cudaMemcpyDeviceToHost));
    }

    int num_accepted = 0;
    int64_t replacement_token = -1;

    for (int i = 0; i < gamma; ++i) {
        int64_t d_i = draft_tokens[i].token_id;
        float q_i = draft_tokens[i].draft_prob;

        std::vector<float> p_target;
        if (i == 0) {
            p_target = p_target_first;
        } else {
            p_target = softmax_from_logits(verify_logits_cpu[i - 1].data(), V);
        }

        float p_i = (d_i >= 0 && d_i < V) ? p_target[d_i] : 0.0f;

        if (greedy_) {
            int target_argmax = argmax(p_target);
            if (static_cast<int64_t>(target_argmax) == d_i) {
                accepted_tokens.push_back(d_i);
                ++num_accepted;
                stats_.total_accepted++;
                stats_.total_draft++;
            } else {
                replacement_token = target_argmax;
                stats_.total_rejected++;
                stats_.total_draft++;
                break;
            }
        } else {
            float accept_prob = (q_i > 1e-8f)
                ? std::min(1.0f, p_i / q_i) : 1.0f;

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float u = dist(rng);

            if (u < accept_prob) {
                accepted_tokens.push_back(d_i);
                ++num_accepted;
                stats_.total_accepted++;
                stats_.total_draft++;
            } else {
                std::vector<float> p_draft_vec(V, 0.0f);
                if (d_i >= 0 && d_i < V) p_draft_vec[d_i] = q_i;
                replacement_token = sample_corrected(p_target, p_draft_vec, rng);
                stats_.total_rejected++;
                stats_.total_draft++;
                break;
            }
        }
    }

    if (num_accepted < gamma) {
        accepted_tokens.push_back(replacement_token);
    }

    return num_accepted;
}

std::vector<int64_t> SpecDecoder::generate(
    const std::vector<int64_t>& prompt_ids,
    int64_t max_new_tokens,
    const std::vector<int64_t>& stop_token_ids) {

    const int64_t prompt_len = prompt_ids.size();
    if (prompt_len > max_seq_len_) {
        throw std::runtime_error("SpecDecoder::generate: prompt too long");
    }

    Tensor draft_prefill_logits = draft_engine_->prefill(prompt_ids);
    Tensor target_prefill_logits = target_prefill_(prompt_ids);

    int64_t cur_len = prompt_len;
    std::vector<int64_t> output = prompt_ids;
    stats_ = SpecDecodeStats{};

    auto is_stop = [&](int64_t tok) {
        return std::find(stop_token_ids.begin(), stop_token_ids.end(), tok)
               != stop_token_ids.end();
    };

    Tensor target_next_logits = target_prefill_logits;
    Tensor draft_next_logits = draft_prefill_logits;

    int64_t total_generated = 0;

    while (total_generated < max_new_tokens && cur_len < max_seq_len_) {
        int gamma = std::min(gamma_,
            static_cast<int>(max_new_tokens - total_generated));
        gamma = std::min(gamma, static_cast<int>(max_seq_len_ - cur_len - 1));
        if (gamma <= 0) break;

        std::vector<DraftToken> draft_tokens =
            draft_engine_->generate_draft(gamma, draft_next_logits);

        const int actual_gamma = static_cast<int>(draft_tokens.size());
        if (actual_gamma == 0) break;

        std::vector<int64_t> draft_token_ids(actual_gamma);
        std::vector<int64_t> draft_positions(actual_gamma);
        for (int i = 0; i < actual_gamma; ++i) {
            draft_token_ids[i] = draft_tokens[i].token_id;
            draft_positions[i] = cur_len + i;
        }

        Tensor target_verify_logits = target_forward_tokens_(
            draft_token_ids, draft_positions);

        const int V = static_cast<int>(target_model_->config().vocab_size);

        std::vector<int64_t> accepted_tokens;
        int num_accepted = verify_and_accept_(
            draft_tokens, target_next_logits,
            target_verify_logits, V, accepted_tokens);

        bool all_accepted = (num_accepted == actual_gamma);

        if (all_accepted) {
            for (int64_t tok : accepted_tokens) {
                output.push_back(tok);
                ++total_generated;
                ++cur_len;
                if (is_stop(tok)) {
                    stats_.total_generated = total_generated;
                    return output;
                }
            }

            int64_t bonus = sample_from_logits_(
                extract_row(target_verify_logits, actual_gamma - 1,
                            device_index_));
            output.push_back(bonus);
            ++total_generated;

            if (is_stop(bonus)) {
                ++cur_len;
                stats_.total_generated = total_generated;
                return output;
            }

            target_next_logits = target_forward_one_(bonus, cur_len);
            draft_next_logits = draft_engine_->forward_one(bonus, cur_len);
            ++cur_len;
        } else {
            for (int64_t tok : accepted_tokens) {
                output.push_back(tok);
                ++total_generated;
                ++cur_len;
                if (is_stop(tok)) {
                    stats_.total_generated = total_generated;
                    return output;
                }
            }

            int64_t correction = accepted_tokens.back();
            target_cur_len_ = cur_len - 1;
            draft_engine_->truncate(cur_len - 1);

            target_next_logits = target_forward_one_(correction, cur_len - 1);
            draft_next_logits = draft_engine_->forward_one(correction, cur_len - 1);
        }

        stats_.total_generated = total_generated;
    }

    return output;
}

}  // namespace mini_infer
