#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/tensor.h"
#include "model/qwen_model.h"
#include "scheduler/kv_cache.h"
#include "speculative/draft_engine.h"

namespace mini_infer {

struct SpecDecodeStats {
    int64_t total_accepted  = 0;
    int64_t total_rejected  = 0;
    int64_t total_draft     = 0;
    int64_t total_generated = 0;
    double  accept_rate() const {
        return total_draft > 0
            ? static_cast<double>(total_accepted) / total_draft : 0.0;
    }
};

class SpecDecoder {
public:
    SpecDecoder(std::shared_ptr<QwenModel> target_model,
                std::shared_ptr<DraftEngine> draft_engine,
                int64_t max_seq_len,
                int gamma = 4,
                int device_index = 0);
    ~SpecDecoder();

    void set_sampling_greedy();
    void set_sampling_top_p(float top_p, float temperature,
                            unsigned long long seed);

    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_ids,
                                  int64_t max_new_tokens,
                                  const std::vector<int64_t>& stop_token_ids = {});

    const SpecDecodeStats& stats() const { return stats_; }

private:
    Tensor target_prefill_(const std::vector<int64_t>& prompt_ids);

    Tensor target_forward_tokens_(const std::vector<int64_t>& tokens,
                                  const std::vector<int64_t>& positions);

    Tensor target_forward_one_(int64_t token, int64_t position);

    int verify_and_accept_(
        const std::vector<DraftToken>& draft_tokens,
        const Tensor& target_last_logits,
        const Tensor& target_verify_logits,
        int vocab_size,
        std::vector<int64_t>& accepted_tokens);

    int64_t sample_from_logits_(const Tensor& logits);

    std::shared_ptr<QwenModel> target_model_;
    std::shared_ptr<DraftEngine> draft_engine_;
    int64_t max_seq_len_;
    int gamma_;
    int device_index_;

    std::unique_ptr<KVCache> target_kv_cache_;
    std::vector<__half*> target_k_ptrs_;
    std::vector<__half*> target_v_ptrs_;
    int64_t target_cur_len_ = 0;

    bool greedy_ = true;
    float top_p_ = 0.9f;
    float temperature_ = 1.0f;
    unsigned long long seed_ = 42;

    Tensor token_ids_gpu_;
    SpecDecodeStats stats_;
};

}  // namespace mini_infer
