#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_fp16.h>

#include "core/tensor.h"
#include "model/transformer_model.h"
#include "scheduler/kv_cache.h"

namespace mini_infer {

struct DraftToken {
    int64_t token_id;
    float   draft_prob;
};

class DraftEngine {
public:
    DraftEngine(std::shared_ptr<TransformerModel> model, int64_t max_seq_len,
                int device_index = 0);
    ~DraftEngine();

    void set_sampling_greedy();
    void set_sampling_top_p(float top_p, float temperature,
                            unsigned long long seed);

    Tensor prefill(const std::vector<int64_t>& prompt_ids);

    std::vector<DraftToken> generate_draft(int gamma,
                                           const Tensor& initial_logits);

    void truncate(int64_t new_len);

    Tensor forward_one(int64_t token, int64_t position);

    DraftToken sample_token(const Tensor& logits);

    void reset();

    int64_t current_length() const { return cur_len_; }

    const ModelConfig& config() const { return model_->config(); }

private:
    DraftToken sample_one_(const Tensor& logits);

    std::shared_ptr<TransformerModel> model_;
    int device_index_;
    int64_t max_seq_len_;
    int64_t cur_len_ = 0;

    std::unique_ptr<KVCache> kv_cache_;
    std::vector<__half*> k_ptrs_;
    std::vector<__half*> v_ptrs_;

    bool greedy_ = true;
    float top_p_ = 0.9f;
    float temperature_ = 1.0f;
    unsigned long long seed_ = 42;

    Tensor token_ids_gpu_;
};

}  // namespace mini_infer
