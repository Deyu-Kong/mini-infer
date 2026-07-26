#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "model/qwen_model.h"
#include "scheduler/kv_cache.h"
#include "scheduler/paged_kv_cache.h"

namespace mini_infer {

/**
 * Sampling mode for `Engine::generate`.
 */
enum class SamplingMode {
    Greedy = 0,   // argmax
    TopP   = 1,   // nucleus sampling
};

/**
 * Engine — orchestrates the autoregressive decode loop on top of QwenModel.
 *
 *   1. Prefill : run the entire prompt in one forward pass.
 *   2. Decode  : for each new token, run a 1-token forward pass and sample.
 *   3. KV cache is owned by the engine and shared across the whole generation.
 *
 * Sampling defaults to greedy. Top-p requires `set_sampling(SamplingMode::TopP,
 * p, temperature)`.
 *
 * Week 5: PagedAttention path is available via `generate_paged`. The naive
 * path (`generate`) is retained for backward-compat and as the reference
 * for correctness testing.
 */
class Engine {
public:
    Engine(std::shared_ptr<QwenModel> model, int64_t max_seq_len,
           int device_index = 0,
           int paged_num_blocks_override = 0);
    ~Engine();

    // Configure sampling.
    void set_sampling(SamplingMode mode, float top_p = 0.9f,
                      float temperature = 1.0f,
                      unsigned long long seed = 42);

    // Generate `max_new_tokens` completions given a host-side int32 prompt.
    //   - prompt_ids : int32 token IDs (host) returned by the tokenizer
    //   - max_new_tokens : how many to generate beyond the prompt
    //   - stop_token_ids : end generation if sampled (default = empty)
    //
    // Returns the full sequence (prompt + generated) on host.
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_ids,
                                  int64_t max_new_tokens,
                                  const std::vector<int64_t>& stop_token_ids = {});

    // PagedAttention variant of `generate`. Same interface, but uses the
    // shared PagedKVCache owned by the engine (allocated on construction
    // with the maximum number of paged blocks implied by max_seq_len).
    //
    // Returns the full sequence (prompt + generated) on host.
    std::vector<int64_t> generate_paged(const std::vector<int64_t>& prompt_ids,
                                        int64_t max_new_tokens,
                                        const std::vector<int64_t>& stop_token_ids = {});

    // Batched paged generation. Takes N prompts and produces N completions
    // sharing the same PagedKVCache pool. Prefill is sequential; decode
    // is fully batched (one forward call serves all active sequences).
    //
    // Each input prompt can have a different length. The returned vector
    // is `prompts.size()` long; element i is the full sequence (prompt +
    // generated) for the i-th prompt.
    //
    // Sequences that hit a stop token are removed from the batch on the
    // next step. Aggregate throughput improves ~linearly with batch size
    // for compute-bound workloads (the GEMMs amortize across the batch).
    std::vector<std::vector<int64_t>> generate_batched_paged(
        const std::vector<std::vector<int64_t>>& prompts,
        int64_t max_new_tokens,
        const std::vector<int64_t>& stop_token_ids = {});

    // Access the shared PagedKVCache (used by tests to inspect fragmentation
    // and by external code that wants to manage multiple sequences).
    const PagedKVCache& paged_kv() const { return *paged_kv_; }
    PagedKVCache&       paged_kv()       { return *paged_kv_; }

    // Diagnostics.
    int64_t prompt_len()  const { return last_prompt_len_; }
    int64_t generated()   const { return last_generated_; }
    double  decode_tokens_per_sec() const { return decode_tps_; }

    // Week 6: tear down all sequences currently in the paged KV cache.
    // Use between static-batched batches so the pool doesn't accumulate
    // sequences from prior runs. No-op for the naive path.
    void clear_paged_sequences();

private:
    Tensor sample_logits_(const Tensor& logits, int vocab_offset = 0);

    std::shared_ptr<QwenModel> model_;
    int device_index_;
    int64_t max_seq_len_;
    std::unique_ptr<KVCache> kv_cache_;
    std::vector<__half*> k_ptrs_;
    std::vector<__half*> v_ptrs_;

    // Week 5: PagedAttention storage.
    std::unique_ptr<PagedKVCache> paged_kv_;
    int                           paged_seq_id_ = 0;
    int64_t                       paged_num_blocks_ = 0;

    // Sampling
    SamplingMode mode_      = SamplingMode::Greedy;
    float        top_p_     = 0.9f;
    float        temperature_ = 1.0f;
    unsigned long long seed_  = 42;

    // Token-id buffer that lives between forward + sample (kept on GPU).
    Tensor token_ids_gpu_;

    // Stats from last generate() call.
    int64_t last_prompt_len_ = 0;
    int64_t last_generated_  = 0;
    double  decode_tps_      = 0.0;
};

}  // namespace mini_infer