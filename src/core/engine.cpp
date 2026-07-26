#include "core/engine.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "kernels/sampling_kernel.cu"
#include "kernels/sampling_kernel.cuh"

namespace mini_infer {

namespace {
void cuda_check_(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 file + ":" + std::to_string(line) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA(expr) cuda_check_((expr), #expr, __FILE__, __LINE__)
}  // namespace

Engine::Engine(std::shared_ptr<QwenModel> model, int64_t max_seq_len,
               int device_index,
               int paged_num_blocks_override)
    : model_(model), device_index_(device_index), max_seq_len_(max_seq_len) {
    if (!model_) throw std::runtime_error("Engine: model is null");
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

    // Week 5: also construct a PagedKVCache. Size the pool so that the
    // worst case (max_seq_len tokens) fits with some headroom; one block
    // holds BLOCK_SIZE tokens. paged_num_blocks_override (Week 6) lets the
    // benchmark binaries grow the pool to host a batched workload.
    const int default_num_blocks = (static_cast<int>(max_seq_len)
                                    + BlockAllocator::kBlockSize - 1)
                                   / BlockAllocator::kBlockSize + 8;
    const int num_blocks = (paged_num_blocks_override > 0)
        ? paged_num_blocks_override : default_num_blocks;
    paged_kv_ = std::make_unique<PagedKVCache>(
        num_blocks,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        /*max_blocks_per_seq=*/num_blocks,
        device_index_);
    paged_num_blocks_ = num_blocks;
}

Engine::~Engine() = default;

void Engine::clear_paged_sequences() {
    if (!paged_kv_) return;
    paged_kv_->clear_all_sequences();
    paged_seq_id_ = -1;
}

void Engine::set_sampling(SamplingMode mode, float top_p, float temperature,
                          unsigned long long seed) {
    mode_ = mode;
    top_p_ = top_p;
    temperature_ = temperature;
    seed_ = seed;
}

Tensor Engine::sample_logits_(const Tensor& logits, int vocab_offset) {
    // logits: [B, S, V] — sample only the last position of the row
    // starting at vocab_offset (default 0; for batched we sample one row
    // at a time by advancing the offset).
    const int B = static_cast<int>(logits.shape()[0]);
    const int S = static_cast<int>(logits.shape()[1]);
    const int V = static_cast<int>(logits.shape()[2]);
    if (B != 1 || S != 1) {
        throw std::runtime_error("Engine::sample_logits_: only B=1 S=1 supported");
    }

    // Debug: print logits stats and top-5 tokens
    std::vector<__half> logits_cpu(V);
    MI_CHECK_CUDA(cudaMemcpy(logits_cpu.data(),
                             static_cast<const __half*>(logits.data()) + vocab_offset,
                             V * sizeof(__half), cudaMemcpyDeviceToHost));

    // Find top-5
    std::vector<std::pair<float, int>> top_k;
    for (int i = 0; i < V; ++i) {
        float v = __half2float(logits_cpu[i]);
        top_k.push_back({v, i});
    }
    std::partial_sort(top_k.begin(), top_k.begin() + 5, top_k.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::printf("[sample] Top-5 logits: ");
    for (int i = 0; i < 5; ++i) {
        std::printf("[%d=%.2f] ", top_k[i].second, top_k[i].first);
    }
    std::printf("\n");

    const __half* logits_ptr =
        static_cast<const __half*>(logits.data()) + vocab_offset;
    int* out_buf = nullptr;
    MI_CHECK_CUDA(cudaMalloc(&out_buf, sizeof(int)));

    if (mode_ == SamplingMode::Greedy) {
        kernels::launch_greedy_sample(logits_ptr, V, out_buf, /*stream=*/0);
    } else {
        kernels::launch_top_p_sample(logits_ptr, V, top_p_, temperature_,
                                     seed_++, out_buf, /*stream=*/0);
    }
    int token = -1;
    MI_CHECK_CUDA(cudaMemcpy(&token, out_buf, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(out_buf);

    std::printf("[sample] Selected token: %d (logit=%.2f)\n", token, top_k[0].first);

    Tensor one({1}, DType::INT64, Device::cpu());
    static_cast<int64_t*>(one.data())[0] = static_cast<int64_t>(token);
    return one;
}

std::vector<int64_t> Engine::generate(const std::vector<int64_t>& prompt_ids,
                                      int64_t max_new_tokens,
                                      const std::vector<int64_t>& stop_token_ids) {
    if (static_cast<int64_t>(prompt_ids.size()) > max_seq_len_) {
        throw std::runtime_error("Engine::generate: prompt longer than max_seq_len");
    }
    const int64_t prompt_len = prompt_ids.size();
    last_prompt_len_ = prompt_len;

    // Upload prompt token_ids to GPU.
    token_ids_gpu_ = Tensor::empty({1, static_cast<int>(prompt_len)},
                                    DType::INT64, Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), prompt_ids.data(),
                             prompt_len * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

    // Generate position IDs: [0, 1, 2, ..., prompt_len-1]
    std::vector<int64_t> positions(prompt_len);
    for (int64_t i = 0; i < prompt_len; ++i) {
        positions[i] = i;
    }

    // ---- Prefill --------------------------------------------------------
    Tensor logits = model_->forward(token_ids_gpu_, positions,
                                    k_ptrs_, v_ptrs_,
                                    max_seq_len_,
                                    /*cur_len=*/0, /*is_prefill=*/true);

    // Sample first new token (from the last prompt position's logits).
    int64_t next_token;
    {
        Tensor last_logits = Tensor(
            std::vector<int64_t>{1, 1, logits.shape()[2]}, DType::FP16,
            Device::cuda(device_index_));
        MI_CHECK_CUDA(cudaMemcpy(last_logits.data(),
            static_cast<const __half*>(logits.data()) +
                (prompt_len - 1) * logits.shape()[2],
            logits.shape()[2] * sizeof(__half), cudaMemcpyDeviceToDevice));
        Tensor sampled = sample_logits_(last_logits);
        next_token = static_cast<const int64_t*>(sampled.data())[0];
    }

    std::vector<int64_t> out = prompt_ids;
    out.push_back(next_token);
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), next_token)
        != stop_token_ids.end()) {
        last_generated_ = 1;
        decode_tps_ = 0.0;
        return out;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t generated = 0;

    // ---- Decode loop ----------------------------------------------------
    for (int64_t step = 0; step < max_new_tokens - 1; ++step) {
        // Check if we would exceed max_seq_len
        if (prompt_len + step + 1 > max_seq_len_) {
            std::printf("[mini-infer] Warning: reached max_seq_len (%ld), stopping generation\n",
                        max_seq_len_);
            break;
        }
        
        // Upload just the new token.
        token_ids_gpu_ = Tensor::empty({1, 1}, DType::INT64,
                                        Device::cuda(device_index_));
        MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), &next_token,
                                 sizeof(int64_t), cudaMemcpyHostToDevice));

        const std::vector<int64_t> pos = {prompt_len + step};
        Tensor step_logits = model_->forward(token_ids_gpu_, pos,
                                              k_ptrs_, v_ptrs_,
                                              max_seq_len_,
                                              /*cur_len=*/prompt_len + step,
                                              /*is_prefill=*/false);
        Tensor sampled = sample_logits_(step_logits);
        next_token = static_cast<const int64_t*>(sampled.data())[0];

        out.push_back(next_token);
        ++generated;

        if (std::find(stop_token_ids.begin(), stop_token_ids.end(), next_token)
            != stop_token_ids.end()) {
            break;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    last_generated_ = generated + 1;   // include the first sampled token
    decode_tps_ = generated / secs;
    return out;
}

// ---------------------------------------------------------------------------
// Paged generation (Week 5).
//
// Same shape as `generate`, but K/V live in PagedKVCache and the kernel
// path goes through `QwenModel::forward_paged`.
// ---------------------------------------------------------------------------
std::vector<int64_t> Engine::generate_paged(
    const std::vector<int64_t>& prompt_ids,
    int64_t max_new_tokens,
    const std::vector<int64_t>& stop_token_ids) {
    if (static_cast<int64_t>(prompt_ids.size()) > max_seq_len_) {
        throw std::runtime_error("Engine::generate_paged: prompt too long");
    }
    const int64_t prompt_len = prompt_ids.size();
    last_prompt_len_ = prompt_len;

    // Create the sequence in the paged cache and grow its block table for
    // every prompt token.
    paged_kv_->create_sequence(paged_seq_id_);
    for (int64_t i = 0; i < prompt_len; ++i) {
        int pos = paged_kv_->append_token(paged_seq_id_);
        if (pos < 0) {
            throw std::runtime_error("Engine::generate_paged: OOM growing paged KV");
        }
    }

    // Upload prompt token_ids to GPU.
    token_ids_gpu_ = Tensor::empty({1, static_cast<int>(prompt_len)},
                                    DType::INT64, Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), prompt_ids.data(),
                             prompt_len * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

    std::vector<int64_t> positions(prompt_len);
    for (int64_t i = 0; i < prompt_len; ++i) positions[i] = i;

    // ---- Prefill ------------------------------------------------------
    Tensor logits = model_->forward_paged(token_ids_gpu_, positions,
                                          *paged_kv_, paged_seq_id_,
                                          /*is_prefill=*/true);
    int64_t next_token;
    {
        Tensor last_logits = Tensor(
            std::vector<int64_t>{1, 1, logits.shape()[2]}, DType::FP16,
            Device::cuda(device_index_));
        MI_CHECK_CUDA(cudaMemcpy(last_logits.data(),
            static_cast<const __half*>(logits.data()) +
                (prompt_len - 1) * logits.shape()[2],
            logits.shape()[2] * sizeof(__half), cudaMemcpyDeviceToDevice));
        Tensor sampled = sample_logits_(last_logits);
        next_token = static_cast<const int64_t*>(sampled.data())[0];
    }

    std::vector<int64_t> out = prompt_ids;
    out.push_back(next_token);
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), next_token)
        != stop_token_ids.end()) {
        last_generated_ = 1;
        decode_tps_ = 0.0;
        return out;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t generated = 0;
    for (int64_t step = 0; step < max_new_tokens - 1; ++step) {
        if (prompt_len + step + 1 > max_seq_len_) {
            std::printf("[mini-infer paged] Warning: reached max_seq_len (%ld)\n",
                        max_seq_len_);
            break;
        }

        // Grow the block table for the new token.
        int pos = paged_kv_->append_token(paged_seq_id_);
        if (pos < 0) {
            std::fprintf(stderr, "[mini-infer paged] OOM growing block table\n");
            break;
        }

        token_ids_gpu_ = Tensor::empty({1, 1}, DType::INT64,
                                        Device::cuda(device_index_));
        MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), &next_token,
                                 sizeof(int64_t), cudaMemcpyHostToDevice));

        const std::vector<int64_t> posv = {prompt_len + step};
        Tensor step_logits = model_->forward_paged(token_ids_gpu_, posv,
                                                   *paged_kv_, paged_seq_id_,
                                                   /*is_prefill=*/false);
        Tensor sampled = sample_logits_(step_logits);
        next_token = static_cast<const int64_t*>(sampled.data())[0];

        out.push_back(next_token);
        ++generated;

        if (std::find(stop_token_ids.begin(), stop_token_ids.end(), next_token)
            != stop_token_ids.end()) {
            break;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    last_generated_ = generated + 1;
    decode_tps_ = generated / secs;
    return out;
}

// ---------------------------------------------------------------------------
// Batched paged generation (Week 5+ throughput path).
//
//   * Prefill is batched via forward_paged_batched (Week 6). The batched
//     path uses the same per-sequence bucket-padding trick as the
//     continuous scheduler: prompts in the batch are padded to the
//     maximum prompt length, and the paged-attention kernel masks
//     positions past each sequence's actual length via seq_len[b].
//   * Decode is fully batched: at every step we run one model forward
//     that serves all active sequences; each sequence samples its own
//     token from the corresponding row of logits.
//   * Sequences that emit a stop token are dropped from the batch on
//     the next step.
//   * Aggregate decode throughput scales linearly with B until GPU
//     compute saturates.
// ---------------------------------------------------------------------------
std::vector<std::vector<int64_t>> Engine::generate_batched_paged(
    const std::vector<std::vector<int64_t>>& prompts,
    int64_t max_new_tokens,
    const std::vector<int64_t>& stop_token_ids) {
    const int N = static_cast<int>(prompts.size());
    std::vector<std::vector<int64_t>> outs(N);
    if (N == 0) return outs;

    // 1) Create sequences and run batched prefill (Week 6: was sequential
    //    in Week 5; now matches the continuous scheduler's prefill path).
    std::vector<int> sids(N);
    static int sid_counter = 0;
    const int sid_base = 100000 + (sid_counter++) * 1000;

    int max_len = 0;
    for (const auto& p : prompts) {
        if (static_cast<int>(p.size()) > max_len) max_len = static_cast<int>(p.size());
    }
    std::vector<int64_t> token_ids_h(static_cast<int64_t>(N) * max_len, 0);
    std::vector<int64_t> positions_h(static_cast<int64_t>(N) * max_len, 0);
    std::vector<int>    start_pos_h(N, 0);
    for (int i = 0; i < N; ++i) {
        sids[i] = sid_base + i;
        paged_kv_->create_sequence(sids[i]);
        const int L = static_cast<int>(prompts[i].size());
        for (int t = 0; t < L; ++t) {
            if (paged_kv_->append_token(sids[i]) < 0) {
                throw std::runtime_error("batched: OOM during prefill");
            }
            token_ids_h[static_cast<int64_t>(i) * max_len + t] = prompts[i][t];
            positions_h[static_cast<int64_t>(i) * max_len + t] = t;
        }
        outs[i] = prompts[i];
    }
    Tensor ids_dev = Tensor::empty({N, max_len}, DType::INT64,
                                    Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemcpy(ids_dev.data(), token_ids_h.data(),
                             static_cast<int64_t>(N) * max_len * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
    Tensor prefill_logits = model_->forward_paged_batched(
        ids_dev, positions_h, *paged_kv_,
        sids, start_pos_h, /*is_prefill=*/true);
    // Sample first token for each sequence from its last real position.
    int* sample_buf = nullptr;
    MI_CHECK_CUDA(cudaMalloc(&sample_buf, N * sizeof(int)));
    {
        __half* gathered = nullptr;
        const int V = static_cast<int>(prefill_logits.shape()[2]);
        MI_CHECK_CUDA(cudaMalloc(&gathered, N * V * sizeof(__half)));
        for (int i = 0; i < N; ++i) {
            const int L = static_cast<int>(prompts[i].size());
            const int64_t src_off = static_cast<int64_t>(i) * max_len * V + (L - 1) * V;
            MI_CHECK_CUDA(cudaMemcpyAsync(
                gathered + i * V,
                static_cast<const __half*>(prefill_logits.data()) + src_off,
                V * sizeof(__half), cudaMemcpyDeviceToDevice, /*stream=*/0));
        }
        MI_CHECK_CUDA(cudaDeviceSynchronize());
        kernels::launch_greedy_sample_batched(gathered, N, V, sample_buf,
                                              /*stream=*/0);
        MI_CHECK_CUDA(cudaFree(gathered));
    }
    std::vector<int> sample_in(N);
    MI_CHECK_CUDA(cudaMemcpy(sample_in.data(), sample_buf, N * sizeof(int),
                             cudaMemcpyDeviceToHost));
    MI_CHECK_CUDA(cudaFree(sample_buf));
    for (int i = 0; i < N; ++i) {
        outs[i].push_back(static_cast<int64_t>(sample_in[i]));
    }

    // 2) Decode loop. Track which sequences are still active.
    std::vector<bool> active(N, true);
    int64_t total_decoded = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int64_t step = 0; step < max_new_tokens - 1; ++step) {
        // Build batched input for the active sequences.
        std::vector<int> active_ids;
        for (int i = 0; i < N; ++i) if (active[i]) active_ids.push_back(i);
        if (active_ids.empty()) break;

        const int B = static_cast<int>(active_ids.size());
        std::vector<int64_t> tokens_h(B);
        std::vector<int64_t> positions_h(B);
        std::vector<int>    sid_h(B);
        std::vector<int>    start_pos_h(B);
        for (int k = 0; k < B; ++k) {
            const int i = active_ids[k];
            tokens_h[k] = outs[i].back();
            const int64_t pos = static_cast<int64_t>(outs[i].size()) - 1;
            positions_h[k] = pos;
            sid_h[k] = sids[i];
            // Append the new token's slot to its block table BEFORE the
            // forward call (kernel reads num_blocks_used to bound the
            // loop over blocks).
            int ap = paged_kv_->append_token(sids[i]);
            if (ap < 0) throw std::runtime_error("batched: OOM during decode");
            start_pos_h[k] = ap;          // == seq_len - 1 after append
        }


        // Upload the batched token tensor [B, 1] int64.
        token_ids_gpu_ = Tensor::empty({B, 1}, DType::INT64,
                                        Device::cuda(device_index_));
        MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), tokens_h.data(),
                                  B * sizeof(int64_t), cudaMemcpyHostToDevice));

        Tensor logits = model_->forward_paged_batched(
            token_ids_gpu_, positions_h, *paged_kv_,
            sid_h, start_pos_h, /*is_prefill=*/false);

        // Sample one row per sequence.
        const int V = static_cast<int>(logits.shape()[2]);
        for (int k = 0; k < B; ++k) {
            const int i = active_ids[k];
            // Extract the k-th row from logits [B, 1, V] -> [1, 1, V]
            Tensor row_logits = Tensor(
                std::vector<int64_t>{1, 1, V}, DType::FP16,
                Device::cuda(device_index_));
            MI_CHECK_CUDA(cudaMemcpy(row_logits.data(),
                static_cast<const __half*>(logits.data()) + k * V,
                V * sizeof(__half), cudaMemcpyDeviceToDevice));
            Tensor sampled = sample_logits_(row_logits);
            int64_t next_token = static_cast<const int64_t*>(sampled.data())[0];
            outs[i].push_back(next_token);
            ++total_decoded;
            if (std::find(stop_token_ids.begin(), stop_token_ids.end(),
                          next_token) != stop_token_ids.end()) {
                active[i] = false;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    decode_tps_ = (secs > 0.0) ? total_decoded / secs : 0.0;
    last_generated_ = total_decoded / N;
    return outs;
}

}  // namespace mini_infer