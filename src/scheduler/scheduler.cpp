#include "scheduler/scheduler.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "kernels/sampling_kernel.cu"
#include "kernels/sampling_kernel.cuh"

namespace mini_infer {

namespace {
inline void cuda_check_(cudaError_t e, const char* expr, const char* f, int l) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error ") + expr + " at " +
                                 f + ":" + std::to_string(l) + " : " +
                                 cudaGetErrorString(e));
    }
}
#define MI_CHECK_CUDA(expr) cuda_check_((expr), #expr, __FILE__, __LINE__)
}  // namespace

Scheduler::Scheduler(std::shared_ptr<QwenModel> model,
                     std::shared_ptr<PagedKVCache> paged_kv,
                     SchedulerConfig cfg,
                     int device_index)
    : model_(std::move(model)),
      paged_kv_(std::move(paged_kv)),
      cfg_(cfg),
      device_index_(device_index),
      t0_(std::chrono::steady_clock::now()) {
    if (!model_)    throw std::runtime_error("Scheduler: model is null");
    if (!paged_kv_) throw std::runtime_error("Scheduler: paged_kv is null");
    std::sort(cfg_.prefill_buckets.begin(), cfg_.prefill_buckets.end());
    if (cfg_.prefill_buckets.empty()) {
        throw std::runtime_error("Scheduler: prefill_buckets must be non-empty");
    }
}

double Scheduler::now_ms() const {
    auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t - t0_).count();
}

void Scheduler::submit(std::unique_ptr<Request> req) {
    if (!req) return;
    req->set_arrival_ms(now_ms());
    waiting_queue_.push_back(std::move(req));
}

int Scheduler::allocate_seq_id_() {
    return next_seq_id_++;
}

bool Scheduler::done() const {
    return waiting_queue_.empty() && running_.empty();
}

std::vector<std::unique_ptr<Request>> Scheduler::drain_finished() {
    std::vector<std::unique_ptr<Request>> out;
    out.swap(finished_);
    return out;
}

// ---------------------------------------------------------------------------
// sweep_finished_ — check each running request for terminal conditions.
// ---------------------------------------------------------------------------
int Scheduler::sweep_finished_(IterationStats& stats) {
    int n_finished = 0;
    for (auto it = running_.begin(); it != running_.end(); ) {
        Request& r = **it;
        bool stop = false;
        std::string reason;
        if (r.metrics().generated_tokens >= r.max_new_tokens()) {
            stop = true;
            reason = "length";
        } else {
            const int64_t last = r.last_token();
            if (last >= 0) {
                for (int64_t s : r.stop_token_ids()) {
                    if (s == last) { stop = true; reason = "stop"; break; }
                }
            }
        }
        if (stop) {
            r.set_state(RequestState::Finished);
            r.metrics().finished_at_ms = now_ms();
            if (reason.empty()) reason = "eos";
            r.metrics().finished_reason = reason;
            int sid = r.seq_id();
            if (sid >= 0) paged_kv_->destroy_sequence(sid);
            std::unique_ptr<Request> owned = std::move(*it);
            finished_.push_back(std::move(owned));
            it = running_.erase(it);
            ++n_finished;
        } else {
            ++it;
        }
    }
    stats.num_finished = n_finished;
    return n_finished;
}

// ---------------------------------------------------------------------------
// admit_prefill_group_ — pop a batch of requests from waiting_ that fits
// in one bucket. Returns owned unique_ptrs in `owned_` plus raw pointers
// (aliasing the same objects) in `reqs`.
// ---------------------------------------------------------------------------
bool Scheduler::admit_prefill_group_(PrefillGroup& out) {
    if (waiting_queue_.empty()) return false;

    int max_len = 0;
    for (auto& r : waiting_queue_) {
        if (static_cast<int>(r->prompt_len()) > max_len) {
            max_len = static_cast<int>(r->prompt_len());
        }
    }
    int chosen_bucket = -1;
    for (int b : cfg_.prefill_buckets) {
        if (b >= max_len) { chosen_bucket = b; break; }
    }
    if (chosen_bucket < 0) {
        // Drop oversized request as "prompt_too_long".
        auto& front = waiting_queue_.front();
        front->set_state(RequestState::Finished);
        front->metrics().finished_reason = "prompt_too_long";
        front->metrics().finished_at_ms = now_ms();
        finished_.push_back(std::move(waiting_queue_.front()));
        waiting_queue_.pop_front();
        return false;
    }
    out.bucket_size = chosen_bucket;

    int free_blocks = paged_kv_->total_free_blocks();
    int taken = 0;
    while (!waiting_queue_.empty() && taken < cfg_.max_prefill_batch) {
        auto& r = waiting_queue_.front();
        const int L = static_cast<int>(r->prompt_len());
        if (L > chosen_bucket) break;
        const int need_blocks =
            (L + PagedKVCache::kBlockSizeHint - 1) / PagedKVCache::kBlockSizeHint;
        if (need_blocks > free_blocks) break;
        free_blocks -= need_blocks;
        out.reqs.push_back(r.get());
        admitted_prefill_.push_back(std::move(waiting_queue_.front()));
        waiting_queue_.pop_front();
        ++taken;
    }
    return !out.reqs.empty();
}

// ---------------------------------------------------------------------------
// do_prefill_ — batched paged prefill on admitted group.
// Each request's prompt is padded with sentinel (token id 0) up to the
// bucket size. The paged-attention kernel already correctly masks padded
// positions because seq_len[b] = L_b (not bucket size) and the scatter
// kernel skips writes past seq_len.
// ---------------------------------------------------------------------------
void Scheduler::do_prefill_(const PrefillGroup& group, IterationStats& stats) {
    const int B = static_cast<int>(group.reqs.size());
    const int S = group.bucket_size;
    if (B == 0 || S == 0) return;

    auto t_prefill_start = std::chrono::steady_clock::now();

    // 1) Build padded token_ids [B, S] int64 on host.
    std::vector<int64_t> token_ids_h(B * S, /*pad=*/0);
    std::vector<int64_t> positions_h(B * S);
    std::vector<int> seq_ids_h(B);
    std::vector<int> start_pos_h(B);

    for (int b = 0; b < B; ++b) {
        Request& r = *group.reqs[b];
        const int L = static_cast<int>(r.prompt_len());
        const int sid = allocate_seq_id_();
        r.set_seq_id(sid);
        r.set_state(RequestState::Prefilling);
        paged_kv_->create_sequence(sid);
        for (int t = 0; t < L; ++t) {
            int pos = paged_kv_->append_token(sid);
            if (pos < 0) {
                throw std::runtime_error("Scheduler: OOM during prefill");
            }
        }
        for (int t = 0; t < L; ++t) {
            token_ids_h[b * S + t] = r.prompt_ids()[t];
        }
        for (int t = 0; t < S; ++t) {
            positions_h[b * S + t] = (t < L) ? t : (L - 1);
        }
        seq_ids_h[b]   = sid;
        start_pos_h[b] = 0;
    }

    // 2) Upload tokens [B, S] int64.
    Tensor token_ids_gpu_ = Tensor::empty({B, S}, DType::INT64,
                                          Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), token_ids_h.data(),
                             B * S * sizeof(int64_t), cudaMemcpyHostToDevice));

    // 3) Batched paged prefill (causal mask).
    Tensor logits = model_->forward_paged_batched(
        token_ids_gpu_, positions_h, *paged_kv_,
        seq_ids_h, start_pos_h, /*is_prefill=*/true);

    // 4) Sample the first token from each request's last-real position.
    auto t_prefill_mid = std::chrono::steady_clock::now();
    stats.prefill_ms += std::chrono::duration<double, std::milli>(
        t_prefill_mid - t_prefill_start).count();

    auto t_sample_start = std::chrono::steady_clock::now();
    const int V = static_cast<int>(logits.shape()[2]);
    // Gather the relevant rows into a contiguous [B, V] FP16 buffer so we
    // can call launch_greedy_sample_batched in one launch.
    __half* gathered = nullptr;
    int* sample_buf  = nullptr;
    MI_CHECK_CUDA(cudaMalloc(&gathered,    B * V * sizeof(__half)));
    MI_CHECK_CUDA(cudaMalloc(&sample_buf,  B * sizeof(int)));
    for (int b = 0; b < B; ++b) {
        const int L = static_cast<int>(group.reqs[b]->prompt_len());
        const int64_t src_off = (int64_t)b * S * V + (L - 1) * V;
        MI_CHECK_CUDA(cudaMemcpyAsync(
            gathered + b * V,
            static_cast<const __half*>(logits.data()) + src_off,
            V * sizeof(__half), cudaMemcpyDeviceToDevice, /*stream=*/0));
    }
    MI_CHECK_CUDA(cudaDeviceSynchronize());
    kernels::launch_greedy_sample_batched(gathered, B, V, sample_buf,
                                          /*stream=*/0);
    std::vector<int> sample_in(B);
    MI_CHECK_CUDA(cudaMemcpy(sample_in.data(), sample_buf, B * sizeof(int),
                             cudaMemcpyDeviceToHost));
    MI_CHECK_CUDA(cudaFree(sample_buf));
    MI_CHECK_CUDA(cudaFree(gathered));

    // 5) Update each admitted request with first generated token + metrics.
    //    Transfer ownership from admitted_prefill_ -> running_.
    const double now = now_ms();
    for (int b = 0; b < B; ++b) {
        Request& r = *group.reqs[b];   // alias of admitted_prefill_[b]
        const int64_t tok = static_cast<int64_t>(sample_in[b]);
        r.set_last_token(tok);
        r.generated_ids().push_back(tok);
        r.metrics().generated_tokens = 1;
        r.metrics().first_token_at_ms = now;
        r.metrics().ttft_ms = now - r.arrival_ms();
        r.set_state(RequestState::Decoding);
    }
    // Move ownership into running_.
    for (auto& up : admitted_prefill_) {
        running_.push_back(std::move(up));
    }
    admitted_prefill_.clear();

    auto t_sample_end = std::chrono::steady_clock::now();
    stats.sample_ms += std::chrono::duration<double, std::milli>(
        t_sample_end - t_sample_start).count();
    stats.num_prefilled = B;
}

// ---------------------------------------------------------------------------
// do_decode_ — batched paged decode (S=1) on all running requests.
// ---------------------------------------------------------------------------
void Scheduler::do_decode_(IterationStats& stats) {
    const int N = static_cast<int>(running_.size());
    if (N == 0) return;

    auto t_dec_start = std::chrono::steady_clock::now();

    std::vector<int64_t> tokens_h(N);
    std::vector<int64_t> positions_h(N);
    std::vector<int>    seq_ids_h(N);
    std::vector<int>    start_pos_h(N);

    for (int i = 0; i < N; ++i) {
        Request& r = *running_[i];
        tokens_h[i] = r.last_token();
        positions_h[i] = r.total_len() - 1;
        const int sid = r.seq_id();
        seq_ids_h[i] = sid;
        int ap = paged_kv_->append_token(sid);
        if (ap < 0) {
            throw std::runtime_error("Scheduler: OOM during decode");
        }
        start_pos_h[i] = ap;
    }

    Tensor token_ids_gpu_ = Tensor::empty({N, 1}, DType::INT64,
                                          Device::cuda(device_index_));
    MI_CHECK_CUDA(cudaMemcpy(token_ids_gpu_.data(), tokens_h.data(),
                             N * sizeof(int64_t), cudaMemcpyHostToDevice));

    Tensor logits = model_->forward_paged_batched(
        token_ids_gpu_, positions_h, *paged_kv_,
        seq_ids_h, start_pos_h, /*is_prefill=*/false);

    auto t_dec_mid = std::chrono::steady_clock::now();
    stats.decode_ms += std::chrono::duration<double, std::milli>(
        t_dec_mid - t_dec_start).count();

    auto t_sample_start = std::chrono::steady_clock::now();
    const int V = static_cast<int>(logits.shape()[2]);
    int* sample_buf = nullptr;
    MI_CHECK_CUDA(cudaMalloc(&sample_buf, N * sizeof(int)));
    kernels::launch_greedy_sample_batched(
        static_cast<const __half*>(logits.data()), N, V, sample_buf,
        /*stream=*/0);
    std::vector<int> sample_in(N);
    MI_CHECK_CUDA(cudaMemcpy(sample_in.data(), sample_buf, N * sizeof(int),
                             cudaMemcpyDeviceToHost));
    MI_CHECK_CUDA(cudaFree(sample_buf));

    const double now = now_ms();
    for (int i = 0; i < N; ++i) {
        Request& r = *running_[i];
        const int64_t tok = static_cast<int64_t>(sample_in[i]);
        r.set_last_token(tok);
        r.generated_ids().push_back(tok);
        ++r.metrics().generated_tokens;
        ++r.metrics().decode_steps;
        const double step_ms = now_ms() - now;
        r.metrics().decode_total_ms += step_ms;
    }
    auto t_sample_end = std::chrono::steady_clock::now();
    stats.sample_ms += std::chrono::duration<double, std::milli>(
        t_sample_end - t_sample_start).count();
    stats.num_decoded = N;
}

// ---------------------------------------------------------------------------
// step() — one iteration of the scheduler.
// ---------------------------------------------------------------------------
IterationStats Scheduler::step() {
    IterationStats stats;
    auto t_start = std::chrono::steady_clock::now();

    // Step 1: sweep finished.
    sweep_finished_(stats);

    // Step 2: admit a prefill group (FIFO, capacity-aware, bucketed).
    PrefillGroup group;
    bool admitted = admit_prefill_group_(group);
    if (admitted) {
        do_prefill_(group, stats);
    }

    // Step 3: decode all currently running sequences.
    do_decode_(stats);

    // Snapshot pool state.
    last_used_blocks_  = paged_kv_->total_in_use_blocks();
    last_total_blocks_ = paged_kv_->total_num_blocks();

    stats.iter         = static_cast<int>(history_.size());
    stats.num_waiting  = num_waiting();
    stats.num_running  = num_running();
    stats.used_blocks  = last_used_blocks_;
    stats.total_blocks = last_total_blocks_;

    auto t_end = std::chrono::steady_clock::now();
    stats.wall_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    history_.push_back(stats);
    return stats;
}

void Scheduler::run_until_done(int64_t max_iters) {
    int64_t iter = 0;
    while (!done() && iter < max_iters) {
        step();
        ++iter;
    }
}

}  // namespace mini_infer