#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace mini_infer {

/**
 * Request lifecycle (Week 6).
 *
 *   Pending     -> just submitted, lives in the scheduler's waiting_queue
 *   Prefilling  -> currently inside a forward() prefill call
 *   Decoding    -> at least one decode token generated; lives in running_queue
 *   Finished    -> terminal state; moved to results and removed from the cache
 *
 * The state machine is owned by Scheduler; Request itself only stores data
 * + metrics. Transitions are explicit so the scheduler can be debugged
 * deterministically.
 */
enum class RequestState {
    Pending    = 0,
    Prefilling = 1,
    Decoding   = 2,
    Finished   = 3,
};

const char* request_state_name(RequestState s);

/**
 * Sampling parameters attached to a request (independent of global default).
 * Used by the scheduler's batched sampling kernel.
 */
struct SamplingParams {
    bool  greedy       = true;     // if true, top_p/temperature ignored
    float top_p        = 0.9f;
    float temperature  = 1.0f;
    unsigned long long seed = 42;

    static SamplingParams greedy_default() {
        SamplingParams s;
        s.greedy = true;
        return s;
    }
};

/**
 * Per-iteration metrics for a single request. Used by the benchmark layer
 * to compute TTFT / TPOT / per-request latency.
 *
 *   ttft_ms          : wall time from request arrival until the first
 *                      generated token is sampled (end of prefill).
 *   first_token_at   : wall-clock time at which the first generated token
 *                      was produced (steady_clock time_point).
 *   decode_total_ms  : sum of decode-step wall times (used for TPOT).
 *   decode_steps     : number of decode steps executed (== number of
 *                      generated tokens after the first one).
 *   generated_tokens : total tokens generated (incl. the first).
 *   finished_at      : wall-clock time when state became Finished.
 *   finished_reason  : "eos" | "stop" | "length"
 */
struct RequestMetrics {
    double ttft_ms         = 0.0;
    double first_token_at_ms = 0.0;     // relative to scheduler start
    double decode_total_ms = 0.0;
    int    decode_steps    = 0;
    int    generated_tokens = 0;
    double finished_at_ms  = 0.0;
    std::string finished_reason;
};

/**
 * Request — one user-submitted prompt.
 *
 * Owns:
 *   - prompt_ids            : tokenized prompt (host)
 *   - generated_ids         : sampled tokens (host, grows each decode step)
 *   - paged_seq_id          : the sequence id in PagedKVCache (-1 until admitted)
 *   - last_decoded_token    : most recent token sampled (used as next input)
 *   - state                 : see RequestState
 *   - max_new_tokens        : hard cap on tokens to generate
 *   - stop_token_ids        : stop generation if any of these is sampled
 *   - sampling              : per-request sampling config
 *   - arrival_ms            : wall-clock time at submission (for TTFT)
 *   - metrics               : per-iteration metrics
 */
class Request {
public:
    Request(std::vector<int64_t> prompt_ids,
            int max_new_tokens,
            std::vector<int64_t> stop_token_ids = {},
            SamplingParams sampling = SamplingParams::greedy_default());

    // Convenience: get full output sequence (prompt + generated).
    std::vector<int64_t> output_ids() const;

    // ----- state accessors (mutable from scheduler) ----------------------
    RequestState state() const { return state_; }
    void set_state(RequestState s) { state_ = s; }

    int seq_id() const { return paged_seq_id_; }
    void set_seq_id(int sid) { paged_seq_id_ = sid; }

    int64_t last_token() const { return last_decoded_token_; }
    void    set_last_token(int64_t t) { last_decoded_token_ = t; }

    int64_t prompt_len() const { return static_cast<int64_t>(prompt_ids_.size()); }
    int64_t total_len()  const { return prompt_len() + metrics_.generated_tokens; }

    bool is_finished() const { return state_ == RequestState::Finished; }

    // ----- data accessors ------------------------------------------------
    const std::vector<int64_t>& prompt_ids() const { return prompt_ids_; }
    std::vector<int64_t>&       generated_ids() { return generated_ids_; }
    const std::vector<int64_t>& generated_ids() const { return generated_ids_; }

    int max_new_tokens() const { return max_new_tokens_; }
    const std::vector<int64_t>& stop_token_ids() const { return stop_token_ids_; }
    const SamplingParams& sampling() const { return sampling_; }

    double arrival_ms() const { return arrival_ms_; }
    void   set_arrival_ms(double v) { arrival_ms_ = v; }

    RequestMetrics& metrics() { return metrics_; }
    const RequestMetrics& metrics() const { return metrics_; }

private:
    std::vector<int64_t> prompt_ids_;
    std::vector<int64_t> generated_ids_;
    std::vector<int64_t> stop_token_ids_;
    SamplingParams sampling_;
    int max_new_tokens_;

    RequestState state_ = RequestState::Pending;
    int paged_seq_id_   = -1;
    int64_t last_decoded_token_ = -1;

    double arrival_ms_ = 0.0;
    RequestMetrics metrics_;
};

}  // namespace mini_infer