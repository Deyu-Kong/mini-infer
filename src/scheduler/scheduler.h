#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "model/qwen_model.h"
#include "scheduler/paged_kv_cache.h"
#include "scheduler/prefix_cache.h"
#include "scheduler/request.h"

namespace mini_infer {

/**
 * SchedulerConfig — runtime knobs for the iteration-level scheduler.
 *
 *   max_num_blocks        : block pool capacity (shared across sequences).
 *   max_blocks_per_seq    : per-sequence block cap (= max_seq_len / BLOCK_SIZE).
 *   prefill_buckets       : padding buckets for batched prefill, ascending.
 *                           Default {64,128,256,512,1024} means: a prefill
 *                           request with prompt_len=300 goes into bucket 512
 *                           and is padded to 512; a request with 600 goes
 *                           into 1024. The scheduler picks the smallest
 *                           bucket that can hold the batch on this iteration.
 *   max_prefill_batch     : max number of sequences admitted in one prefill
 *                           batch (prevents huge batches from blowing up
 *                           memory).
 *   max_decode_batch      : soft cap on decode-batch size; if exceeded,
 *                           oldest running sequences are evicted (rare in
 *                           Week 6 because we don't have admission control
 *                           beyond block capacity).
 */
struct SchedulerConfig {
    int max_num_blocks       = 4096;
    int max_blocks_per_seq   = 256;
    std::vector<int> prefill_buckets = {64, 128, 256, 512, 1024};
    int max_prefill_batch    = 8;
    int max_decode_batch     = 64;
};

/**
 * IterationStats — diagnostics from a single scheduler iteration.
 * Used by the benchmark layer to plot VRAM utilization, throughput, etc.
 */
struct IterationStats {
    int iter                       = 0;
    int num_waiting                = 0;
    int num_running                = 0;
    int num_prefilled              = 0;    // requests that entered running this iter
    int num_decoded                = 0;    // requests that produced a decode token
    int num_finished               = 0;    // requests that hit a stop condition
    int used_blocks                = 0;
    int total_blocks               = 0;
    double wall_ms                 = 0.0;  // step() wall time
    double prefill_ms              = 0.0;  // breakdown
    double decode_ms               = 0.0;
    double sample_ms               = 0.0;
};

/**
 * Scheduler — iteration-level dynamic batch dispatcher (Week 6).
 *
 * Owns:
 *   - waiting_queue_   : deque<unique_ptr<Request>>  (FIFO of pending reqs)
 *   - running_         : vector<unique_ptr<Request>> (active seqs)
 *   - finished_        : vector<unique_ptr<Request>> (terminal)
 *   - paged_kv_        : the shared PagedKVCache
 *   - prefix_cache_    : Week 7-8 placeholder (no-op now)
 *
 * Iteration loop (`step()`):
 *   1. Sweep running_, drop finished, free their blocks.
 *   2. Admit up to max_prefill_batch from waiting_ (FIFO), grouped into
 *      the smallest prefill_bucket that contains all admitted prompts.
 *      If even one bucket cannot accommodate the batch on this iteration,
 *      try a larger bucket. Admission is gated by `paged_kv_` block
 *      availability — we only admit a prefill if its block cost fits.
 *   3. Run batched paged prefill (variable-length padded to bucket size).
 *      Sample the FIRST token for each newly-prefilled request.
 *   4. Run batched paged decode for all currently-running requests
 *      (single token per sequence, S=1).
 *   5. Sample one token per running sequence, append to its generated_ids,
 *      and update metrics.
 *
 * Sampling is greedy for Week 6 (top-p path is a stub).
 *
 * The host-side iteration loop is:
 *
 *   while (!done()) {
 *       auto stats = step();
 *       log(stats);
 *   }
 *
 * `done()` is true when both queues are empty AND every request that ever
 * ran has reached Finished.
 */
class Scheduler {
public:
    Scheduler(std::shared_ptr<QwenModel> model,
              std::shared_ptr<PagedKVCache> paged_kv,
              SchedulerConfig cfg = SchedulerConfig{},
              int device_index = 0);

    // ---- request lifecycle ----------------------------------------------
    // Add a request to the waiting_queue. seq_id is assigned on admission.
    void submit(std::unique_ptr<Request> req);

    // Run one scheduler iteration. Returns IterationStats for logging.
    IterationStats step();

    // Run until waiting + running are empty.
    //   max_iters : safety cap (default 1e6, well above any practical run).
    void run_until_done(int64_t max_iters = 1000000);

    // ---- queries --------------------------------------------------------
    bool done() const;

    int num_waiting() const { return static_cast<int>(waiting_queue_.size()); }
    int num_running() const { return static_cast<int>(running_.size()); }
    int num_finished() const { return static_cast<int>(finished_.size()); }

    // Snapshot of finished requests (transferred ownership).
    std::vector<std::unique_ptr<Request>> drain_finished();

    const std::vector<IterationStats>& history() const { return history_; }

    // Block-pool snapshot (last seen).
    int used_blocks() const  { return last_used_blocks_; }
    int total_blocks() const { return last_total_blocks_; }

    // Wall-clock time of scheduler creation (for relative metrics).
    double now_ms() const;

private:
    // Step 1: remove finished requests from running_, free their blocks.
    int sweep_finished_(IterationStats& stats);

    // Step 2: pick a prefill bucket + admit up to max_prefill_batch
    // requests from waiting_. Returns the admitted requests + bucket size
    // in `out_*`. Returns false if nothing to admit.
    struct PrefillGroup {
        std::vector<Request*> reqs;
        int bucket_size = 0;
    };
    bool admit_prefill_group_(PrefillGroup& out);

    // Step 3: run batched prefill on the admitted group, sample first
    // token for each, transition them to Decoding.
    void do_prefill_(const PrefillGroup& group, IterationStats& stats);

    // Step 4 + 5: run batched decode (S=1) on all running_ requests,
    // sample one token per request, update Request state.
    void do_decode_(IterationStats& stats);

    // Assign a fresh paged sequence id, register with PagedKVCache.
    int allocate_seq_id_();

    std::shared_ptr<QwenModel>    model_;
    std::shared_ptr<PagedKVCache> paged_kv_;
    SchedulerConfig               cfg_;
    int                           device_index_;
    PrefixCache                   prefix_cache_;

    std::deque<std::unique_ptr<Request>>  waiting_queue_;
    std::vector<std::unique_ptr<Request>> running_;
    std::vector<std::unique_ptr<Request>> finished_;
    // Admitted in the current iteration's prefill step; ownership is
    // transferred to running_ after sampling the first token.
    std::vector<std::unique_ptr<Request>> admitted_prefill_;

    int next_seq_id_ = 100000;       // match Engine's existing range
    int last_used_blocks_  = 0;
    int last_total_blocks_ = 0;
    std::chrono::steady_clock::time_point t0_;

    std::vector<IterationStats> history_;
};

}  // namespace mini_infer