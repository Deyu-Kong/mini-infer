#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "scheduler/request.h"

namespace mini_infer {

/**
 * BenchMetrics — aggregate metrics for one benchmark run.
 *
 *   - num_requests            : total requests submitted
 *   - num_completed           : requests that reached Finished
 *   - total_prompt_tokens     : sum of prompt_len across all reqs
 *   - total_generated_tokens  : sum of generated_ids.size() across reqs
 *   - wall_time_ms            : from first submit to last finish
 *   - ttft_avg_ms / ttft_p99_ms : time-to-first-token (per-request)
 *   - tpot_avg_ms / tpot_p99_ms : time-per-output-token (per-request)
 *   - aggregate_throughput_tps : total_generated_tokens / wall_time_ms * 1000
 *   - per_request_throughput_tps : aggregate_throughput_tps / num_completed
 *   - peak_used_blocks / total_blocks
 *
 * `tpot` here is the *wall* time per output token, including the time
 * spent waiting for the scheduler (it's the user-visible latency). This
 * is what production serving frameworks measure.
 */
struct BenchMetrics {
    int num_requests            = 0;
    int num_completed           = 0;
    int64_t total_prompt_tokens    = 0;
    int64_t total_generated_tokens = 0;
    double wall_time_ms            = 0.0;

    double ttft_avg_ms    = 0.0;
    double ttft_p50_ms    = 0.0;
    double ttft_p99_ms    = 0.0;

    double tpot_avg_ms    = 0.0;
    double tpot_p50_ms    = 0.0;
    double tpot_p99_ms    = 0.0;

    double aggregate_throughput_tps  = 0.0;
    double per_request_throughput_tps = 0.0;

    int peak_used_blocks  = 0;
    int total_blocks      = 0;

    // Comma-separated tag used in CSV output (e.g. "continuous,B=8").
    std::string tag;
};

/**
 * Aggregate per-request metrics into BenchMetrics.
 */
BenchMetrics aggregate_metrics(const std::vector<std::unique_ptr<Request>>& reqs,
                               double wall_time_ms,
                               int peak_used_blocks,
                               int total_blocks,
                               const std::string& tag);

/**
 * Write metrics to a CSV row (header written separately). Returns the
 * CSV header string.
 */
std::string csv_header();
void        csv_write_row(std::ofstream& f, const BenchMetrics& m);

/**
 * Write a Markdown report summarizing one or more BenchMetrics.
 * Returns the markdown body.
 */
std::string markdown_report(const std::vector<BenchMetrics>& rows);

/**
 * Compute p99 / p50 of a sorted vector (the vector is not modified).
 */
double percentile(const std::vector<double>& v, double q);

}  // namespace mini_infer