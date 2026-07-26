#include "benchmarks/common/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace mini_infer {

double percentile(const std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const double pos = q * (s.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(pos));
    const size_t hi  = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return s[lo];
    const double frac = pos - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

BenchMetrics aggregate_metrics(const std::vector<std::unique_ptr<Request>>& reqs,
                               double wall_time_ms,
                               int peak_used_blocks,
                               int total_blocks,
                               const std::string& tag) {
    BenchMetrics m;
    m.tag              = tag;
    m.wall_time_ms     = wall_time_ms;
    m.peak_used_blocks = peak_used_blocks;
    m.total_blocks     = total_blocks;
    m.num_requests     = static_cast<int>(reqs.size());

    std::vector<double> ttfts, tpots;
    ttfts.reserve(reqs.size());
    tpots.reserve(reqs.size());

    int64_t total_gen = 0;
    int64_t total_prompt = 0;
    int completed = 0;
    for (auto& r : reqs) {
        if (!r->is_finished()) continue;
        ++completed;
        total_prompt += r->prompt_len();
        total_gen    += r->metrics().generated_tokens;
        ttfts.push_back(r->metrics().ttft_ms);
        if (r->metrics().generated_tokens > 1) {
            const double tpot =
                r->metrics().decode_total_ms
                / std::max(1, r->metrics().decode_steps);
            tpots.push_back(tpot);
        }
    }
    m.num_completed            = completed;
    m.total_prompt_tokens      = total_prompt;
    m.total_generated_tokens   = total_gen;
    if (wall_time_ms > 0.0) {
        m.aggregate_throughput_tps = total_gen / (wall_time_ms / 1000.0);
        m.per_request_throughput_tps = (completed > 0)
            ? m.aggregate_throughput_tps / completed : 0.0;
    }
    if (!ttfts.empty()) {
        double sum = 0.0;
        for (double v : ttfts) sum += v;
        m.ttft_avg_ms = sum / ttfts.size();
        m.ttft_p50_ms = percentile(ttfts, 0.50);
        m.ttft_p99_ms = percentile(ttfts, 0.99);
    }
    if (!tpots.empty()) {
        double sum = 0.0;
        for (double v : tpots) sum += v;
        m.tpot_avg_ms = sum / tpots.size();
        m.tpot_p50_ms = percentile(tpots, 0.50);
        m.tpot_p99_ms = percentile(tpots, 0.99);
    }
    return m;
}

std::string csv_header() {
    return "tag,num_requests,num_completed,wall_ms,prompt_tokens,"
           "gen_tokens,ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,"
           "tpot_avg_ms,tpot_p50_ms,tpot_p99_ms,"
           "aggregate_tps,per_req_tps,peak_blocks,total_blocks";
}

void csv_write_row(std::ofstream& f, const BenchMetrics& m) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "%s,%d,%d,%.3f,%ld,%ld,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%d,%d\n",
        m.tag.c_str(), m.num_requests, m.num_completed,
        m.wall_time_ms,
        static_cast<long>(m.total_prompt_tokens),
        static_cast<long>(m.total_generated_tokens),
        m.ttft_avg_ms, m.ttft_p50_ms, m.ttft_p99_ms,
        m.tpot_avg_ms, m.tpot_p50_ms, m.tpot_p99_ms,
        m.aggregate_throughput_tps, m.per_request_throughput_tps,
        m.peak_used_blocks, m.total_blocks);
    f << buf;
}

std::string markdown_report(const std::vector<BenchMetrics>& rows) {
    std::ostringstream o;
    o << "# mini-infer benchmark report\n\n";
    if (rows.empty()) {
        o << "_no rows_\n";
        return o.str();
    }
    o << "## Summary\n\n";
    o << "| tag | reqs | wall_ms | gen_tokens | aggregate_tps | ttft_p50_ms | "
         "tpot_p50_ms | peak_blocks/total |\n";
    o << "|-----|------|---------|------------|---------------|-------------|"
         "-------------|-------------------|\n";
    for (const auto& m : rows) {
        o << "| " << m.tag
          << " | " << m.num_completed << "/" << m.num_requests
          << " | " << static_cast<int>(m.wall_time_ms)
          << " | " << m.total_generated_tokens
          << " | " << static_cast<int>(m.aggregate_throughput_tps)
          << " | " << static_cast<int>(m.ttft_p50_ms)
          << " | " << (m.tpot_p50_ms > 0.0
                          ? std::to_string(static_cast<int>(m.tpot_p50_ms))
                          : std::string("-"))
          << " | " << m.peak_used_blocks << "/" << m.total_blocks
          << " |\n";
    }
    o << "\n## Detailed metrics\n\n";
    o << "| tag | ttft_avg_ms | ttft_p99_ms | tpot_avg_ms | tpot_p99_ms | "
         "per_req_tps |\n";
    o << "|-----|-------------|-------------|-------------|-------------|"
         "-------------|\n";
    for (const auto& m : rows) {
        o << "| " << m.tag
          << " | " << m.ttft_avg_ms
          << " | " << m.ttft_p99_ms
          << " | " << m.tpot_avg_ms
          << " | " << m.tpot_p99_ms
          << " | " << static_cast<int>(m.per_request_throughput_tps)
          << " |\n";
    }
    return o.str();
}

}  // namespace mini_infer