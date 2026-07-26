/**
 * bench_continuous — Week 6 main benchmark.
 *
 * Drives the iteration-level Scheduler on N prompts (loaded from a JSON
 * file or synthesized) and reports TTFT / TPOT / throughput per the
 * standardized metrics in benchmarks/common/metrics.h.
 *
 * Usage:
 *   bench_continuous --model DIR --dataset PATH_OR_EMPTY \
 *                    --num-prompts N [--max-new-tokens M] \
 *                    [--seed S] [--bucket 64,128,...] \
 *                    [--max-num-blocks K] [--device D] \
 *                    [--out-prefix OUT] [--quiet]
 *
 * Writes:
 *   <out-prefix>.csv : one CSV row + header
 *   <out-prefix>.md  : human-readable markdown summary
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "benchmarks/common/dataset.h"
#include "benchmarks/common/metrics.h"
#include "core/engine.h"
#include "core/tokenizer.h"
#include "model/model_config.h"
#include "model/qwen_model.h"
#include "model/safetensors_loader.h"
#include "scheduler/paged_kv_cache.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"

using namespace mini_infer;

namespace {

struct Args {
    std::string model_dir;
    std::string dataset;
    int         num_prompts       = 100;
    int         max_new_tokens    = 64;
    int         max_seq_len       = 2048;
    int         device            = 0;
    int         seed              = 42;
    int         max_num_blocks    = 4096;
    int         max_prefill_batch = 8;
    // Arrival pattern: 0 = all at once (worst case for continuous),
    // 1 = uniform stagger across arrival_window_ms.
    int         arrival_mode      = 0;
    int         arrival_window_ms = 0;
    std::vector<int> prefill_buckets = {64, 128, 256, 512, 1024};
    std::string out_prefix        = "bench_continuous";
    bool        quiet             = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if      (s == "--model")         a.model_dir        = need(s);
        else if (s == "--dataset")       a.dataset          = need(s);
        else if (s == "--num-prompts")   a.num_prompts      = std::atoi(need(s).c_str());
        else if (s == "--max-new-tokens")a.max_new_tokens   = std::atoi(need(s).c_str());
        else if (s == "--max-seq-len")   a.max_seq_len      = std::atoi(need(s).c_str());
        else if (s == "--device")        a.device           = std::atoi(need(s).c_str());
        else if (s == "--seed")          a.seed             = std::atoi(need(s).c_str());
        else if (s == "--max-num-blocks")a.max_num_blocks   = std::atoi(need(s).c_str());
        else if (s == "--max-prefill-batch") a.max_prefill_batch = std::atoi(need(s).c_str());
        else if (s == "--bucket") {
            a.prefill_buckets.clear();
            std::string v = need(s);
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) {
                a.prefill_buckets.push_back(std::atoi(item.c_str()));
            }
        }
        else if (s == "--out-prefix")    a.out_prefix       = need(s);
        else if (s == "--quiet")         a.quiet            = true;
        else if (s == "--arrival-mode")    a.arrival_mode     = std::atoi(need(s).c_str());
        else if (s == "--arrival-window-ms") a.arrival_window_ms = std::atoi(need(s).c_str());
        else if (s == "-h" || s == "--help") {
            std::printf(
                "Usage: %s --model DIR [--dataset PATH] [--num-prompts N] "
                "[--max-new-tokens M] [--max-seq-len L] [--device D] "
                "[--seed S] [--bucket B1,B2,...] [--max-num-blocks K] "
                "[--max-prefill-batch B] [--out-prefix OUT] [--quiet] "
                "[--arrival-mode M] [--arrival-window-ms W]\n",
                argv[0]);
            std::exit(0);
        }
        else {
            std::fprintf(stderr, "unknown flag: %s\n", s.c_str());
            std::exit(2);
        }
    }
    if (a.model_dir.empty()) {
        std::fprintf(stderr, "--model is required\n");
        std::exit(2);
    }
    return a;
}

std::vector<int64_t> tokenize_prompt(const Tokenizer& tok,
                                     const std::string& text) {
    return tok.encode("user\n" + text + "\nassistant\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const Args args = parse_args(argc, argv);

    // 1) Load model + tokenizer.
    if (!args.quiet) {
        std::printf("[bench_continuous] loading model from %s ...\n",
                    args.model_dir.c_str());
    }
    auto cfg = ModelConfig::load(args.model_dir + "/config.json");
    auto idx = WeightIndex::load(args.model_dir);
    auto model = std::make_shared<QwenModel>(cfg, args.device);
    model->load_weights(idx);
    Tokenizer tok(args.model_dir + "/tokenizer.json",
                  "/data1/kdy/anaconda3/envs/vllm/bin/python");

    // 2) Load or synthesize prompts.
    auto prompts = load_or_synthesize(args.dataset, args.num_prompts,
                                      cfg.vocab_size,
                                      static_cast<uint64_t>(args.seed));
    if (static_cast<int>(prompts.size()) > args.num_prompts) {
        prompts.resize(args.num_prompts);
    }
    if (prompts.empty()) {
        std::fprintf(stderr, "no prompts loaded\n");
        return 1;
    }

    // 3) Tokenize JSON-loaded prompts (synthetic already has ids).
    std::vector<BenchPrompt> final_prompts;
    final_prompts.reserve(prompts.size());
    int dropped = 0;
    const int max_prompt_len = args.max_seq_len - args.max_new_tokens;
    for (auto& p : prompts) {
        BenchPrompt q;
        q.prompt_text = p.prompt_text;
        if (!p.token_ids.empty()) {
            q.token_ids = p.token_ids;   // synthetic: use pre-computed ids
        } else {
            q.token_ids = tokenize_prompt(tok, p.prompt_text);
        }
        // Truncate so prompt_len + max_new_tokens <= max_seq_len, otherwise
        // prefill will OOM the paged pool.
        if (q.prompt_len() > max_prompt_len) {
            q.token_ids.resize(max_prompt_len);
            q.prompt_text += " [truncated]";
        }
        if (q.prompt_len() <= 0) { ++dropped; continue; }
        final_prompts.push_back(std::move(q));
    }
    if (final_prompts.empty()) {
        std::fprintf(stderr, "all prompts exceed max_seq_len\n");
        return 1;
    }
    if (dropped > 0 && !args.quiet) {
        std::fprintf(stderr, "[bench_continuous] dropped %d prompts "
                             "(prompt_len + max_new > max_seq_len)\n",
                     dropped);
    }

    // 4) Build scheduler.
    auto paged_kv = std::make_shared<PagedKVCache>(
        args.max_num_blocks, cfg.num_hidden_layers,
        cfg.num_key_value_heads, cfg.kv_head_dim(),
        /*max_blocks_per_seq=*/args.max_seq_len / BlockAllocator::kBlockSize,
        args.device);

    SchedulerConfig sc;
    sc.max_num_blocks    = args.max_num_blocks;
    sc.max_blocks_per_seq = args.max_seq_len / BlockAllocator::kBlockSize;
    sc.prefill_buckets    = args.prefill_buckets;
    sc.max_prefill_batch  = args.max_prefill_batch;
    sc.max_decode_batch   = 64;

    Scheduler sched(model, paged_kv, sc, args.device);

    // 5) Build Request objects. For Qwen2.5 we add the chat-format wrapping.
    std::vector<int64_t> stop_ids;
    if (tok.im_end_id()   >= 0) stop_ids.push_back(tok.im_end_id());
    if (tok.eos_token_id()>= 0) stop_ids.push_back(tok.eos_token_id());

    std::vector<std::unique_ptr<Request>> reqs;
    reqs.reserve(final_prompts.size());
    for (auto& p : final_prompts) {
        auto r = std::make_unique<Request>(
            p.token_ids, args.max_new_tokens, stop_ids);
        reqs.push_back(std::move(r));
    }

    // 6) Submit all requests and run scheduler.
    auto t_start = std::chrono::steady_clock::now();
    if (args.arrival_mode == 0 || args.arrival_window_ms <= 0) {
        // All requests submitted at t=0 (worst case for continuous batching).
        for (auto& r : reqs) sched.submit(std::move(r));
    } else {
        // Stagger submissions uniformly across arrival_window_ms.
        const int N = static_cast<int>(reqs.size());
        const int step_ms = (N > 1) ? args.arrival_window_ms / (N - 1) : 0;
        auto submit_t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            sched.submit(std::move(reqs[i]));
            if (i + 1 < N && step_ms > 0) {
                // Sleep until next arrival slot.
                auto target = submit_t0 + std::chrono::milliseconds(
                    static_cast<long long>((i + 1) * step_ms));
                std::this_thread::sleep_until(target);
            }
        }
    }
    sched.run_until_done();
    auto t_end = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    auto finished = sched.drain_finished();

    // 7) Aggregate metrics.
    int peak_used = 0;
    for (const auto& s : sched.history()) {
        if (s.used_blocks > peak_used) peak_used = s.used_blocks;
    }
    char tag[128];
    std::snprintf(tag, sizeof(tag), "continuous;N=%d;B=%d",
                  args.num_prompts, args.max_prefill_batch);
    BenchMetrics m = aggregate_metrics(finished, wall_ms,
                                       peak_used, args.max_num_blocks, tag);

    if (!args.quiet) {
        std::printf("[bench_continuous] wall=%.1fms  gen=%ld tokens  "
                    "agg=%.1f tok/s  ttft_p50=%.1fms  tpot_p50=%.1fms  "
                    "blocks=%d/%d\n",
                    m.wall_time_ms, m.total_generated_tokens,
                    m.aggregate_throughput_tps,
                    m.ttft_p50_ms, m.tpot_p50_ms,
                    m.peak_used_blocks, m.total_blocks);
    }

    // 8) Write CSV + Markdown.
    std::ofstream csv(args.out_prefix + ".csv");
    csv << csv_header() << "\n";
    csv_write_row(csv, m);
    csv.close();
    std::ofstream md(args.out_prefix + ".md");
    md << markdown_report({m});
    md.close();

    return 0;
}