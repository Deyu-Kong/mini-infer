/**
 * bench_static — Week 6 static batching baseline.
 *
 * Splits N prompts into chunks of size `--batch-size B`. Each chunk is
 * processed by `Engine::generate_batched_paged` and the wall time is the
 * sum over chunks. This is the "static batching" baseline that
 * continuous batching is supposed to beat.
 *
 * It uses the same metrics as bench_continuous so the comparison is
 * apples-to-apples.
 *
 * Usage:
 *   bench_static --model DIR --dataset PATH_OR_EMPTY \
 *                --num-prompts N --batch-size B \
 *                [--max-new-tokens M] [--seed S] [--device D] \
 *                [--out-prefix OUT] [--quiet]
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
#include <vector>

#include "benchmarks/common/dataset.h"
#include "benchmarks/common/metrics.h"
#include "core/engine.h"
#include "core/tokenizer.h"
#include "model/model_config.h"
#include "model/transformer_model.h"
#include "model/safetensors_loader.h"
#include "scheduler/request.h"

using namespace mini_infer;

namespace {

struct Args {
    std::string model_dir;
    std::string dataset;
    int num_prompts    = 100;
    int batch_size     = 8;
    int max_new_tokens = 64;
    int max_seq_len    = 2048;
    int device         = 0;
    int seed           = 42;
    std::string out_prefix = "bench_static";
    bool quiet = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", flag.c_str()); std::exit(2); }
            return argv[++i];
        };
        if      (s == "--model")         a.model_dir      = need(s);
        else if (s == "--dataset")       a.dataset        = need(s);
        else if (s == "--num-prompts")   a.num_prompts    = std::atoi(need(s).c_str());
        else if (s == "--batch-size")    a.batch_size     = std::atoi(need(s).c_str());
        else if (s == "--max-new-tokens")a.max_new_tokens = std::atoi(need(s).c_str());
        else if (s == "--max-seq-len")   a.max_seq_len    = std::atoi(need(s).c_str());
        else if (s == "--device")        a.device         = std::atoi(need(s).c_str());
        else if (s == "--seed")          a.seed           = std::atoi(need(s).c_str());
        else if (s == "--out-prefix")    a.out_prefix     = need(s);
        else if (s == "--quiet")         a.quiet          = true;
        else if (s == "-h" || s == "--help") {
            std::printf("Usage: %s --model DIR [--dataset P] --num-prompts N "
                        "--batch-size B [--max-new-tokens M] [--device D] "
                        "[--seed S] [--out-prefix OUT] [--quiet]\n", argv[0]);
            std::exit(0);
        }
        else {
            std::fprintf(stderr, "unknown flag: %s\n", s.c_str());
            std::exit(2);
        }
    }
    if (a.model_dir.empty()) { std::fprintf(stderr, "--model required\n"); std::exit(2); }
    if (a.batch_size <= 0)   { std::fprintf(stderr, "--batch-size must be > 0\n"); std::exit(2); }
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

    if (!args.quiet) {
        std::printf("[bench_static] loading model from %s ...\n",
                    args.model_dir.c_str());
    }
    auto cfg = ModelConfig::load(args.model_dir + "/config.json");
    auto idx = WeightIndex::load(args.model_dir);
    auto model = std::make_shared<TransformerModel>(cfg, args.device);
    model->load_weights(idx);
    Tokenizer tok(args.model_dir + "/tokenizer.json",
                  "/data1/kdy/anaconda3/envs/vllm/bin/python");

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

    std::vector<BenchPrompt> final_prompts;
    final_prompts.reserve(prompts.size());
    int dropped = 0;
    const int max_prompt_len = args.max_seq_len - args.max_new_tokens;
    for (auto& p : prompts) {
        BenchPrompt q;
        q.prompt_text = p.prompt_text;
        if (!p.token_ids.empty()) {
            q.token_ids = p.token_ids;
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
    if (dropped > 0 && !args.quiet) {
        std::fprintf(stderr, "[bench_static] dropped %d prompts\n", dropped);
    }
    if (final_prompts.empty()) {
        std::fprintf(stderr, "all prompts exceed max_seq_len\n");
        return 1;
    }

    // Engine for static batching. Make the paged pool big enough for one
    // batch's worth of KV at full max_seq_len + per-request max_new_tokens;
    // otherwise prefill OOMs. The synthetic generator may produce prompts
    // up to ~1024 tokens, so we use max_seq_len as the upper bound.
    const int per_seq_blocks = (args.max_seq_len + BlockAllocator::kBlockSize - 1)
                               / BlockAllocator::kBlockSize;
    const int pool_blocks = per_seq_blocks * (args.batch_size + 1) + 16;
    Engine engine(model, args.max_seq_len, args.device, pool_blocks);
    engine.set_sampling(SamplingMode::Greedy);

    std::vector<int64_t> stop_ids;
    if (tok.im_end_id()    >= 0) stop_ids.push_back(tok.im_end_id());
    if (tok.eos_token_id() >= 0) stop_ids.push_back(tok.eos_token_id());

    // For static batching we also need a Request per chunk-entry so we can
    // reuse aggregate_metrics(). The Request is constructed from the actual
    // outputs of generate_batched_paged.
    std::vector<std::unique_ptr<Request>> all_reqs;
    all_reqs.reserve(final_prompts.size());
    for (auto& p : final_prompts) {
        all_reqs.push_back(std::make_unique<Request>(
            p.token_ids, args.max_new_tokens, stop_ids));
    }

    int peak_used_blocks = 0;
    auto t_start = std::chrono::steady_clock::now();

    // Run static-batched: split into chunks of batch_size, run each chunk.
    const int N = static_cast<int>(final_prompts.size());
    for (int off = 0; off < N; off += args.batch_size) {
        const int B = std::min(args.batch_size, N - off);
        std::vector<std::vector<int64_t>> chunk_prompts(B);
        for (int i = 0; i < B; ++i) {
            chunk_prompts[i] = final_prompts[off + i].token_ids;
        }
        auto outs = engine.generate_batched_paged(
            chunk_prompts, args.max_new_tokens, stop_ids);

        // Backfill Request objects in this chunk with measured metrics.
        for (int i = 0; i < B; ++i) {
            Request& r = *all_reqs[off + i];
            const int gen_count = static_cast<int>(outs[i].size())
                                - static_cast<int>(chunk_prompts[i].size());
            r.metrics().generated_tokens = std::max(0, gen_count);
            r.metrics().decode_steps     = std::max(0, gen_count - 1);
            r.metrics().ttft_ms         = 0.0;
            r.metrics().decode_total_ms = 0.0;
            r.set_last_token(outs[i].empty() ? -1 : outs[i].back());
            for (size_t k = chunk_prompts[i].size(); k < outs[i].size(); ++k) {
                r.generated_ids().push_back(outs[i][k]);
            }
            r.set_state(RequestState::Finished);
            r.metrics().finished_reason = "length";
        }

        const int used = engine.paged_kv().total_in_use_blocks();
        if (used > peak_used_blocks) peak_used_blocks = used;

        // Engine::generate_batched_paged doesn't tear down its sequences,
        // so we do it here between chunks to keep the pool small.
        engine.clear_paged_sequences();
    }
    auto t_end = std::chrono::steady_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    char tag[128];
    std::snprintf(tag, sizeof(tag), "static;B=%d;N=%d",
                  args.batch_size, args.num_prompts);
    BenchMetrics m = aggregate_metrics(all_reqs, wall_ms,
                                       peak_used_blocks,
                                       engine.paged_kv().total_num_blocks(),
                                       tag);

    if (!args.quiet) {
        std::printf("[bench_static] wall=%.1fms  gen=%ld tokens  "
                    "agg=%.1f tok/s  blocks=%d/%d\n",
                    m.wall_time_ms, m.total_generated_tokens,
                    m.aggregate_throughput_tps,
                    m.peak_used_blocks, m.total_blocks);
    }

    std::ofstream csv(args.out_prefix + ".csv");
    csv << csv_header() << "\n";
    csv_write_row(csv, m);
    csv.close();
    std::ofstream md(args.out_prefix + ".md");
    md << markdown_report({m});
    md.close();

    return 0;
}