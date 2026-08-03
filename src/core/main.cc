/**
 * main.cc — CLI entry point for mini-infer.
 *
 * Subcommands:
 *   mini_infer serve   --model DIR [--port 8000] ...   # online HTTP service
 *   mini_infer client  --remote http://host:port ...   # online client
 *   mini_infer generate --model DIR --prompt T ...     # offline (explicit)
 *   mini_infer --model DIR --prompt T ...              # offline (legacy, default)
 *
 * Offline usage:
 *   ./build/mini_infer --model /path/to/Qwen2.5-7B-Instruct \
 *       --prompt "介绍一下你自己" --max-new-tokens 100 \
 *       [--temperature 1.0 --top-p 0.9 --seed 42] \
 *       [--max-seq-len 2048 --device 0] [--python python3]
 *       [--raw]                                # use prompt verbatim (no ChatML wrap)
 *       [--paged]                              # PagedAttention (legacy flag, kept for compat)
 *       [--spec-draft /path/to/Qwen2.5-0.5B-Instruct --gamma 4]  # speculative
 *
 * For Qwen2.5, the model directory must contain:
 *   config.json, tokenizer.json, model-*.safetensors
 */
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/tokenizer.h"
#include "core/server.h"
#include "model/model_config.h"
#include "model/transformer_model.h"
#include "model/safetensors_loader.h"
#include "speculative/draft_engine.h"
#include "speculative/spec_decoder.h"

namespace {

struct Args {
    std::string model_dir;
    std::string prompt = "你好，请介绍一下你自己。";
    int64_t max_new_tokens = 100;
    int64_t max_seq_len    = 2048;
    int     device         = 0;
    float   temperature    = 1.0f;
    float   top_p          = 0.9f;
    bool    greedy         = false;
    bool    paged          = false;
    bool    raw            = false;   // skip ChatML wrapping
    uint64_t seed          = 42;
    std::string spec_draft_dir;
    int     gamma          = 4;
    std::string python     = "python3";
};

// Expects argv[0] = program name (ignored); flags begin at argv[1].
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
        if      (s == "--model")         a.model_dir      = need(s);
        else if (s == "--prompt")        a.prompt         = need(s);
        else if (s == "--max-new-tokens")a.max_new_tokens = std::stoll(need(s));
        else if (s == "--max-seq-len")   a.max_seq_len    = std::stoll(need(s));
        else if (s == "--device")        a.device         = std::stoi(need(s));
        else if (s == "--temperature")   a.temperature    = std::stof(need(s));
        else if (s == "--top-p")         a.top_p          = std::stof(need(s));
        else if (s == "--greedy")        a.greedy         = true;
        else if (s == "--paged")         a.paged          = true;
        else if (s == "--raw")           a.raw            = true;
        else if (s == "--python")        a.python         = need(s);
        else if (s == "--seed")          a.seed           = std::stoull(need(s));
        else if (s == "--spec-draft")    a.spec_draft_dir = need(s);
        else if (s == "--gamma")         a.gamma          = std::stoi(need(s));
        else if (s == "-h" || s == "--help") {
            std::printf(
                "Usage: %s [--model DIR] [--prompt T] [--max-new-tokens N] "
                "[--max-seq-len M] [--device D] [--temperature T] [--top-p P] "
                "[--greedy] [--paged] [--raw] [--python PY] [--seed S] "
                "[--spec-draft DRAFT_DIR] [--gamma G]\n"
                "  Subcommands: serve | client | generate\n",
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

int run_offline(const Args& args) {
    std::printf("[mini-infer] loading model from %s ...\n", args.model_dir.c_str());

    auto cfg = mini_infer::ModelConfig::load(args.model_dir + "/config.json");
    std::printf("  H=%ld I=%ld L=%ld Hq=%ld Hkv=%ld V=%ld theta=%.0f\n",
                cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.num_key_value_heads * cfg.kv_head_dim(),
                cfg.vocab_size, cfg.rope_theta);

    auto weight_idx = mini_infer::WeightIndex::load(args.model_dir);
    auto model = std::make_shared<mini_infer::TransformerModel>(cfg, args.device);
    model->load_weights(weight_idx);

    mini_infer::Tokenizer tok(args.model_dir + "/tokenizer.json", args.python);
    if (tok.eos_token_id() >= 0) {
        std::printf("  eos=%ld bos=%ld im_start=%ld im_end=%ld\n",
                    tok.eos_token_id(), tok.bos_token_id(),
                    tok.im_start_id(), tok.im_end_id());
    }

    std::vector<int64_t> prompt_ids;
    std::vector<int64_t> stop_ids;
    const bool has_im = (tok.im_start_id() >= 0 && tok.im_end_id() >= 0);
    if (!args.raw && has_im) {
        // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        std::string chat = "<|im_start|>user\n" + args.prompt +
                           "<|im_end|>\n<|im_start|>assistant\n";
        prompt_ids = tok.encode(chat);
        stop_ids = {tok.im_end_id()};
        if (tok.eos_token_id() >= 0) stop_ids.push_back(tok.eos_token_id());
    } else {
        prompt_ids = tok.encode(args.prompt);
        if (tok.eos_token_id() >= 0) stop_ids = {tok.eos_token_id()};
        if (has_im) stop_ids.push_back(tok.im_end_id());
    }

    if (static_cast<int64_t>(prompt_ids.size()) > args.max_seq_len) {
        std::fprintf(stderr, "[mini-infer] Error: prompt length (%zu) exceeds max_seq_len (%ld)\n",
                     prompt_ids.size(), args.max_seq_len);
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> out_ids;

    if (!args.spec_draft_dir.empty()) {
        std::printf("[mini-infer] loading draft model from %s ...\n",
                    args.spec_draft_dir.c_str());
        auto draft_cfg = mini_infer::ModelConfig::load(
            args.spec_draft_dir + "/config.json");
        std::printf("  Draft: H=%ld I=%ld L=%ld Hq=%ld Hkv=%ld V=%ld\n",
                    draft_cfg.hidden_size, draft_cfg.intermediate_size,
                    draft_cfg.num_hidden_layers,
                    draft_cfg.num_attention_heads * draft_cfg.head_dim(),
                    draft_cfg.num_key_value_heads * draft_cfg.kv_head_dim(),
                    draft_cfg.vocab_size);

        auto draft_weight_idx = mini_infer::WeightIndex::load(args.spec_draft_dir);
        auto draft_model = std::make_shared<mini_infer::TransformerModel>(
            draft_cfg, args.device);
        draft_model->load_weights(draft_weight_idx);

        auto draft_engine = std::make_shared<mini_infer::DraftEngine>(
            draft_model, args.max_seq_len, args.device);

        mini_infer::SpecDecoder spec_decoder(
            model, draft_engine, args.max_seq_len, args.gamma, args.device);

        if (args.greedy) {
            spec_decoder.set_sampling_greedy();
        } else {
            spec_decoder.set_sampling_top_p(args.top_p, args.temperature, args.seed);
        }

        out_ids = spec_decoder.generate(prompt_ids, args.max_new_tokens, stop_ids);

        const auto& stats = spec_decoder.stats();
        std::printf("[spec-decode] accepted=%ld rejected=%ld draft=%ld "
                    "accept_rate=%.2f%%\n",
                    stats.total_accepted, stats.total_rejected,
                    stats.total_draft, stats.accept_rate() * 100.0);
    } else {
        mini_infer::Engine engine(model, args.max_seq_len, args.device);
        if (args.greedy) {
            engine.set_sampling(mini_infer::SamplingMode::Greedy);
        } else {
            engine.set_sampling(mini_infer::SamplingMode::TopP,
                                 args.top_p, args.temperature, args.seed);
        }

        if (args.paged) {
            out_ids = engine.generate_paged(prompt_ids, args.max_new_tokens, stop_ids);
        } else {
            out_ids = engine.generate(prompt_ids, args.max_new_tokens, stop_ids);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    int64_t gen = static_cast<int64_t>(out_ids.size()) - static_cast<int64_t>(prompt_ids.size());
    std::vector<int64_t> gen_ids(out_ids.begin() + prompt_ids.size(), out_ids.end());
    std::string gen_text = tok.decode(gen_ids);

    std::printf("\n--- generated ---\n%s\n--- end ---\n", gen_text.c_str());
    std::printf("[mini-infer] %ld tokens in %.2fs (%.2f tok/s)\n",
                gen, secs, gen / secs);

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string cmd = argv[1];
        if (cmd == "serve")            return mini_infer::run_serve(argc - 2, argv + 2);
        if (cmd == "client")           return mini_infer::run_client(argc - 2, argv + 2);
        if (cmd == "generate")         return run_offline(parse_args(argc - 1, argv + 1));
        // fall through: legacy "mini_infer --model ... --prompt ..." form
    }
    return run_offline(parse_args(argc, argv));
}
