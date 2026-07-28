/**
 * bench_batched — measure aggregate batched-paged decode throughput.
 *
 * Build with the project's libraries. Runs the same prompt N times
 * through `Engine::generate_batched_paged`, varying N from 1 to 16,
 * and reports aggregate tokens/sec across all sequences.
 *
 * Usage:
 *     bench_batched <model_dir> <prompt_text> <max_new_tokens>
 *                   <batch_size> <max_seq_len> <gpu>
 */
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/engine.h"
#include "core/tokenizer.h"
#include "model/model_config.h"
#include "model/transformer_model.h"
#include "model/safetensors_loader.h"

using namespace mini_infer;

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 7) {
        std::fprintf(stderr,
            "Usage: %s <model_dir> <prompt> <max_new_tokens> "
            "<batch_size> <max_seq_len> <gpu>\n", argv[0]);
        return 1;
    }
    const std::string model_dir  = argv[1];
    const std::string prompt     = argv[2];
    const int max_new_tokens      = std::atoi(argv[3]);
    const int batch_size          = std::atoi(argv[4]);
    const int max_seq_len         = std::atoi(argv[5]);
    const int gpu                 = std::atoi(argv[6]);

                model_dir.c_str(), (int)prompt.size(), batch_size,
                max_new_tokens, max_seq_len, gpu);
    std::fflush(stdout);

    // 1) Load model.
    auto cfg = ModelConfig::load(model_dir + "/config.json");
    auto idx = WeightIndex::load(model_dir);
    auto model = std::make_shared<TransformerModel>(cfg, gpu);
    model->load_weights(idx);
    Engine engine(model, max_seq_len, gpu);
    engine.set_sampling(SamplingMode::Greedy);
    std::fflush(stdout);

    // 2) Tokenize once; we use the same prompt for all N sequences.
    Tokenizer tok(model_dir + "/tokenizer.json",
                  "/data1/kdy/anaconda3/envs/vllm/bin/python");
    std::string chat = "user\n" + prompt + "\nassistant\n";
    std::vector<int64_t> prompt_ids = tok.encode(chat);
    for (size_t i = 0; i < std::min<size_t>(5, prompt_ids.size()); ++i)
        std::printf("%ld ", prompt_ids[i]);
    std::printf("\n");
    std::fflush(stdout);

    // Add a single explicit forward_paged test (single-seq path) to see
    // if it works outside the batched loop.
    std::fflush(stdout);
    {
        const int sid0 = 1999;
        engine.paged_kv().create_sequence(sid0);
        for (int64_t t = 0; t < static_cast<int64_t>(prompt_ids.size()); ++t) {
            engine.paged_kv().append_token(sid0);
        }
        Tensor ids_dev = Tensor::empty({1, static_cast<int>(prompt_ids.size())},
                                        DType::INT64, Device::cuda(0));
        cudaMemcpy(ids_dev.data(), prompt_ids.data(),
                   prompt_ids.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        std::vector<int64_t> pos(prompt_ids.size());
        for (size_t k = 0; k < prompt_ids.size(); ++k) pos[k] = (int64_t)k;
        Tensor logits = model->forward_paged(ids_dev, pos, engine.paged_kv(),
                                              sid0, true);
        for (size_t i = 0; i < logits.shape().size(); ++i)
            std::printf("%ld%s", logits.shape()[i], i+1<logits.shape().size()?",":"");
        std::printf("]\n");
        engine.paged_kv().destroy_sequence(sid0);
        std::fflush(stdout);
    }

    // 3) Build N copies of the prompt.
    std::vector<std::vector<int64_t>> prompts(batch_size, prompt_ids);

    // Warmup: small batch to JIT kernels + warm allocator.
    {
        std::fflush(stdout);
        std::vector<std::vector<int64_t>> wps(1, prompt_ids);
        auto wout = engine.generate_batched_paged(wps, 4, {});
        std::fflush(stdout);
    }

    // 4) Real benchmark.
    auto t0 = std::chrono::high_resolution_clock::now();
    auto outs = engine.generate_batched_paged(prompts, max_new_tokens, {});
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    int64_t total_new = 0;
    int matches = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        const int64_t n = static_cast<int64_t>(outs[i].size())
                          - static_cast<int64_t>(prompts[i].size());
        total_new += n;
        if (outs[i] == outs[0]) ++matches;
    }
    double aggregate_tps = (secs > 0.0) ? total_new / secs : 0.0;
    double per_seq_tps   = (secs > 0.0 && batch_size > 0)
                           ? aggregate_tps / batch_size : 0.0;

    std::printf("RESULT B=%d total_new=%ld wall=%.3fs aggregate=%.1f tok/s "
                "per_seq=%.1f tok/s identical=%d/%d\n",
                batch_size, total_new, secs, aggregate_tps, per_seq_tps,
                matches, batch_size);
    std::fflush(stdout);
    return 0;
}