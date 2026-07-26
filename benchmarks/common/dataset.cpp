#include "benchmarks/common/dataset.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mini_infer {

namespace {

// Rough templates — not for NLP, just for shape variance.
const std::vector<std::string>& template_bank() {
    static const std::vector<std::string> bank = {
        // Short
        "Hi.",
        "Translate to French: %s.",
        "What is 2 + 2?",
        "Summarize this: %s.",
        "Continue: Once upon a time %s",
        // Medium
        "Write a paragraph explaining %s to a 10-year-old.",
        "Compare and contrast %s and %s in detail.",
        "Translate the following passage from English to Mandarin:\n%s",
        "Read the following text and answer the question.\nText: %s\nQuestion: %s",
        "Explain how the %s algorithm works and give an example.",
        // Long
        "Write an essay of roughly 500 words on the following topic:\n%s\n"
        "Your essay should have an introduction, two body paragraphs, and "
        "a conclusion. Use clear topic sentences and supporting evidence.",
        "You are a senior software engineer. Review the following code "
        "for bugs, style issues, and performance. Output a numbered list "
        "of specific issues with line-level suggestions.\n\n```\n%s\n```",
    };
    return bank;
}

}  // namespace

std::vector<BenchPrompt> load_prompts_from_json(const std::string& path,
                                                int max_count) {
    std::vector<BenchPrompt> out;
    if (path.empty()) return out;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[dataset] WARN: cannot open %s\n", path.c_str());
        return out;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[dataset] WARN: invalid JSON in %s: %s\n",
                     path.c_str(), e.what());
        return out;
    }
    if (!j.is_array()) {
        std::fprintf(stderr, "[dataset] WARN: %s is not a JSON array\n",
                     path.c_str());
        return out;
    }

    int idx = 0;
    for (const auto& item : j) {
        if (idx >= max_count) break;
        std::string text;
        if (item.is_string()) {
            text = item.get<std::string>();
        } else if (item.is_object()) {
            // ShareGPT-style: pick the first "human" turn, fall back to
            // the first non-empty string field.
            if (item.contains("conversations") && item["conversations"].is_array()) {
                for (const auto& turn : item["conversations"]) {
                    if (turn.contains("from") && turn["from"] == "human"
                        && turn.contains("value")) {
                        text = turn["value"].get<std::string>();
                        break;
                    }
                }
            }
            if (text.empty() && item.contains("prompt")) {
                text = item["prompt"].get<std::string>();
            }
            if (text.empty() && item.contains("text")) {
                text = item["text"].get<std::string>();
            }
        }
        if (text.empty()) continue;
        BenchPrompt p;
        p.prompt_text = std::move(text);
        out.push_back(std::move(p));
        ++idx;
    }
    return out;
}

std::vector<BenchPrompt> make_synthetic_prompts(int count, int vocab_size,
                                               uint64_t seed) {
    std::vector<BenchPrompt> out;
    out.reserve(count);
    if (count <= 0 || vocab_size <= 1) return out;

    std::mt19937_64 rng(seed);
    const auto& bank = template_bank();

    // Length buckets.
    struct Bucket { int lo, hi, weight; };
    const Bucket buckets[] = {
        {10, 32, 5},     // short
        {64, 256, 3},    // medium
        {512, 1024, 2},  // long
    };
    std::discrete_distribution<int> len_dist({5, 3, 2});

    for (int i = 0; i < count; ++i) {
        const auto& b = buckets[len_dist(rng)];
        std::uniform_int_distribution<int> len_dist2(b.lo, b.hi);
        const int target_len = len_dist2(rng);

        // Build a prompt of approximately target_len *whitespace* tokens.
        // We use the same template multiple times, then truncate to size.
        std::ostringstream oss;
        while (static_cast<int>(oss.tellp()) < target_len * 6) {
            const auto& tmpl = bank[static_cast<size_t>(rng()) % bank.size()];
            oss << tmpl << " ";
        }
        std::string text = oss.str();

        BenchPrompt p;
        p.prompt_text = "synthetic#" + std::to_string(i)
                       + " (target_tokens=" + std::to_string(target_len) + ")";
        // Tokenize deterministically: hash each char + position into vocab.
        std::vector<int64_t> ids;
        ids.reserve(target_len);
        std::seed_seq ss{seed, static_cast<uint64_t>(i)};
        std::mt19937_64 trng(ss);
        std::uniform_int_distribution<int> vocab_dist(10, vocab_size - 1);
        // Pad / truncate to target_len.
        for (int t = 0; t < target_len; ++t) {
            ids.push_back(static_cast<int64_t>(vocab_dist(trng)));
        }
        p.token_ids = std::move(ids);
        out.push_back(std::move(p));
    }
    return out;
}

std::vector<BenchPrompt> load_or_synthesize(const std::string& json_path,
                                            int count, int vocab_size,
                                            uint64_t seed) {
    auto prompts = load_prompts_from_json(json_path, count);
    if (!prompts.empty()) {
        std::fprintf(stderr, "[dataset] loaded %zu prompts from %s\n",
                     prompts.size(), json_path.c_str());
        return prompts;
    }
    std::fprintf(stderr,
        "[dataset] no JSON at %s; generating %d synthetic prompts "
        "(vocab_size=%d, seed=%llu)\n",
        json_path.c_str(), count, vocab_size,
        static_cast<unsigned long long>(seed));
    return make_synthetic_prompts(count, vocab_size, seed);
}

}  // namespace mini_infer