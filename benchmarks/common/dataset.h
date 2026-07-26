#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_infer {

/**
 * A single benchmark prompt.
 *   - prompt_text : human-readable source (used for reporting).
 *   - token_ids   : pre-tokenized int64 ids (size = prompt_len).
 *
 * We store token_ids rather than the text so the benchmark binary doesn't
 * need to call the tokenizer inside the timed loop.
 */
struct BenchPrompt {
    std::string            prompt_text;
    std::vector<int64_t>   token_ids;
    int                    prompt_len() const { return static_cast<int>(token_ids.size()); }
};

/**
 * Load prompts from a JSON file with one of the following shapes:
 *
 *   1. ShareGPT-like (list of objects with a `conversations` field):
 *      [{"conversations": [{"from":"human","value":"..."}, ...]}, ...]
 *
 *   2. Plain list-of-strings:
 *      ["prompt 1", "prompt 2", ...]
 *
 * Returns at most `max_count` prompts.
 *
 * If `path` is empty or the file cannot be opened, returns an empty
 * vector — callers should fall back to synthetic generation.
 */
std::vector<BenchPrompt> load_prompts_from_json(const std::string& path,
                                                int max_count);

/**
 * Generate `count` synthetic prompts of varying lengths, drawn from a
 * fixed template bank. Token ids are produced by hashing the template
 * index into the vocabulary range [10, vocab_size); this means generated
 * prompts have no semantic content but exercise the full pipeline
 * (tokenize + prefill + decode) deterministically.
 *
 * Lengths are chosen from buckets:
 *   short  : [10, 32]      (≈ 1 sentence)
 *   medium : [64, 256]     (≈ 1 paragraph)
 *   long   : [512, 1024]   (≈ 1 page)
 * with weights 0.5 / 0.3 / 0.2 — matches the rough ShareGPT distribution.
 *
 * `vocab_size` is required so token ids stay in-range for the model.
 */
std::vector<BenchPrompt> make_synthetic_prompts(int count, int vocab_size,
                                               uint64_t seed = 42);

/**
 * Convenience: try JSON first, fall back to synthetic.
 */
std::vector<BenchPrompt> load_or_synthesize(const std::string& json_path,
                                            int count, int vocab_size,
                                            uint64_t seed = 42);

}  // namespace mini_infer