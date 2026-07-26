#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mini_infer {

/**
 * Tokenizer — thin wrapper around HuggingFace `tokenizers` (Python).
 *
 * We avoid linking the C++ binding directly because that would require
 * shipping a matching libtokenizers.so. Instead, this class shells out
 * to a tiny Python helper script (`scripts/tokenize_helper.py`) which
 * loads the tokenizer once, then handles encode/decode in two file-based
 * round-trips.
 *
 * The Python helper writes a tmp file with the input (one item per line),
 * then writes the output tokens / text back to another tmp file. This is
 * an order of magnitude faster than spawning one Python process per
 * encode/decode.
 */
class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path,
                       const std::string& python_exe = "python3");

    // encode(text) -> token ids.
    std::vector<int64_t> encode(const std::string& text) const;

    // decode([ids]) -> text.
    std::string decode(const std::vector<int64_t>& ids) const;

    // Common special tokens (best-effort; absent if not in tokenizer).
    int64_t eos_token_id() const { return eos_id_; }
    int64_t bos_token_id() const { return bos_id_; }
    int64_t im_start_id()  const { return im_start_id_; }
    int64_t im_end_id()    const { return im_end_id_; }

private:
    std::string helper_path_;
    std::string tokenizer_path_;
    std::string python_exe_;
    int64_t eos_id_      = -1;
    int64_t bos_id_      = -1;
    int64_t im_start_id_ = -1;
    int64_t im_end_id_   = -1;
};

}  // namespace mini_infer