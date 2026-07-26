#include "scheduler/request.h"

#include <stdexcept>
#include <utility>

namespace mini_infer {

const char* request_state_name(RequestState s) {
    switch (s) {
        case RequestState::Pending:    return "Pending";
        case RequestState::Prefilling: return "Prefilling";
        case RequestState::Decoding:   return "Decoding";
        case RequestState::Finished:   return "Finished";
    }
    return "?";
}

Request::Request(std::vector<int64_t> prompt_ids,
                 int max_new_tokens,
                 std::vector<int64_t> stop_token_ids,
                 SamplingParams sampling)
    : prompt_ids_(std::move(prompt_ids)),
      stop_token_ids_(std::move(stop_token_ids)),
      sampling_(sampling),
      max_new_tokens_(max_new_tokens) {
    if (prompt_ids_.empty()) {
        throw std::runtime_error("Request: empty prompt");
    }
    if (max_new_tokens <= 0) {
        throw std::runtime_error("Request: max_new_tokens must be > 0");
    }
}

std::vector<int64_t> Request::output_ids() const {
    std::vector<int64_t> out;
    out.reserve(prompt_ids_.size() + generated_ids_.size());
    out.insert(out.end(), prompt_ids_.begin(), prompt_ids_.end());
    out.insert(out.end(), generated_ids_.begin(), generated_ids_.end());
    return out;
}

}  // namespace mini_infer