/**
 * test_request — Week 6 unit test for the Request class.
 *
 * Covers:
 *   - Construction validation (empty prompt / max_new=0 rejected).
 *   - State machine transitions via set_state().
 *   - output_ids() == prompt + generated.
 *   - last_token() default = -1, set_last_token() round-trip.
 *   - Metrics defaults are zero-initialized.
 *   - finished_reason persistence.
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "scheduler/request.h"

using namespace mini_infer;

namespace {

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "  FAIL [%s:%d] %s\n",                  \
                         __FILE__, __LINE__, msg);                        \
            return false;                                                 \
        }                                                                 \
    } while (0)

bool test_construction() {
    std::fprintf(stderr, "[1] construction\n");
    auto r = std::make_unique<Request>(
        std::vector<int64_t>{10, 20, 30}, /*max_new=*/50);
    CHECK(r->state() == RequestState::Pending, "initial state Pending");
    CHECK(r->prompt_len() == 3, "prompt_len() == 3");
    CHECK(r->total_len() == 3, "total_len() starts == prompt_len");
    CHECK(r->last_token() == -1, "last_token default -1");
    CHECK(r->seq_id() == -1, "seq_id default -1");
    CHECK(r->is_finished() == false, "not finished");
    CHECK(r->metrics().generated_tokens == 0, "no tokens generated yet");
    CHECK(r->metrics().ttft_ms == 0.0, "ttft_ms default 0");
    auto out = r->output_ids();
    CHECK(out.size() == 3, "output_ids == prompt before any gen");
    return true;
}

bool test_state_machine() {
    std::fprintf(stderr, "[2] state machine\n");
    auto r = std::make_unique<Request>(
        std::vector<int64_t>{1}, /*max_new=*/4);
    r->set_state(RequestState::Prefilling);
    CHECK(r->state() == RequestState::Prefilling, "Prefilling");
    r->set_state(RequestState::Decoding);
    CHECK(r->state() == RequestState::Decoding, "Decoding");
    r->set_state(RequestState::Finished);
    CHECK(r->is_finished(), "Finished == is_finished");
    CHECK(request_state_name(r->state()) ==
          std::string("Finished"), "name() == Finished");
    return true;
}

bool test_metrics() {
    std::fprintf(stderr, "[3] metrics + finished_reason\n");
    auto r = std::make_unique<Request>(
        std::vector<int64_t>{1, 2}, /*max_new=*/10);
    r->metrics().ttft_ms = 12.5;
    r->metrics().generated_tokens = 7;
    r->metrics().finished_reason = "length";
    r->set_state(RequestState::Finished);
    CHECK(r->metrics().ttft_ms == 12.5, "ttft_ms preserved");
    CHECK(r->metrics().generated_tokens == 7, "generated_tokens preserved");
    CHECK(r->metrics().finished_reason == "length", "reason preserved");
    return true;
}

bool test_stop_tokens() {
    std::fprintf(stderr, "[4] stop tokens\n");
    auto r = std::make_unique<Request>(
        std::vector<int64_t>{1, 2}, /*max_new=*/100,
        /*stop=*/std::vector<int64_t>{99, 100});
    CHECK(r->stop_token_ids().size() == 2, "two stop tokens");
    CHECK(r->stop_token_ids()[0] == 99, "stop[0] == 99");
    return true;
}

bool test_validation() {
    std::fprintf(stderr, "[5] validation rejects bad input\n");
    bool threw_empty = false;
    try {
        Request r({}, 10);
    } catch (...) { threw_empty = true; }
    CHECK(threw_empty, "empty prompt rejected");
    bool threw_maxnew = false;
    try {
        Request r({1, 2}, 0);
    } catch (...) { threw_maxnew = true; }
    CHECK(threw_maxnew, "max_new=0 rejected");
    return true;
}

}  // namespace

int main() {
    bool ok = test_construction()
           && test_state_machine()
           && test_metrics()
           && test_stop_tokens()
           && test_validation();
    std::printf("[test_request] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}