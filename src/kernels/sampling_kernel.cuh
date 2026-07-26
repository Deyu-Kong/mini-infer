/**
 * Header declarations for the sampling kernels. The actual implementations
 * live in sampling_kernel.cu. Engine.cpp #includes the .cu file directly
 * (single-TU sampling is the simplest path in Week 4).
 *
 * Week 6: added batched greedy sample; top-p batched is a no-op stub for now.
 */
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Single-row greedy argmax.
void launch_greedy_sample(const __half* logits, int vocab, int* out,
                          cudaStream_t stream);

// Single-row top-p (nucleus) sample.
void launch_top_p_sample(const __half* logits, int vocab,
                         float p, float temperature,
                         unsigned long long seed,
                         int* out, cudaStream_t stream);

// Week 6: B-row greedy argmax. logits: [B, vocab] FP16 row-major;
// out: [B] int32, one chosen token per row.
void launch_greedy_sample_batched(const __half* logits, int B, int vocab,
                                  int* out, cudaStream_t stream);

// Week 6: B-row top-p. Currently a stub — callers must fall back to
// per-row launch_greedy_sample (or extend the .cu file).
void launch_top_p_sample_batched(const __half* logits, int B, int vocab,
                                 float p, float temperature,
                                 unsigned long long seed,
                                 int* out, cudaStream_t stream);

// Week 7: Speculative decoding accept/reject kernel.
//   target_logits : [vocab] FP16 — target model logits for this position
//   draft_logits  : [vocab] FP16 — draft model logits for this position
//   vocab         : vocabulary size
//   draft_token   : the token sampled by the draft model
//   draft_prob    : the draft model's probability for draft_token
//   seed          : curand seed
//   out_token     : [1] int32 — accepted token or corrected sample
//   out_accepted  : [1] int32 — 1 if accepted, 0 if rejected
void launch_spec_accept_reject(
    const __half* target_logits,
    const __half* draft_logits,
    int vocab,
    int draft_token,
    float draft_prob,
    unsigned long long seed,
    int* out_token,
    int* out_accepted,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer