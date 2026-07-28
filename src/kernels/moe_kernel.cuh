#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace mini_infer {
namespace kernels {

/**
 * Fused top-K expert selection + softmax weighting.
 *
 * For each token (row), selects the top-K experts from router_logits,
 * computes softmax weights over the selected logits, and outputs:
 *   - expert_weights: [B, K] float — softmax weights for selected experts
 *   - expert_indices: [B, K] int32 — indices of selected experts
 *
 * @param router_logits  [B, num_experts] FP16 — router gate output
 * @param expert_weights [B, top_k] float — output weights (softmax over selected)
 * @param expert_indices [B, top_k] int32 — output expert indices
 * @param batch_size     B
 * @param num_experts    E
 * @param top_k          K (num_experts_per_tok)
 * @param stream         CUDA stream
 */
void launch_moe_topk_select(const __half* router_logits,
                            float* expert_weights,
                            int* expert_indices,
                            int batch_size,
                            int num_experts,
                            int top_k,
                            cudaStream_t stream);

/**
 * Scatter-add: accumulate expert outputs into the final MoE output.
 *
 * For each token b, if expert_indices[b, k] == current_expert:
 *   output[b, :] += expert_weights[b, k] * expert_output[b, :]
 *
 * The caller runs one expert at a time on all tokens, then calls this
 * kernel to accumulate only for tokens that selected this expert.
 *
 * @param output         [B, hidden] FP16 — accumulated output (pre-zeroed)
 * @param expert_output  [B, hidden] FP16 — current expert's output for all B tokens
 * @param expert_weights [B, K] float — weights for selected experts
 * @param expert_indices [B, K] int32 — indices of selected experts
 * @param expert_k       which of the K slots to check (0..K-1)
 * @param current_expert the expert index being processed
 * @param batch_size     B
 * @param hidden         H
 * @param stream         CUDA stream
 */
void launch_moe_scatter_add(__half* output,
                            const __half* expert_output,
                            const float* expert_weights,
                            const int* expert_indices,
                            int expert_k,
                            int current_expert,
                            int batch_size,
                            int hidden,
                            int top_k,
                            cudaStream_t stream);

}  // namespace kernels
}  // namespace mini_infer
