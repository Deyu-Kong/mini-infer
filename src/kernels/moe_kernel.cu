#include "kernels/moe_kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace mini_infer {
namespace kernels {

__global__ void moe_topk_select_kernel(const __half* __restrict__ router_logits,
                                       float* __restrict__ expert_weights,
                                       int* __restrict__ expert_indices,
                                       int batch_size,
                                       int num_experts,
                                       int top_k) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const __half* logits_row = router_logits + b * num_experts;
    float* weights_row = expert_weights + b * top_k;
    int* indices_row = expert_indices + b * top_k;

    // Simple top-K selection (K is typically small, e.g., 2 or 8)
    // For each of K slots, find the max logit, record it, then mask it out
    float selected_logits[32];  // max top_k we support
    for (int k = 0; k < top_k && k < 32; ++k) {
        float max_logit = -FLT_MAX;
        int max_idx = -1;
        for (int e = 0; e < num_experts; ++e) {
            float logit = __half2float(logits_row[e]);
            bool already_selected = false;
            for (int prev = 0; prev < k; ++prev) {
                if (indices_row[prev] == e) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && logit > max_logit) {
                max_logit = logit;
                max_idx = e;
            }
        }
        indices_row[k] = max_idx;
        selected_logits[k] = max_logit;
    }

    // Softmax over selected logits
    float max_logit = -FLT_MAX;
    for (int k = 0; k < top_k; ++k) {
        if (selected_logits[k] > max_logit) max_logit = selected_logits[k];
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float exp_val = expf(selected_logits[k] - max_logit);
        weights_row[k] = exp_val;
        sum_exp += exp_val;
    }
    for (int k = 0; k < top_k; ++k) {
        weights_row[k] /= sum_exp;
    }
}

__global__ void moe_scatter_add_kernel(__half* __restrict__ output,
                                       const __half* __restrict__ expert_output,
                                       const float* __restrict__ expert_weights,
                                       const int* __restrict__ expert_indices,
                                       int expert_k,
                                       int current_expert,
                                       int batch_size,
                                       int hidden,
                                       int top_k_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden;
    if (idx >= total) return;

    int b = idx / hidden;
    int h = idx % hidden;

    // Check if this token selected the current expert in slot expert_k
    int selected_expert = expert_indices[b * top_k_stride + expert_k];
    if (selected_expert != current_expert) return;

    float weight = expert_weights[b * top_k_stride + expert_k];

    // Accumulate: output[b, h] += weight * expert_output[b, h]
    float out_val = __half2float(output[idx]);
    float expert_val = __half2float(expert_output[idx]);
    out_val += weight * expert_val;
    output[idx] = __float2half(out_val);
}

void launch_moe_topk_select(const __half* router_logits,
                            float* expert_weights,
                            int* expert_indices,
                            int batch_size,
                            int num_experts,
                            int top_k,
                            cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    moe_topk_select_kernel<<<blocks, threads, 0, stream>>>(
        router_logits, expert_weights, expert_indices,
        batch_size, num_experts, top_k);
}

void launch_moe_scatter_add(__half* output,
                            const __half* expert_output,
                            const float* expert_weights,
                            const int* expert_indices,
                            int expert_k,
                            int current_expert,
                            int batch_size,
                            int hidden,
                            int top_k,
                            cudaStream_t stream) {
    int total = batch_size * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    moe_scatter_add_kernel<<<blocks, threads, 0, stream>>>(
        output, expert_output, expert_weights, expert_indices,
        expert_k, current_expert, batch_size, hidden, top_k);
}

__global__ void moe_shared_expert_add_kernel(__half* __restrict__ output,
                                              const __half* __restrict__ shared_output,
                                              const __half* __restrict__ gate_logits,
                                              int batch_size,
                                              int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden;
    if (idx >= total) return;

    int b = idx / hidden;

    // Compute sigmoid(gate_logits[b])
    float gate_val = __half2float(gate_logits[b]);
    float gate_sigmoid = 1.0f / (1.0f + expf(-gate_val));

    // output[b, h] += sigmoid(gate) * shared_output[b, h]
    float out_val = __half2float(output[idx]);
    float shared_val = __half2float(shared_output[idx]);
    out_val += gate_sigmoid * shared_val;
    output[idx] = __float2half(out_val);
}

void launch_moe_shared_expert_add(__half* output,
                                   const __half* shared_output,
                                   const __half* gate_logits,
                                   int batch_size,
                                   int hidden,
                                   cudaStream_t stream) {
    int total = batch_size * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    moe_shared_expert_add_kernel<<<blocks, threads, 0, stream>>>(
        output, shared_output, gate_logits, batch_size, hidden);
}

}  // namespace kernels
}  // namespace mini_infer
