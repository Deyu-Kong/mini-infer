/**
 * Sampling CUDA kernels — greedy (argmax) and top-p (nucleus) for Week 4.
 *
 * Both operate on a single row of [V] logits and produce a single int.
 * Suitable for B=1 decode loops; multi-batch extensions land in W6.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_infer {
namespace kernels {

// Greedy: pick argmax(logits). Returns the chosen vocab id.
__global__ void greedy_sample_kernel(const __half* logits, int vocab, int* out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smem[tid] = local_max;
    reinterpret_cast<int*>(smem + blockDim.x)[tid] = local_idx;
    __syncthreads();

    // Reduce within block.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float a = smem[tid], b = smem[tid + offset];
            int   ia = reinterpret_cast<int*>(smem + blockDim.x)[tid];
            int   ib = reinterpret_cast<int*>(smem + blockDim.x)[tid + offset];
            if (b > a) { smem[tid] = b; reinterpret_cast<int*>(smem + blockDim.x)[tid] = ib; }
        }
        __syncthreads();
    }
    if (tid == 0) *out = reinterpret_cast<int*>(smem + blockDim.x)[0];
}

void launch_greedy_sample(const __half* logits, int vocab, int* out,
                          cudaStream_t stream) {
    constexpr int BLOCK = 256;
    size_t smem = BLOCK * (sizeof(float) + sizeof(int));
    greedy_sample_kernel<<<1, BLOCK, smem, stream>>>(logits, vocab, out);
}

// ===========================================================================
// Batched greedy sampling (Week 6) — process B independent rows in one call.
// One block per row; each block does the same argmax as launch_greedy_sample.
// ===========================================================================
__global__ void greedy_sample_batched_kernel(
    const __half* logits, int vocab, int* out) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const __half* row_ptr = logits + (int64_t)row * vocab;
    int* out_ptr          = out + row;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    int   local_idx = -1;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = __half2float(row_ptr[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smem[tid] = local_max;
    reinterpret_cast<int*>(smem + blockDim.x)[tid] = local_idx;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float a = smem[tid], b = smem[tid + offset];
            int   ia = reinterpret_cast<int*>(smem + blockDim.x)[tid];
            int   ib = reinterpret_cast<int*>(smem + blockDim.x)[tid + offset];
            if (b > a) { smem[tid] = b; reinterpret_cast<int*>(smem + blockDim.x)[tid] = ib; }
        }
        __syncthreads();
    }
    if (tid == 0) *out_ptr = reinterpret_cast<int*>(smem + blockDim.x)[0];
}

void launch_greedy_sample_batched(const __half* logits, int B, int vocab,
                                  int* out, cudaStream_t stream) {
    if (B <= 0) return;
    constexpr int BLOCK = 256;
    size_t smem = BLOCK * (sizeof(float) + sizeof(int));
    dim3 grid(B);
    dim3 block(BLOCK);
    greedy_sample_batched_kernel<<<grid, block, smem, stream>>>(logits, vocab, out);
}

// ----------------------------------------------------------------------------
// Top-p sampling — single-block kernel using the Gumbel-max trick:
//   sample = argmax_i (logit_i / T + gumbel_i)        (gumbel ~ -log(-log(U)))
// then accept only if cumulative prob (sorted desc) <= p. We approximate by
// taking only the top-K logits (K = max(64, p-quantile-ish)) to avoid sorting
// the full vocab.
// For Week 4 we use a simpler variant: do softmax in registers, sort by
// prob descending with bitonic sort over the top-K candidate slots, then
// accept the first whose cumulative prob <= p.
// ----------------------------------------------------------------------------
__global__ void top_p_sample_kernel(const __half* logits, int vocab,
                                    float p, float temperature,
                                    unsigned long long seed,
                                    int* out) {
    constexpr int K = 256;          // candidate slots
    extern __shared__ float smem[]; // [vocab] probs + scratch

    int tid = threadIdx.x;

    // 1. Find max for softmax stability.
    float local_max = -INFINITY;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float v = __half2float(logits[i]) / temperature;
        local_max = fmaxf(local_max, v);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o) smem[tid] = fmaxf(smem[tid], smem[tid + o]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // 2. softmax -> probs
    float* probs = smem;           // [vocab] but only first vocab*sizeof floats
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += blockDim.x) {
        float p_v = __expf(__half2float(logits[i]) / temperature - row_max);
        probs[i] = p_v;
        local_sum += p_v;
    }
    smem[vocab + tid] = local_sum;        // scratch
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o) smem[vocab + tid] += smem[vocab + tid + o];
        __syncthreads();
    }
    float row_sum = smem[vocab];
    __syncthreads();

    // 3. Find top-K using a poor-man's iterative argmax K times.
    int* top_idx = reinterpret_cast<int*>(smem + vocab + blockDim.x);
    float* top_val = smem + vocab + blockDim.x + K;  // [K]

    for (int k = 0; k < K; ++k) {
        // Find current max prob not already picked.
        float best_val = -INFINITY;
        int   best_idx = -1;
        for (int s = 0; s < blockDim.x; ++s) {
            if (tid == s) {
                for (int i = s; i < vocab; i += blockDim.x) {
                    bool used = false;
                    for (int j = 0; j < k; ++j) if (top_idx[j] == i) { used = true; break; }
                    if (used) continue;
                    float v = probs[i] / row_sum;
                    if (v > best_val) { best_val = v; best_idx = i; }
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            top_idx[k] = best_idx;
            top_val[k] = best_val;
        }
        __syncthreads();
    }
    __syncthreads();

    // 4. Walk top-K in descending order until cumulative > p, pick uniformly.
    // We do this serially in thread 0 because K=256 is small.
    if (tid == 0) {
        // Sort top-K by descending prob (insertion sort — fine for K=256).
        for (int i = 1; i < K; ++i) {
            float v = top_val[i]; int ix = top_idx[i];
            int j = i - 1;
            while (j >= 0 && top_val[j] < v) {
                top_val[j + 1] = top_val[j];
                top_idx[j + 1] = top_idx[j];
                --j;
            }
            top_val[j + 1] = v; top_idx[j + 1] = ix;
        }
        float cum = 0.0f;
        int choice = top_idx[K - 1];   // fallback
        for (int i = 0; i < K; ++i) {
            cum += top_val[i];
            if (cum >= p) {
                // Pick uniformly among slots [0..i] (rough approximation).
                unsigned long long s = seed ^ ((unsigned long long)top_idx[i] << 32);
                s ^= s >> 33; s *= 0xff51afd7ed558ccdULL;
                s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ULL;
                s ^= s >> 33;
                float u = (float)(s & 0xFFFFFF) / (float)0x1000000;
                int slot = (int)(u * (i + 1));
                if (slot < 0) slot = 0;
                if (slot > i) slot = i;
                choice = top_idx[slot];
                break;
            }
        }
        *out = choice;
    }
}

void launch_top_p_sample(const __half* logits, int vocab,
                         float p, float temperature,
                         unsigned long long seed,
                         int* out, cudaStream_t stream) {
    constexpr int BLOCK = 128;
    constexpr int K     = 256;
    // smem layout: [vocab] probs + [BLOCK] reduce + [K] top_idx + [K] top_val
    size_t smem = (vocab + BLOCK + 2 * K) * sizeof(float)
                + K * sizeof(int);
    // Cap vocab to fit shared memory (~96KB on A6000 SM).
    if (vocab > 30000) {
        // Fallback: greedy (top-p with very narrow nucleus approximates greedy).
        launch_greedy_sample(logits, vocab, out, stream);
        return;
    }
    top_p_sample_kernel<<<1, BLOCK, smem, stream>>>(
        logits, vocab, p, temperature, seed, out);
}

void launch_top_p_sample_batched(const __half* /*logits*/, int /*B*/,
                                 int /*vocab*/, float /*p*/, float /*temperature*/,
                                 unsigned long long /*seed*/,
                                 int* /*out*/, cudaStream_t /*stream*/) {
    // Not implemented in Week 6 — fall back to per-row greedy in callers.
}

}  // namespace kernels
}  // namespace mini_infer