/**
 * test_paged_attention — Week 5 PagedAttention correctness vs. naive.
 *
 * Tests:
 *  [1]  PagedAttention output matches naive output (single seq, all
 *       positions in cache).
 *  [2]  PagedAttention handles arbitrary block_table layouts (e.g.
 *       random-order physical blocks; same logical sequence should
 *       yield the same output).
 *  [3]  PagedAttention causal-masked prefill output matches naive
 *       causal-masked prefill output for S_q > 1.
 *  [4]  PagedKVCache + block_table routing works for 4 sequences of
 *       different lengths.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kernels/naive_attn_kernel.cuh"
#include "kernels/paged_attn_kernel.cuh"
#include "scheduler/paged_kv_cache.h"

using mini_infer::BlockAllocator;
using mini_infer::PagedKVCache;
namespace kn = mini_infer::kernels;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

static void cuda_check_(cudaError_t e, const char* expr, const char* f, int l) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s at %s:%d : %s\n",
                     expr, f, l, cudaGetErrorString(e));
        std::exit(1);
    }
}
#define MI_CUDA_CHECK(expr) cuda_check_((expr), #expr, __FILE__, __LINE__)

// Fill a host buffer with deterministic-but-unique values per element.
static void fill_seq(__half* dst, int n, float base, float step) {
    for (int i = 0; i < n; ++i) {
        dst[i] = __float2half(base + step * static_cast<float>(i));
    }
}

static bool nearly_equal(const std::vector<float>& a, const std::vector<float>& b,
                          float rtol = 5e-3f, float atol = 5e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float ref  = std::fabs(b[i]);
        if (diff > atol + rtol * ref) return false;
    }
    return true;
}

static std::vector<float> to_host_fp32(const __half* dev, int n) {
    std::vector<__half> tmp(n);
    MI_CUDA_CHECK(cudaMemcpy(tmp.data(), dev, n * sizeof(__half),
                             cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = __half2float(tmp[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Build a "logical" K/V cache (continuous layout, like the W4 naive path).
// We use this as the reference answer.
// ---------------------------------------------------------------------------
struct NaiveCtx {
    int B = 1;
    int H_q = 4;
    int H_kv = 2;
    int num_kv_groups;
    int head_dim = 128;
    int seq_len;
    int S_q;
    bool is_prefill;
    __half* Q_dev;       // [B, S_q, H_q, head_dim]
    __half* K_dev;       // [B, seq_len, H_kv, head_dim]
    __half* V_dev;
    __half* out_dev;     // [B, S_q, H_q, head_dim]
    float scale;

    NaiveCtx(int S_q_, int seq_len_, bool prefill)
        : S_q(S_q_), seq_len(seq_len_), is_prefill(prefill) {
        num_kv_groups = H_q / H_kv;
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int qsz = B * S_q * H_q * head_dim;
        const int ksz = B * seq_len * H_kv * head_dim;
        MI_CUDA_CHECK(cudaMalloc(&Q_dev, qsz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&K_dev, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&V_dev, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&out_dev, qsz * sizeof(__half)));
    }
    ~NaiveCtx() {
        cudaFree(Q_dev); cudaFree(K_dev); cudaFree(V_dev); cudaFree(out_dev);
    }

    // Fill K/V with deterministic values, Q with random-ish values.
    void fill() {
        const int qsz = B * S_q * H_q * head_dim;
        const int ksz = B * seq_len * H_kv * head_dim;
        std::vector<__half> qh(qsz), kh(ksz), vh(ksz);
        fill_seq(qh.data(), qsz, 0.01f, 0.001f);
        fill_seq(kh.data(), ksz, 0.10f, 0.002f);
        fill_seq(vh.data(), ksz, 0.20f, 0.003f);
        MI_CUDA_CHECK(cudaMemcpy(Q_dev, qh.data(), qsz * sizeof(__half),
                                 cudaMemcpyHostToDevice));
        MI_CUDA_CHECK(cudaMemcpy(K_dev, kh.data(), ksz * sizeof(__half),
                                 cudaMemcpyHostToDevice));
        MI_CUDA_CHECK(cudaMemcpy(V_dev, vh.data(), ksz * sizeof(__half),
                                 cudaMemcpyHostToDevice));
    }

    std::vector<float> run_naive() {
        kn::launch_naive_attn(
            Q_dev, K_dev, V_dev, out_dev,
            B, S_q, seq_len,
            H_q, H_kv, head_dim,
            num_kv_groups, scale, is_prefill ? 1 : 0, /*stream=*/0);
        MI_CUDA_CHECK(cudaDeviceSynchronize());
        return to_host_fp32(out_dev, B * S_q * H_q * head_dim);
    }
};

// ---------------------------------------------------------------------------
// Build the equivalent PagedAttention context. We allocate a PagedKVCache
// (1 layer) and scatter the same K/V into it through a (single) block table.
// ---------------------------------------------------------------------------
struct PagedCtx {
    int B = 1;
    int H_q = 4;
    int H_kv = 2;
    int num_kv_groups;
    int head_dim = 128;
    int block_size = 16;
    int seq_len;
    int S_q;
    int num_blocks_pool;
    int max_blocks_per_seq;
    __half* Q_dev;
    __half* out_dev;
    __half* K_cache;     // [num_layers, num_blocks_pool, H_kv, block_size, head_dim]
    __half* V_cache;
    int* block_table_dev;        // [B, max_blocks_per_seq]
    int* num_blocks_used_dev;    // [B]
    int* seq_len_dev;            // [B]
    float scale;
    bool is_prefill;

    PagedCtx(int S_q_, int seq_len_, bool prefill)
        : S_q(S_q_), seq_len(seq_len_), is_prefill(prefill) {
        num_kv_groups = H_q / H_kv;
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        // Allocate enough physical blocks for the worst case.
        num_blocks_pool = (seq_len + block_size - 1) / block_size + 4;
        max_blocks_per_seq = num_blocks_pool;
        const int qsz = B * S_q * H_q * head_dim;
        const int ksz = 1 * num_blocks_pool * H_kv * block_size * head_dim;
        MI_CUDA_CHECK(cudaMalloc(&Q_dev, qsz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&out_dev, qsz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&K_cache, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&V_cache, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&block_table_dev, max_blocks_per_seq * sizeof(int)));
        MI_CUDA_CHECK(cudaMalloc(&num_blocks_used_dev, sizeof(int)));
        MI_CUDA_CHECK(cudaMalloc(&seq_len_dev, sizeof(int)));
    }
    ~PagedCtx() {
        cudaFree(Q_dev); cudaFree(out_dev);
        cudaFree(K_cache); cudaFree(V_cache);
        cudaFree(block_table_dev); cudaFree(num_blocks_used_dev); cudaFree(seq_len_dev);
    }

    // Build K_cache / V_cache with logical [0, seq_len) in the order
    // specified by block_table (so we can test both sequential and
    // shuffled mappings). Q is copied through.
    void fill_from(const NaiveCtx& n, const std::vector<int>& block_table) {
        const int qsz = B * S_q * H_q * head_dim;
        MI_CUDA_CHECK(cudaMemcpy(Q_dev, n.Q_dev, qsz * sizeof(__half),
                                 cudaMemcpyDeviceToDevice));
        // Zero K_cache and V_cache, then write token s into
        // block_table[s/BS] at offset s%BS for each kv_head.
        const int ksz = num_blocks_pool * H_kv * block_size * head_dim;
        MI_CUDA_CHECK(cudaMemset(K_cache, 0, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMemset(V_cache, 0, ksz * sizeof(__half)));
        // Per-token copy on host then upload. seq_len <= 128 in tests so
        // the cost is negligible.
        const int ksz_logical = B * seq_len * H_kv * head_dim;
        std::vector<__half> k_logical(ksz_logical), v_logical(ksz_logical);
        MI_CUDA_CHECK(cudaMemcpy(k_logical.data(), n.K_dev,
                                 ksz_logical * sizeof(__half),
                                 cudaMemcpyDeviceToHost));
        MI_CUDA_CHECK(cudaMemcpy(v_logical.data(), n.V_dev,
                                 ksz_logical * sizeof(__half),
                                 cudaMemcpyDeviceToHost));
        std::vector<__half> k_phys(ksz), v_phys(ksz);
        for (int s = 0; s < seq_len; ++s) {
            const int phys = block_table[s / block_size];
            const int off  = s % block_size;
            for (int h = 0; h < H_kv; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    const int dst = ((phys * H_kv + h) * block_size + off) * head_dim + d;
                    const int src = (s * H_kv + h) * head_dim + d;
                    k_phys[dst] = k_logical[src];
                    v_phys[dst] = v_logical[src];
                }
            }
        }
        MI_CUDA_CHECK(cudaMemcpy(K_cache, k_phys.data(), ksz * sizeof(__half),
                                 cudaMemcpyHostToDevice));
        MI_CUDA_CHECK(cudaMemcpy(V_cache, v_phys.data(), ksz * sizeof(__half),
                                 cudaMemcpyHostToDevice));
        // Upload metadata.
        std::vector<int> bt_h(max_blocks_per_seq, 0);
        for (size_t i = 0; i < block_table.size(); ++i) bt_h[i] = block_table[i];
        MI_CUDA_CHECK(cudaMemcpy(block_table_dev, bt_h.data(),
                                 max_blocks_per_seq * sizeof(int),
                                 cudaMemcpyHostToDevice));
        const int n_blocks_used = (seq_len + block_size - 1) / block_size;
        MI_CUDA_CHECK(cudaMemcpy(num_blocks_used_dev, &n_blocks_used,
                                 sizeof(int), cudaMemcpyHostToDevice));
        MI_CUDA_CHECK(cudaMemcpy(seq_len_dev, &seq_len, sizeof(int),
                                 cudaMemcpyHostToDevice));
    }

    std::vector<float> run_paged() {
        kn::launch_paged_attn(
            Q_dev, K_cache, V_cache,
            block_table_dev, num_blocks_used_dev, seq_len_dev,
            B, S_q, H_q, H_kv, head_dim,
            num_kv_groups, max_blocks_per_seq,
            /*layer=*/0, num_blocks_pool, scale, is_prefill ? 1 : 0,
            out_dev, /*stream=*/0);
        MI_CUDA_CHECK(cudaDeviceSynchronize());
        return to_host_fp32(out_dev, B * S_q * H_q * head_dim);
    }
};

int main() {
    std::printf("mini-infer :: paged_attention test\n");
    std::printf("---------------------------------\n");
    int prev_failures = 0;
    auto report = [&](const char* label) {
        const int diff = g_failures - prev_failures;
        std::printf(diff == 0 ? "ok\n" : "FAIL (%d EXPECT)\n", diff);
        prev_failures = g_failures;
    };

    // ------------------------------------------------------------------
    // [1] Decode (S_q=1): paged vs naive, sequential block_table
    // ------------------------------------------------------------------
    {
        std::printf("[1] decode (S_q=1), sequential block_table ... ");
        const int seq_len = 37;
        NaiveCtx naive(1, seq_len, /*prefill=*/false);
        naive.fill();
        const auto ref = naive.run_naive();

        PagedCtx paged(1, seq_len, /*prefill=*/false);
        std::vector<int> bt;
        const int nblocks = (seq_len + 15) / 16;
        for (int i = 0; i < nblocks; ++i) bt.push_back(i);
        paged.fill_from(naive, bt);
        const auto got = paged.run_paged();
        EXPECT(nearly_equal(got, ref), "paged decode matches naive");
        report("[1]");
    }

    // ------------------------------------------------------------------
    // [2] Decode (S_q=1) with shuffled block_table
    // ------------------------------------------------------------------
    {
        std::printf("[2] decode, shuffled block_table ... ");
        const int seq_len = 50;
        NaiveCtx naive(1, seq_len, false);
        naive.fill();
        const auto ref = naive.run_naive();

        PagedCtx paged(1, seq_len, false);
        const int nblocks = (seq_len + 15) / 16;
        std::vector<int> bt(nblocks);
        std::iota(bt.begin(), bt.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(bt.begin(), bt.end(), rng);
        std::printf("(perm: ");
        for (int b : bt) std::printf("%d ", b);
        std::printf(") ");
        paged.fill_from(naive, bt);
        const auto got = paged.run_paged();
        EXPECT(nearly_equal(got, ref), "shuffled block_table matches naive");
        report("[2]");
    }

    // ------------------------------------------------------------------
    // [3] Prefill (S_q > 1, causal)
    // ------------------------------------------------------------------
    {
        std::printf("[3] prefill (S_q=8), causal mask ... ");
        const int seq_len = 8;
        NaiveCtx naive(seq_len, seq_len, /*prefill=*/true);
        naive.fill();
        const auto ref = naive.run_naive();

        PagedCtx paged(seq_len, seq_len, true);
        std::vector<int> bt;
        for (int i = 0; i < (seq_len + 15) / 16; ++i) bt.push_back(i);
        paged.fill_from(naive, bt);
        const auto got = paged.run_paged();
        EXPECT(nearly_equal(got, ref), "paged prefill matches naive");
        report("[3]");
    }

    // ------------------------------------------------------------------
    // [4] PagedKVCache + block table routing for multiple sequences.
    // ------------------------------------------------------------------
    {
        std::printf("[4] PagedKVCache multi-sequence routing ... ");
        // 4 sequences, different lengths, different block tables.
        std::vector<int> seq_lens = {10, 22, 5, 17};
        const int n_seq = static_cast<int>(seq_lens.size());
        const int H_q_unused __attribute__((unused)) = 2;
        const int H_kv = 2, head_dim = 32;
        PagedKVCache cache(/*num_blocks=*/32, /*num_layers=*/1,
                           H_kv, head_dim, /*max_blocks_per_seq=*/8,
                           /*device=*/0);
        std::vector<int> sids;
        std::vector<std::vector<__half>> all_naive_k(n_seq);
        std::vector<std::vector<__half>> all_naive_v(n_seq);
        for (int i = 0; i < n_seq; ++i) {
            cache.create_sequence(1000 + i);
            for (int t = 0; t < seq_lens[i]; ++t) {
                int pos = cache.append_token(1000 + i);
                EXPECT(pos >= 0, "append_token ok");
            }
            sids.push_back(1000 + i);
            const int sl = seq_lens[i];
            const int sz = H_kv * head_dim;
            all_naive_k[i].resize(sl * sz);
            all_naive_v[i].resize(sl * sz);
            fill_seq(all_naive_k[i].data(), sl * sz, 0.5f + 0.1f * i, 0.001f);
            fill_seq(all_naive_v[i].data(), sl * sz, 1.0f + 0.1f * i, 0.002f);
        }
        // For each sequence, write the expected K/V into the cache and
        // then read it back via k_ptr_for / v_ptr_for.
        for (int i = 0; i < n_seq; ++i) {
            const int sl = seq_lens[i];
            const int sz = H_kv * head_dim;
            for (int t = 0; t < sl; ++t) {
                auto* kptr = static_cast<__half*>(cache.k_ptr_for(sids[i], 0, t));
                auto* vptr = static_cast<__half*>(cache.v_ptr_for(sids[i], 0, t));
                MI_CUDA_CHECK(cudaMemcpy(kptr, all_naive_k[i].data() + t * sz,
                                         sz * sizeof(__half),
                                         cudaMemcpyHostToDevice));
                MI_CUDA_CHECK(cudaMemcpy(vptr, all_naive_v[i].data() + t * sz,
                                         sz * sizeof(__half),
                                         cudaMemcpyHostToDevice));
            }
        }
        // Read back and compare.
        for (int i = 0; i < n_seq; ++i) {
            const int sl = seq_lens[i];
            const int sz = H_kv * head_dim;
            for (int t = 0; t < sl; ++t) {
                auto* kptr = static_cast<__half*>(cache.k_ptr_for(sids[i], 0, t));
                auto* vptr = static_cast<__half*>(cache.v_ptr_for(sids[i], 0, t));
                std::vector<__half> kh(sz), vh(sz);
                MI_CUDA_CHECK(cudaMemcpy(kh.data(), kptr, sz * sizeof(__half),
                                         cudaMemcpyDeviceToHost));
                MI_CUDA_CHECK(cudaMemcpy(vh.data(), vptr, sz * sizeof(__half),
                                         cudaMemcpyDeviceToHost));
                for (int j = 0; j < sz; ++j) {
                    EXPECT(__half2float(kh[j]) == __half2float(all_naive_k[i][t * sz + j]),
                           "K mismatch");
                    EXPECT(__half2float(vh[j]) == __half2float(all_naive_v[i][t * sz + j]),
                           "V mismatch");
                }
            }
        }
        for (int sid : sids) cache.destroy_sequence(sid);
        report("[4]");
    }

    // ------------------------------------------------------------------
    // [5] Real-shape test: Qwen2.5-1.5B dims, single block, all positions
    //     within BLOCK_SIZE so the partial-block edge case doesn't trigger.
    // ------------------------------------------------------------------
    {
        std::printf("[5] real-shape (Qwen2.5-1.5B dims), single block ... ");
        const int H_q_real = 12, H_kv_real = 2, head_dim_real = 128;
        const int seq_len_local = 11;
        NaiveCtx naive(1, seq_len_local, false);
        naive.H_q = H_q_real; naive.H_kv = H_kv_real; naive.head_dim = head_dim_real;
        naive.num_kv_groups = H_q_real / H_kv_real;
        naive.scale = 1.0f / std::sqrt(static_cast<float>(head_dim_real));
        const int qsz = naive.B * 1 * H_q_real * head_dim_real;
        const int ksz = naive.B * seq_len_local * H_kv_real * head_dim_real;
        __half *Q2, *K2, *V2, *O2;
        MI_CUDA_CHECK(cudaMalloc(&Q2, qsz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&K2, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&V2, ksz * sizeof(__half)));
        MI_CUDA_CHECK(cudaMalloc(&O2, qsz * sizeof(__half)));
        naive.Q_dev = Q2; naive.K_dev = K2; naive.V_dev = V2; naive.out_dev = O2;
        naive.fill();
        const auto ref = naive.run_naive();

        PagedCtx paged(1, seq_len_local, false);
        paged.H_q = H_q_real; paged.H_kv = H_kv_real; paged.head_dim = head_dim_real;
        paged.num_kv_groups = H_q_real / H_kv_real;
        paged.scale = naive.scale;
        std::vector<int> bt = {0};
        paged.Q_dev = Q2;        // share the same Q buffer
        paged.out_dev = O2;      // share the same output buffer
        paged.fill_from(naive, bt);
        const auto got = paged.run_paged();

        float max_d = 0.0f; int bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = std::fabs(ref[i] - got[i]);
            if (d > max_d) max_d = d;
            if (d > 5e-3f && d / (std::fabs(ref[i]) + 1e-6f) > 5e-3f) ++bad;
        }
        std::printf("(max_diff=%.4f bad=%d) ", max_d, bad);
        EXPECT(bad == 0, "real-shape decode matches naive");

        cudaFree(Q2); cudaFree(K2); cudaFree(V2); cudaFree(O2);
        report("[5]");
    }

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d test(s) failed.\n", g_failures);
    return 1;
}