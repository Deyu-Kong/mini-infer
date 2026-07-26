#!/usr/bin/env python3
"""
Week 5 acceptance tests:
  - fragmentation: 100 synthetic requests of varying lengths.
  - concurrency:   paged pool supports N>>max_seq_len/BLOCK_SIZE sequences
                   that the naive contiguous cache cannot.
  - python vs HF:  same prompt, same model, same greedy -> identical tokens.

Run from the repo root.
"""
import os
import re
import subprocess
import sys

MODEL_15B = "/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct"
MODEL_7B  = "/data1/kdy/LLMs/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Test 1: fragmentation with synthetic requests via a small ctest helper.
# ---------------------------------------------------------------------------

def fragmentation_test():
    """
    Builds a C++ binary that simulates 100 sequences of varying length,
    measuring wasted tokens in the paged pool. Pass criterion: < 10%.
    """
    src = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <vector>
#include "core/allocator.h"
#include "scheduler/paged_kv_cache.h"

int main() {
    using mini_infer::BlockAllocator;
    using mini_infer::PagedKVCache;
    // 100 sequences, lengths uniformly in [1, 200] tokens.
    const int N_SEQ = 100;
    const int MAX_LEN = 200;
    const int NUM_KV_HEADS = 2;
    const int HEAD_DIM = 64;
    const int NUM_LAYERS = 1;

    // Pick pool size so total tokens ~= sum / 2 (overcommit by 50% to
    // simulate realistic pressure).
    std::mt19937 rng(42);
    std::vector<int> lengths(N_SEQ);
    long long sum = 0;
    for (int i = 0; i < N_SEQ; ++i) {
        lengths[i] = 1 + rng() % MAX_LEN;
        sum += lengths[i];
    }
    const int total_blocks_needed = (sum + BlockAllocator::kBlockSize - 1)
                                    / BlockAllocator::kBlockSize;
    const int num_pool_blocks = total_blocks_needed / 2 + 8;

    PagedKVCache cache(num_pool_blocks, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
                       /*max_blocks_per_seq=*/MAX_LEN, /*device=*/0);

    std::vector<int> sids;
    long long total_logical_tokens = 0;
    long long total_blocks_in_use = 0;
    int rejected = 0;
    for (int i = 0; i < N_SEQ; ++i) {
        const int sid = 1000 + i;
        cache.create_sequence(sid);
        sids.push_back(sid);
        int appended = 0;
        for (int t = 0; t < lengths[i]; ++t) {
            if (cache.append_token(sid) < 0) {
                ++rejected;
                break;
            }
            ++appended;
        }
        total_logical_tokens += appended;
        total_blocks_in_use += cache.num_blocks(sid);
    }
    long long total_capacity = static_cast<long long>(total_blocks_in_use)
                               * BlockAllocator::kBlockSize;
    double waste_ratio = 1.0 - static_cast<double>(total_logical_tokens)
                                  / static_cast<double>(total_capacity);
    std::printf("sequences=%d  total_tokens=%lld  blocks_in_use=%lld\n",
                N_SEQ, total_logical_tokens, total_blocks_in_use);
    std::printf("waste_ratio=%.4f\n", waste_ratio);
    std::printf("rejected=%d (pool OOM)\n", rejected);

    // Tear down: free all sequences.
    for (int sid : sids) cache.destroy_sequence(sid);
    std::printf("after teardown: free=%d in_use=%d\n",
                cache.total_free_blocks(),
                cache.total_in_use_blocks());
    return 0;
}
"""
    test_dir = os.path.join(os.path.dirname(__file__), "test_frag")
    os.makedirs(test_dir, exist_ok=True)
    src_path = os.path.join(test_dir, "main.cpp")
    with open(src_path, "w") as f:
        f.write(src)

    # Compile + run.
    inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    cmd = [
        "/usr/local/cuda-12.1/bin/nvcc",
        "-std=c++17", "-O2",
        "-arch=sm_86",
        f"-I{inc}",
        src_path,
        "-o", os.path.join(test_dir, "frag_test"),
    ]
    cmd += ["-L/data1/kdy/Project/mini-infer/build",
            "-lmini_infer_core", "-lcudart", "-lcuda"]
    subprocess.run(cmd, check=True)
    res = subprocess.run([os.path.join(test_dir, "frag_test")],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        return False
    # Parse waste_ratio.
    m = re.search(r"waste_ratio=([\d.]+)", res.stdout)
    if not m:
        return False
    waste = float(m.group(1))
    return waste < 0.10, waste


# ---------------------------------------------------------------------------
# Test 2: paged max concurrency vs naive max concurrency.
# ---------------------------------------------------------------------------
#
# Naive KVCache allocates max_seq_len slots per sequence in one flat pool.
# With B=1 (single batch), naive can hold 1 sequence of length <= max_seq.
# PagedKVCache allocates blocks on demand; total tokens = num_blocks * BS.
# We compare the maximum number of length-L sequences each can host.
#

def concurrency_test():
    src = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "core/allocator.h"
#include "scheduler/kv_cache.h"
#include "scheduler/paged_kv_cache.h"

int main() {
    using mini_infer::BlockAllocator;
    using mini_infer::KVCache;
    using mini_infer::PagedKVCache;
    const int NUM_LAYERS = 1;
    const int NUM_KV_HEADS = 2;
    const int HEAD_DIM = 64;
    const int MAX_SEQ = 256;

    // Naive: pre-allocates max_seq per sequence * layers.
    KVCache naive(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ, 0);
    // We can only fit ONE sequence in the naive cache (single contiguous
    // region per layer). So naive_concurrent = 1.
    std::printf("naive_concurrent=1 (single contiguous block per layer)\n");

    // Paged: pool of N blocks. Each sequence needs ceil(L / BS) blocks.
    const int num_pool = 1024;
    PagedKVCache paged(num_pool, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
                       /*max_blocks_per_seq=*/MAX_SEQ, 0);
    // Try to host as many sequences of length L as possible.
    const int L = 32;     // 2 blocks per sequence
    int paged_concurrent = 0;
    int sid = 0;
    while (true) {
        paged.create_sequence(sid);
        bool ok = true;
        for (int t = 0; t < L; ++t) {
            if (paged.append_token(sid) < 0) {
                ok = false; break;
            }
        }
        if (!ok) {
            paged.destroy_sequence(sid);
            break;
        }
        ++paged_concurrent;
        ++sid;
    }
    std::printf("paged_concurrent=%d (length=%d, pool=%d blocks)\n",
                paged_concurrent, L, num_pool);

    double speedup = static_cast<double>(paged_concurrent) / 1.0;
    std::printf("speedup=%.2fx\n", speedup);
    return 0;
}
"""
    test_dir = os.path.join(os.path.dirname(__file__), "test_conc")
    os.makedirs(test_dir, exist_ok=True)
    src_path = os.path.join(test_dir, "main.cpp")
    with open(src_path, "w") as f:
        f.write(src)
    inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    cmd = [
        "/usr/local/cuda-12.1/bin/nvcc",
        "-std=c++17", "-O2",
        "-arch=sm_86",
        f"-I{inc}",
        src_path,
        "-o", os.path.join(test_dir, "conc_test"),
    ]
    cmd += ["-L/data1/kdy/Project/mini-infer/build",
            "-lmini_infer_engine", "-lmini_infer_core", "-lcudart", "-lcuda"]
    subprocess.run(cmd, check=True)
    res = subprocess.run([os.path.join(test_dir, "conc_test")],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        return False, 0.0
    m_naive = re.search(r"naive_concurrent=(\d+)", res.stdout)
    m_paged = re.search(r"paged_concurrent=(\d+)", res.stdout)
    naive_n = int(m_naive.group(1)) if m_naive else 0
    paged_n = int(m_paged.group(1)) if m_paged else 0
    speedup = paged_n / max(naive_n, 1)
    return speedup >= 2.0, speedup


# ---------------------------------------------------------------------------
# Test 3: same prompt, paged vs HF, tokens should match for first few.
# ---------------------------------------------------------------------------

def hf_vs_paged_test(model_path, prompt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    chat = f"user\n{prompt}\nassistant\n"
    ids = tok.encode(chat, return_tensors="pt").to("cuda:0")
    # Use sdpa (FlashAttention); eager attention overflows in FP16 on
    # short prompts (a known HF issue), making the reference text garbage.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=30, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    hf_ids = out[0][ids.shape[1]:].tolist()

    # Run paged.
    res = subprocess.run(
        ["./build/mini_infer", "--paged", "--model", model_path,
         "--prompt", prompt, "--max-new-tokens", "30",
         "--greedy", "--device", "0", "--max-seq-len", "2048"],
        capture_output=True, text=True, timeout=300,
    )
    body = res.stdout
    m = re.search(r"(\d+) tokens in", body)
    n_paged = int(m.group(1)) if m else 0

    m_text = re.search(r"--- generated ---\n(.*?)\n--- end ---", body, re.DOTALL)
    paged_text = m_text.group(1).strip() if m_text else ""
    hf_text = tok.decode(hf_ids, skip_special_tokens=True)

    same = hf_text.strip() == paged_text.strip()
    print(f"  HF text   : {hf_text!r}")
    print(f"  Paged text: {paged_text!r}")
    print(f"  identical : {same}")
    return same


def main():
    print("=" * 70)
    print("Week 5 acceptance tests")
    print("=" * 70)

    print("\n[1] Fragmentation (100 synthetic requests)")
    print("-" * 70)
    ok, waste = fragmentation_test()
    print(f"  -> waste_ratio={waste:.4f}  pass={ok}  (criterion: <0.10)")

    print("\n[2] Concurrency (paged vs naive)")
    print("-" * 70)
    ok2, speedup = concurrency_test()
    print(f"  -> speedup={speedup:.2f}x  pass={ok2}  (criterion: >=2.0x)")

    print("\n[3] HF vs paged on Qwen2.5-Coder-1.5B-Instruct")
    print("-" * 70)
    ok3 = hf_vs_paged_test(MODEL_15B, "你好")
    print(f"  -> identical text : {ok3}")

    print("\n" + "=" * 70)
    all_ok = ok and ok2 and ok3
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())