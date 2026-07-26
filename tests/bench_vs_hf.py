#!/usr/bin/env python3
"""
Side-by-side performance benchmark: mini-infer vs HuggingFace transformers.

Same model, same prompt, same max_new_tokens, same GPU, same sampling
(greedy). Warmup both before timing. Times:

  - prefill latency     (model.forward on the prompt)
  - decode tok/s        (decoding tokens / pure-decode wall time)
  - end-to-end latency  (total wall time / total tokens)

Usage:
    python3 tests/bench_vs_hf.py [--model PATH] [--prompt TEXT]
                                 [--max-new-tokens N] [--runs R]
"""
import argparse
import re
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------- HF reference ----------------------------------

def hf_bench(model_path: str, prompt: str, max_new_tokens: int,
             runs: int, device: str, attn_impl: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation=attn_impl
    ).to(device)
    model.eval()

    chat = f"user\n{prompt}\nassistant\n"
    ids = tok.encode(chat, return_tensors="pt").to(device)
    n_prompt = ids.shape[1]

    # Warmup (CUDA kernel JIT, allocator, etc.)
    with torch.no_grad():
        _ = model.generate(
            ids, max_new_tokens=8, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()

    prefill_ms = []
    decode_per_tok_ms = []
    e2e_ms = []
    total_new_tokens = []

    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            # 1) Prefill-only pass to measure prefill cost.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        prefill_ms.append((t1 - t0) * 1000)

        # 2) Full generation, count new tokens.
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        n_new = out.shape[1] - n_prompt
        total_new_tokens.append(n_new)

        e2e = (t3 - t2) * 1000
        # Decode portion = e2e - prefill.
        decode = max(e2e - prefill_ms[-1], 1e-6)
        e2e_ms.append(e2e)
        decode_per_tok_ms.append(decode / max(n_new - 1, 1))

    return {
        "prompt_len":    n_prompt,
        "n_new_avg":     statistics.mean(total_new_tokens),
        "prefill_ms":    statistics.median(prefill_ms),
        "decode_ms_per_tok": statistics.median(decode_per_tok_ms),
        "e2e_ms":        statistics.median(e2e_ms),
        "decode_tok_s":  1000.0 / statistics.median(decode_per_tok_ms),
    }


# ---------------------------- mini-infer ------------------------------------

def mini_bench(model_path: str, prompt: str, max_new_tokens: int,
               runs: int, device: str):
    # Warmup: a small generation to JIT kernels + warm allocator + cache.
    cmd_warm = [
        "./build/mini_infer", "--model", model_path,
        "--prompt", "hi", "--max-new-tokens", "8",
        "--greedy", "--device", str(device), "--max-seq-len", "4096",
    ]
    subprocess.run(cmd_warm, capture_output=True, timeout=180)

    cmd = [
        "./build/mini_infer", "--model", model_path,
        "--prompt", prompt, "--max-new-tokens", str(max_new_tokens),
        "--greedy", "--device", str(device), "--max-seq-len", "4096",
    ]

    decode_per_tok_ms = []
    e2e_ms = []
    total_new_tokens = []

    for _ in range(runs):
        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        t1 = time.perf_counter()
        if res.returncode != 0:
            print("STDERR:", res.stderr[-300:])
            raise RuntimeError("mini_infer failed")

        body = res.stdout
        m_speed = re.search(r"(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)", body)
        if not m_speed:
            print("OUT:", body)
            raise RuntimeError("cannot parse mini_infer output")
        n_new = int(m_speed.group(1))
        wall_s = float(m_speed.group(2))
        # mini_infer prints (n_new - 1) as "tokens in ...s" — that's the
        # number of *decode* steps (prefill produces the first token).
        # wall_s = pure decode time (per the engine code: t1-t0 is started
        # AFTER prefill completes).
        decode_per_tok_ms.append((wall_s * 1000.0) / max(n_new - 1, 1))
        total_new_tokens.append(n_new)
        e2e_ms.append((t1 - t0) * 1000)

    return {
        "prompt_len":    None,  # unknown without parsing
        "n_new_avg":     statistics.mean(total_new_tokens),
        "prefill_ms":    None,  # not separately reported by mini_infer
        "decode_ms_per_tok": statistics.median(decode_per_tok_ms),
        "e2e_ms":        statistics.median(e2e_ms),
        "decode_tok_s":  1000.0 / statistics.median(decode_per_tok_ms),
    }


# ---------------------------- driver ----------------------------------------

def fmt(d, key):
    if d.get(key) is None:
        return "   n/a "
    return f"{d[key]:8.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--prompt", default="Write a Python function to compute Fibonacci numbers")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="0")
    ap.add_argument("--hf-attn", default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    help="HF attention implementation")
    args = ap.parse_args()

    print(f"model     : {args.model}")
    print(f"prompt    : {args.prompt!r}")
    print(f"max_new   : {args.max_new_tokens}")
    print(f"runs      : {args.runs} (median reported)")
    print(f"device    : cuda:{args.device}")
    print(f"torch     : {torch.__version__}")
    print(f"hf-attn   : {args.hf_attn}")
    print()

    print(f">>> HuggingFace transformers ({args.hf_attn}, FP16) ...")
    hf = hf_bench(args.model, args.prompt, args.max_new_tokens, args.runs,
                  f"cuda:{args.device}", args.hf_attn)
    print("done.")

    print(">>> mini-infer ...")
    mi = mini_bench(args.model, args.prompt, args.max_new_tokens, args.runs, args.device)
    print("done.")

    print()
    print("=" * 72)
    print(f"{'metric':<26}  {'HF transformers':>16}  {'mini-infer':>16}  {'speedup':>8}")
    print("-" * 72)
    rows = [
        ("prompt tokens",         "prompt_len"),
        ("generated tokens (avg)","n_new_avg"),
        ("prefill (ms)",          "prefill_ms"),
        ("decode ms / token",     "decode_ms_per_tok"),
        ("decode tok/s",          "decode_tok_s"),
        ("end-to-end (ms)*",      "e2e_ms"),
    ]
    for label, key in rows:
        h = hf.get(key); m = mi.get(key)
        speedup = ""
        if isinstance(h, (int, float)) and isinstance(m, (int, float)) and h and m:
            if "tok/s" in key:
                speedup = f"{m/h:.2f}x"
            elif "ms" in key or "ms_per_tok" in key:
                speedup = f"{h/m:.2f}x"
        print(f"{label:<26}  {fmt(hf, key):>16}  {fmt(mi, key):>16}  {speedup:>8}")
    print("=" * 72)
    print("* HF e2e is in-process GPU time. mini-infer e2e is wall-clock")
    print("  including subprocess startup (~5-10s) + model load + tokenize.")
    print("  Compare decode ms/token and decode tok/s for an apples-to-apples")
    print("  GPU-only comparison.")


if __name__ == "__main__":
    main()