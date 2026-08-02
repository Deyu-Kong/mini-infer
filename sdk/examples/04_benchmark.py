#!/usr/bin/env python3
"""
04_benchmark.py — Benchmark different configurations with performance metrics.

Usage:
    python sdk/examples/04_benchmark.py --model /path/to/Qwen2.5-7B-Instruct \
        --draft /path/to/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_infer_sdk import MiniInfer, InferenceConfig, SamplingConfig, SamplingMode


BENCH_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Explain the difference between TCP and UDP.",
    "Implement a simple LRU cache in Python.",
    "What is the difference between a process and a thread?",
    "Write a SQL query to find the top 3 highest salaries.",
]


def run_benchmark(engine: MiniInfer, config: InferenceConfig, label: str,
                  prompts: List[str]) -> dict:
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label}")
    print(f"{'='*60}")

    batch = engine.batch(config=config)
    t0 = time.perf_counter()
    stats = batch.process(prompts)
    t1 = time.perf_counter()

    total_tokens = stats.total_generated_tokens
    wall_time = t1 - t0

    print(f"  Tokens: {total_tokens}")
    print(f"  Time:   {wall_time:.2f}s")
    print(f"  TPS:    {total_tokens / wall_time:.1f} tok/s")
    print(f"  Avg TPS per prompt: {sum(r.result.tokens_per_sec for r in stats.results) / len(stats.results):.1f}")

    return {
        "label": label,
        "total_tokens": total_tokens,
        "wall_time": wall_time,
        "tps": total_tokens / wall_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--draft", default=None, help="Path to draft model")
    args = parser.parse_args()

    print("=" * 60)
    print("  mini-infer Benchmark Comparison")
    print("=" * 60)

    engine = MiniInfer(
        model_path=args.model,
        draft_path=args.draft,
    )

    results = []

    base_config = InferenceConfig(
        max_new_tokens=64,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
    )

    # E0: Naive
    cfg = InferenceConfig(**{**base_config.__dict__, "paged": False})
    results.append(run_benchmark(engine, cfg, "E0: Naive autoregressive", BENCH_PROMPTS))

    # E1: PagedAttention
    cfg = InferenceConfig(**{**base_config.__dict__, "paged": True})
    results.append(run_benchmark(engine, cfg, "E1: PagedAttention", BENCH_PROMPTS))

    # E2: Speculative decoding
    if args.draft:
        cfg = InferenceConfig(**{**base_config.__dict__, "draft_path": args.draft, "gamma": 4})
        results.append(run_benchmark(engine, cfg, "E2: Speculative decoding (γ=4)", BENCH_PROMPTS))

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Configuration':<35} {'TPS':>10} {'Speedup':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")

    if results:
        baseline_tps = results[0]["tps"]
        for r in results:
            speedup = r["tps"] / baseline_tps if baseline_tps > 0 else 0
            print(f"  {r['label']:<35} {r['tps']:>8.1f} tok/s {speedup:>8.2f}x")


if __name__ == "__main__":
    main()