#!/usr/bin/env python3
"""
03_batch_demo.py — Batch processing with performance comparison.

Usage:
    python sdk/examples/03_batch_demo.py --model /path/to/Qwen2.5-7B-Instruct
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_infer_sdk import MiniInfer, InferenceConfig, SamplingConfig, SamplingMode


PROMPTS = [
    "Explain Python decorators in one sentence.",
    "Write a binary search function in Python.",
    "What is the time complexity of quicksort?",
    "Explain the difference between list and tuple.",
    "What is a context manager in Python?",
    "Explain the GIL in Python.",
    "What is the purpose of __init__.py?",
    "Explain list comprehension with an example.",
    "What is the difference between deep and shallow copy?",
    "How does Python's garbage collection work?",
]


def main():
    parser = argparse.ArgumentParser(description="Batch inference demo")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--draft", default=None, help="Path to draft model")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers")
    args = parser.parse_args()

    print("=" * 60)
    print("  Batch Inference Demo")
    print("=" * 60)

    engine = MiniInfer(
        model_path=args.model,
        draft_path=args.draft,
    )

    config = InferenceConfig(
        max_new_tokens=64,
        sampling=SamplingConfig(mode=SamplingMode.GREEDY),
    )

    batch = engine.batch(config=config)

    print(f"\nProcessing {len(PROMPTS)} prompts with {args.workers} workers...")
    t0 = time.perf_counter()
    stats = batch.process(PROMPTS, workers=args.workers)
    t1 = time.perf_counter()

    print(f"\n--- Results ---")
    for r in stats.results:
        print(f"  [{r.index}] Q: {r.prompt}")
        print(f"      A: {r.result.text[:80]}...")
        print(f"      tps: {r.result.tokens_per_sec:.1f}")
        print()

    print(f"--- Summary ---")
    print(f"  Prompts:   {stats.total_prompts}")
    print(f"  Tokens:    {stats.total_generated_tokens}")
    print(f"  Wall time: {t1 - t0:.2f}s")
    print(f"  Throughput: {stats.tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()