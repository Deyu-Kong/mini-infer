#!/usr/bin/env python3
"""
Ablation Study: E1 - PagedAttention

This script measures the performance improvement from PagedAttention over
naive KV cache management.

Metrics:
- Memory utilization (fragmentation ratio)
- Maximum concurrent sequences
- Time per token with paged vs naive
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_experiment(model_path: str, prompt: str, max_tokens: int,
                   use_paged: bool, binary: str = "./build/mini_infer") -> dict:
    """Run a single experiment and collect metrics."""
    cmd = [
        binary,
        "--model", model_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens),
        "--greedy"
    ]
    if use_paged:
        cmd.append("--paged")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    output = result.stdout
    metrics = {
        "total_time": elapsed,
        "max_tokens": max_tokens,
        "use_paged": use_paged,
    }
    
    import re
    speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)', output)
    if speed_match:
        metrics["tokens"] = int(speed_match.group(1))
        metrics["reported_time"] = float(speed_match.group(2))
        metrics["tok_per_sec"] = float(speed_match.group(3))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E1: PagedAttention")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--binary", default="./build/mini_infer", help="Path to binary")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="benchmarks/ablation/results/e1_paged.json")
    args = parser.parse_args()
    
    prompts = [
        "Hello, how are you?",
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the concept of attention in transformers.",
    ]
    
    results = {"naive": [], "paged": []}
    
    for i, prompt in enumerate(prompts):
        print(f"[E1] Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Naive
        print("  Running naive...")
        m_naive = run_experiment(args.model, prompt, args.max_tokens, False, args.binary)
        if m_naive:
            results["naive"].append(m_naive)
        
        # Paged
        print("  Running paged...")
        m_paged = run_experiment(args.model, prompt, args.max_tokens, True, args.binary)
        if m_paged:
            results["paged"].append(m_paged)
    
    # Calculate speedup
    naive_avg = sum(r.get("tok_per_sec", 0) for r in results["naive"]) / len(results["naive"])
    paged_avg = sum(r.get("tok_per_sec", 0) for r in results["paged"]) / len(results["paged"])
    speedup = paged_avg / naive_avg if naive_avg > 0 else 0
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E1_paged_attention",
            "model": args.model,
            "results": results,
            "summary": {
                "naive_avg_tok_per_sec": naive_avg,
                "paged_avg_tok_per_sec": paged_avg,
                "speedup": speedup
            }
        }, f, indent=2)
    
    print(f"\n[E1] Results saved to {output_path}")
    print(f"[E1] Naive: {naive_avg:.2f} tok/s, Paged: {paged_avg:.2f} tok/s, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
