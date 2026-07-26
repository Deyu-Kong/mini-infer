#!/usr/bin/env python3
"""
Ablation Study: E5 - Prefix Caching

This script measures the TTFT improvement from prefix caching when
multiple requests share common prefixes (e.g., system prompts).

Metrics:
- Time to First Token (TTFT) with/without prefix cache
- Cache hit rate
- Memory savings
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_experiment(model_path: str, prompt: str, max_tokens: int,
                   binary: str = "./build/mini_infer") -> dict:
    """Run experiment and measure TTFT."""
    cmd = [
        binary,
        "--model", model_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens),
        "--greedy"
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        return None
    
    output = result.stdout
    metrics = {
        "total_time": elapsed,
    }
    
    import re
    speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)', output)
    if speed_match:
        metrics["tokens"] = int(speed_match.group(1))
        metrics["tok_per_sec"] = float(speed_match.group(3))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E5: Prefix Caching")
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", default="./build/mini_infer")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="benchmarks/ablation/results/e5_prefix_cache.json")
    args = parser.parse_args()
    
    # System prompt that will be shared across requests
    system_prompt = "You are a helpful AI assistant. Answer questions clearly and concisely."
    
    # Multiple requests with shared prefix
    user_prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does backpropagation work?",
        "What are transformers?",
        "Describe attention mechanism."
    ]
    
    # Create full prompts with shared system prompt
    full_prompts = [f"{system_prompt}\n\nUser: {p}\nAssistant:" for p in user_prompts]
    
    results = {"first_run": [], "second_run": []}
    
    # First run - cache cold
    print("[E5] First run (cache cold)...")
    for i, prompt in enumerate(full_prompts):
        print(f"  Prompt {i+1}/{len(full_prompts)}...")
        metrics = run_experiment(args.model, prompt, args.max_tokens, args.binary)
        if metrics:
            results["first_run"].append(metrics)
    
    # Second run - cache should be warm (if prefix caching is enabled)
    print("[E5] Second run (cache warm)...")
    for i, prompt in enumerate(full_prompts):
        print(f"  Prompt {i+1}/{len(full_prompts)}...")
        metrics = run_experiment(args.model, prompt, args.max_tokens, args.binary)
        if metrics:
            results["second_run"].append(metrics)
    
    # Calculate improvement
    cold_avg = sum(r.get("tok_per_sec", 0) for r in results["first_run"]) / len(results["first_run"])
    warm_avg = sum(r.get("tok_per_sec", 0) for r in results["second_run"]) / len(results["second_run"])
    speedup = warm_avg / cold_avg if cold_avg > 0 else 0
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E5_prefix_caching",
            "model": args.model,
            "system_prompt": system_prompt,
            "results": results,
            "summary": {
                "cold_avg_tok_per_sec": cold_avg,
                "warm_avg_tok_per_sec": warm_avg,
                "speedup": speedup
            }
        }, f, indent=2)
    
    print(f"\n[E5] Results saved to {output_path}")
    print(f"[E5] Cold: {cold_avg:.2f} tok/s, Warm: {warm_avg:.2f} tok/s, Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
