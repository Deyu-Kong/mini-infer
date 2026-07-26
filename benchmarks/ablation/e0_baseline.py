#!/usr/bin/env python3
"""
Ablation Study: E0 - Baseline Naive Autoregressive Generation

This script measures the baseline performance of naive autoregressive decoding
without any optimizations (no PagedAttention, no batching, no speculative decoding).

Metrics:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT)
- Total generation time
- Tokens per second (throughput)
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_experiment(model_path: str, prompt: str, max_tokens: int, 
                   binary: str = "./build/mini_infer") -> dict:
    """Run a single experiment and collect metrics."""
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
        print(f"ERROR: {result.stderr}")
        return None
    
    # Parse output
    output = result.stdout
    metrics = {
        "total_time": elapsed,
        "max_tokens": max_tokens,
    }
    
    # Extract token count and speed from output
    import re
    speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)', output)
    if speed_match:
        metrics["tokens"] = int(speed_match.group(1))
        metrics["reported_time"] = float(speed_match.group(2))
        metrics["tok_per_sec"] = float(speed_match.group(3))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E0: Baseline naive autoregressive")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--binary", default="./build/mini_infer", help="Path to binary")
    parser.add_argument("--prompts", default="benchmarks/ablation/prompts.json",
                        help="Path to prompts file")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--output", default="benchmarks/ablation/results/e0_baseline.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    # Load prompts
    prompts_path = Path(args.prompts)
    if prompts_path.exists():
        with open(prompts_path) as f:
            prompts = json.load(f)
    else:
        # Default prompts
        prompts = [
            "Hello, how are you?",
            "Write a Python function to calculate fibonacci numbers.",
            "Explain the concept of attention in transformers.",
            "What are the benefits of speculative decoding?",
            "Describe the PagedAttention algorithm."
        ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[E0] Running prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        metrics = run_experiment(args.model, prompt, args.max_tokens, args.binary)
        if metrics:
            metrics["prompt"] = prompt
            metrics["prompt_id"] = i
            results.append(metrics)
            print(f"  -> {metrics.get('tok_per_sec', 0):.2f} tok/s")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E0_baseline",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "results": results
        }, f, indent=2)
    
    print(f"\n[E0] Results saved to {output_path}")
    print(f"[E0] Average throughput: {sum(r.get('tok_per_sec', 0) for r in results) / len(results):.2f} tok/s")


if __name__ == "__main__":
    main()
