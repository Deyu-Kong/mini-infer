#!/usr/bin/env python3
"""
Ablation Study: E4 - Speculative Decoding (gamma=8)

Same as E3 but with gamma=8 to measure the impact of longer draft sequences.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_experiment(model_path: str, draft_path: str, prompt: str, 
                   max_tokens: int, gamma: int,
                   binary: str = "./build/mini_infer") -> dict:
    """Run speculative decoding experiment."""
    cmd = [
        binary,
        "--model", model_path,
        "--spec-draft", draft_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens),
        "--greedy",
        "--gamma", str(gamma)
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None
    
    output = result.stdout
    metrics = {
        "total_time": elapsed,
        "gamma": gamma,
    }
    
    import re
    speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)', output)
    if speed_match:
        metrics["tokens"] = int(speed_match.group(1))
        metrics["reported_time"] = float(speed_match.group(2))
        metrics["tok_per_sec"] = float(speed_match.group(3))
    
    accept_match = re.search(r'accept_rate=([\d.]+)%', output)
    if accept_match:
        metrics["accept_rate"] = float(accept_match.group(1))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="E4: Speculative Decoding gamma=8")
    parser.add_argument("--model", required=True, help="Target model path")
    parser.add_argument("--draft", required=True, help="Draft model path")
    parser.add_argument("--binary", default="./build/mini_infer")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--output", default="benchmarks/ablation/results/e4_spec_g8.json")
    args = parser.parse_args()
    
    prompts = [
        "Hello, how are you?",
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the concept of attention in transformers.",
        "What are the benefits of speculative decoding?",
        "Describe the PagedAttention algorithm."
    ]
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[E4] Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        metrics = run_experiment(args.model, args.draft, prompt, 
                                 args.max_tokens, args.gamma, args.binary)
        if metrics:
            metrics["prompt"] = prompt
            results.append(metrics)
            print(f"  -> {metrics.get('tok_per_sec', 0):.2f} tok/s, "
                  f"accept_rate={metrics.get('accept_rate', 0):.1f}%")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": f"E4_speculative_gamma{args.gamma}",
            "model": args.model,
            "draft_model": args.draft,
            "gamma": args.gamma,
            "results": results
        }, f, indent=2)
    
    avg_throughput = sum(r.get("tok_per_sec", 0) for r in results) / len(results)
    avg_accept = sum(r.get("accept_rate", 0) for r in results) / len(results)
    print(f"\n[E4] Results saved to {output_path}")
    print(f"[E4] Avg throughput: {avg_throughput:.2f} tok/s, Avg accept rate: {avg_accept:.1f}%")


if __name__ == "__main__":
    main()
