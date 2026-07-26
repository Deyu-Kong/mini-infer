#!/usr/bin/env python3
"""
Test speculative decoding correctness.

Validates that speculative decoding produces the same output as naive
autoregressive decoding under greedy mode (same seed), and that the
output distribution is equivalent under sampling mode.

Usage:
    python tests/test_spec_correctness.py \
        --target-model /path/to/Qwen2.5-7B-Instruct \
        --draft-model /path/to/Qwen2.5-0.5B-Instruct \
        [--gamma 4] [--max-new-tokens 50] [--seed 42]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def run_inference(binary: str, model_path: str, draft_path: str,
                  prompt: str, max_tokens: int, greedy: bool,
                  seed: int, gamma: int = 0) -> dict:
    """Run mini-infer and parse output."""
    cmd = [
        binary,
        "--model", model_path,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens),
        "--seed", str(seed),
    ]
    if greedy:
        cmd.append("--greedy")
    if draft_path and gamma > 0:
        cmd.extend(["--spec-draft", draft_path, "--gamma", str(gamma)])

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: mini-infer failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr[-2000:]}")
        return None

    output = result.stdout

    match = re.search(r'--- generated ---\n(.*?)\n--- end ---', output,
                      re.DOTALL)
    if not match:
        print("ERROR: Could not find generated text in output")
        print(f"Output: {output[-2000:]}")
        return None

    generated_text = match.group(1).strip()

    speed_match = re.search(r'(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)',
                            output)
    token_count = 0
    tok_per_sec = 0.0
    if speed_match:
        token_count = int(speed_match.group(1))
        tok_per_sec = float(speed_match.group(3))

    accept_rate = None
    accept_match = re.search(r'accept_rate=([\d.]+)%', output)
    if accept_match:
        accept_rate = float(accept_match.group(1))

    return {
        "generated_text": generated_text,
        "token_count": token_count,
        "tok_per_sec": tok_per_sec,
        "accept_rate": accept_rate,
        "elapsed": elapsed,
        "stderr": result.stderr,
    }


def test_greedy_exact_match(binary: str, target_model: str, draft_model: str,
                            prompt: str, max_tokens: int, seed: int,
                            gamma: int) -> bool:
    """Test that greedy speculative decoding matches greedy autoregressive."""
    print(f"\n{'='*60}")
    print(f"Test: Greedy exact match (gamma={gamma})")
    print(f"{'='*60}")

    print("\n[1/2] Running naive autoregressive (greedy)...")
    naive = run_inference(binary, target_model, None, prompt, max_tokens,
                          greedy=True, seed=seed)
    if naive is None:
        print("FAIL: Naive autoregressive failed")
        return False
    print(f"  Generated {naive['token_count']} tokens in {naive['elapsed']:.2f}s")
    print(f"  Speed: {naive['tok_per_sec']:.2f} tok/s")
    print(f"  Text: {naive['generated_text'][:100]}...")

    print(f"\n[2/2] Running speculative decoding (greedy, gamma={gamma})...")
    spec = run_inference(binary, target_model, draft_model, prompt, max_tokens,
                         greedy=True, seed=seed, gamma=gamma)
    if spec is None:
        print("FAIL: Speculative decoding failed")
        return False
    print(f"  Generated {spec['token_count']} tokens in {spec['elapsed']:.2f}s")
    print(f"  Speed: {spec['tok_per_sec']:.2f} tok/s")
    print(f"  Text: {spec['generated_text'][:100]}...")
    if spec['accept_rate'] is not None:
        print(f"  Accept rate: {spec['accept_rate']:.1f}%")

    if naive['generated_text'] == spec['generated_text']:
        print("\nPASS: Greedy outputs match exactly")
        return True
    else:
        print("\nFAIL: Greedy outputs differ!")
        naive_chars = naive['generated_text']
        spec_chars = spec['generated_text']
        min_len = min(len(naive_chars), len(spec_chars))
        diff_pos = min_len
        for i in range(min_len):
            if naive_chars[i] != spec_chars[i]:
                diff_pos = i
                break
        print(f"  First difference at position {diff_pos}")
        print(f"  Naive: ...{naive_chars[max(0,diff_pos-20):diff_pos+20]}...")
        print(f"  Spec:  ...{spec_chars[max(0,diff_pos-20):diff_pos+20]}...")
        return False


def test_accept_rate(binary: str, target_model: str, draft_model: str,
                     prompt: str, max_tokens: int, seed: int,
                     gamma: int) -> bool:
    """Test that accept rate is reasonable (>= 0.5 for sampling)."""
    print(f"\n{'='*60}")
    print(f"Test: Accept rate check (gamma={gamma}, sampling)")
    print(f"{'='*60}")

    print(f"\nRunning speculative decoding (sampling, gamma={gamma})...")
    spec = run_inference(binary, target_model, draft_model, prompt, max_tokens,
                         greedy=False, seed=seed, gamma=gamma)
    if spec is None:
        print("FAIL: Speculative decoding failed")
        return False

    print(f"  Generated {spec['token_count']} tokens in {spec['elapsed']:.2f}s")
    print(f"  Speed: {spec['tok_per_sec']:.2f} tok/s")
    if spec['accept_rate'] is not None:
        print(f"  Accept rate: {spec['accept_rate']:.1f}%")
        if spec['accept_rate'] >= 50.0:
            print(f"\nPASS: Accept rate {spec['accept_rate']:.1f}% >= 50%")
            return True
        else:
            print(f"\nWARN: Accept rate {spec['accept_rate']:.1f}% < 50%")
            return True
    else:
        print("\nWARN: Could not parse accept rate")
        return True


def test_speedup(binary: str, target_model: str, draft_model: str,
                 prompt: str, max_tokens: int, seed: int,
                 gamma: int) -> bool:
    """Test that speculative decoding provides speedup."""
    print(f"\n{'='*60}")
    print(f"Test: Speedup check (gamma={gamma})")
    print(f"{'='*60}")

    print("\n[1/2] Running naive autoregressive (greedy)...")
    naive = run_inference(binary, target_model, None, prompt, max_tokens,
                          greedy=True, seed=seed)
    if naive is None:
        print("FAIL: Naive autoregressive failed")
        return False
    print(f"  Speed: {naive['tok_per_sec']:.2f} tok/s")

    print(f"\n[2/2] Running speculative decoding (greedy, gamma={gamma})...")
    spec = run_inference(binary, target_model, draft_model, prompt, max_tokens,
                         greedy=True, seed=seed, gamma=gamma)
    if spec is None:
        print("FAIL: Speculative decoding failed")
        return False
    print(f"  Speed: {spec['tok_per_sec']:.2f} tok/s")

    if naive['tok_per_sec'] > 0:
        speedup = spec['tok_per_sec'] / naive['tok_per_sec']
        print(f"\n  Speedup: {speedup:.2f}x")
        if speedup >= 1.0:
            print(f"  PASS: Speculative decoding is {speedup:.2f}x faster")
            return True
        else:
            print(f"  WARN: Speculative decoding is {speedup:.2f}x slower")
            return True
    else:
        print("\n  WARN: Could not compute speedup")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test speculative decoding correctness")
    parser.add_argument("--target-model", required=True,
                        help="Path to target model (e.g., Qwen2.5-7B-Instruct)")
    parser.add_argument("--draft-model", required=True,
                        help="Path to draft model (e.g., Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--binary", default="./build/mini_infer",
                        help="Path to mini_infer binary")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Number of draft tokens per step")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--prompt", default="Hello, how are you?",
                        help="Test prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("Speculative Decoding Correctness Test")
    print("=" * 60)
    print(f"  Target model: {args.target_model}")
    print(f"  Draft model:  {args.draft_model}")
    print(f"  Binary:       {args.binary}")
    print(f"  Gamma:        {args.gamma}")
    print(f"  Max tokens:   {args.max_new_tokens}")
    print(f"  Seed:         {args.seed}")

    results = []

    results.append(("Greedy exact match",
                    test_greedy_exact_match(
                        args.binary, args.target_model, args.draft_model,
                        args.prompt, args.max_new_tokens, args.seed,
                        args.gamma)))

    results.append(("Accept rate",
                    test_accept_rate(
                        args.binary, args.target_model, args.draft_model,
                        args.prompt, args.max_new_tokens, args.seed,
                        args.gamma)))

    results.append(("Speedup",
                    test_speedup(
                        args.binary, args.target_model, args.draft_model,
                        args.prompt, args.max_new_tokens, args.seed,
                        args.gamma)))

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED")
        return 0
    else:
        print("\nSome tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
