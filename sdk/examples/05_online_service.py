#!/usr/bin/env python3
"""
05_online_service.py — the SDK as a *service*: launch it once, then keep
sending requests.

This demonstrates the Python SDK's online support end-to-end:

  1. `MiniInferServer` brings up a persistent online service (the C++
     `mini_infer serve` binary, launched and managed by the SDK). The model
     is loaded **once** into GPU memory and stays resident for the life of
     the service.
  2. `MiniInfer(endpoint=...)` is the SDK client. We send a *continuous
     stream* of requests — first sequentially, then concurrently — against
     the same resident model. Every request reuses the loaded weights (no
     per-call reload), which is the whole point of online serving.

Run:
    python sdk/examples/05_online_service.py --model /path/to/Qwen2.5-7B-Instruct
    # (optionally --draft /path/to/draft for speculative decoding, --port 8000)
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini_infer_sdk import (
    MiniInfer, MiniInferServer, InferenceConfig, SamplingConfig, SamplingMode,
)


# A small but varied workload so the demo looks like real traffic, not a
# one-shot toy. Each item is (label, prompt).
WORKLOAD = [
    ("code",   "Write a Python function that reverses a string."),
    ("code",   "Write a Python one-liner to compute factorial(n)."),
    ("math",   "What is 2+2? Answer with just the number."),
    ("qa",     "What is the capital of France? One word."),
    ("code",   "Reverse a linked list in Python, 3 lines."),
    ("math",   "What is 7 * 8? Number only."),
    ("qa",     "Name the largest planet. One word."),
    ("code",   "Write hello world in C."),
    ("math",   "Sum 1 to 100. Number only."),
    ("qa",     "What language has the keyword 'def'? One word."),
]


def run_once(engine, prompt, cfg):
    t0 = time.perf_counter()
    r = engine.generate(prompt, config=cfg)
    return r, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description="mini-infer online service demo")
    p.add_argument("--model", required=True, help="HuggingFace model directory")
    p.add_argument("--draft", default=None, help="Draft model (speculative decoding)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=4,
                   help="workers for the concurrent phase")
    args = p.parse_args()

    cfg = InferenceConfig(max_new_tokens=args.max_tokens,
                          sampling=SamplingConfig(mode=SamplingMode.GREEDY))

    print("=" * 64)
    print("  mini-infer ONLINE SERVICE demo")
    print("  SDK launches a persistent service, then floods it with requests")
    print("=" * 64)

    # --- 1. Bring up the service (model loaded once, resident) ----------
    print(f"\n[1/3] starting service  model={Path(args.model).name}  port={args.port}")
    t0 = time.perf_counter()
    with MiniInferServer(model=args.model, port=args.port, host=args.host,
                         device=args.device, draft=args.draft) as srv:
        print(f"      service up in {time.perf_counter() - t0:.1f}s  ->  {srv}")

        engine = MiniInfer(endpoint=srv.endpoint)
        print(f"      client:  {engine}")

        # --- 2. Sequential traffic: continuous request stream -----------
        print("\n[2/3] sequential traffic (each request reuses resident model)")
        seq_tokens, seq_wall = 0, 0.0
        for i, (label, prompt) in enumerate(WORKLOAD, 1):
            r, dt = run_once(engine, prompt, cfg)
            seq_tokens += r.generated_tokens
            seq_wall += dt
            print(f"      #{i:<2d} [{label:4s}] {r.generated_tokens:>3d} tok  "
                  f"{r.tokens_per_sec:5.1f} tok/s  ->  {r.text.strip()[:42]!r}")
        print(f"      sequential total: {seq_tokens} tok in {seq_wall:.2f}s "
              f"= {seq_tokens / seq_wall if seq_wall else 0:.1f} tok/s")

        # --- 3. Concurrent traffic: the service fields ongoing requests -
        print(f"\n[3/3] concurrent traffic ({args.concurrency} parallel clients)")
        prompts = [w[1] for w in WORKLOAD]
        c_tokens, c_wall = 0, 0.0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(run_once, engine, q, cfg): q for q in prompts}
            for fut in as_completed(futs):
                r, _ = fut.result()
                c_tokens += r.generated_tokens
        c_wall = time.perf_counter() - t0
        print(f"      concurrent total: {c_tokens} tok in {c_wall:.2f}s "
              f"wall = {c_tokens / c_wall if c_wall else 0:.1f} tok/s")
        print("      (generation is serialized on one resident model; "
              "this shows the service fielding parallel clients gracefully)")

        print("\n  service stayed up the whole time — you could keep sending")
        print("  requests indefinitely; only the client side closed here.")
        print("=" * 64)
        print("  Done.")
        print("=" * 64)


if __name__ == "__main__":
    main()
