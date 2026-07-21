#!/usr/bin/env python3
"""inspect_weights.py — dump per-tensor shape / dtype / mean / std for a
HuggingFace safetensors model (single-shard or multi-shard).

Usage:
    scripts/inspect_weights.py <model_dir>
    scripts/inspect_weights.py path/to/model-00001-of-00003.safetensors

If a directory is given, all `model-*.safetensors` inside are scanned. The
output format matches `QwenModel::summarize` so they can be diffed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def fmt_shape(shape):
    return "[" + ",".join(str(s) for s in shape) + "]"


def stats(t: torch.Tensor):
    f = t.to(torch.float32).reshape(-1)
    if f.numel() > 8192:
        f = f[:8192]
    if f.numel() == 0:
        return 0.0, 0.0
    mean = float(f.mean())
    std = float(f.std())
    return mean, std


def inspect_one(path: str, name_filter=None):
    rows = []
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    keys.sort()
    for k in keys:
        if name_filter and name_filter not in k:
            continue
        with safe_open(path, framework="pt") as f:
            t = f.get_tensor(k)
        m, s = stats(t)
        rows.append((k, tuple(t.shape), str(t.dtype).removeprefix("torch."), m, s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="model dir or path to a single .safetensors file")
    ap.add_argument("--filter", default=None, help="only print keys containing this substring")
    ap.add_argument("--show-shard", action="store_true", help="include shard filename")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        paths = [str(p)]
    elif p.is_dir():
        paths = sorted(str(x) for x in p.glob("model-*.safetensors"))
        if not paths:
            sys.exit(f"no model-*.safetensors found in {p}")
    else:
        sys.exit(f"not a file or directory: {p}")

    all_rows = []
    for shard in paths:
        rows = inspect_one(shard, args.filter)
        for r in rows:
            all_rows.append((shard, *r))

    width = max((len(r[1]) for r in all_rows), default=0)
    for shard, name, shape, dtype, mean, std in all_rows:
        prefix = f"{os.path.basename(shard):40s} " if args.show_shard else ""
        print(f"{prefix}{name:<{width}}  shape={fmt_shape(shape)}  "
              f"dtype={dtype:<8s}  mean={mean:+.4e}  std={std:.4e}")

    if not all_rows:
        print("(no tensors matched)", file=sys.stderr)


if __name__ == "__main__":
    main()