"""Generic reference-op verifier for C++ kernel tests.

Usage:
    python3 verify.py <op> <input1.bin> [<input2.bin> ...] <our_output.bin>

The first non-flag argument selects the op. Each subsequent positional
argument is either an input tensor (FP16 little-endian raw bytes) or the
final argument which is our output. The op knows how many inputs it needs.

Output (stdout):
    OK  if torch.allclose passes (rtol=1e-3, atol=1e-3 by default).
    FAIL with max abs/rel error if not.

Exit code: 0 on pass, 1 on mismatch, 2 on bad invocation.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch


def load_f16(path):
    return torch.from_numpy(np.fromfile(path, dtype=np.float16))


def to_f32(t):
    return t.to(torch.float32)


def op_rmsnorm(args):
    # args: weight.bin input.bin our.bin [shape N D] [eps]
    weight = to_f32(load_f16(args.inputs[0]))
    x      = to_f32(load_f16(args.inputs[1])).reshape(args.shape)
    ours   = to_f32(load_f16(args.our_output)).reshape(args.shape)
    ref    = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + args.eps) * weight
    return ref, ours


def op_rope(args):
    # args: x.bin cos.bin sin.bin pos.bin our.bin [shape B S H D]
    x       = to_f32(load_f16(args.inputs[0])).reshape(args.shape)
    cos     = to_f32(load_f16(args.inputs[1]))
    sin     = to_f32(load_f16(args.inputs[2]))
    pos     = np.fromfile(args.inputs[3], dtype=np.int64)
    ours    = to_f32(load_f16(args.our_output)).reshape(args.shape)
    B, S, H, D = args.shape
    half = D // 2
    # Reference: same algorithm, computed via reshape trick.
    # x: [B, S, H, D]  -> split into first half / second half along last dim
    x1 = x[..., :half]                                  # [B,S,H,half]
    x2 = x[..., half:]                                  # [B,S,H,half]
    # Reshape cos/sin so right-aligned broadcasting lands [1, S, 1, half].
    cos_p = cos.reshape(S, half).reshape(1, S, 1, half).to(torch.float32)
    sin_p = sin.reshape(S, half).reshape(1, S, 1, half).to(torch.float32)
    y1 = x1 * cos_p - x2 * sin_p
    y2 = x2 * cos_p + x1 * sin_p
    ref = torch.cat([y1, y2], dim=-1)
    return ref, ours


def op_softmax(args):
    x    = to_f32(load_f16(args.inputs[0])).reshape(args.shape)
    ours = to_f32(load_f16(args.our_output)).reshape(args.shape)
    ref  = torch.softmax(x, dim=-1)
    return ref, ours


def op_swiglu(args):
    gate = to_f32(load_f16(args.inputs[0])).reshape(args.shape)
    up   = to_f32(load_f16(args.inputs[1])).reshape(args.shape)
    ours = to_f32(load_f16(args.our_output)).reshape(args.shape)
    silu = torch.nn.functional.silu(gate)
    ref  = silu * up
    return ref, ours


def op_mlp(args):
    # inputs: x, w_gate, w_up, w_down
    # shape: [B, H, I]  (hidden H, intermediate I)
    B, H, I = args.shape
    x      = to_f32(load_f16(args.inputs[0])).reshape(B, H)
    w_gate = to_f32(load_f16(args.inputs[1])).reshape(I, H)
    w_up   = to_f32(load_f16(args.inputs[2])).reshape(I, H)
    w_down = to_f32(load_f16(args.inputs[3])).reshape(H, I)
    ours   = to_f32(load_f16(args.our_output)).reshape(B, H)
    gate = x @ w_gate.T
    up   = x @ w_up.T
    h    = torch.nn.functional.silu(gate) * up
    ref  = h @ w_down.T
    return ref, ours


OPS = {
    "rmsnorm": op_rmsnorm,
    "rope":    op_rope,
    "softmax": op_softmax,
    "swiglu":  op_swiglu,
    "mlp":     op_mlp,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=list(OPS.keys()))
    ap.add_argument("inputs", nargs="+", help="input .bin files in op order")
    ap.add_argument("--our-output", required=True)
    ap.add_argument("--shape", type=int, nargs="+", required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    if args.op not in OPS:
        print(f"unknown op {args.op}", file=sys.stderr)
        return 2
    ref, ours = OPS[args.op](args)
    if ref.shape != ours.shape:
        print(f"FAIL shape ref={tuple(ref.shape)} ours={tuple(ours.shape)}")
        return 1
    diff = (ref - ours).abs()
    max_abs = float(diff.max())
    denom = ref.abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max())
    ok = torch.allclose(ours, ref, rtol=1e-3, atol=1e-3)
    if ok:
        print(f"OK   max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
        return 0
    print(f"FAIL max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())