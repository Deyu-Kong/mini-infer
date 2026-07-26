#!/usr/bin/env python3
"""tokenize_helper.py — backing process for mini-infer's Tokenizer class.

Three sub-commands:
  specials          -> prints "key=value" lines for known special tokens
  encode --input X --output Y    -> reads text from X, writes int64 ids to Y
  decode --input X --output Y    -> reads int64 ids from X, writes text to Y

The Python tokenizer is loaded fresh on every invocation — slow per call, but
the model side runs at most a handful of times per generation, so the cost
is dwarfed by GPU work.
"""
import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer


def probe_specials(tok):
    out = []
    for name in ("eos_token", "bos_token"):
        v = tok.token_to_id(name)
        if v is not None:
            out.append(f"{name.split('_')[0]}={v}")
    for s in ("<|im_start|>", "<|im_end|>"):
        v = tok.token_to_id(s)
        if v is not None:
            out.append(f"{s.strip('<>|').split('|')[0]}={v}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("specials")
    sp.add_argument("--tokenizer", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("--tokenizer", required=True)
    enc.add_argument("--input",    required=True)
    enc.add_argument("--output",   required=True)

    dec = sub.add_parser("decode")
    dec.add_argument("--tokenizer", required=True)
    dec.add_argument("--input",    required=True)
    dec.add_argument("--output",   required=True)

    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)

    if args.cmd == "specials":
        for line in probe_specials(tok):
            print(line)
        return 0

    if args.cmd == "encode":
        text = Path(args.input).read_text(encoding="utf-8")
        ids  = tok.encode(text).ids
        Path(args.output).write_bytes(
            __import__("array").array("q", ids).tobytes())
        return 0

    if args.cmd == "decode":
        import array
        data = Path(args.input).read_bytes()
        ids  = array.array("q")
        ids.frombytes(data)
        ids  = [max(0, x) for x in ids]   # tokenizers raises on negative ids
        text = tok.decode(ids)
        Path(args.output).write_text(text, encoding="utf-8")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())