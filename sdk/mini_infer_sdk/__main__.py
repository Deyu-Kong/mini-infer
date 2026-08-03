"""
Unified Python CLI for mini-infer.

    python -m mini_infer_sdk generate --model /path --prompt "Hi"      # offline
    python -m mini_infer_sdk generate --endpoint http://h:8000 --p "Hi" # online
    python -m mini_infer_sdk chat --model /path                         # offline REPL
    python -m mini_infer_sdk serve --model /path --port 8000           # launch server

Both offline (--model) and online (--endpoint) modes share the same SDK.
"""

import argparse
import sys

from . import MiniInfer, InferenceConfig, SamplingConfig, SamplingMode
from .backends.local import _find_binary


def _engine_from_args(args) -> MiniInfer:
    if args.endpoint:
        return MiniInfer(endpoint=args.endpoint)
    if args.model:
        return MiniInfer(model_path=args.model, draft_path=args.draft,
                         python_path=args.python)
    raise SystemExit("provide --model (offline) or --endpoint (online)")


def _config_from_args(args) -> InferenceConfig:
    mode = SamplingMode.GREEDY if args.greedy else SamplingMode.TOP_P
    return InferenceConfig(
        max_new_tokens=args.max_tokens,
        sampling=SamplingConfig(mode=mode, temperature=args.temperature,
                                top_p=args.top_p, seed=args.seed),
    )


def cmd_generate(args):
    engine = _engine_from_args(args)
    cfg = _config_from_args(args)
    result = engine.generate(args.prompt, config=cfg)
    print(result.text)
    print(f"\n[mini-infer] {result.generated_tokens} tokens, "
          f"{result.tokens_per_sec:.1f} tok/s", file=sys.stderr)


def cmd_chat(args):
    engine = _engine_from_args(args)
    cfg = _config_from_args(args)
    chat = engine.chat(system_prompt=args.system)
    print(f"mini-infer chat ({engine}). Type 'exit' to quit.")
    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in ("exit", "quit"):
            break
        resp = chat.send(user)
        print(f"assistant> {resp.content}")


def cmd_serve(args):
    import subprocess
    binary = _find_binary(args.binary)
    if not binary.exists():
        raise SystemExit(f"mini_infer binary not found at {binary}")
    cmd = [str(binary), "serve", "--model", args.model,
           "--host", args.host, "--port", str(args.port),
           "--device", str(args.device), "--python", args.python]
    if args.draft:
        cmd += ["--draft", args.draft, "--gamma", str(args.gamma)]
    print(f"[sdk] launching: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mini_infer_sdk",
                                description="mini-infer unified CLI (offline + online)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        g = sp.add_mutually_exclusive_group(required=False)
        g.add_argument("--model", help="HuggingFace model directory (offline)")
        g.add_argument("--endpoint", "--remote", dest="endpoint",
                       help="Server URL, e.g. http://localhost:8000 (online)")
        sp.add_argument("--draft", help="Draft model (offline, speculative decoding)")
        sp.add_argument("--python", default="python3", help="Python executable for tokenizer")
        sp.add_argument("--max-tokens", type=int, default=128)
        sp.add_argument("--temperature", type=float, default=1.0)
        sp.add_argument("--top-p", type=float, default=0.9)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--greedy", action="store_true")

    g = sub.add_parser("generate", help="Single-prompt generation")
    g.add_argument("--prompt", "--p", dest="prompt", required=True)
    add_common(g)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("chat", help="Interactive multi-turn chat")
    c.add_argument("--system", default=None)
    add_common(c)
    c.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="Launch the mini-infer HTTP server")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--draft", default=None)
    s.add_argument("--gamma", type=int, default=4)
    s.add_argument("--python", default="python3")
    s.add_argument("--binary", default=None, help="Path to mini_infer binary")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
