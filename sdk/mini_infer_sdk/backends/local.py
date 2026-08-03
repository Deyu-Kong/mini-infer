"""Local backend — wraps the mini-infer C++/CUDA binary via subprocess."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..config import InferenceConfig, SamplingMode
from ..result import GenerationResult
from .base import Backend


def _find_binary(explicit: Optional[str] = None) -> Path:
    """Auto-discover the mini_infer binary.

    Search order:
      1. explicit argument / $MINI_INFER_BINARY
      2. ./build/mini_infer (cwd)
      3. <sdk>/../../build/mini_infer (project root)
      4. $MINI_INFER_ROOT/build/mini_infer
      5. `which mini_infer`
    """
    candidates = []
    env_bin = explicit or os.environ.get("MINI_INFER_BINARY")
    if env_bin:
        p = Path(env_bin)
        if p.exists():
            return p
        candidates.append(p)
    candidates.extend([
        Path.cwd() / "build" / "mini_infer",
        # sdk/mini_infer_sdk/backends/local.py -> ../../../build
        Path(__file__).resolve().parent.parent.parent.parent / "build" / "mini_infer",
    ])
    root = os.environ.get("MINI_INFER_ROOT")
    if root:
        candidates.append(Path(root) / "build" / "mini_infer")
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("mini_infer")
    if found:
        return Path(found)
    return candidates[0] if candidates else Path("mini_infer")


def _build_chatml(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build a ChatML prompt (or plain fallback) from chat messages."""
    has_im = tokenizer.token_to_id("<|im_start|>") is not None
    if has_im:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    parts = []
    for m in messages:
        role = m["role"]
        if role == "system":
            parts.append(f"System: {m['content']}")
        elif role == "assistant":
            parts.append(f"Assistant: {m['content']}")
        else:
            parts.append(f"User: {m['content']}")
    parts.append("Assistant: ")
    return "\n".join(parts)


class LocalBackend(Backend):
    """Offline backend: spawns the mini_infer binary per call.

    The model is reloaded on each invocation (the C++ binary loads weights,
    generates, exits). For persistent in-memory serving use `RemoteBackend`
    against a running `mini_infer serve` instance.
    """

    def __init__(self, model_path: str,
                 draft_path: Optional[str] = None,
                 binary: Optional[str] = None,
                 python_path: str = "python3",
                 config: Optional[InferenceConfig] = None):
        self.model_path = Path(model_path)
        self.draft_path = Path(draft_path) if draft_path else None
        self.python_path = python_path
        self.config = config or InferenceConfig(python_path=python_path)
        self.config.draft_path = draft_path
        self.binary = _find_binary(binary)
        if not self.binary.exists():
            raise FileNotFoundError(
                f"mini_infer binary not found at {self.binary}. "
                f"Build with: scripts/build.sh"
            )
        self._tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        from tokenizers import Tokenizer
        path = self.model_path / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_path}")
        return Tokenizer.from_file(str(path))

    @property
    def model_name(self) -> str:
        return self.model_path.name

    # -- Backend API --------------------------------------------------------
    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def detokenize(self, ids: List[int]) -> str:
        return self._tokenizer.decode([max(0, x) for x in ids])

    def _merge(self, config: Optional[InferenceConfig]) -> InferenceConfig:
        if config is None:
            return self.config
        return InferenceConfig(
            max_new_tokens=config.max_new_tokens,
            max_seq_len=config.max_seq_len,
            device=config.device if config.device != 0 else self.config.device,
            paged=config.paged,
            gamma=config.gamma,
            sampling=config.sampling,
            draft_path=config.draft_path or self.config.draft_path,
            python_path=config.python_path or self.config.python_path,
        )

    def _build_cmd(self, prompt: str, config: InferenceConfig,
                   raw: bool = False) -> List[str]:
        cmd = [
            str(self.binary),
            "--model", str(self.model_path),
            "--prompt", prompt,
            "--max-new-tokens", str(config.max_new_tokens),
            "--max-seq-len", str(config.max_seq_len),
            "--device", str(config.device),
            "--seed", str(config.sampling.seed),
            "--python", config.python_path or self.python_path,
        ]
        if config.sampling.mode == SamplingMode.GREEDY:
            cmd.append("--greedy")
        else:
            cmd.extend(["--temperature", str(config.sampling.temperature),
                        "--top-p", str(config.sampling.top_p)])
        if raw:
            cmd.append("--raw")
        if config.paged and not config.draft_path and not raw:
            cmd.append("--paged")
        if config.draft_path:
            cmd.extend(["--spec-draft", str(config.draft_path),
                        "--gamma", str(config.gamma)])
        return cmd

    def _env(self) -> dict:
        env = os.environ.copy()
        env["PATH"] = f"{self.python_path}:{env.get('PATH', '')}"
        return env

    @staticmethod
    def _parse_output(stdout: str, stderr: str, prompt: str,
                     wall_sec: float, tokenizer) -> GenerationResult:
        output = stdout + "\n" + stderr
        text = ""
        accept_rate = None
        infer_tps = None
        infer_sec = None
        in_generated = False
        for line in output.split("\n"):
            if line.strip() == "--- generated ---":
                in_generated = True
                continue
            if line.strip() == "--- end ---":
                in_generated = False
                continue
            if in_generated:
                text += line + "\n"
            if "[mini-infer]" in line and "tok/s" in line:
                m = re.search(r"(\d+\.?\d*)\s+tok/s", line)
                if m:
                    infer_tps = float(m.group(1))
                m2 = re.search(r"in\s+([\d.]+)s", line)
                if m2:
                    infer_sec = float(m2.group(1))
            if "accept_rate" in line:
                try:
                    accept_rate = float(line.split("accept_rate=")[-1].split("%")[0].strip()) / 100
                except (ValueError, IndexError):
                    pass
        text = text.strip()
        prompt_ids = tokenizer.encode(prompt).ids
        gen_ids = tokenizer.encode(text).ids
        return GenerationResult(
            text=text,
            tokens=prompt_ids + gen_ids,
            prompt_tokens=len(prompt_ids),
            elapsed_sec=infer_sec if infer_sec is not None else wall_sec,
            accept_rate=accept_rate,
            tokens_per_sec=infer_tps,
            wall_sec=wall_sec,
        )

    def _run(self, prompt: str, config: InferenceConfig,
             raw: bool = False) -> GenerationResult:
        cmd = self._build_cmd(prompt, config, raw=raw)
        env = self._env()
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=config.max_new_tokens * 2 + 120,
        )
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"mini_infer failed (rc={proc.returncode}):\n{proc.stderr}")
        return self._parse_output(proc.stdout, proc.stderr, prompt, wall, self._tokenizer)

    def generate(self, prompt: str, config: Optional[InferenceConfig] = None) -> GenerationResult:
        return self._run(prompt, self._merge(config), raw=False)

    def chat(self, messages: List[Dict[str, str]],
             config: Optional[InferenceConfig] = None) -> str:
        prompt = _build_chatml(self._tokenizer, messages)
        return self._run(prompt, self._merge(config), raw=True).text

    def _run_stream(self, prompt: str, config: InferenceConfig,
                    raw: bool = False) -> Iterator[str]:
        cmd = self._build_cmd(prompt, config, raw=raw)
        env = self._env()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        in_generated = False
        try:
            for line in proc.stdout:
                if line.strip() == "--- generated ---":
                    in_generated = True
                    continue
                if line.strip() == "--- end ---":
                    break
                if in_generated:
                    yield line
        finally:
            proc.terminate()
            proc.wait()

    def stream(self, prompt: str, config: Optional[InferenceConfig] = None) -> Iterator[str]:
        yield from self._run_stream(prompt, self._merge(config), raw=False)

    def chat_stream(self, messages: List[Dict[str, str]],
                    config: Optional[InferenceConfig] = None) -> Iterator[str]:
        prompt = _build_chatml(self._tokenizer, messages)
        yield from self._run_stream(prompt, self._merge(config), raw=True)
