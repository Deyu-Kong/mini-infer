"""Core engine wrapper for mini-infer C++ CLI."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict

from .config import InferenceConfig, SamplingConfig, SamplingMode


class GenerationResult:
    """Result of a generation call.

    Attributes:
        text:           Generated text.
        tokens:         Full token list (prompt + generated).
        generated_tokens: Number of tokens generated.
        elapsed_sec:    Pure inference time (excludes model loading).
        wall_sec:       Wall-clock time including model loading.
        tokens_per_sec: Pure inference throughput (tok/s).
        accept_rate:    Speculative decoding acceptance rate (or None).
    """
    def __init__(self, text: str, tokens: List[int], prompt_tokens: int,
                 elapsed_sec: float, accept_rate: Optional[float] = None,
                 tokens_per_sec: Optional[float] = None,
                 wall_sec: Optional[float] = None):
        self.text = text
        self.tokens = tokens
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = len(tokens) - prompt_tokens
        self.elapsed_sec = elapsed_sec
        self.wall_sec = wall_sec if wall_sec is not None else elapsed_sec
        if tokens_per_sec is not None:
            self.tokens_per_sec = tokens_per_sec
        else:
            self.tokens_per_sec = self.generated_tokens / elapsed_sec if elapsed_sec > 0 else 0
        self.accept_rate = accept_rate

    def __repr__(self):
        extra = f", accept_rate={self.accept_rate:.0%}" if self.accept_rate else ""
        return (f"GenerationResult(generated={self.generated_tokens}, "
                f"tps={self.tokens_per_sec:.1f}{extra})")


class MiniInfer:
    """Python interface for the mini-infer C++/CUDA inference engine.

    Wraps the CLI binary via subprocess and provides a clean Python API.

    Args:
        model_path: Path to HuggingFace model directory.
        draft_path: Optional path to draft model for speculative decoding.
        binary: Path to mini_infer binary (auto-detected if None).
        python_path: Path to Python executable for tokenizer subprocess.
        config: Default inference configuration.
    """

    def __init__(
        self,
        model_path: str,
        draft_path: Optional[str] = None,
        binary: Optional[str] = None,
        python_path: str = "python3",
        config: Optional[InferenceConfig] = None,
    ):
        self.model_path = Path(model_path)
        self.draft_path = Path(draft_path) if draft_path else None
        self.python_path = python_path
        self.config = config or InferenceConfig(python_path=python_path)
        self.config.draft_path = draft_path

        if binary:
            self.binary = Path(binary)
        else:
            self.binary = self._find_binary()

        if not self.binary.exists():
            raise FileNotFoundError(
                f"mini_infer binary not found at {self.binary}. "
                f"Build with: scripts/build.sh"
            )

        self._tokenizer = self._init_tokenizer()

    def _find_binary(self) -> Path:
        """Auto-discover the mini_infer binary.

        Search order:
          1. $MINI_INFER_BINARY env var
          2. ./build/mini_infer (current working directory)
          3. ../../build/mini_infer relative to this file (project root)
          4. $MINI_INFER_ROOT/build/mini_infer
          5. `which mini_infer` on PATH
        """
        # 1. Explicit env var
        env_bin = os.environ.get("MINI_INFER_BINARY")
        if env_bin:
            p = Path(env_bin)
            if p.exists():
                return p

        # 2. CWD/build
        candidates = [
            Path.cwd() / "build" / "mini_infer",
            # 3. Relative to this file: sdk/mini_infer_sdk/engine.py -> ../../build
            Path(__file__).resolve().parent.parent.parent / "build" / "mini_infer",
        ]

        # 4. $MINI_INFER_ROOT/build
        root = os.environ.get("MINI_INFER_ROOT")
        if root:
            candidates.append(Path(root) / "build" / "mini_infer")

        for c in candidates:
            if c.exists():
                return c

        # 5. Search PATH
        import shutil
        found = shutil.which("mini_infer")
        if found:
            return Path(found)

        # Nothing found — return the most likely candidate for a helpful error
        return candidates[0] if candidates else Path("mini_infer")

    def _init_tokenizer(self):
        from tokenizers import Tokenizer
        tokenizer_path = self.model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_path}")
        return Tokenizer.from_file(str(tokenizer_path))

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def detokenize(self, ids: List[int]) -> str:
        return self._tokenizer.decode([max(0, x) for x in ids])

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id("<|im_end|>") or self._tokenizer.token_to_id("eos_token") or -1

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id("bos_token") or -1

    def _merge_config(self, config: Optional[InferenceConfig]) -> InferenceConfig:
        """Merge a user-provided config with the engine defaults.

        Ensures model-level settings (draft_path, device) are preserved
        even when the user passes a partial config.
        """
        if config is None:
            return self.config
        merged = InferenceConfig(
            max_new_tokens=config.max_new_tokens,
            max_seq_len=config.max_seq_len,
            device=config.device if config.device != 0 else self.config.device,
            paged=config.paged,
            gamma=config.gamma,
            sampling=config.sampling,
            draft_path=config.draft_path or self.config.draft_path,
            python_path=config.python_path or self.config.python_path,
        )
        return merged

    def _build_cmd(self, prompt: str, config: Optional[InferenceConfig] = None) -> List[str]:
        cfg = self._merge_config(config)
        cmd = [
            str(self.binary),
            "--model", str(self.model_path),
            "--prompt", prompt,
            "--max-new-tokens", str(cfg.max_new_tokens),
            "--max-seq-len", str(cfg.max_seq_len),
            "--device", str(cfg.device),
            "--seed", str(cfg.sampling.seed),
        ]

        if cfg.sampling.mode == SamplingMode.GREEDY:
            cmd.append("--greedy")
        else:
            cmd.extend(["--temperature", str(cfg.sampling.temperature),
                        "--top-p", str(cfg.sampling.top_p)])

        if cfg.paged and not cfg.draft_path:
            cmd.append("--paged")

        if cfg.draft_path:
            cmd.extend(["--spec-draft", str(cfg.draft_path), "--gamma", str(cfg.gamma)])

        return cmd

    def generate(self, prompt: str, config: Optional[InferenceConfig] = None) -> GenerationResult:
        """Generate text from a single prompt.

        Args:
            prompt: Input text prompt.
            config: Override default inference configuration.

        Returns:
            GenerationResult with text, tokens, and timing info.
        """
        cfg = config or self.config
        cmd = self._build_cmd(prompt, cfg)

        env = os.environ.copy()
        env["PATH"] = f"{self.python_path}:{env.get('PATH', '')}"

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=cfg.max_new_tokens * 2 + 120,
        )
        t1 = time.perf_counter()

        if proc.returncode != 0:
            raise RuntimeError(f"mini_infer failed (rc={proc.returncode}):\n{proc.stderr}")

        output = proc.stdout + "\n" + proc.stderr
        text = ""
        accept_rate = None
        infer_tps = None      # pure inference TPS (from CLI, excludes model load)
        infer_sec = None       # pure inference time

        # Parse output
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

            # Parse: [mini-infer] 10 tokens in 0.26s (38.33 tok/s)
            if "[mini-infer]" in line and "tok/s" in line:
                try:
                    import re
                    m = re.search(r"(\d+\.?\d*)\s+tok/s", line)
                    if m:
                        infer_tps = float(m.group(1))
                    m2 = re.search(r"in\s+([\d.]+)s", line)
                    if m2:
                        infer_sec = float(m2.group(1))
                except (ValueError, IndexError):
                    pass

            if "accept_rate" in line:
                try:
                    accept_rate = float(line.split("accept_rate=")[-1].split("%")[0].strip()) / 100
                except (ValueError, IndexError):
                    pass

        text = text.strip()
        prompt_ids = self.tokenize(prompt)
        gen_ids = self.tokenize(text)

        wall_sec = t1 - t0
        return GenerationResult(
            text=text,
            tokens=prompt_ids + gen_ids,
            prompt_tokens=len(prompt_ids),
            elapsed_sec=infer_sec if infer_sec is not None else wall_sec,
            accept_rate=accept_rate,
            tokens_per_sec=infer_tps,
            wall_sec=wall_sec,
        )

    def stream(self, prompt: str, config: Optional[InferenceConfig] = None):
        """Stream generated tokens as they arrive.

        Args:
            prompt: Input text prompt.
            config: Override default inference configuration.

        Yields:
            String chunks of generated text.
        """
        cfg = config or self.config
        cmd = self._build_cmd(prompt, cfg)

        env = os.environ.copy()
        env["PATH"] = f"{self.python_path}:{env.get('PATH', '')}"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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

    def chat(self, system_prompt: Optional[str] = None) -> "ChatSession":
        """Create a multi-turn chat session.

        Args:
            system_prompt: Optional system instruction.

        Returns:
            ChatSession instance for multi-turn conversation.
        """
        from .chat import ChatSession
        return ChatSession(self, system_prompt=system_prompt)

    def batch(self, config: Optional[InferenceConfig] = None) -> "BatchInfer":
        """Create a batch inference runner.

        Args:
            config: Override default inference configuration.

        Returns:
            BatchInfer instance for processing multiple prompts.
        """
        from .batch import BatchInfer
        return BatchInfer(self, config=config or self.config)

    def __repr__(self):
        draft = f", draft={self.draft_path.name}" if self.draft_path else ""
        return f"MiniInfer(model={self.model_path.name}{draft})"