"""MiniInferServer — launch and manage a persistent online inference service.

This is the SDK's "service" side of online inference. It spawns the C++
`mini_infer serve` binary as a managed subprocess: the model is loaded
**once** into GPU memory and kept resident, so every subsequent request
reuses the loaded weights (no per-call reload — the key win over the
offline LocalBackend, which reloads on each call).

Pair it with `MiniInfer(endpoint=server.endpoint)` (the SDK client) to send
a continuous stream of requests to the long-lived service.

    from mini_infer_sdk import MiniInfer, MiniInferServer

    with MiniInferServer(model="/path/to/model", port=8000) as srv:
        engine = MiniInfer(endpoint=srv.endpoint)
        for q in questions:
            print(engine.generate(q).text)

The Python interpreter running the SDK already has the `tokenizers` package
(a SDK dependency), so by default `sys.executable` is handed to the binary —
no `--python` plumbing required.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .backends.local import _find_binary

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


class MiniInferServer:
    """A managed, persistent mini-infer HTTP inference service.

    Args:
        model:           HuggingFace model directory (loaded once, resident).
        port:            TCP port to listen on.
        host:            Bind address (use 0.0.0.0 to expose remotely).
        device:          GPU device index.
        draft:           Optional draft model for speculative decoding.
        gamma:           Speculative-decoding gamma.
        python:          Python executable for the engine's tokenizer.
                         Defaults to ``sys.executable`` (the interpreter
                         running the SDK — it already has `tokenizers`).
        binary:          Path to the `mini_infer` binary (auto-discovered).
        max_new_tokens:  Default per-request generation cap.
        startup_timeout: Seconds to wait for /health before giving up.
    """

    def __init__(
        self,
        model: str,
        port: int = 8000,
        host: str = "127.0.0.1",
        device: int = 0,
        draft: Optional[str] = None,
        gamma: int = 4,
        python: Optional[str] = None,
        binary: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        startup_timeout: int = 180,
    ):
        self.model = model
        self.port = port
        self.host = host
        self._endpoint = f"http://{host}:{port}"
        self._python = python or sys.executable
        self._binary = _find_binary(binary)
        if not self._binary.exists():
            raise FileNotFoundError(
                f"mini_infer binary not found at {self._binary}. "
                f"Build with: scripts/build.sh"
            )

        cmd: List[str] = [
            str(self._binary), "serve",
            "--model", str(model),
            "--host", host,
            "--port", str(port),
            "--device", str(device),
            "--python", self._python,
        ]
        if draft:
            cmd += ["--draft", str(draft), "--gamma", str(gamma)]
        if max_new_tokens:
            cmd += ["--max-new-tokens", str(max_new_tokens)]
        self._cmd = cmd
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[Path] = None
        self._start(startup_timeout)

    # -- lifecycle ---------------------------------------------------------
    def _start(self, timeout: int) -> None:
        import tempfile
        self._log_path = Path(tempfile.mktemp(prefix="mini_infer_serve_", suffix=".log"))
        # Inherit nothing; redirect so the subprocess never holds the
        # caller's stdout/stderr pipes.
        log_fh = open(self._log_path, "w")
        try:
            self._proc = subprocess.Popen(
                self._cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            log_fh.close()
            raise RuntimeError(f"failed to launch server: {e}\ncmd: {' '.join(self._cmd)}")
        # Close the handle in this process; the child keeps its copy.
        log_fh.close()

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (rc={self._proc.returncode}); "
                    f"see log: {self._log_path}\n--- log ---\n{self.logs()}"
                )
            if requests is not None:
                try:
                    if requests.get(f"{self._endpoint}/health", timeout=1).ok:
                        return
                except Exception:
                    pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(
            f"server not ready after {timeout}s; see log: {self._log_path}\n"
            f"--- log ---\n{self.logs()}"
        )

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        self._proc = None

    def logs(self) -> str:
        if self._log_path and self._log_path.exists():
            return self._log_path.read_text(errors="replace")
        return ""

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def __enter__(self) -> "MiniInferServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        alive = "up" if (self._proc and self._proc.poll() is None) else "down"
        return f"MiniInferServer({self._endpoint}, model={Path(self.model).name}, {alive})"
