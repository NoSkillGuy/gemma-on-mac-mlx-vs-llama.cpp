"""
llama.cpp (llama-server + GGUF) benchmark: TTFT, throughput, total time, server RSS.

Spawns a local llama-server with Metal GPU layers (-ngl), streams OpenAI-compatible
chat completions, parses SSE, then shuts the server down.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import psutil
import requests
from huggingface_hub import hf_hub_download


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _iter_llama_server_candidates() -> List[str]:
    """Paths to try when `llama-server` is not on PATH (common Homebrew layouts)."""
    out: List[str] = []
    w = shutil.which("llama-server")
    if w:
        out.append(w)
    try:
        r = subprocess.run(
            ["brew", "--prefix", "llama.cpp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            out.append(os.path.join(r.stdout.strip(), "bin", "llama-server"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    out.extend(
        [
            "/opt/homebrew/bin/llama-server",
            "/usr/local/bin/llama-server",
        ]
    )
    return out


def _resolve_llama_server_bin(override: Optional[str]) -> str:
    if override:
        p = os.path.expanduser(override)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        raise FileNotFoundError(f"llama_server_bin not executable or missing: {p}")

    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        p = os.path.expanduser(env)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        raise FileNotFoundError(f"LLAMA_SERVER_BIN not executable or missing: {p}")

    seen: set[str] = set()
    for candidate in _iter_llama_server_candidates():
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "llama-server not found. Install a Metal build, e.g. "
        "`brew install llama.cpp` (Apple Silicon: ensure /opt/homebrew/bin is on PATH), "
        "or set llama_cpp.llama_server_bin / env LLAMA_SERVER_BIN to the full path."
    )


def _resolve_gguf_path(cfg: Dict[str, Any]) -> str:
    lc = cfg.get("llama_cpp", {})
    local = lc.get("local_gguf_path")
    if local:
        p = os.path.expanduser(local)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"local_gguf_path not found: {p}")
        return p
    repo = lc.get("gguf_repo", "unsloth/gemma-4-E2B-it-GGUF")
    filename = lc.get("gguf_filename", "gemma-4-E2B-it-Q4_K_M.gguf")
    return hf_hub_download(repo_id=repo, filename=filename)


def _resolve_mmproj_path(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Gemma 4 E2B (multimodal) GGUFs need a projector file for llama-server chat.
    Set llama_cpp.mmproj_filename to null to skip (only if your GGUF is text-only).
    """
    lc = cfg.get("llama_cpp", {})
    local = lc.get("local_mmproj_path")
    if local:
        p = os.path.expanduser(local)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"local_mmproj_path not found: {p}")
        return p
    fn = lc.get("mmproj_filename", "mmproj-F16.gguf")
    if fn is None or fn == "":
        return None
    repo = lc.get("gguf_repo", "unsloth/gemma-4-E2B-it-GGUF")
    return hf_hub_download(repo_id=repo, filename=str(fn))


def _delta_text(delta: Any) -> str:
    """
    Normalize OpenAI-style delta fields from llama-server.

    llama.cpp may stream `reasoning_content` (thinking) and/or `content` (see
    common_chat_msg_diff_to_json_oaicompat in llama.cpp). Ignoring
    `reasoning_content` yields an empty stream for models that only emit
    reasoning deltas first or use a reasoning-capable template.
    """
    if not isinstance(delta, dict):
        return ""

    def one(field: str) -> str:
        v = delta.get(field)
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts: List[str] = []
            for part in v:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        parts.append(str(part["text"]))
                    elif "text" in part:
                        parts.append(str(part["text"]))
            return "".join(parts)
        return str(v)

    # Prefer visible answer; include reasoning so TTFT / throughput reflect real work.
    return one("content") + one("reasoning_content")


def _read_file_tail(path: str, max_bytes: int = 12000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _diagnose_non_stream_chat(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    """If streaming yields nothing, check whether non-streaming chat works (same endpoint)."""
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        r = requests.post(url, json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        choices = j.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        text = str(msg.get("content") or "") + str(msg.get("reasoning_content") or "")
        if text.strip():
            return f"non-streaming /v1/chat/completions returned {len(text)} characters (first 120 chars): {text[:120]!r}"
    except requests.RequestException:
        pass
    return None


def _wait_for_server(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.2)
    raise TimeoutError(f"llama-server did not become ready at {base_url} within {timeout_s}s")


def _peak_rss_while(pid: int, work: Callable[[], None]) -> int:
    stop = threading.Event()
    peak = [0]

    def sample() -> None:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        while not stop.wait(0.05):
            try:
                peak[0] = max(peak[0], proc.memory_info().rss)
            except psutil.NoSuchProcess:
                break

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        work()
    finally:
        stop.set()
        th.join(timeout=3.0)
    try:
        peak[0] = max(peak[0], psutil.Process(pid).memory_info().rss)
    except psutil.NoSuchProcess:
        pass
    return int(peak[0])


def _stream_chat_completion(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    server_pid: int,
    server_stderr_path: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    ttft_s: Optional[float] = None
    total_time_s: Optional[float] = None
    usage_completion_tokens: Optional[int] = None
    text_parts: List[str] = []

    def run_req() -> None:
        nonlocal ttft_s, total_time_s, usage_completion_tokens
        t0 = time.perf_counter()
        first_token = True
        with requests.post(url, json=body, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                err = obj.get("error")
                if err is not None:
                    if isinstance(err, dict):
                        msg = err.get("message", json.dumps(err))
                    else:
                        msg = str(err)
                    raise RuntimeError(f"llama-server returned error in stream: {msg}")
                u = obj.get("usage")
                if isinstance(u, dict):
                    ct = u.get("completion_tokens")
                    if ct is not None:
                        usage_completion_tokens = int(ct)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0]
                delta = ch0.get("delta") or {}
                piece = _delta_text(delta)
                if not piece and isinstance(ch0.get("message"), dict):
                    m = ch0["message"]
                    piece = str(m.get("content") or "") + str(m.get("reasoning_content") or "")
                if not piece and isinstance(ch0.get("text"), str):
                    piece = ch0["text"]
                if piece:
                    text_parts.append(piece)
                    now = time.perf_counter()
                    if first_token:
                        ttft_s = now - t0
                        first_token = False
        total_time_s = time.perf_counter() - t0

    peak_rss = _peak_rss_while(server_pid, run_req)

    if ttft_s is None:
        extra = _diagnose_non_stream_chat(
            base_url, messages, max_tokens, temperature
        )
        tail = (
            _read_file_tail(server_stderr_path)
            if server_stderr_path
            else ""
        )
        msg = (
            "No streamed token received from llama-server (empty stream or parse error). "
            "Streaming deltas may use `reasoning_content` without `content` on some models; "
            "this client counts both. "
            "For Gemma 4 E2B multimodal GGUF, ensure `--mmproj` is set (see config / README)."
        )
        if extra:
            msg += f" Diagnostic: {extra}"
        if tail.strip():
            msg += f" llama-server stderr (tail): {tail.strip()}"
        raise RuntimeError(msg)
    assert total_time_s is not None

    full_text = "".join(text_parts)
    if usage_completion_tokens is not None and usage_completion_tokens > 0:
        n_tokens = usage_completion_tokens
        tokens_from_usage = True
    else:
        # Rough estimate when stream omits usage (common on some llama-server builds).
        n_tokens = max(1, len(full_text) // 4)
        tokens_from_usage = False

    gen_wall = total_time_s - ttft_s
    tokens_per_sec = (n_tokens / gen_wall) if gen_wall > 0 else float("inf")
    tokens_per_sec_all = (n_tokens / total_time_s) if total_time_s > 0 else 0.0

    return {
        "ttft_s": ttft_s,
        "total_time_s": total_time_s,
        "tokens_generated": n_tokens,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_including_prefill": tokens_per_sec_all,
        "peak_memory_rss_bytes": peak_rss,
        "tokens_from_usage": tokens_from_usage,
    }


class _LlamaServer:
    def __init__(self, cmd: List[str], port: int, stderr_file: Optional[Any] = None) -> None:
        self.cmd = cmd
        self.port = port
        self._stderr_file = stderr_file
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "_LlamaServer":
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file if self._stderr_file is not None else subprocess.DEVNULL,
            text=True,
        )
        base = f"http://127.0.0.1:{self.port}"
        try:
            _wait_for_server(base)
        except Exception:
            self._terminate()
            raise
        return self

    def _terminate(self) -> None:
        if not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def __exit__(self, *args: Any) -> None:
        self._terminate()
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except OSError:
                pass


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    lc = config.get("llama_cpp", {})
    gguf_path = _resolve_gguf_path(config)
    mmproj_path = _resolve_mmproj_path(config)
    llama_bin = _resolve_llama_server_bin(lc.get("llama_server_bin"))
    context_size = int(lc.get("context_size", 8192))
    n_gpu_layers = lc.get("n_gpu_layers", -1)
    if n_gpu_layers is None:
        n_gpu_layers = -1

    prompt = config["prompt"]
    max_tokens = int(config.get("max_tokens", 200))
    temperature = float(config.get("temperature", 0.7))
    num_runs = int(config.get("num_runs", 3))

    port = _pick_free_port()
    host = "127.0.0.1"
    cmd = [
        llama_bin,
        "-m",
        gguf_path,
        "-c",
        str(context_size),
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        str(n_gpu_layers),
    ]
    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])

    messages = [{"role": "user", "content": prompt}]
    base_url = f"http://{host}:{port}"

    stderr_f = tempfile.NamedTemporaryFile(
        prefix="llama-server-",
        suffix=".log",
        delete=False,
        mode="w",
        encoding="utf-8",
        errors="replace",
    )
    stderr_path = stderr_f.name

    runs: List[Dict[str, Any]] = []
    load_time_s = 0.0
    t0 = time.perf_counter()
    try:
        with _LlamaServer(cmd, port, stderr_file=stderr_f) as srv:
            load_time_s = time.perf_counter() - t0
            assert srv.proc is not None
            server_pid = srv.proc.pid

            for i in range(num_runs):
                row = _stream_chat_completion(
                    base_url,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    server_pid=server_pid,
                    server_stderr_path=stderr_path,
                )
                row["run_index"] = i
                runs.append(row)
    finally:
        try:
            os.unlink(stderr_path)
        except OSError:
            pass

    def avg(key: str) -> float:
        return sum(r[key] for r in runs) / len(runs)

    average = {
        "ttft_s": avg("ttft_s"),
        "total_time_s": avg("total_time_s"),
        "tokens_generated": avg("tokens_generated"),
        "tokens_per_sec": avg("tokens_per_sec"),
        "peak_memory_rss_bytes": avg("peak_memory_rss_bytes"),
    }

    return {
        "backend": "llama_cpp",
        "gguf_path": gguf_path,
        "mmproj_path": mmproj_path,
        "gguf_repo": lc.get("gguf_repo"),
        "gguf_filename": lc.get("gguf_filename"),
        "quantization": lc.get("quantization"),
        "llama_server_bin": llama_bin,
        "n_gpu_layers": n_gpu_layers,
        "load_time_s": load_time_s,
        "runs": runs,
        "average": average,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="llama.cpp server benchmark")
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    defaults = {
        "prompt": "Explain transformers architecture in simple terms.",
        "max_tokens": 200,
        "temperature": 0.7,
        "num_runs": 3,
        "llama_cpp": {
            "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
            "gguf_filename": "gemma-4-E2B-it-Q4_K_M.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
        },
    }
    merged = {**defaults, **cfg}
    merged["llama_cpp"] = {**defaults["llama_cpp"], **merged.get("llama_cpp", {})}
    out = run_benchmark(merged)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
