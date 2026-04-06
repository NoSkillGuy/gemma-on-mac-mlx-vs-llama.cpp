#!/usr/bin/env python3
"""
Run MLX and llama.cpp benchmarks with shared config, print comparison table, write results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from llama_benchmark import run_benchmark as run_llama
from mlx_benchmark import run_benchmark as run_mlx


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_config() -> Dict[str, Any]:
    example = _repo_root() / "config.example.json"
    if example.is_file():
        with open(example, encoding="utf-8") as f:
            return json.load(f)
    return {
        "prompt": "Explain transformers architecture in simple terms.",
        "max_tokens": 200,
        "temperature": 0.7,
        "num_runs": 3,
        "mlx": {
            "model_id": "mlx-community/gemma-4-e2b-it-mxfp4",
            "quantization": "mxfp4",
        },
        "llama_cpp": {
            "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
            "gguf_filename": "gemma-4-E2B-it-Q4_K_M.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
            "local_mmproj_path": None,
            "local_gguf_path": None,
            "n_gpu_layers": -1,
            "context_size": 8192,
            "llama_server_bin": None,
            "quantization": "Q4_K_M",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env(cfg: Dict[str, Any]) -> None:
    if os.environ.get("MLX_MODEL_ID"):
        cfg.setdefault("mlx", {})["model_id"] = os.environ["MLX_MODEL_ID"]
    if os.environ.get("GGUF_REPO"):
        cfg.setdefault("llama_cpp", {})["gguf_repo"] = os.environ["GGUF_REPO"]
    if os.environ.get("GGUF_FILENAME"):
        cfg.setdefault("llama_cpp", {})["gguf_filename"] = os.environ["GGUF_FILENAME"]
    if os.environ.get("LOCAL_GGUF_PATH"):
        cfg.setdefault("llama_cpp", {})["local_gguf_path"] = os.environ["LOCAL_GGUF_PATH"]
    if os.environ.get("LLAMA_SERVER_BIN"):
        cfg.setdefault("llama_cpp", {})["llama_server_bin"] = os.environ["LLAMA_SERVER_BIN"]


def _fmt_s(x: float) -> str:
    if x >= 10:
        return f"{x:.2f}s"
    return f"{x:.3f}s"


def _fmt_tok_s(x: float) -> str:
    if x >= 100:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_mem(rss_bytes: float) -> str:
    mb = rss_bytes / (1024**2)
    return f"{mb:.0f} MB"


def _print_table(mlx_avg: Dict[str, Any], llama_avg: Dict[str, Any]) -> None:
    print()
    print("| Runtime   | TTFT   | Tokens/sec | Total time | Memory  |")
    print("| --------- | ------ | ---------- | ---------- | ------- |")
    for name, a in (("MLX", mlx_avg), ("llama.cpp", llama_avg)):
        print(
            f"| {name:<9} | {_fmt_s(a['ttft_s']):>6} | {_fmt_tok_s(a['tokens_per_sec']):>10} | "
            f"{_fmt_s(a['total_time_s']):>10} | {_fmt_mem(a['peak_memory_rss_bytes']):>7} |"
        )
    print()


def _print_one_row(label: str, a: Dict[str, Any]) -> None:
    print()
    print("| Runtime   | TTFT   | Tokens/sec | Total time | Memory  |")
    print("| --------- | ------ | ---------- | ---------- | ------- |")
    print(
        f"| {label:<9} | {_fmt_s(a['ttft_s']):>6} | {_fmt_tok_s(a['tokens_per_sec']):>10} | "
        f"{_fmt_s(a['total_time_s']):>10} | {_fmt_mem(a['peak_memory_rss_bytes']):>7} |"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gemma 4 E2B: MLX vs llama.cpp (GGUF via llama-server)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config path (default: config.json if present, else config.example.json fields)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Write full results here (default: results.json)",
    )
    parser.add_argument("--mlx-only", action="store_true", help="Only run MLX benchmark")
    parser.add_argument(
        "--llama-only", action="store_true", help="Only run llama.cpp benchmark"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-run details")
    args = parser.parse_args()

    cfg_path = args.config or os.environ.get("BENCHMARK_CONFIG")
    base = _default_config()
    if cfg_path and Path(cfg_path).is_file():
        with open(cfg_path, encoding="utf-8") as f:
            user = json.load(f)
        cfg = _deep_merge(base, user)
    else:
        local = _repo_root() / "config.json"
        if local.is_file():
            with open(local, encoding="utf-8") as f:
                user = json.load(f)
            cfg = _deep_merge(base, user)
        else:
            cfg = base

    _apply_env(cfg)

    if args.mlx_only and args.llama_only:
        print("Choose at most one of --mlx-only / --llama-only", file=sys.stderr)
        sys.exit(2)

    results: Dict[str, Any] = {
        "meta": {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": cfg,
        },
        "mlx": None,
        "llama_cpp": None,
    }

    t_all = time.perf_counter()
    err: Optional[str] = None

    try:
        if not args.llama_only:
            print("Running MLX benchmark…", flush=True)
            results["mlx"] = run_mlx(cfg)
            if args.verbose:
                print(json.dumps(results["mlx"], indent=2))

        if not args.mlx_only:
            print("Running llama.cpp (llama-server) benchmark…", flush=True)
            results["llama_cpp"] = run_llama(cfg)
            if args.verbose:
                print(json.dumps(results["llama_cpp"], indent=2))

        results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
        results["meta"]["total_wall_time_s"] = time.perf_counter() - t_all

        mlx_avg = (results["mlx"] or {}).get("average") or {}
        llama_avg = (results["llama_cpp"] or {}).get("average") or {}

        if mlx_avg and llama_avg:
            _print_table(mlx_avg, llama_avg)
        elif mlx_avg:
            _print_one_row("MLX", mlx_avg)
        elif llama_avg:
            _print_one_row("llama.cpp", llama_avg)

        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_path.resolve()}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        results["meta"]["error"] = err
        results["meta"]["traceback"] = traceback.format_exc()
        print(err, file=sys.stderr)
        try:
            out_path = Path(args.output)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Partial results written to {out_path.resolve()}", file=sys.stderr)
        except OSError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
