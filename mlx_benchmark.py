"""
MLX (mlx-lm) benchmark for Gemma / chat models: TTFT, throughput, total time, peak RSS.

Uses stream_generate with a sampler from make_sampler(temp=...). Text-only: chat template
when available. mlx-vlm is not required for this path; install per project instructions.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import psutil
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


def _build_prompt_ids(tokenizer, user_text: str) -> mx.array:
    messages = [{"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None) is None:
        ids = tokenizer.encode(user_text)
    else:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
    return mx.array(ids)


def _peak_rss_while(pid: int, work: Callable[[], None]) -> int:
    """Sample RSS every 50ms during work; return peak bytes."""
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


def _one_generation(
    model,
    tokenizer,
    prompt_mx: mx.array,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    sampler = make_sampler(temp=temperature, top_p=1.0)
    pid = os.getpid()

    ttft_s: Optional[float] = None
    total_time_s: Optional[float] = None
    n_tokens = 0

    def run_stream() -> None:
        nonlocal ttft_s, total_time_s, n_tokens
        t0 = time.perf_counter()
        first = True
        for response in stream_generate(
            model,
            tokenizer,
            prompt_mx,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            now = time.perf_counter()
            if first:
                ttft_s = now - t0
                gen_start = now
                first = False
            # stream_generate emits a duplicate finalize response; take max count.
            n_tokens = max(n_tokens, response.generation_tokens)
        total_time_s = time.perf_counter() - t0

    peak_rss = _peak_rss_while(pid, run_stream)

    assert ttft_s is not None and total_time_s is not None
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
    }


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load model once, run ``num_runs`` generations, return per-run metrics and averages.

    Expected keys: prompt, max_tokens, temperature, num_runs, mlx.model_id
    """
    mlx_cfg = config.get("mlx", {})
    model_id = mlx_cfg.get("model_id", "mlx-community/gemma-4-e2b-it-mxfp4")
    prompt = config["prompt"]
    max_tokens = int(config.get("max_tokens", 200))
    temperature = float(config.get("temperature", 0.7))
    num_runs = int(config.get("num_runs", 3))

    t_load0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_time_s = time.perf_counter() - t_load0

    prompt_mx = _build_prompt_ids(tokenizer, prompt)

    runs: List[Dict[str, Any]] = []
    for i in range(num_runs):
        mx.clear_cache()
        row = _one_generation(model, tokenizer, prompt_mx, max_tokens, temperature)
        row["run_index"] = i
        runs.append(row)

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
        "backend": "mlx",
        "model_id": model_id,
        "quantization": mlx_cfg.get("quantization"),
        "load_time_s": load_time_s,
        "runs": runs,
        "average": average,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="MLX-only benchmark")
    p.add_argument("--config", type=str, default=None, help="JSON config path")
    p.add_argument("--model", type=str, default=None, help="Override mlx model id")
    args = p.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    if args.model:
        cfg.setdefault("mlx", {})["model_id"] = args.model

    # Defaults match config.example.json
    defaults = {
        "prompt": "Explain transformers architecture in simple terms.",
        "max_tokens": 200,
        "temperature": 0.7,
        "num_runs": 3,
        "mlx": {"model_id": "mlx-community/gemma-4-e2b-it-mxfp4"},
    }
    merged = {**defaults, **cfg}
    merged["mlx"] = {**defaults["mlx"], **merged.get("mlx", {})}

    out = run_benchmark(merged)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
