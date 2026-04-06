# Gemma 4 E2B: MLX vs llama.cpp benchmark (macOS / Apple Silicon)

Local prototype to compare **Gemma 4 E2B** inference on the same prompt across:

1. **MLX** (`mlx-lm`) — Hugging Face MLX weights (default: quantized `mxfp4`)
2. **llama.cpp** — GGUF via **`llama-server`** (default: Unsloth `Q4_K_M` GGUF)

Metrics: **TTFT**, **tokens/sec** (generation only, after first token), **total time**, **peak RSS** memory. Each backend loads once, then runs **`num_runs`** generations (default **3**) and averages.

## Benchmark report (sample run)

*Example hardware: Apple Silicon (Metal), macOS. Same prompt, 200 max tokens (requested), temperature 0.7, **3 averaged runs** per backend (2026-04-06). Your numbers will differ by chip, OS build, and model files.*

### Summary

On that setup, **MLX and llama.cpp were broadly in the same ballpark** for time-to-first-token (TTFT) and tokens per second, with **MLX using far less resident memory (RSS)**. **llama.cpp** showed slightly **faster first token** on average and **slightly lower** sustained tokens/sec in the aggregate table; **total time** was longer for llama.cpp partly because **it generated more tokens** per run (~223 vs 200) under the same `max_tokens` cap.

The first attempt to benchmark **llama-server** failed until the stack matched **Gemma 4 E2B’s multimodal GGUF** requirements and streaming behavior (see *Integration issues* below).

### Configuration compared

| Aspect | MLX | llama.cpp |
|--------|-----|-----------|
| Model source | `mlx-community/gemma-4-e2b-it-mxfp4` | `unsloth/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-Q4_K_M.gguf` |
| Quantization | mxfp4 | Q4_K_M |
| Server / runtime | In-process `mlx-lm` | Homebrew `llama-server` (e.g. `/opt/homebrew/bin/llama-server`) |
| Multimodal | Native MLX weights | GGUF + **`mmproj-F16.gguf`** (`--mmproj`) |

The two backends are **not identical** in weight format and quantization. Treat throughput and memory as **directional**, not a strict apples-to-apples lab result, unless you match quantization and token counts.

### Results (successful run)

Printed comparison (3-run averages from one session):

| Runtime   | TTFT   | Tokens/sec | Total time | Memory  |
| --------- | ------ | ---------- | ---------- | ------- |
| MLX       | 0.810s | 13.10      | 16.08s     | 473 MB  |
| llama.cpp | 0.721s | 12.56      | 18.44s     | 3345 MB |

Detail from `results.json` for that run:

- **MLX:** `tokens_generated` **200** per run (fixed); TTFT varied a lot on run 0 (~1.39s) vs runs 1–2 (~0.52s), pulling the average up.
- **llama.cpp:** `tokens_generated` **~217–236** per run (**average ~223**); `tokens_from_usage: false` (estimate from text length), so reported tokens/sec is approximate.

**Interpretation:** **TTFT** — llama.cpp ~0.72s vs MLX ~0.81s on average; MLX’s first run had a cold-start-like TTFT spike. **Throughput** — MLX ~13.1 tok/s vs llama.cpp ~12.6 tok/s (methodology differs slightly because llama.cpp emitted more tokens per run). **Memory** — MLX ~496 MB RSS (average) vs llama.cpp ~3.35 GB RSS (llama-server holds the full GGUF + projector + Metal path in a separate process). Full benchmark wall clock was **~119 s** including both loads and three runs each.

### Integration issues (earlier failure)

**Symptom:** `RuntimeError: No streamed token received from llama-server` after downloading weights and `mmproj-F16.gguf`.

**Causes:**

1. **Multimodal GGUF:** Gemma 4 E2B needs **`--mmproj`**; without it, chat streaming can be empty.
2. **Streaming deltas:** llama.cpp can stream **`delta.reasoning_content`** (thinking) separately from **`delta.content`**. Clients that only read `content` see “no tokens.”

This repo’s `llama_benchmark.py` expects **mmproj** and counts **reasoning + content** in streams. See [Troubleshooting](#troubleshooting).

### Recommendations

1. **Choose MLX** for **lower memory**, **simpler Python-native** inference, and MLX tooling on Mac—especially interactive or notebook use.
2. **Choose llama.cpp** for **GGUF** portability, **`llama-server`** (OpenAI-compatible HTTP), or existing llama.cpp tooling. Expect **much higher RSS** for this model class unless you tune context, batching, and offloading.
3. **Fair comparisons:** Align **quantization**, **same effective max tokens**, and use **`usage.completion_tokens`** when the server provides it so token throughput is comparable.
4. **Gemma 4 E2B + llama-server:** Pair the **main GGUF** with the matching **`mmproj`** from the same repo; streaming parsers must handle **`reasoning_content`** and **`content`**.
5. **Repeatability:** Run multiple trials (this harness averages 3 runs); MLX showed **high TTFT variance** between first and later runs—consider a **warmup** discard for steady-state TTFT.
6. **Production:** **Pin** `llama-server` version (e.g. Homebrew) and document **`LLAMA_SERVER_BIN`** when `/opt/homebrew/bin` is not on `PATH`.

### Reproducibility

- Config snapshot: `results.json` → `meta.config` after a run.
- **`results.json`** is gitignored by default; copy or commit a run if you need a permanent record.

## Prerequisites

- **macOS** on **Apple Silicon** (Metal)
- **Python 3.10+**
- **`llama-server`** with a Metal build. Install with Homebrew: `brew install llama.cpp` — the binary is usually `/opt/homebrew/bin/llama-server` on Apple Silicon. Ensure that directory is on your **`PATH`** in the same shell where you run `python benchmark_runner.py` (the benchmark also probes Homebrew locations if `which llama-server` fails). Or set **`LLAMA_SERVER_BIN`** to the full path.
- **Hugging Face**: models download on first run. Set `HF_TOKEN` for higher rate limits (`export HF_TOKEN=...` or add to `~/.zshrc` and `source ~/.zshrc`)

## Quick start

```bash
cd /path/to/gemma-on-mac-mlx-vs-llama.cpp
python3 -m venv .venv
source .venv/bin/activate   # or: source .venv/bin/activate.fish
pip install -r requirements.txt
python benchmark_runner.py
```

You should see a comparison table on stdout and a written **`results.json`** (gitignored by default).

**`mlx-lm` from GitHub:** Gemma 4 requires `mlx-lm` builds that include the `gemma4` model type. This repo pins **`mlx-lm` from the upstream Git repo** (see [`requirements.txt`](requirements.txt)) because the current PyPI release may still raise `ValueError: Model type gemma4 not supported.` After changing `requirements.txt`, reinstall: `pip install -r requirements.txt --upgrade`.

## Configuration

- Copy **[`config.example.json`](config.example.json)** to **`config.json`** and edit, or pass **`--config path/to/config.json`**.
- If **`config.json`** is missing, defaults come from **`config.example.json`** when that file exists.

### Environment overrides (optional)

| Variable | Effect |
|----------|--------|
| `BENCHMARK_CONFIG` | Path to JSON config |
| `MLX_MODEL_ID` | MLX Hugging Face repo id |
| `GGUF_REPO` | GGUF repo id (llama.cpp) |
| `GGUF_FILENAME` | GGUF file name inside that repo |
| `LOCAL_GGUF_PATH` | Use a local `.gguf` file instead of downloading |
| `LLAMA_SERVER_BIN` | Full path to `llama-server` if not on `PATH` |

### Swapping models

- **MLX**: set `mlx.model_id` (e.g. `mlx-community/gemma-4-e2b-it-mxfp8`) or `MLX_MODEL_ID`.
- **llama.cpp**: set `llama_cpp.gguf_repo`, `llama_cpp.gguf_filename` (e.g. `gemma-4-E2B-it-Q8_0.gguf`), or `LOCAL_GGUF_PATH` for an on-disk GGUF.

GPU offload for llama.cpp: **`llama_cpp.n_gpu_layers`** — `-1` means all layers (see your `llama-server --help`).

## Scripts

| File | Role |
|------|------|
| [`benchmark_runner.py`](benchmark_runner.py) | Runs both backends, prints table, writes `results.json` |
| [`mlx_benchmark.py`](mlx_benchmark.py) | MLX-only benchmark (JSON to stdout) |
| [`llama_benchmark.py`](llama_benchmark.py) | llama-server + GGUF benchmark (JSON to stdout) |

### `benchmark_runner.py` flags

```text
--config PATH     JSON config
--output PATH     Results file (default: results.json)
--mlx-only        Only MLX
--llama-only      Only llama.cpp
--verbose         Dump per-backend JSON
```

## Metrics (definitions)

- **TTFT**: wall time from start of generation until the first streamed token.
- **Total time**: wall time until the stream ends.
- **Tokens/sec**: `tokens_generated / (total_time - TTFT)` (throughput after first token).
- **Memory**: peak **RSS** sampled during the run (Python process for MLX; `llama-server` process for GGUF).

If the llama.cpp stream does not include `usage.completion_tokens`, token count falls back to a rough estimate from generated text (see per-run `tokens_from_usage` in JSON).

## Project layout

```text
benchmark_runner.py   # entry point
mlx_benchmark.py
llama_benchmark.py
config.example.json
requirements.txt
results.json          # generated; listed in .gitignore
```

The **Benchmark report** section above summarizes a sample run; regenerate numbers anytime with `python benchmark_runner.py`.

## Troubleshooting

| Symptom | What to do |
|--------|------------|
| `ValueError: Model type gemma4 not supported` | Reinstall deps so `mlx-lm` comes from GitHub as in `requirements.txt`: `pip install -r requirements.txt --upgrade`. PyPI-only `mlx-lm` may lag behind Gemma 4 support. |
| `ModuleNotFoundError` (e.g. `psutil`) | Use the venv where you ran `pip install -r requirements.txt`, or run `python -m pip install -r requirements.txt` for the same interpreter you use to start `benchmark_runner.py`. |
| `llama-server` not found | Run `brew install llama.cpp`, then `which llama-server` (or `ls /opt/homebrew/bin/llama-server`). If you use GUI/IDE launches without Homebrew PATH, set `export LLAMA_SERVER_BIN=/opt/homebrew/bin/llama-server` or add `llama_server_bin` in `config.json`. |
| `No streamed token received from llama-server` | Gemma 4 **E2B** GGUF is multimodal: `llama-server` needs a **projector** (`--mmproj`). The benchmark downloads `mmproj-F16.gguf` from the same Hugging Face repo as the GGUF (`llama_cpp.mmproj_filename`). Set `local_mmproj_path` if you already have it cached, or `mmproj_filename` to `null` only for text-only GGUFs. If you still see this after `--mmproj`, llama.cpp may be streaming **`reasoning_content`** only (thinking); the client treats **`delta.reasoning_content` and `delta.content`** as generated text. The error message may include a non-streaming diagnostic and a tail of `llama-server` stderr. |

## License

[MIT](LICENSE)
