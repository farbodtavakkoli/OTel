# `inference/mlx` — MLX serving on Apple silicon (Metal)

One MLX install serves all three workloads on Apple silicon Macs using unified memory. The LLM leaf is served via `mlx_lm.server` (OpenAI-compatible HTTP); the embedding and reranker leaves run in-process.

> **Format note:** MLX consumes safetensors. The embedding and reranker leaves serve ungated `mlx-community/...` conversions of the base models. The LLM leaf serves `OTel-LLM-E4B-IT`, which is converted from PyTorch `.bin` using [`llm/convert_otel_e4b.py`](llm/convert_otel_e4b.py).

## Leaves

| Leaf | Model | Status on M4 24 GB |
|---|---|---|
| [`llm/`](llm/README.md) | `OTel-LLM-E4B-IT` (MLX 4-bit) | **works** — 35 tok/s generation. Parity option `mlx-community/Qwen3.8-27B-4bit`: see leaf README. |
| [`embedding/`](embedding/README.md) | `mlx-community/embeddinggemma-300m-bf16` | **works** — 768-d L2-normalised vectors, 0.65 GB peak memory. |
| [`reranker/`](reranker/README.md) | `mlx-community/Qwen3-Reranker-0.6B-4bit` | **works** — yes/no logit scoring, 0.48 GB peak memory. |

All three default models fit together in under 6 GB, allowing a single laptop to run retrieval, reranking, and grounded generation concurrently.

## Install

Python 3.12, in its own venv. The MLX wheels on PyPI are the Metal build.

```bash
cd inference/mlx
python3.12 -m venv .env_mlx && source .env_mlx/bin/activate
pip install -r requirements.txt
```

One venv serves all three leaves. `torch` and `safetensors` are only needed for the one-off `llm/convert_otel_e4b.py` script and can be installed ad hoc.

## Environment & secrets

Scripts load `dev.env` from their own folder (symlinked to the repo root). Nothing served here is gated. `HF_TOKEN` is only needed if overriding the default model to pull the gated `google/embeddinggemma-300m`.

```bash
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # reference artifacts
```

## How artifacts map to MLX

The generated JSON reference artifacts use slightly different keys than the `transformers` baseline:

| transformers artifact | MLX artifact |
|---|---|
| `torch`, `hip` | `mlx`, `mlx_lm` (and `mlx_embeddings`) |
| `gpu: "gfx950"` | `gpu: "Apple M4"` (`sysctl machdep.cpu.brand_string`) |
| `devices: ["cuda:0"]` | `devices: ["gpu"]` (Metal, one device) |
| `VRAM(MiB)` | `peak_memory_mib` (Unified memory peak bytes) |
| `attn_implementation` | `"mlx"` |
| `seed` | `null` (leaves are deterministic) |

## Hardware support

- **Apple M4, 24 GB (macOS 26.6, Metal)**: works. The three default models need < 6 GB (supports any M-series Mac from 8 GB). The `Qwen3.8-27B` parity model needs 24 GB. Intel Macs are not supported.

## Known fixes (applied automatically)

- **`lm_head.weight` dropping**: `convert_otel_e4b.py` drops the redundant `lm_head.weight` tensor from the source checkpoint so `mlx_lm.convert` accepts it.
- **`mlx-embeddings` kwargs**: `embed_mlx.py` bypasses `mlx_embeddings.generate()` and calls the Gemma3 text model directly to avoid a keyword argument mismatch.
- **Thinking channel suppression**: `serve.sh` starts `mlx_lm.server` with `enable_thinking: false`. The model may still reason in a separate `reasoning` field for difficult questions.
