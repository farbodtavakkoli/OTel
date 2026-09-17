# `inference/transformers` — Transformers / sentence-transformers baseline

The correctness baseline for the inference stacks: plain PyTorch + HF Transformers +
sentence-transformers, no serving engine. Every other stack's output is diffed against the
reference artifacts these leaves write to `$OUTPUT_DIR/inference_*_transformers/`. Use a
serving stack (vLLM, SGLang, llama.cpp) for throughput.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2) · NVIDIA H100 80GB (CUDA 13)

## Leaves

| Leaf | Model |
|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` — FP8 loads natively, single-GPU or `--device_map auto` |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` — reference vectors |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` — `CrossEncoder` scores |

## Setup

One venv at this stack root serves all three leaves.

NVIDIA / CUDA 13:

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0          # the PyPI wheel is the CUDA 13 build
pip install -r requirements.txt
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

AMD / ROCm 7.2:

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
python -c "import torch;print(torch.cuda.is_available(), torch.version.hip)"   # True 7.2.x
```

Install torch (and `torchvision`, pinned) first on both platforms so the requirements
install does not re-resolve it. Do not install `flash-attn`. On CUDA add
`--index-url https://download.pytorch.org/whl/cu130` if you want the `+cu130` tag.

## Environment

```bash
export HF_HOME=/path/to/hf_cache       # model cache, large volume
export OUTPUT_DIR=/path/to/outputs     # reference artifacts
```

Each leaf loads `dev.env` from its own folder: `ln -sf ../../../dev.env dev.env`. It must
supply `HF_TOKEN` — `google/embeddinggemma-300m` is gated. Offline runs (cache already
populated): `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

For multi-GPU pass `--device_map auto` (llm) or `--devices cuda:0,cuda:1`
(embedding/reranker data-parallel).
