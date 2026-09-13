# `inference/transformers` — Transformers / sentence-transformers baseline

The **correctness baseline** for the inference stacks: plain PyTorch + HF Transformers +
sentence-transformers, no serving engine. Every other stack's output is compared against
the reference artifacts these folders produce. Leaves: [`llm/`](llm/) ·
[`embedding/`](embedding/) · [`reranker/`](reranker/).

| Leaf | Model | Status |
|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **works — FP8 loaded natively** (no BF16 fallback), single-GPU and 2-GPU `device_map` |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **works** — reference vectors saved for cross-stack comparison |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **works** — `CrossEncoder` scores, reference artifact saved |

Reference artifacts (generated text, embedding vectors, rerank scores) are written under
`$OUTPUT_DIR/inference_*_transformers/` and are what the other stacks diff against.

## Install — AMD / ROCm

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # inference reference artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache

cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

One venv serves all three leaves; build it from this requirements file. Never install
flash-attn on ROCm — the attention path is `sdpa`.

## Environment & secrets

Scripts load `dev.env` from their own folder (leaves symlink the repo root:
`ln -sf ../../../dev.env dev.env`). `google/embeddinggemma-300m` is gated — the token is
required for the first download.

## Install — NVIDIA / CUDA 13

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0          # CUDA 13 build from PyPI; no --index-url needed
pip install -r requirements.txt    # the torch pin is already satisfied
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.11.0 13.0
```

One venv serves all three leaves (offline: `HF_HOME=$HF_HOME HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`). The PyPI `torch==2.11.0` wheel is a CUDA 13 build, so the pin in
requirements holds here as it does on ROCm; install torch first so the requirements install
does not re-resolve it. Add `--index-url https://download.pytorch.org/whl/cu130` if you want
the `+cu130` local version tag. Never install flash-attn here either — there is no prebuilt
cu130 wheel; all three leaves run `sdpa`.

## Known fix — reranker pair template

The reranker leaf needs one code fix, enabled by default via `--fix_pair_template`: a bare
cached `Qwen/Qwen3-Reranker-0.6B` checkpoint carries only the generic Qwen3 *generation* chat
template, and sentence-transformers **5.7.0** refuses a template that cannot render the
`query`/`document` pair roles, so `predict()` raises `ValueError`. The script installs a
Query/Document yes-no template when the loaded one cannot carry both roles.

## Hardware support

Works on **AMD MI355X (gfx950, ROCm 7.2)** — all three workloads, including a native FP8 load
of the 27B checkpoint (OCP E4M3FN), single-GPU and 2-GPU `device_map` — and on
**NVIDIA H100 80GB (CUDA 13)** single-GPU with `torch 2.13.0+cu130` and attn `sdpa`.

For multi-GPU use `--device_map auto` (llm) or `--devices cuda:0,cuda:1`
(embedding/reranker data-parallel).
