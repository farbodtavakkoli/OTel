# `inference/transformers` — Transformers / sentence-transformers baseline on AMD MI355X

The **correctness baseline** for the inference stacks: plain PyTorch + HF Transformers +
sentence-transformers, no serving engine. Every other stack's output is compared against
the reference artifacts these folders produce. Leaves: [`llm/`](llm/) ·
[`embedding/`](embedding/) · [`reranker/`](reranker/).

| Leaf | Model | Verdict on MI355X (gfx950, ROCm 7.2.4) |
|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **works — FP8 loaded natively** (no BF16 fallback), single-GPU and 2-GPU `device_map` |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **works** — reference vectors saved for cross-stack comparison |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **works** — `CrossEncoder` scores, reference artifact saved |

Reference artifacts (generated text, embedding vectors, rerank scores) are written under
`$OUTPUT_DIR/inference_*_transformers/` and are what the other stacks diff
against — e.g. Ollama's embedding path matched to worst |Δcosine| 0.0034 with the correct
prompt prefixes.

## Install (AMD / ROCm — the route that worked)

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

## Install (NVIDIA / CUDA 13 — H100, verified)

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch numpy            # -> torch 2.13.0+cu130 (native CUDA 13; no --index-url)
pip install -r requirements.txt    # torch/torchvision ROCm pins are skipped on CUDA
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.13.0+cu130 13.0
```

One venv serves all three leaves (offline: `HF_HOME=$HF_HOME HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`). The `torch==2.11.0` pin in requirements has no cu130 wheel, so
H100 uses the default stable **2.13.0+cu130** (documented deviation). flash-attn has no
prebuilt cu130 wheel and did not build in-budget, so all three leaves run `sdpa` on H100
too — `torch.nn.functional.scaled_dot_product_attention` selects an efficient Hopper
kernel, so there is no correctness cost. Never install flash-attn from source here.

## H100 verification summary (1× H100 80GB, single physical GPU)

Single-GPU smoke of all three leaves, `torch 2.13.0+cu130`, `transformers 5.5.0`,
`sentence-transformers 5.7.0`, `accelerate 1.14.0`, `kernels 0.12.3`, driver 580.173.02.
Model swaps vs the MI355X reference are cache-driven (see each leaf README):

| Leaf | Model on H100 | dtype/attn | Result | GPU-5 residency (by PID) |
|---|---|---|---|---|
| [`llm/`](llm/) | `LiquidAI/LFM2.5-350M` (Qwen3.8-27B-FP8 not cached) | bf16 / sdpa | **works** — `Paris`, `negative` correct | ~1196 MiB |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | bf16 / sdpa | **works** — dim 768, SANITY PASS both queries | 623 MiB (torch) |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | fp32 / sdpa | **works-with-changes** — SANITY PASS, one code fix | 2378→3106 MiB |

Reranker required one **hardware-neutral** code fix (`--fix_pair_template`, default on):
the bare cached checkpoint carries only the generic Qwen3 generation chat template, and
sentence-transformers **5.7.0** now refuses a template that can't render the `query`/`document`
pair roles (`_verify_pair_roles_supported`), so `predict()` raised `ValueError`. The script
now installs a Query/Document yes-no template when the loaded one can't carry both roles.
This would bite MI355X on the same ST version + a bare cache too.

Reference artifacts (what vLLM / SGLang / TensorRT-LLM diff against) written under
`$OUTPUT_DIR/transformers/{llm,embedding,reranker}/`:
`reference_llm_lfm2.5-350m_1gpu.json`, `reference_embedding_embeddinggemma-300m_1gpu.json`,
`reference_reranker_qwen3-reranker-0.6b_fp32_1gpu.json`.

**Multi-GPU not verified here.** A 2-GPU pass would use
`--device_map auto` (llm) or `--devices cuda:0,cuda:1` (embedding/reranker data-parallel);
none of these models need sharding (all < 3 GiB), so multi-GPU here is throughput-only.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4): verified** — all three workloads, incl. native FP8
  load of the 27B checkpoint (OCP E4M3FN).
- **NVIDIA H100 (Hopper cc9.0, CUDA 13.0, driver 580.173.02): verified** —
  all three leaves, single-GPU, `torch 2.13.0+cu130`, attn `sdpa`. LLM used the cached
  `LiquidAI/LFM2.5-350M` (the FP8 27B is not in this node's cache); embedding and reranker
  used the same models as MI355X. Reranker needed the pair-template fix above (ST 5.7.0).
- **Other hardware (upstream claims — not verified here):** Apple Silicon (MPS), Intel
  XPU, TPU via torch/XLA, and CPU — torch-level device support; sentence-transformers
  follows whatever backend torch provides.
