# `inference/vllm` — vLLM serving on AMD MI355X

One vLLM install serves all three workloads. The shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

| Leaf | Model | Verdict on MI355X (gfx950, ROCm 7.2.4) |
|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **works** — TP=1 and TP=2 (+36%), FP8 native; **requires `VLLM_ROCM_USE_AITER=0`** |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **works, unmodified** (`--runner pooling`); TP=2 architecturally impossible for this model — replicate instead |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **works, unmodified** with the `--hf_overrides` JSON + `qwen3_reranker.jinja` |

## Install (AMD / ROCm — the route that worked)

The verified route is the **ROCm container** (a pip vLLM-ROCm venv was not needed). The
campaign used the `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`
image already present on the box (vLLM `0.20.2rc1`); `vllm/vllm-openai-rocm:nightly`
(~11.5 GB compressed) is the upstream image if starting fresh. Canonical AMD flags:

```bash
docker run -d --name vllm_serve \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /mnt/data_1.5t/hf_cache:/root/.cache/huggingface \
  <IMAGE> sleep infinity
```

Some ROCm images have no `render` group — drop `--group-add render` if it errors. Serve
commands per workload live in each leaf README. Ports: llm 8000, embedding 8001,
reranker 8002.

## ⚠️ Shared quirk — the AITER trap (silent corruption)

`VLLM_ROCM_USE_AITER=1` is the gfx950 **default** and, with the FP8 27B checkpoint,
selects `AiterFp8BlockScaledMMKernel`, which emits garbage text at HTTP 200 — and is
*faster* than the correct path, so throughput benchmarks pick the broken config. **Always
set `VLLM_ROCM_USE_AITER=0`** for FP8 serving on gfx950. `VLLM_ROCM_USE_AITER_MOE=0` is
the wrong workaround (this is a GEMM bug, not MoE). Details and reproduction in
[`llm/README.md`](llm/README.md) and
[`docs/mi355x_inference_notes.md`](../../docs/mi355x_inference_notes.md).

## Environment & secrets

Clients load `dev.env` from their own folder (symlinked to the repo root — leaves use
`ln -sf ../../../dev.env dev.env`). One client venv serves all three leaves:

```bash
cd inference/vllm
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install -r requirements.txt
```

Campaign venvs were removed in the 2026-08 reorg; rebuild from this requirements file.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4): verified** — all three workloads, evidence in the leaves.
- **NVIDIA H100 (80GB HBM3, CUDA 13.0): verified 2026-08-22** — all three workloads, via
  the **pip** route (native CUDA wheel, no container). Evidence in the leaves and below.
- **Other hardware (upstream claims — not verified here):** Intel XPU, Google TPU,
  AWS Neuron, and CPU via vLLM's hardware-plugin backends, per upstream installation docs.

## H100 (NVIDIA) — verified 2026-08-22

All three workloads serve on a single H100 80GB. On NVIDIA the **pip route works natively**
(unlike ROCm, which is container-only here), so no `vllm/vllm-openai` container was needed.

### Install (pip route that worked — one venv for engine + clients)

`pip install vllm` gives a native CUDA-13 wheel. In a tmpfs venv, with the proxy unset
(pypi.nvidia.com / download.pytorch.org are proxy-blocked; pypi.org is allowlisted):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy       # -> torch 2.13.0+cu130 (native CUDA 13.0, no --index-url)
pip install vllm              # -> vllm 0.27.1 (pulls flashinfer + cutlass-dsl[cu13])
pip install python-dotenv     # for the client scripts
# re-verify: torch is NOT clobbered by the vllm install ->
python -c "import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
#   2.13.0+cu130 13.0 0.27.1
```

| Component | Version | Notes |
|---|---|---|
| vLLM | `0.27.1` | pip wheel, CUDA-only |
| torch | `2.13.0+cu130` | native CUDA 13.0; survives the vLLM install |
| transformers | `5.15.1` | bundled; **knows `qwen3_5`** (see llm finding) |
| driver / CUDA | 580.173.02 / 13.0 | H100 80GB HBM3, cc(9,0), native FP8 |

Ports on H100 were consolidated to **8500** (one workload served at a time on the single
free GPU); the ROCm campaign used 8000/8001/8002 for concurrent serving.

### Results (one H100 80GB, physical GPU 5)

| Leaf | Model | Verdict | Key evidence |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **WORKS-WITH-CHANGES** | `Resolved architecture: Qwen3_5ForConditionalGeneration`; "capital of France → **Paris**", 68–73 tok/s; needs `--max-num-seqs 256` (Mamba cache); 73 GB on GPU 5 |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **WORKS, unmodified** | 768-dim; cos **+0.7106** (ROCm doc) vs **+0.2008** (postgres); 0.61 GiB weights |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **WORKS, unmodified** | ranking correct, 4-orders separation (0.9994 vs 0.00013); `--hf_overrides`+jinja mandatory; query-conditioned |

### ⚠️ The 27B-FP8 architecture-support finding (the headline)

The documented `Qwen/Qwen3.8-27B-FP8` is **not a plain LLM** — its `config.json` is
`architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`, a
**vision-language model with a Mamba/GDN (gated delta net) linear-attention text backbone**
(`quantization_config.modules_to_not_convert` lists `visual.blocks.*`). **TensorRT-LLM
1.2.1 rejected this checkpoint** as an unrecognized `qwen3_5` arch. **vLLM 0.27.1 serves
it**: its registry contains `Qwen3_5ForConditionalGeneration`, the bundled transformers
`5.15.1` knows `qwen3_5`/`qwen3_5_vision`/`qwen3_5_text`, and it auto-detects the E4M3 FP8
quantization (selecting `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` — the CUDA FP8 GEMM
path). The one required change is `--max-num-seqs 256`: because the backbone is a
Mamba/GDN hybrid, each decode sequence needs a Mamba cache block, and the default
`max_num_seqs=1024` exceeds the ~694 blocks that fit on a single 80GB card. **No fallback to
`Qwen/Qwen3-0.6B` was needed** — the real 27B FP8 served correctly. Full detail in
[`llm/README.md`](llm/README.md).

### Reverse of the ROCm quirks on NVIDIA

- The **AITER silent-corruption trap does not apply** — `VLLM_ROCM_USE_AITER` is ROCm-only;
  NVIDIA uses the FlashInfer/DeepGEMM FP8 path, which produced correct text on every prompt
  (verified, not just HTTP 200). No garbage.
- Drop `HIP_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`; use plain
  `CUDA_VISIBLE_DEVICES`.
- No `--device /dev/kfd`, no `--group-add video/render`, no ROCm container — the pip venv
  serves directly.

### Multi-GPU (deferred)

Single-GPU smoke only this wave (GPUs 0–3 were a co-tenant production job; only physical
GPU 5 was used). A TP pass would add `--tensor-parallel-size N`; note embeddinggemma cannot
TP (3 heads, indivisible — replicate instead), and the 27B `qwen3_5` head/GDN divisibility
must be checked before a TP launch. Not run here.
