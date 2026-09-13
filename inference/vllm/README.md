# `inference/vllm` — vLLM serving on AMD MI355X and NVIDIA H100

One vLLM install serves all three workloads. The shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

On AMD the engine runs in a ROCm container; on NVIDIA it installs natively via pip.

| Leaf | Model | MI355X (gfx950, ROCm 7.2) | H100 (CUDA 13) |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **works** — TP=1 and TP=2, FP8 native; **requires `VLLM_ROCM_USE_AITER=0`** | **works** — needs `--max-num-seqs 256` (Mamba cache) |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **works, unmodified** (`--runner pooling`); TP=2 architecturally impossible for this model — replicate instead | **works, unmodified** |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **works, unmodified** with the `--hf_overrides` JSON + `qwen3_reranker.jinja` | **works, unmodified** — same `--hf_overrides` JSON + jinja |

## Install (AMD / ROCm — the container route)

On ROCm the engine runs in a **container** (a pip vLLM-ROCm venv is not needed). Known
working image: `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`
(vLLM `0.20.2rc1`); `vllm/vllm-openai-rocm:nightly`
(~11.5 GB compressed) is the upstream image if starting fresh. Canonical AMD flags:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache

docker run -d --name vllm_serve \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$HF_HOME":/root/.cache/huggingface \
  <IMAGE> sleep infinity
```

Some ROCm images have no `render` group — drop `--group-add render` if it errors. Serve
commands per workload live in each leaf README. Ports: llm 8000, embedding 8001,
reranker 8002.

## Shared quirk — AITER silent corruption on gfx950 FP8

`VLLM_ROCM_USE_AITER=1` is the gfx950 **default** and, with the FP8 27B checkpoint, selects
`AiterFp8BlockScaledMMKernel`, which emits garbage text at HTTP 200. **Always set
`VLLM_ROCM_USE_AITER=0`** for FP8 serving on gfx950. `VLLM_ROCM_USE_AITER_MOE=0` is the wrong
workaround (this is a GEMM bug, not MoE). See [`llm/README.md`](llm/README.md) and
[`docs/mi355x_inference_notes.md`](../../docs/mi355x_inference_notes.md).

## Environment & secrets

Clients load `dev.env` from their own folder (symlinked to the repo root — leaves use
`ln -sf ../../../dev.env dev.env`). One client venv serves all three leaves:

```bash
cd inference/vllm
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install -r requirements.txt
```

One shared venv at the stack root (`inference/vllm/.env_vllm`) serves the `llm/`,
`embedding/` and `reranker/` leaves — build it once here, activate it from any leaf.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4)** — all three workloads, container route.
- **NVIDIA H100 (80GB HBM3, CUDA 13.0)** — all three workloads, **pip** route (native CUDA
  wheel, no container).

## H100 (NVIDIA)

All three workloads serve on a single H100 80GB. On NVIDIA the **pip route works natively**
(unlike ROCm, which is container-only here), so no `vllm/vllm-openai` container is needed.

### Install (pip route — one venv for engine + clients)

`pip install vllm` gives a native CUDA-13 wheel. Unset any proxy first — pypi.nvidia.com
and download.pytorch.org are commonly proxy-blocked:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy       # -> the current CUDA 13 build (no --index-url)
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
| transformers | `5.15.1` | bundled; **knows `qwen3_5`** |
| CUDA | 13.0 | H100 80GB HBM3, cc(9,0), native FP8 |

The NVIDIA leaf examples all use port **8500**, serving one workload at a time on a single
GPU; the ROCm container maps 8000/8001/8002 so all three can run concurrently.

All three leaves serve on one 80GB card with these versions: the 27B-FP8 LLM (with
`--max-num-seqs 256`), embeddinggemma unmodified at 768 dims, and the reranker unmodified
with its `--hf_overrides` JSON + jinja template. Per-leaf commands and expected output are
in each leaf README.

### The 27B-FP8 `--max-num-seqs` requirement

`Qwen/Qwen3.8-27B-FP8` has a Mamba/GDN hybrid backbone, so each decode sequence needs a Mamba
cache block and the default `max_num_seqs=1024` does not fit on one 80GB card. Pass
`--max-num-seqs 256`. Detail in [`llm/README.md`](llm/README.md).

### Reverse of the ROCm quirks on NVIDIA

- The **AITER silent corruption does not apply** — `VLLM_ROCM_USE_AITER` is ROCm-only;
  NVIDIA uses the FlashInfer/DeepGEMM FP8 path, which produces correct text.
- Drop `HIP_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`; use plain
  `CUDA_VISIBLE_DEVICES`.
- No `--device /dev/kfd`, no `--group-add video/render`, no ROCm container — the pip venv
  serves directly.

### Multi-GPU on NVIDIA

The NVIDIA commands here are single-GPU. For tensor parallelism add
`--tensor-parallel-size N`, but note embeddinggemma cannot TP (3 heads, indivisible —
replicate instead), and the 27B `qwen3_5` head/GDN divisibility must be checked before a
TP launch.
