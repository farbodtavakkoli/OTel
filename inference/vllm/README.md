# `inference/vllm` — vLLM serving on AMD MI355X and NVIDIA H100

One vLLM install serves all three workloads. The shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

On AMD the engine runs in a ROCm container; on NVIDIA it installs natively via pip.

**Hardware:** AMD MI355X (gfx950, host ROCm 7.2.4) · NVIDIA H100 80GB HBM3 (CUDA 13.0)

## Workloads

| Leaf | Model | ROCm port | H100 port |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | 8000 | 8500 |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | 8001 | 8500 |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | 8002 | 8500 |

The ROCm container maps 8000/8001/8002 so all three serve concurrently. The H100 examples
use 8500 and serve one workload at a time on a single GPU.

## Setup — NVIDIA (pip, no container)

`pip install vllm` gives a native CUDA-13 wheel that serves the engine *and* runs the
clients from one venv. Unset any proxy first — pypi.nvidia.com and download.pytorch.org
are commonly proxy-blocked.

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd inference/vllm
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy          # -> the current CUDA 13 build (no --index-url)
pip install vllm                 # -> vllm 0.27.1 (pulls flashinfer + cutlass-dsl[cu13])
pip install -r requirements.txt  # python-dotenv, for the client scripts
python -c "import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
#   2.13.0+cu130 13.0 0.27.1
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA-only) |
| torch | `2.13.0+cu130` (survives the vLLM install) |
| transformers | `5.15.1` (bundled; knows `qwen3_5`) |
| CUDA | 13.0, H100 80GB HBM3, cc(9,0), native FP8 |

Use plain `CUDA_VISIBLE_DEVICES`; the ROCm variables below do not apply.

## Setup — AMD / ROCm (container)

There is no ROCm vLLM wheel, so the engine runs in a container. Known-working image:
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` (vLLM `0.20.2rc1`,
torch `2.9.1.dev20251204+rocm7.0.2`, transformers `5.14.1`, container ROCm 7.0.2).

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export VLLM_IMAGE=rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2

docker run -d --name vllm_serve \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$HF_HOME":/root/.cache/huggingface \
  "$VLLM_IMAGE" sleep infinity
```

Some ROCm images have no `render` group, so `--group-add render` fails; pass the host's
numeric GIDs instead:

```bash
--group-add "$(getent group video | cut -d: -f3)" \
--group-add "$(getent group render | cut -d: -f3)"
```

To pin specific cards rather than exposing all of `/dev/dri`, pass their render nodes
(`--device /dev/dri/renderD128 --device /dev/dri/renderD136` — match yours with
`ls /dev/dri`).

Build the client venv on the host, outside the container:

```bash
cd inference/vllm
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install -r requirements.txt
```

## Client environment

One shared venv at the stack root (`inference/vllm/.env_vllm`) serves all three leaves —
build it once here, activate it from any leaf. Clients load `dev.env` from their own
folder; symlink it in each leaf with `ln -sf ../../../dev.env dev.env`. It supplies
`HF_TOKEN`, required for the gated `google/embeddinggemma-300m` repo.

## Notes

- **Set `VLLM_ROCM_USE_AITER=0` when serving the FP8 27B on gfx950.** The AITER default
  selects an FP8 GEMM kernel that returns corrupt text at HTTP 200.
  `VLLM_ROCM_USE_AITER_MOE=0` does not substitute for it. See
  [`llm/README.md`](llm/README.md) and
  [`docs/mi355x_inference_notes.md`](../../docs/mi355x_inference_notes.md).
- **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every GPU.
- **EmbeddingGemma cannot use tensor parallelism** (3 attention heads); replicate one
  server per GPU instead. For the 27B, check `qwen3_5` head/GDN divisibility by N before
  passing `--tensor-parallel-size N`.
