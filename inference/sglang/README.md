# `inference/sglang` — SGLang serving on AMD MI355X and NVIDIA H100

One SGLang runtime serves all three workloads. Shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

There are two routes. On NVIDIA a plain `pip install "sglang[all]"` venv serves all three
workloads — no container. On AMD/ROCm the pip route **cannot** serve (a packaging gap in
`sglang-kernel`) and the vendor container serves everything.

| Leaf | Model | AMD MI355X (gfx950, ROCm 7.2) | NVIDIA H100 (cc9.0, CUDA 13) |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **container** (pip blocked) | **pip** |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **container** (pip blocked) | **pip** |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **container** (pip blocked) | **pip** |

## Install (NVIDIA / CUDA — the pip route)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
```

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip
pip install torch numpy                 # -> the current CUDA 13 build (plain PyPI)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  # 2.13.0+cu130 13.0 True
# unset any proxy: the sgl_kernel/flashinfer wheel hosts are commonly 403'd through one
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"               # -> sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17, flash-attn-4 4.0.0b19
python -c "import sgl_kernel; print('sgl_kernel OK')"   # <- THE import that is impossible on ROCm; it succeeds here
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # re-verify: still 2.13.0+cu130 (sglang did NOT clobber torch)
```

All three leaves share this one venv; each leaf README has the exact serve+client commands.
The leaf examples all use port **8600**, serving one workload at a time on a single GPU; in
production give each workload its own port. Set `CUDA_VISIBLE_DEVICES` to whichever GPU is
free on your machine.

## Environment & GPU pinning (NVIDIA)

```bash
export HF_HOME=/path/to/hf_cache              # Hugging Face model cache
export HF_HUB_OFFLINE=1                       # cache-first; nothing is downloaded when the models are already cached
export CUDA_VISIBLE_DEVICES=<free-gpu>        # plain CUDA — no HIP_VISIBLE_DEVICES / RAY_EXPERIMENTAL_NOSET_* needed
ln -sf ../../dev.env dev.env                  # stack-root symlink -> repo-root dev.env (supplies HF_TOKEN for gated embeddinggemma)
```

None of the ROCm workarounds apply here: plain `CUDA_VISIBLE_DEVICES`, no
`HIP_VISIBLE_DEVICES`, no `--disable-custom-all-reduce`, no `aiter` backend. For multi-GPU,
add `--tp N` and list N GPUs in `CUDA_VISIBLE_DEVICES`.

## Hardware support

- **NVIDIA H100 (cc9.0, CUDA 13.0): pip route serves all three workloads.** No container
  required.
- **AMD MI355X (gfx950, ROCm 7.2.4): container route only** (pip blocked).

## Install (AMD / ROCm — the container route)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache

docker pull lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819   # ~23 GB compressed / ~90 GB on disk
docker run -d --name sglang_serve \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -p 8100:8100 -p 8101:8101 -p 8102:8102 \
  -v "$HF_HOME":/hf_cache -e HF_HOME=/hf_cache \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

Pick the tag matching your GPU family (`mi35x` = gfx950) and ROCm line. Launch commands
per workload live in each leaf README. Ports: llm 8100, embedding 8101, reranker 8102.

## Why pip cannot work on ROCm

On ROCm, `uv pip install sglang` breaks the venv — the base wheel depends on CUDA torch,
flashinfer and `sglang-kernel`, which publishes **CUDA-only wheels**, so installing it
replaces ROCm torch and still fails at the first native kernel. Never `pip install aiter`
either: the PyPI project of that name is not AMD's AITER.

## Environment & secrets

Clients load `dev.env` from their own folder (leaves symlink the repo root:
`ln -sf ../../../dev.env dev.env`). A client venv (optional — clients are stdlib+dotenv):

```bash
cd inference/sglang
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -r requirements.txt
```

One shared venv at the stack root (`inference/sglang/.env_sglang`) serves the `llm/`,
`embedding/` and `reranker/` leaves — build it once here, activate it from any leaf.
