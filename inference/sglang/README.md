# `inference/sglang` — SGLang serving on AMD MI355X and NVIDIA H100

One SGLang runtime serves all three workloads. The shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

There are two routes. On NVIDIA a plain `pip install "sglang[all]"` venv serves all three
workloads. On AMD/ROCm the vendor container serves them; a pip venv there runs the client
scripts only, because `sgl_kernel` and `aiter` publish no ROCm wheels.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (cc9.0, CUDA 13.0)

## Workloads

| Leaf | Model | ROCm port | H100 port |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | 8100 | 8600 |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | 8101 | 8600 |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | 8102 | 8600 |

The container maps 8100/8101/8102 so siblings run concurrently. The H100 examples use 8600
and serve one workload at a time on a single GPU.

## Setup — NVIDIA (pip, no container)

```bash
cd inference/sglang
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip
pip install torch numpy                 # -> the current CUDA 13 build (plain PyPI)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  # 2.13.0+cu130 13.0 True
# unset any proxy: the sgl_kernel/flashinfer wheel hosts are commonly 403'd through one
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"               # -> sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17, flash-attn-4 4.0.0b19
python -c "import sgl_kernel; print('sgl_kernel OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # still 2.13.0+cu130 — sglang did not clobber torch
```

All three leaves share this one venv. Each leaf README has the exact serve and client
commands.

## Setup — AMD / ROCm (container)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs written by the leaf commands

docker pull lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819   # ~23 GB compressed / ~90 GB on disk
docker run -d --name sglang_bringup \
  --device /dev/kfd --device /dev/dri --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -p 8100:8100 -p 8101:8101 -p 8102:8102 \
  -v "$HF_HOME":/hf_cache -e HF_HOME=/hf_cache \
  -v "$OUTPUT_DIR":"$OUTPUT_DIR" -e OUTPUT_DIR="$OUTPUT_DIR" \
  -v /path/to/OTel:/work \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

Pick the tag matching your GPU family (`mi35x` = gfx950) and ROCm line. Allow >150 GB free
on the filesystem holding the image. The `-v /path/to/OTel:/work` mount is what makes the
leaves' `docker exec -w /work/...` client commands and container-visible template paths
resolve, and `-e OUTPUT_DIR` is what lets their serve commands redirect logs. Every leaf
command targets this container by name (`sglang_bringup`). Launch commands per workload
live in each leaf README.

Keep one long-lived container (`sleep infinity` + `docker exec -d`) rather than one
`docker run` per server, so AITER's JIT build cache in `/root/.aiter/build/` is reused. To
pin specific cards instead of exposing all of `/dev/dri`, pass their render nodes
(`--device /dev/dri/renderD128 --device /dev/dri/renderD129` — match yours with
`ls /dev/dri`); they then appear as `cuda:0`/`cuda:1` inside. If the minimal flag set hits a
permission or memory error, add `--group-add render` and `--shm-size 64G`.

Build the client venv on the host:

```bash
cd inference/sglang
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -r requirements.txt
```

## Environment

```bash
export HF_HOME=/path/to/hf_cache              # weights must not land on /
export HF_HUB_OFFLINE=1                       # cache-first; nothing downloads when models are cached
export CUDA_VISIBLE_DEVICES=<free-gpu>        # NVIDIA
export HIP_VISIBLE_DEVICES=0,1                # ROCm — export alongside CUDA_VISIBLE_DEVICES
ln -sf ../../dev.env dev.env                  # HF_TOKEN, for the gated embeddinggemma repo
```

Leaves symlink their own copy with `ln -sf ../../../dev.env dev.env`. One shared venv at
the stack root (`inference/sglang/.env_sglang`) serves all three leaves.

## Notes

- **Do not `pip install aiter`** — that PyPI project is an unrelated async-iterator
  library, not AMD's AITER, and it shadows the real one.
- **Do not `pip install flash-attn` on ROCm** — CUDA-only wheel.
- **Never set `CUDA_VISIBLE_DEVICES` empty on ROCm**, and re-export both it and
  `HIP_VISIBLE_DEVICES` after activating the venv.
- **Leave `--attention-backend` unset.** SGLang picks `aiter` on gfx950 and `fa3` on
  Hopper; `flashinfer` is NVIDIA-only.
