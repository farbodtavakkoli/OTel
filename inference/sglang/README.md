# `inference/sglang` — SGLang serving on AMD MI355X

One SGLang runtime serves all three workloads. Shared setup lives here; each leaf
documents only its workload: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

On MI355X the pip route cannot serve; the vendor container serves everything. Both facts
are evidenced in the leaves.

| Leaf | Model | Status on MI355X (gfx950, ROCm 7.2.4) |
|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **works in the container** — FP8 27B generates at TP=1 **and** TP=2; pip route blocked |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **works in the container** — `/v1/embeddings`, dim 768, correct cosine ordering; pip route blocked |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **works in the container** — `/v1/rerank`, correct ordering (0.7773 / 0.1403 / 0.00002); pip route blocked |

---

# `inference/sglang` — SGLang serving on NVIDIA H100

On NVIDIA the pip route works, which flips the MI355X "pip cannot serve" limitation: the
exact import that killed the ROCm pip route (`from sgl_kernel import rotary_embedding`)
succeeds, because `sglang-kernel` ships native CUDA wheels on PyPI. All three workloads
serve from a plain `pip install "sglang[all]"` venv — no container needed. The notes below
come from a single-GPU pass on one free GPU of an 8×H100 node; set `CUDA_VISIBLE_DEVICES`
to whichever GPU is free on yours.

| Leaf | Model | Status on H100 (cc9.0, CUDA 13.0, driver 580.173.02) — **pip route** |
|---|---|---|
| [`llm/`](llm/) | `Qwen/Qwen3.8-27B-FP8` | **WORKS** — FP8 27B loads as `Qwen3_5ForConditionalGeneration` (hybrid-GDN VL arch) and generates correct text; capital-of-France=Paris, ROCm→AMD/CUDA→NVIDIA |
| [`embedding/`](embedding/) | `google/embeddinggemma-300m` | **WORKS** — `/v1/embeddings`, dim 768, correct cosine ordering (+0.7931 rel > +0.2750 irrel) |
| [`reranker/`](reranker/) | `Qwen/Qwen3-Reranker-0.6B` | **WORKS** — `/v1/rerank`, correct 3-tier ordering (0.7773 / 0.1480 / 0.000033) |

## Install (NVIDIA / CUDA — the pip route)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
```

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip
pip install torch numpy                 # -> torch 2.13.0+cu130 (native CUDA 13, proxy ON: pypi.org allowlisted)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  # 2.13.0+cu130 13.0 True
# unset the proxy: sgl_kernel/flashinfer wheels live on NVIDIA-adjacent hosts pypi 403s
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"               # -> sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17, flash-attn-4 4.0.0b19
python -c "import sgl_kernel; print('sgl_kernel OK')"   # <- THE import that is impossible on ROCm; it succeeds here
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # re-verify: still 2.13.0+cu130 (sglang did NOT clobber torch)
```

`pip install "sglang[all]"` pulls CUDA `torchvision 0.28.0+cu130` / `torchaudio 2.11.0+cu130`
but leaves the cu130 `torch` untouched — no force-reinstall needed (contrast the ROCm route,
where a plain install replaces ROCm torch with a CUDA wheel). All three leaves share this one
venv; each leaf README has the exact serve+client commands. Ports here: everything served on
**8600** one workload at a time (single free GPU); in production use distinct ports per workload.

## Why the pip route works on NVIDIA but not ROCm (the limitation, flipped)

The MI355X blocker was purely a packaging gap: SGLang's HIP path does
`from sgl_kernel import rotary_embedding`, and `sglang-kernel` publishes **CUDA-only wheels**
(no ROCm build), so the ROCm pip route dies at the first native kernel. On H100 that same
wheel is exactly the right artifact — `sgl_kernel 0.4.6.post1` installs and imports cleanly,
`flashinfer` and `flash-attn-4` install, and the full server import chain
(`sglang.srt.entrypoints.http_server`) loads. **The ROCm finding stated the gap was packaging,
not hardware; the H100 result confirms that directly.**

## Environment & GPU pinning (H100)

```bash
export HF_HOME=/path/to/hf_cache              # Hugging Face model cache
export HF_HUB_OFFLINE=1                       # cache-first; nothing is downloaded when the models are already cached
export CUDA_VISIBLE_DEVICES=<free-gpu>        # plain CUDA — no HIP_VISIBLE_DEVICES / RAY_EXPERIMENTAL_NOSET_* needed
ln -sf ../../dev.env dev.env                  # stack-root symlink -> repo-root dev.env (supplies HF_TOKEN for gated embeddinggemma)
```

Reversed ROCm workarounds: plain `CUDA_VISIBLE_DEVICES`, no `HIP_VISIBLE_DEVICES`, no
`--disable-custom-all-reduce`, no `aiter` backend. SGLang auto-selects `fa3` attention +
`flashinfer` sampling on Hopper. Multi-GPU (TP=2/8) is not covered here; it just adds
`--tp N` with `CUDA_VISIBLE_DEVICES` listing N free GPUs.

## Hardware support

- **NVIDIA H100 (cc9.0, CUDA 13.0, driver 580.173.02): verified — the PIP ROUTE serves all
  three workloads.** No container required.
- **AMD MI355X (gfx950, ROCm 7.2.4): verified — container route only** (pip blocked; see below).

---

## Install (AMD / ROCm — the route that works)

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

`uv pip install sglang` is actively harmful on ROCm — the base wheel depends on CUDA
torch, flashinfer and `sglang-kernel`, and SGLang's HIP code path does
`from sgl_kernel import rotary_embedding` while `sglang-kernel` publishes **CUDA-only
wheels** (and PyPI `aiter` is a name-squat, not AMD's AITER). Everything up to the first
native kernel call works — weights load, KV cache allocates, TP=2 shards with per-rank
weights exactly halved, FP8 resolves to `torch.float8_e4m3fn` (correct OCP E4M3FN for
gfx950) — so this is a packaging gap, not a hardware one. Full analysis in the leaves.

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

## Upstream images and other hardware

- **NVIDIA:** `lmsysorg/sglang` is the upstream CUDA image, with an exact Qwen3.8-27B
  cookbook; the pip route documented above also serves all three workloads on H100.
- **Other hardware (upstream claims — not verified here):** none beyond NVIDIA CUDA and
  AMD ROCm in upstream's supported-hardware documentation.
