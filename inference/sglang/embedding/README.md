# `inference/sglang/embedding` — SGLang embedding serving (EmbeddingGemma-300m)

## Overview & when to use

Serve `google/embeddinggemma-300m` behind an OpenAI-compatible `POST /v1/embeddings`
endpoint with **SGLang** (`python -m sglang.launch_server --is-embedding`), and hit it with
`inference_embedding_sglang.py`, which embeds a query plus documents and prints the real
vector dimension and cosine scores.

Use this folder when you already run SGLang for generation and want one serving stack for
embeddings too. On **AMD gfx950 a pip install does not serve; the vendor container does** —
see "Container route — `lmsysorg/sglang-rocm`". If you want an embedding endpoint without
pulling a ~90 GB image, the Transformers or vLLM folder is lighter.

## H100 (NVIDIA) — the pip route

**On NVIDIA the pip route works** — the `aiter`/`sgl_kernel` kernels ship as CUDA wheels, so
a plain `pip install "sglang[all]"` serves `/v1/embeddings` with **no container**. The
commands below are single-GPU.

### H100 install (pip route)

Identical shared venv as the LLM leaf (see `llm/README.md` H100 section for the full block):

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and run artifacts
```

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip && pip install torch numpy            # the current CUDA 13 build
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"                                 # sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17
python -c "import sgl_kernel; print('sgl_kernel OK')"     # OK — impossible on ROCm
```

### H100 serve — single GPU

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1   # HF_HOME set above
# HF_TOKEN (gated model) comes from the stack-root dev.env symlink:
set -a; source ../dev.env; set +a
python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m \
  --is-embedding \
  --mem-fraction-static 0.5 \
  --host 127.0.0.1 --port 8600
```

`--is-embedding` is correct here (true bidirectional embedding model). No `--attention-backend`
flag — on Hopper SGLang auto-selects the CUDA path (`breakable` CUDA prefill graph for embeddings).
Plain `CUDA_VISIBLE_DEVICES`; no `HIP_VISIBLE_DEVICES`.

**Expected output** (server log):

```
EmbeddingGemma detected: disabling radix cache and chunked prefill; using breakable CUDA graph for CUDA prefill.
The server is fired up and ready to roll!
```

### H100 client / smoke command

```bash
python inference_embedding_sglang.py --port 8600
```

**Expected output:**

```
[health] server ready after 1.0s
[latency] 0.02s | n_vectors=3 dim=768
[vector0 head] [-0.0957, -0.08154, 0.04419, -0.00885, 0.03015, 0.03687, -0.04492, 0.0437]
[cosine] +0.7931  <- vLLM supports AMD ROCm.
[cosine] +0.2750  <- SQLite is an embedded database.
```

The relevant document must score above the irrelevant one, and `dim` must be **768**
(EmbeddingGemma's native width).

### H100 GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader
```

Only the card named by `CUDA_VISIBLE_DEVICES` is touched. The VRAM it holds is the
`--mem-fraction-static 0.5` pool, not real demand — lower it to pack more replicas per card.

### H100 quirks / notes

- Same benign `torchcodec`/`libavutil` startup traceback as the LLM leaf — ignore it.
- Scale with independent single-GPU replicas (one `CUDA_VISIBLE_DEVICES` each), not `--tp`.

## Container route — `lmsysorg/sglang-rocm`

**This is the route that works on gfx950.**

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk** (allow
>150 GB free on the filesystem holding it).

### Start the container

```bash
docker run -d --name sglang_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD129 \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HF_HOME="$HF_HOME" -e HF_HUB_OFFLINE=1 \
  -v "$HF_HOME":"$HF_HOME" \
  -v "$OUTPUT_DIR":"$OUTPUT_DIR" \
  -v /path/to/OTel:/work \
  --network bridge \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

The `renderD*` entries are the render nodes of the two target GPUs — match them to your own
cards (`ls /dev/dri`). Only those two nodes are passed in, so inside the container they appear
as `cuda:0` and `cuda:1` — `HIP_VISIBLE_DEVICES=0` in the container means the first passed-in
GPU. Weights and logs stay on the mounted volumes, never on `/`. If the minimal flag set hits
a permission or memory error, add `--group-add render --cap-add=SYS_PTRACE --shm-size 64G`.

Sanity check:

```bash
docker exec sglang_bringup python3 -c \
  "import torch, sgl_kernel; from importlib.metadata import version; \
   print(version('sglang'), torch.__version__, torch.version.hip, torch.cuda.device_count())"
```

### Serve — single GPU

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8101 \
     > $OUTPUT_DIR/inference_embedding_sglang/embed_rep_a.log 2>&1"
```

**Expected output** (`embed_rep_a.log`):

```
INFO:     "POST /v1/embeddings HTTP/1.1" 200 OK
```

`GET /get_model_info` confirms the server is in embedding mode (`"is_generation":false`,
`"task":"embed"`).

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/embedding sglang_bringup \
  python inference_embedding_sglang.py --port 8101
```

**Expected output:**

```
[health] server ready after 1.0s
[latency] 0.03s | n_vectors=3 dim=768
[vector0 head] [-0.1543, -0.02832, -0.00574, 0.00757, -0.00922, 0.02148, -0.05151, 0.05591]
[cosine] +0.8720  <- vLLM supports AMD ROCm.
[cosine] +0.8061  <- SQLite is an embedded database.
```

The relevant document must rank above the irrelevant one; the absolute gap is narrow when the
client sends raw text, but the ordering is what matters.

### Two concurrent single-GPU replicas (instead of TP=2)

For a 300M model scale with replicas, not tensor parallelism — one independent server per GPU
behind a load balancer:

```bash
# replica A -> first GPU, port 8101
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8101 > $OUTPUT_DIR/inference_embedding_sglang/embed_rep_a.log 2>&1"
# replica B -> second GPU, port 8111
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8111 > $OUTPUT_DIR/inference_embedding_sglang/embed_rep_b.log 2>&1"
```

Hit both concurrently (`--port 8101` and `--port 8111`); the replicas return identical
vectors, so a pool is safe to load-balance across. Each holds the
`--mem-fraction-static 0.5` pool rather than real demand — lower it to pack more replicas per
GPU.

### Container-route quirks

- **`sleep infinity` + `docker exec -d`**, not `docker run` per server — the AITER JIT warm-up
  is one-off per container, so one long-lived container reuses it.
- **`HF_HUB_OFFLINE=1`** is set in the image env, so nothing is downloaded when the models are
  already in `$HF_HOME`. Unset it if you need a fresh pull — and point `HF_HOME` at a volume
  with space first, never at `/`.
- **AITER is the default attention backend** (`attention_backend='aiter'`) and JIT-compiles
  kernels on first use. Expect a burst of `[aiter] import [mha_batch_prefill_...]` lines.
- **`not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default
  config!`** appears for every GEMM shape — harmless; AITER falls back to a torch GEMM.
- **`Ignore import error when loading sglang.srt.models.inkling: No module named 'cutlass'`** —
  benign, an unrelated CUDA-only model class failing to register.
- `FastAPIDeprecationWarning: ORJSONResponse is deprecated` — cosmetic, upstream.

## Install

Python 3.12; the shared [`../requirements.txt`](../requirements.txt) is the pinned set. The
stack is identical to the other two SGLang leaves, so one venv can be shared.

> **Note:** on ROCm the pip route below does not serve — it is documented so the limitation
> is clear and so the client scripts have an environment. For a working ROCm setup use the
> container route above.

### AMD / ROCm

```bash
python3 -m venv .env_sglang
source .env_sglang/bin/activate
pip install -U pip
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install --no-deps sglang==0.5.17
pip install -r ../requirements.txt
python -c "import torch; print(torch.__version__, torch.version.hip)"   # 2.11.0+rocm7.2 7.2.26015
```

`--no-deps` is mandatory: sglang 0.5.17 declares CUDA packages and the PyPI CUDA
`torch==2.11.0` as **base** dependencies, so a plain install replaces ROCm torch with a CUDA
wheel. If that happens:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` (CUDA-only). Never `pip install aiter` — the PyPI package of
that name is an unrelated async-iterator library, not AMD's AITER.

The three SGLang leaves share one identical environment: build a single `.env_sglang` at the
software root (`inference/sglang/`) and activate it from `llm/`, `embedding/` and `reranker/`.

This venv cannot serve on ROCm (`aiter` and `sgl_kernel` have no ROCm wheel), so use it only
for the client; serve from the container.

### NVIDIA / CUDA (see the H100 section above for the full variant)

```bash
pip install --upgrade pip && pip install uv
uv pip install sglang==0.5.17
```

## Environment & secrets

`dev.env` is symlinked to the repo root (`ln -sf ../../../dev.env dev.env`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

**EmbeddingGemma is a gated Google model** — the token is required for the first pull
(accept Google's terms on the model page). The client loads it with
`load_dotenv("dev.env")`; run from inside this folder. Never print or commit the token.

```bash
export HF_HOME=/path/to/hf_cache           # weights must not land on /
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1            # never empty on ROCm
```

## Run

### Launch — single GPU

```bash
source .env_sglang/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HOME=/path/to/hf_cache

python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m \
  --is-embedding \
  --mem-fraction-static 0.5 \
  --host 0.0.0.0 --port 8101
```

`--is-embedding` is correct **here** (a true bidirectional embedding model) and is exactly
what you must *not* pass to the Qwen3 reranker. No `--attention-backend` flag: on ROCm
SGLang auto-selects `aiter`; `flashinfer` is NVIDIA-only.

### Two concurrent single-GPU replicas (recommended over TP for a 300M model)

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m --is-embedding --port 8101 &
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m --is-embedding --port 8111 &
```

### Client / smoke command

```bash
python inference_embedding_sglang.py --port 8101 \
  --query "Which inference engines support AMD ROCm?" \
  --texts "vLLM supports AMD ROCm." "SQLite is an embedded database."
```

Expected on a working build:

```
[health] server ready after <N>s
[latency] 0.0x s | n_vectors=3 dim=768
[vector0 head] [...]
[cosine] +0.7xxx  <- vLLM supports AMD ROCm.
[cosine] +0.1xxx  <- SQLite is an embedded database.
```

## Where the ROCm pip route stops

Under pip on gfx950 the model loads and the warm-up prefill then dies inside AMD's attention
entry point `aiter.mha_batch_prefill_func`, which has no installable implementation (no
`aiter` ROCm wheel on PyPI), so the endpoint never reaches `/health`. TP=2 hits the same wall
in `sgl_kernel.rotary_embedding`, and `--disable-custom-all-reduce` does not change it. Use
the container route above.

## Arguments

Client (`inference_embedding_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8101` | Server port (8101 = SGLang embedding slot in the repo port map) |
| `--model` | `google/embeddinggemma-300m` | Model id as served; must match `--model-path` |
| `--query` | ROCm engines question | Query embedded and scored against each document |
| `--texts` | 2 sample documents | Documents to embed (space-separated list) |
| `--endpoint` | `/v1/embeddings` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

Server flags that matter:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `google/embeddinggemma-300m` | Gated; needs `HF_TOKEN` |
| `--is-embedding` | on | Correct for this model (and forbidden for the Qwen3 reranker) |
| `--tp` | `1` | Correct for a 300M model; scale with replicas, not TP |
| `--mem-fraction-static` | `0.5` | KV/static pool fraction; auto-reduced at TP>1 |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang selects `aiter` |

## Output

No artifacts are written by the server. Logs land wherever you redirect them, e.g.
`$OUTPUT_DIR/inference_embedding_sglang/`. Weights live in `$HF_HOME`. The client
prints dimensions, a vector head, and cosine scores to stdout only.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Status | **Works** — H100 80GB, CUDA 13.0, pip route (see the H100 section above) | **pip route blocked** — MI355X (gfx950), ROCm 7.2.4; container route works |
| Install | `uv pip install sglang` | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Model class | `EmbeddingGemmaModel` | `EmbeddingGemmaModel` — loads on gfx950 |

## Notes / quirks

- `chunked_prefill_size` is forced to `-1` (disabled) for embedding runs — expected.
- **KV cache is large by default** even for a 300M model. Set `--mem-fraction-static` or
  `--max-total-tokens` explicitly when sharing the box.
- `Failed to import amdsmi` on every launch — harmless; install `amdsmi` for AMD telemetry.
- `Ignoring corrupted tree cache file ... Permission denied` — shared HF cache owned by
  another user; cosmetic, the snapshot is still found locally.
- **Ports** — 8101 is the SGLang embedding slot; 8100/8102 belong to sibling folders.
- Re-export `HIP_VISIBLE_DEVICES` **and** `CUDA_VISIBLE_DEVICES` after activating the venv;
  never set `CUDA_VISIBLE_DEVICES` empty on ROCm.
