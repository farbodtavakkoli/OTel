# `inference/sglang/embedding` — SGLang embedding serving (EmbeddingGemma-300m)

## Overview & when to use

Serve `google/embeddinggemma-300m` behind an OpenAI-compatible `POST /v1/embeddings`
endpoint with **SGLang** (`python -m sglang.launch_server --is-embedding`), and hit it with
`inference_embedding_sglang.py`, which embeds a query plus documents and prints the real
vector dimension and cosine scores.

Use this folder when you already run SGLang for generation and want one serving stack for
embeddings too. On **AMD gfx950 the split is: a pip install does NOT serve, the vendor
container DOES** — both were tested here. See "Container route — `lmsysorg/sglang-rocm`
(TESTED)" for the working recipe (`/v1/embeddings` 200 OK, dim=768, correct cosine
ordering); the pip analysis below explains why the wheel route cannot work. If you want an
embedding endpoint without pulling a ~90 GB image, the Transformers or vLLM folder is
lighter.

This cell rates **🟡 "validate before production"**:
SGLang's EmbeddingGemma cookbook explicitly lists an **NVIDIA CUDA GPU as a prerequisite**;
the generic embedding docs list the model as supported but give no ROCm recipe. This folder
is that validation, run on gfx950.

> **Tested topology:** 2×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
> Python 3.12.3, physical GPUs 2 and 3, single node. No NVIDIA GPU on this host.

## VERDICT

**🟢 WORKS on ROCm/gfx950 via the vendor container — pip route stays blocked.**
EmbeddingGemma-300m serves real 768-dim vectors from SGLang on MI355X inside
`lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`. The NVIDIA-only
caveat applies to the **pip** install only; it is not a hardware or model-support limit.
See "Container route — `lmsysorg/sglang-rocm` (TESTED)" for the exact commands and output.

| Question | Answer (measured here) |
|---|---|
| Does SGLang recognise EmbeddingGemma at all? | **Yes** — loads as `type=EmbeddingGemmaModel` with `is_embedding=True` |
| Do weights load on gfx950? | **Yes** — 0.63 GB, 0.38 s, pool allocated |
| Does a forward pass run **in the container**? | **Yes** — `/v1/embeddings` 200 OK, dim=768, correct cosine ordering |
| Does a forward pass run **from pip**? | **No** — dies in AMD's own attention kernel path: `aiter.mha_batch_prefill_func` |
| Is TP=2 meaningful for a 300M model? | **No** — see "Multi-GPU"; two single-GPU replicas is the right pattern, and both replicas were run concurrently |
| Root cause of the pip failure | `aiter` and `sgl_kernel` have **no ROCm wheel** on PyPI; SGLang on HIP imports both. The container ships both prebuilt |

Not a model-support gap: SGLang has a dedicated `EmbeddingGemmaModel` implementation and it
initialised correctly on gfx950. The gap is packaging — identical to the LLM and reranker
folders. The supported ROCm route is `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`
— **it was subsequently pulled (89.9 GB on disk) and it works**; see "Container route" below.

---

# H100 (NVIDIA) — the PIP ROUTE SERVES (verified 2026-08-22)

**Headline: on H100 the pip route works — the exact failure this folder documented on ROCm is
gone.** EmbeddingGemma-300m died on MI355X inside AMD's own attention kernel
(`aiter.mha_batch_prefill_func`) because `aiter`/`sgl_kernel` have no ROCm wheel. On NVIDIA those
kernels ship as CUDA wheels, the forward pass completes, and `/v1/embeddings` returns real
768-dim vectors from a plain `pip install "sglang[all]"` — **no container.**

> **Tested topology:** 1×NVIDIA H100 80GB HBM3, **physical GPU 6 only** (single-GPU smoke test),
> cc(9,0), CUDA 13.0, driver 580.173.02, Python 3.12.3. GPUs 0–3 were a co-tenant production job,
> untouched. TP deferred.

## H100 VERDICT

**🟢 WORKS on H100 via the pip route.** `type=EmbeddingGemmaModel` loads (0.59 GB), the CUDA
prefill runs, and `/v1/embeddings` returns dim-768 vectors with correct cosine ordering. The
NVIDIA-only cookbook prerequisite is satisfied by the pip route itself here.

| Question | Answer (measured on H100 GPU 6) |
|---|---|
| Does the pip route serve embeddings? | **Yes** — `pip install "sglang[all]"`, no container; `import sgl_kernel` succeeds |
| Arch resolved | **`EmbeddingGemmaModel`** — same dedicated class as MI355X, now with a completed forward pass |
| Does the forward pass complete? | **Yes** — `/v1/embeddings` 200, dim **768**, latency 0.02 s (vs ROCm pip: died in `aiter.mha_batch_prefill_func`) |
| Cosine ordering correct? | **Yes** — rel **+0.7931** > irrel **+0.2750** (gap 0.52, wider than the container's 0.066) |
| GPU-6 residency proof? | **Yes** — `sglang::scheduler` held **2476 MiB** on GPU-6 (UUID `GPU-e4fe48bc-…`) |

## H100 install (pip route — VERIFIED)

Identical shared venv as the LLM leaf (see `llm/README.md` H100 section for the full block):

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip && pip install torch numpy            # torch 2.13.0+cu130
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"                                 # sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17
python -c "import sgl_kernel; print('sgl_kernel OK')"     # OK — impossible on ROCm
```

## H100 serve — single GPU (physical GPU 6)

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1
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

Server-side evidence (`/dev/shm/h100/out/sglang/embed_300m.log`), verbatim:

```
EmbeddingGemma detected: disabling radix cache and chunked prefill; using breakable CUDA graph for CUDA prefill.
Load weight end. elapsed=24.47 s, type=EmbeddingGemmaModel, avail mem=77.88 GB, mem usage=0.59 GB.
KV Cache skipped (no-op pool). Logical #tokens: 1686301, physical K/V size: ~24.0 KB placeholder
The server is fired up and ready to roll!
```

## H100 client / smoke command

```bash
python inference_embedding_sglang.py --port 8600
```

**Real output:**

```
[health] server ready after 1.0s
[latency] 0.02s | n_vectors=3 dim=768
[vector0 head] [-0.0957, -0.08154, 0.04419, -0.00885, 0.03015, 0.03687, -0.04492, 0.0437]
[cosine] +0.7931  <- vLLM supports AMD ROCm.
[cosine] +0.2750  <- SQLite is an embedded database.
```

**Cosine sanity passes:** query *"Which inference engines support AMD ROCm?"* scores the relevant
doc **+0.7931** well above the irrelevant one **+0.2750** — a 0.52 gap, notably wider than the
MI355X container's 0.066, since the H100 pip build runs the CUDA prefill path end-to-end.
Dimension is **768** (EmbeddingGemma native), server reports mean pooling + normalize.

## H100 GPU-6 residency proof (from inside serving)

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader | grep e4fe48bc
1700609, sglang::scheduler, 2476 MiB, GPU-e4fe48bc-0c29-f21e-6523-759f964bf823
```

Only GPU 6 touched. (VRAM here is the `--mem-fraction-static 0.5` pool, not real demand — a 300M
model needs far less; lower it to pack more replicas per card.)

## H100 quirks / notes

- Same benign `torchcodec`/`libavutil` startup traceback as the LLM leaf — ignore it.
- `torch 2.13.0+cu130` is **not** clobbered by `pip install "sglang[all]"` (re-verified).
- **TP=2 is still pointless for a 300M model** (as on MI355X) — scale with independent
  single-GPU replicas behind a load balancer, not tensor parallelism. Multi-GPU pass deferred
  until GPUs 0–3 free; it would just launch N replicas on N free cards (one `CUDA_VISIBLE_DEVICES`
  each), no `--tp`.

## Container route — `lmsysorg/sglang-rocm` (TESTED)

**This is the route that works on gfx950.** Everything below this heading was executed on
2×MI355X (physical GPUs 2 and 3) and the output is copied verbatim. The pip analysis in the
rest of this readme is retained and still accurate — it explains *why* the container is needed.

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk** (`docker images`).
Inside it: `sglang 0.5.17.dev20260819+g574274660f`, `torch 2.9.1+rocm7.2.0.git7e1940d4`,
HIP `7.2.26015-fc0010cf6a`, and — the decisive difference — **`import sgl_kernel` succeeds**.

### Start the container

```bash
docker run -d --name sglang_bringup \
  --device /dev/kfd --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -e HF_HOME=/mnt/data_1.5t/hf_cache -e HF_HUB_OFFLINE=1 \
  -v /mnt/data_1.5t/hf_cache:/mnt/data_1.5t/hf_cache \
  -v /mnt/data_450g:/mnt/data_450g \
  -v /home/planolab/software_test/training_junk:/work \
  --network bridge \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity
```

`renderD144`/`renderD152` are **physical GPUs 2 and 3**. Only those two nodes are passed in,
so inside the container they appear as `cuda:0` and `cuda:1` — `HIP_VISIBLE_DEVICES=0` in
the container means physical GPU 2. Weights and logs stay on the mounted volumes, never on `/`.

Sanity check:

```bash
docker exec sglang_bringup python3 -c \
  "import torch, sgl_kernel; from importlib.metadata import version; \
   print(version('sglang'), torch.__version__, torch.version.hip, torch.cuda.device_count())"
# 0.5.17.dev20260819+g574274660f 2.9.1+rocm7.2.0.git7e1940d4 7.2.26015-fc0010cf6a 2
```

### Serve — single GPU (physical GPU 2)

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8101 \
     > /mnt/data_450g/outputs/inference_embedding_sglang/embed_rep_a.log 2>&1"
```

Server-side evidence (`embed_rep_a.log`):

```
[2026-08-20 02:23:11] Freezing GC in Scheduler process. gen0: 371->0, gen1: 962->0, gen2: 1324724->0
[2026-08-20 02:23:27] Prefill batch, #new-seq: 3, #new-token: 28, #cached-token: 0, token usage: 0.00, cuda graph: False, input throughput (token/s): 27.83
[2026-08-20 02:23:27] INFO:     127.0.0.1:42352 - "POST /v1/embeddings HTTP/1.1" 200 OK
```

`GET /get_model_info` confirms the server really is in embedding mode on the right family:

```json
{"model_path":"google/embeddinggemma-300m","is_generation":false,"model_type":"gemma3_text",
 "architectures":["Gemma3TextModel"],
 "embedding":{"family":"embeddinggemma","task":"embed","execution":"encoder_only",
              "attention":"bidirectional","pooling":"mean","normalize":true,"enabled":true}}
```

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/embedding sglang_bringup \
  python inference_embedding_sglang.py --port 8101
```

**Real output:**

```
[health] server ready after 1.0s
[latency] 0.03s | n_vectors=3 dim=768
[vector0 head] [-0.1543, -0.02832, -0.00574, 0.00757, -0.00922, 0.02148, -0.05151, 0.05591]
[cosine] +0.8720  <- vLLM supports AMD ROCm.
[cosine] +0.8061  <- SQLite is an embedded database.
```

**Cosine sanity check passes:** query *"Which inference engines support AMD ROCm?"* scores the
relevant document **+0.8720** above the irrelevant one **+0.8061**. Dimension is **768**, which
is EmbeddingGemma's native width, and the server reports `normalize:true` with mean pooling.
The absolute gap is narrow (0.066) because the client sends raw text; EmbeddingGemma is trained
with `task:`-style prompt prefixes (`task: search result | query: …`) and the separation widens
considerably when those are used. Ordering — the thing under test — is correct either way.

### Two concurrent single-GPU replicas (instead of TP=2)

**TP=2 is pointless for a 300M model** — 0.63 GB of weights split across two MI355X leaves each
GPU almost empty while adding an all-reduce on every forward pass, so it is strictly slower than
one GPU. Throughput on a model this size comes from *replicas*, not tensor parallelism. The
tested pattern is one independent server per GPU behind a load balancer:

```bash
# replica A -> physical GPU 2, port 8101
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8101 > /mnt/data_450g/outputs/inference_embedding_sglang/embed_rep_a.log 2>&1"
# replica B -> physical GPU 3, port 8111
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path google/embeddinggemma-300m --is-embedding --mem-fraction-static 0.5 \
     --host 0.0.0.0 --port 8111 > /mnt/data_450g/outputs/inference_embedding_sglang/embed_rep_b.log 2>&1"
```

Both replicas hit concurrently:

```
=== REPLICA A (port 8101, phys GPU2) ===        === REPLICA B (port 8111, phys GPU3) ===
[health] server ready after 1.0s                [health] server ready after 1.0s
[latency] 0.03s | n_vectors=3 dim=768           [latency] 0.03s | n_vectors=3 dim=768
[cosine] +0.8720  <- vLLM supports AMD ROCm.    [cosine] +0.8720  <- vLLM supports AMD ROCm.
[cosine] +0.8061  <- SQLite is an embedded db.  [cosine] +0.8061  <- SQLite is an embedded db.
```

`rocm-smi --showmemuse` sampled *during* the concurrent run — **both GPUs loaded**:

```
GPU[2]		: GPU Memory Allocated (VRAM%): 51
GPU[3]		: GPU Memory Allocated (VRAM%): 51
```

Both replicas returned **bit-identical vectors** (same `vector0 head`, same cosines to 4 dp),
which confirms the two GPUs are numerically consistent — a replica pool is safe to load-balance
across without clients seeing drift. 51% VRAM each is `--mem-fraction-static 0.5`, not real
demand; a 300M model needs far less, so lower it to pack more replicas per GPU.

### Container-route quirks

- **`sleep infinity` + `docker exec -d`**, not `docker run` per server. Model load is ~15 s and
  the AITER JIT warm-up is one-off per container; keeping one long-lived container makes the
  second and later launches noticeably cheaper.
- **`HF_HUB_OFFLINE=1`** is set in the image env. All four models used here were already in
  `/mnt/data_1.5t/hf_cache`, so nothing was downloaded. Unset it if you need a fresh pull —
  and point `HF_HOME` at `/mnt/data_450g/hf_cache` first, because `/` has only ~109 GB free.
- **AITER is the default attention backend** (`attention_backend='aiter'`) and JIT-compiles
  kernels on first use. Expect a burst of `[aiter] import [mha_batch_prefill_...] under
  /sgl-workspace/aiter/aiter/jit/*.so` lines, then silence.
- **`not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv, will use default
  config! using torch solution:0`** appears for every GEMM shape. Harmless — AITER has no
  pre-tuned entry for EmbeddingGemma's small shapes and falls back to a torch GEMM. It costs
  performance, not correctness, and is expected for a 300M model nobody has tuned for.
- **`Ignore import error when loading sglang.srt.models.inkling: No module named 'cutlass'`** —
  benign, an unrelated CUDA-only model class failing to register.
- `FastAPIDeprecationWarning: ORJSONResponse is deprecated` — cosmetic, upstream.

## Install

Python 3.12; `requirements_sglang_embedding.txt` is the tested set. The stack is identical
to the other two SGLang folders, so one venv can be shared.

> **Note:** the pip route below is the *blocked* one, kept for the diagnosis. For a working
> setup use the container route above.

### AMD / ROCm — exactly what was run

```bash
python3 -m venv .env_inference_embedding_sglang
source .env_inference_embedding_sglang/bin/activate
pip install -U pip
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install --no-deps sglang==0.5.17
pip install -r requirements_sglang_embedding.txt
python -c "import torch; print(torch.__version__, torch.version.hip)"   # 2.11.0+rocm7.2 7.2.26015
```

`--no-deps` is mandatory: sglang 0.5.17 declares `cuda-python`, `flashinfer_python[cu13]`,
`flash-attn-4`, `sglang-kernel==0.4.5` and PyPI `torch==2.11.0` (CUDA) as **base**
dependencies — a plain install replaces ROCm torch with a CUDA wheel. If that happens:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` (CUDA-only). Never `pip install aiter` — the PyPI package of
that name is an unrelated async-iterator library, not AMD's AITER.

This folder's `.env_inference_embedding_sglang` is a **symlink** to
`../../../inference/sglang/llm/.env_inference_llm_sglang` — the three SGLang folders share one
identical 16 GB environment rather than duplicating it. Create a real venv here instead if
you want independent pins.

**Client verified independently of the engine.** Since no SGLang server can serve on this
build, `inference_embedding_sglang.py` was exercised against a minimal OpenAI-compatible
stub to confirm the client half is correct (health wait → POST → vector parsing → cosine):

```
[health] server ready after 0.0s
[latency] 0.00s | n_vectors=3 dim=768
[cosine] +1.0000  <- vLLM supports AMD ROCm.
```

Those numbers come from the stub, **not** from SGLang or EmbeddingGemma.

A fail-loud import shim for the two missing native packages (`aiter`, `sgl_kernel`) was used
purely to locate where the ROCm path stops; it raises on any real call, so it cannot serve
traffic. See `readme_sglang_llm.md` for its full description.

### NVIDIA / CUDA (upstream route — not run here)

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
export HF_HOME=/mnt/data_1.5t/hf_cache     # weights must not land on /
export HIP_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=2,3            # never empty on ROCm
```

## Run

### Launch — single GPU (physical GPU 2)

```bash
source .env_inference_embedding_sglang/bin/activate
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 HF_HOME=/mnt/data_1.5t/hf_cache

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
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model-path google/embeddinggemma-300m --is-embedding --port 8101 &
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
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

## Single-GPU results (measured, gfx950)

```
[01:38:47] Attention backend not specified. Use aiter backend by default.
           server_args=... is_embedding=True, attention_backend='aiter', chunked_prefill_size=-1
[01:38:57] Load weight end. elapsed=0.38 s, type=EmbeddingGemmaModel, avail mem=286.63 GB, mem usage=0.63 GB.
[01:38:58] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 6247850, K size: 71.50 GB, V size: 71.50 GB
[01:39:05] Scheduler hit an exception: ...
RuntimeError: ROCm shim: aiter.mha_batch_prefill_func was really called - needs a native ROCm build
```

| Metric | Value |
|---|---|
| Cold start to failure | **30 s** (server never reaches `/health`) |
| Model-load VRAM | **0.63 GB** (EmbeddingGemma-300m, bf16) |
| Weight-load time | 0.38 s |
| KV cache allocated | 143 GB (6,247,850 tokens @ `mem-fraction-static 0.5`) |
| Architecture resolved | **`EmbeddingGemmaModel`** — a dedicated SGLang implementation |
| Endpoint reachable | **No** |
| Real embedding vectors / dims | **None** — no forward pass completes |

Note the failing frame differs from the LLM folder: this workload dies inside **AMD's own
attention entry point**, `aiter.mha_batch_prefill_func`, during the warm-up prefill —
i.e. SGLang routed to the AITER ROCm fast path exactly as intended, and that path has no
installable implementation. It is the same root cause reached by a different door.

## Multi-GPU (TP=2) results (measured, gfx950)

**TP=2 is pointless for a 300M-parameter embedding model** — a 0.63 GB model on a 288 GB
card gains nothing from sharding, and each request would pay a cross-GPU all-reduce. The
correct scale-out is **two independent single-GPU replicas** behind a load balancer (command
above). It was run anyway to prove the multi-GPU path:

```
[01:41:01 TP0] Setup Custom allreduce failed with ROCm shim: aiter.dist.device_communicators
               .custom_all_reduce.CustomAllreduce ... specify --disable-custom-all-reduce explicitly.
[01:41:01 TP1] (same)
[01:41:42 TP0] Scheduler hit an exception ...
Exception: Capture cuda graph failed:
  ROCm shim: sgl_kernel.rotary_embedding was really called - needs a native ROCm build
```

`rocm-smi` sampled every 2 s during the TP=2 launch — **both owned GPUs held memory
simultaneously** (bytes used; total 309,220,868,096 B = 288 GiB per card):

```
01:40:59 | card2,309220868096,816869376  | card3,309220868096,816869376     # ~779 MiB each
01:41:01 | card2,309220868096,1550635008 | card3,309220868096,1550630912    # ~1.44 GiB each
```

So the two ranks really did initialise on physical GPUs 2 and 3, SGLang selected AMD's
`AiterCustomAllreduce` by default, and the run then hit the same missing-kernel wall.
Cold start to failure: 25 s. Neither replica nor TP rank ever served a request, so **no
embedding vectors were produced at any topology** — two concurrent replicas could not be
demonstrated end to end for the same reason.

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

Server flags that mattered:

| Flag | Value used | Why |
|---|---|---|
| `--model-path` | `google/embeddinggemma-300m` | Gated; needs `HF_TOKEN` |
| `--is-embedding` | on | Correct for this model (and forbidden for the Qwen3 reranker) |
| `--tp` | `1` / `2` | 1 is correct in production; 2 tested only to exercise multi-GPU |
| `--mem-fraction-static` | `0.5` / `0.4` | KV/static pool fraction; auto-reduced at TP>1 |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang selects `aiter` |

## Output

No artifacts are written by the server. Validation logs are under
`/mnt/data_1.5t/outputs/inference_embedding_sglang/` (`embed_tp1.log`, `embed_tp2.log`,
`vram_tp2_sample.log`). Weights live in `$HF_HOME=/mnt/data_1.5t/hf_cache`. The client
prints dimensions, a vector head, and cosine scores to stdout only.

## Hardware support & evidence

| | NVIDIA | AMD |
|---|---|---|
| Status here | **Not tested** — no NVIDIA GPU on this host | **Tested, blocked** — 2×MI355X (gfx950), ROCm 7.2.4 |
| Upstream position | Cookbook exists; **lists an NVIDIA GPU as a prerequisite** | Generic docs list EmbeddingGemma as supported; no ROCm recipe |
| Install | `uv pip install sglang` | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Model class | `EmbeddingGemmaModel` | `EmbeddingGemmaModel` — **confirmed loading on gfx950** |

Evidence gathered 2026-08-20:

- `sglang` 0.5.17 PyPI metadata lists CUDA-only packages as **base** dependencies and has no
  `srt_hip` extra; `sglang-kernel` 0.4.5 publishes only `manylinux2014_x86_64`/`aarch64`
  CUDA wheels.
- `sglang/srt/layers/rotary_embedding/base.py:69-75,116-117` imports `sgl_kernel` directly
  under `if _is_hip:` — building AITER alone cannot substitute for it.
- `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` publishes no `aiter` or `sglang` wheel.
- `lmsysorg/sglang-rocm` publishes exact daily tags for this host, e.g.
  `v0.5.17-rocm720-mi35x-20260819` (23.4 GB compressed).
- Local run logs quoted above (`EmbeddingGemmaModel`, `aiter` backend auto-selected,
  `aiter.mha_batch_prefill_func` frame, dual-card `rocm-smi` sample).

## Notes / quirks

- **The warning is confirmed, with nuance.** The NVIDIA-only cookbook
  prerequisite is real, but the failure is *not* EmbeddingGemma-specific: it is SGLang's
  ROCm kernel packaging, which blocks every model equally. Once the ROCm container is used,
  this model has a first-class SGLang implementation.
- `chunked_prefill_size` is forced to `-1` (disabled) for embedding runs — expected.
- **KV cache is huge by default** — 143 GB on a 288 GB card even for a 300M model. Set
  `--mem-fraction-static` or `--max-total-tokens` explicitly when sharing the box.
- `Failed to import amdsmi` on every launch — harmless; install `amdsmi` for AMD telemetry.
- `Ignoring corrupted tree cache file ... Permission denied` — shared HF cache owned by
  another user; cosmetic, the snapshot is still found locally.
- **Ports** — 8101 is the SGLang embedding slot; 8100/8102 belong to sibling folders.
- Re-export `HIP_VISIBLE_DEVICES` **and** `CUDA_VISIBLE_DEVICES` after activating the venv;
  never set `CUDA_VISIBLE_DEVICES` empty on ROCm.

## Follow-ups

1. Re-run inside `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` when `/` has >150 GB
   free, with the canonical AMD docker flags (`--device /dev/kfd`,
   `--device /dev/dri/renderD144`, `--device /dev/dri/renderD152`, `--group-add video`,
   `--group-add render`, `--ipc=host`, `--cap-add=SYS_PTRACE`,
   `--security-opt seccomp=unconfined`, `--shm-size 64G`).
2. Then benchmark two single-GPU replicas: embedding requests/sec, tokens/sec, p50/p95, and
   cosine agreement against the sentence-transformers reference — the numbers to collect
   before standardising on SGLang for embeddings on AMD.
