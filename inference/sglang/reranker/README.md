# `inference/sglang/reranker` — SGLang reranker serving (Qwen3-Reranker-0.6B)

## Overview & when to use

Serve `Qwen/Qwen3-Reranker-0.6B` behind `POST /v1/rerank` with **SGLang**
(`python -m sglang.launch_server`), and score query/document pairs with
`inference_reranker_sglang.py`, which prints the real relevance scores in ranked order.

Qwen3-Reranker is **not** a cross-encoder with a classification head: it is a
**decoder-only yes/no reranker**, scored from the logits of the `yes`/`no` tokens under a
specific prompt. Two consequences drive this whole folder:

- **Never launch it with `--is-embedding`.** That flag is for true embedding models
  (see the sibling EmbeddingGemma folder); using it here mis-configures the runner.
- **It needs the `qwen3_reranker` jinja chat template**, shipped here as
  `qwen3_reranker.jinja` (fetched verbatim from `sgl-project/sglang` tag `v0.5.17`,
  `examples/chat_template/qwen3_reranker.jinja`), which renders the
  Instruct/Query/Document prompt the model was trained on.

Use this folder when SGLang already serves your generation traffic and you want reranking
in the same engine. On **AMD gfx950 it does not currently serve from a pip install** — see
the VERDICT; use the sentence-transformers `CrossEncoder` or vLLM reranker route instead.

> **Tested topology:** 2×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
> Python 3.12.3, physical GPUs 2 and 3, single node. No NVIDIA GPU on this host.

## VERDICT

**🟢 WORKS on ROCm/gfx950 via the vendor container — pip route stays blocked.**
`Qwen/Qwen3-Reranker-0.6B` produces real, correctly-ordered relevance scores from
`POST /v1/rerank` on MI355X inside `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`.
The reranker configuration this folder documents (no `--is-embedding`, jinja template,
`--disable-radix-cache`) is confirmed correct — it was accepted by SGLang at TP=1 and TP=2
under pip, and it *serves* under the container. See "Container route — TESTED".

| Question | Answer (measured here) |
|---|---|
| Does SGLang accept the reranker config? | **Yes** — template loaded, `is_embedding=False`, radix cache disabled |
| Do weights load on gfx950? | **Yes** — 1.21 GB in 0.45 s; TP=2 shards to 0.61 GB/rank |
| Does TP=2 bring-up work? | **Yes** — 2 ranks, `AiterCustomAllreduce (AMD default)` |
| Does a forward pass run **in the container**? | **Yes** — `/v1/rerank` 200 OK in 0.14 s, 3 scored pairs |
| Real rerank scores produced? | **Yes, in the container** — 0.7773 / 0.1403 / 0.000024, correctly ordered |
| Does a forward pass run **from pip**? | **No** — dies at `sgl_kernel.rotary_embedding` |
| Root cause of the pip failure | `sgl_kernel` + `aiter` have **no ROCm wheel**; SGLang on HIP imports both. The container ships both prebuilt |

Supported ROCm route: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **pulled
(89.9 GB on disk) and verified working**; see "Container route" immediately below.

---

# H100 (NVIDIA) — the PIP ROUTE SERVES (verified 2026-08-22)

**Headline: on H100 the pip route works — the ROCm pip failure this folder documents is gone.**
The root cause on MI355X was that `sgl_kernel` + `aiter` have no ROCm wheel; on NVIDIA
`sgl_kernel` ships as a CUDA wheel, so `pip install "sglang[all]"` serves `/v1/rerank` directly —
**no container.** The reranker configuration rules this folder established (no `--is-embedding`,
the `qwen3_reranker.jinja` template, `--disable-radix-cache`) are unchanged and still correct.

> **Tested topology:** 1×NVIDIA H100 80GB HBM3, **physical GPU 6 only** (single-GPU smoke test),
> cc(9,0), CUDA 13.0, driver 580.173.02, Python 3.12.3. GPUs 0–3 = co-tenant production job,
> untouched. TP deferred.

## H100 VERDICT

**🟢 WORKS on H100 via the pip route.** `Qwen/Qwen3-Reranker-0.6B` loads as `Qwen3ForCausalLM`
(`is_generation: true`), the jinja template renders the yes/no judging prompt, and `/v1/rerank`
returns correctly-ordered relevance scores with a decisive 3-tier spread.

| Question | Answer (measured on H100 GPU 6) |
|---|---|
| Does the pip route serve `/v1/rerank`? | **Yes** — `pip install "sglang[all]"`, no container; `import sgl_kernel` succeeds |
| Arch / mode | **`Qwen3ForCausalLM`**, `is_generation: true` (from `/get_model_info`) — the "no `--is-embedding`" rule holds |
| Jinja template loaded? | **Yes** — "Loading chat template from argument: ./qwen3_reranker.jinja" |
| Real scores, correct order? | **Yes** — 0.7773 / 0.1480 / 0.000033, correctly ranked (~23,500× margin rel-vs-irrel) |
| GPU-6 residency proof? | **Yes** — `sglang::scheduler` held **26346 MiB** on GPU-6 (UUID `GPU-e4fe48bc-…`) |

## H100 install (pip route — VERIFIED)

Same shared venv as the LLM/embedding leaves (full block in `llm/README.md` H100 section):

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip && pip install torch numpy            # torch 2.13.0+cu130
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"                                 # sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17
python -c "import sgl_kernel; print('sgl_kernel OK')"     # OK — impossible on ROCm
```

## H100 serve — single GPU (physical GPU 6)

Same flags as the ROCm container command, minus the HIP env: **no `--is-embedding`**, the folder's
jinja template, `--disable-radix-cache`.

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1   # plain CUDA — no HIP_VISIBLE_DEVICES
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --chat-template ./qwen3_reranker.jinja \
  --disable-radix-cache --mem-fraction-static 0.3 \
  --host 127.0.0.1 --port 8600
```

On the pip route `--chat-template` takes a **host** path (`./qwen3_reranker.jinja`) — there is no
container bind-mount, so the `/work/...` path the ROCm section uses does not apply here.

Server-side evidence (`/dev/shm/h100/out/sglang/rerank_0p6b.log`), verbatim:

```
Loading chat template from argument: ./qwen3_reranker.jinja
Load weight end. elapsed=24.78 s, type=Qwen3ForCausalLM, avail mem=77.29 GB, mem usage=1.18 GB.
max_total_num_tokens=208651, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2608, context_len=40960, available_gpu_mem=53.50 GB
The server is fired up and ready to roll!
```

`/get_model_info` confirms the reranker is served as a causal LM (not a pooling encoder):

```json
{"model_type": "qwen3", "is_generation": true, "architectures": ["Qwen3ForCausalLM"]}
```

## H100 client / smoke command

```bash
python inference_reranker_sglang.py --port 8600
```

**Real output — ordering correct, spread decisive:**

```
[health] server ready after 1.0s
[latency] 0.06s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.777300  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.148047  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.000033  PostgreSQL is a relational database.
```

The directly-relevant doc scores **0.7773**; the topically-related-but-non-answering ROCm
sentence lands mid-pack at **0.1480**; the irrelevant PostgreSQL sentence is driven to
**0.000033** — a ~23,500× margin. This three-tier signature matches the MI355X container almost
exactly (0.7773 / 0.1403 / 0.000024 there): **the RANKING is identical and the scores agree to
~3 decimals**, so the yes/no-logit reranking is hardware-portable pip↔container / NVIDIA↔AMD.

## H100 GPU-6 residency proof (from inside serving)

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader | grep e4fe48bc
1704595, sglang::scheduler, 26346 MiB, GPU-e4fe48bc-0c29-f21e-6523-759f964bf823
```

Only GPU 6 touched. (26 GB is the `--mem-fraction-static 0.3` KV pool; the 0.6B weights are 1.18
GB — this reranker is throughput-bound over short pairs, so scale by replicas, not TP.)

## H100 quirks / notes

- Same benign `torchcodec`/`libavutil` startup traceback as the other leaves — ignore it.
- `torch 2.13.0+cu130` is **not** clobbered by `pip install "sglang[all]"` (re-verified).
- **`--disable-radix-cache` is still correct** — every rerank pair is a distinct query+document
  prompt with no shared prefix, so prefix caching only wastes memory.
- **TP=2 remains pointless for 0.6B** — scale with independent single-GPU replicas (one
  `CUDA_VISIBLE_DEVICES` each). Multi-GPU/replica pass deferred until GPUs 0–3 free.

## Container route — `lmsysorg/sglang-rocm` (TESTED)

**This is the route that works on gfx950.** Everything below this heading was executed on

**This is the route that works on gfx950.** Everything below this heading was executed on
2×MI355X (physical GPUs 2 and 3); output is copied verbatim. The pip analysis further down is
retained and still accurate — it explains *why* the container is required.

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk**. Inside it:
`sglang 0.5.17.dev20260819+g574274660f`, `torch 2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015`,
and **`import sgl_kernel` succeeds** — the exact import that kills the pip route at
`sgl_kernel.rotary_embedding`.

The container is started once and reused (see `readme_sglang_embedding.md` for the full
`docker run`); `renderD144`/`renderD152` map physical GPUs 2 and 3 to `cuda:0`/`cuda:1` inside.

### Serve — single GPU (physical GPU 2)

Note the flags: **no `--is-embedding`**, the folder's jinja template, and `--disable-radix-cache`.

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8102 \
     > /mnt/data_450g/outputs/inference_reranker_sglang/rerank_single.log 2>&1"
```

Server-side evidence (`rerank_single.log`):

```
[2026-08-20 02:53:48] max_total_num_tokens=325376, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=4067, context_len=40960, available_gpu_mem=97.73 GB
[2026-08-20 02:53:48] INFO:     Uvicorn running on http://0.0.0.0:8102 (Press CTRL+C to quit)
[2026-08-20 02:53:50] The server is fired up and ready to roll!
```

**`GET /get_model_info` proves the "no `--is-embedding`" rule is right:**

```json
{"model_path": "Qwen/Qwen3-Reranker-0.6B", "is_generation": true,
 "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
```

`is_generation: true` and the model's `embedding` block reports `"enabled": false`,
`"family": "none"`. The reranker is served as a **causal LM**: `/v1/rerank` renders each
(query, document) pair through `qwen3_reranker.jinja` into the yes/no judging prompt and
converts the `yes` logit into a relevance score. Adding `--is-embedding` would switch the
model into a pooling encoder, destroy the LM head the score is read from, and break `/v1/rerank`.

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/reranker sglang_bringup \
  python inference_reranker_sglang.py --port 8102
```

**Real output:**

```
[health] server ready after 1.0s
[latency] 0.14s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.777300  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.140336  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.000024  PostgreSQL is a relational database.
```

**Ordering is correct and the separation is decisive.** The directly relevant document scores
**0.7773**; the topically-related-but-non-answering ROCm sentence lands mid-pack at **0.1403**;
the irrelevant PostgreSQL sentence is driven to **0.000024** — a ~32,000× margin over the
relevant doc. This three-tier spread is the signature of a working cross-encoder and is far
sharper than the bi-encoder cosine separation in the embedding folder, which is exactly why a
reranker is worth the extra pass over top-k retrieval results.

### Two concurrent single-GPU replicas (instead of TP=2)

**TP=2 is pointless for a 0.6B model** — 1.21 GB of weights sharded to ~0.61 GB per rank leaves
both MI355X nearly idle while adding an all-reduce per layer per forward pass; the collective
costs more than the matmuls it splits. Rerankers are throughput-bound over many short pairs, so
the right scaling axis is replicas. One server per GPU:

```bash
# replica A -> physical GPU 2, port 8102   (command above)
# replica B -> physical GPU 3, port 8112
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8112 \
     > /mnt/data_450g/outputs/inference_reranker_sglang/rerank_rep_b.log 2>&1"
```

Both replicas hit concurrently:

```
=== REPLICA A (8102, phys GPU2) ===                === REPLICA B (8112, phys GPU3) ===
[latency] 0.03s | pairs=3                          [latency] 0.08s | pairs=3
[rank 1] score=0.777300  vLLM and SGLang both...   [rank 1] score=0.777300  vLLM and SGLang both...
[rank 2] score=0.140336  ROCm is AMD's open...     [rank 2] score=0.140336  ROCm is AMD's open...
[rank 3] score=0.000024  PostgreSQL is a rel...    [rank 3] score=0.000024  PostgreSQL is a rel...
```

`rocm-smi --showmemuse` sampled *during* the concurrent run — **both GPUs loaded**:

```
GPU[2]		: GPU Memory Allocated (VRAM%): 66
GPU[3]		: GPU Memory Allocated (VRAM%): 66
```

(66% = this reranker's `--mem-fraction-static 0.3` on top of the embedding replica's 0.5 pool
still resident on the same two GPUs from the embedding test; the reranker alone is ~15%.)

Scores are **bit-identical across the two GPUs** (0.777300 / 0.140336 / 0.000024 on both), so a
replica pool can be load-balanced without rank-dependent score drift — important for a reranker,
where inconsistent scores across replicas would reshuffle result ordering between requests.

### Container-route quirks

- **`--chat-template` takes a container-visible path.** `/work` is the bind-mounted repo, so
  `/work/inference/sglang/reranker/qwen3_reranker.jinja` resolves. A host path will not.
- **`--disable-radix-cache` is correct here.** Every rerank pair is a distinct
  query+document prompt with essentially no shared prefix, so prefix caching only burns memory
  and bookkeeping. The server confirms `disable_radix_cache=True`.
- **AITER JIT probes a compiler flag and fails loudly at startup — this is benign:**
  ```
  clang (LLVM option parsing): Unknown command line argument '-amdgpu-coerce-illegal-types=1'.
  [aiter] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.
  ```
  AITER probes for the flag, sees ROCm 7.2's clang reject it, drops it and rebuilds without it.
  Startup continues normally. Do not chase this error — it is a feature probe, not a failure.
- **First launch takes ~2 minutes**, dominated by AITER JIT-compiling kernels into
  `/root/.aiter/build/`. Subsequent launches in the same container are much faster. Destroying
  the container throws that cache away, which is a further reason to keep one long-lived.
- `Ignore import error when loading sglang.srt.models.inkling: No module named 'cutlass'` —
  benign, a CUDA-only model class failing to register.

## Install

Python 3.12; `requirements_sglang_reranker.txt` is the tested set. Same stack as the other
two SGLang folders, so one venv can be shared.

### AMD / ROCm — exactly what was run

```bash
python3 -m venv .env_inference_reranker_sglang
source .env_inference_reranker_sglang/bin/activate
pip install -U pip
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install --no-deps sglang==0.5.17
pip install -r requirements_sglang_reranker.txt
python -c "import torch; print(torch.__version__, torch.version.hip)"   # 2.11.0+rocm7.2 7.2.26015
```

`--no-deps` is mandatory: sglang 0.5.17 declares `cuda-python`, `flashinfer_python[cu13]`,
`flash-attn-4`, `sglang-kernel==0.4.5` and PyPI `torch==2.11.0` (CUDA build) as **base**
dependencies, so a plain install replaces ROCm torch with a CUDA wheel. Recovery:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` (CUDA-only). Never `pip install aiter` — that PyPI name is
an unrelated async-iterator library, not AMD's AITER.

This folder's `.env_inference_reranker_sglang` is a **symlink** to
`../../../inference/sglang/llm/.env_inference_llm_sglang` — the three SGLang folders share one
identical 16 GB environment rather than duplicating it. Create a real venv here instead if
you want independent pins.

**Client verified independently of the engine.** Since no SGLang server can serve on this
build, `inference_reranker_sglang.py` was exercised against a minimal OpenAI-compatible
stub to confirm the client half is correct (health wait → POST → score parsing → ranking):

```
[health] server ready after 0.0s
[latency] 0.00s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=1.000000  vLLM and SGLang both support AMD ROCm GPUs.
```

Those scores come from the stub, **not** from SGLang or Qwen3-Reranker.

A fail-loud import shim for the missing native packages (`aiter`, `sgl_kernel`) was used to
locate exactly where the ROCm path stops; it raises on any real call and cannot serve
traffic. Full description in `readme_sglang_llm.md`.

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

`Qwen/Qwen3-Reranker-0.6B` is ungated, so the token is only needed if you swap in a gated
model. The client loads it via `load_dotenv("dev.env")`; run from inside this folder. Never
print or commit the token.

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache     # weights must not land on /
export HIP_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=2,3            # never empty on ROCm
```

## Run

### Launch — single GPU (physical GPU 2)

```bash
source .env_inference_reranker_sglang/bin/activate
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 HF_HOME=/mnt/data_1.5t/hf_cache

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --chat-template qwen3_reranker.jinja \
  --mem-fraction-static 0.5 \
  --host 0.0.0.0 --port 8102
```

Note what is **absent**: no `--is-embedding` (wrong for a decoder-only yes/no reranker) and
no `--attention-backend` (on ROCm SGLang auto-selects `aiter`; `flashinfer` is NVIDIA-only).
`--disable-radix-cache` matches the upstream reranker recipe — prefix reuse across unrelated
query/document pairs is not useful here.

### Launch — multi-GPU, TP=2 (physical GPUs 2+3)

```bash
export HIP_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=2,3
python -m sglang.launch_server --model-path Qwen/Qwen3-Reranker-0.6B --trust-remote-code \
  --disable-radix-cache --chat-template qwen3_reranker.jinja \
  --tp 2 --mem-fraction-static 0.4 --host 0.0.0.0 --port 8102
```

### Client / smoke command

```bash
python inference_reranker_sglang.py --port 8102 \
  --query "Which inference engines support AMD ROCm?" \
  --documents "vLLM and SGLang both support AMD ROCm GPUs." \
              "PostgreSQL is a relational database." \
              "ROCm is AMD's open compute platform for Instinct accelerators."
```

Expected on a working build — the database sentence must rank last:

```
[health] server ready after <N>s
[latency] 0.0x s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.9xxxxx  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.xxxxxx  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.0xxxxx  PostgreSQL is a relational database.
```

## Single-GPU results (measured, gfx950)

```
[01:39:42] Attention backend not specified. Use aiter backend by default.
           server_args=... is_embedding=False, disable_radix_cache=True,
           chat_template='.../qwen3_reranker.jinja', attention_backend='aiter'
[01:39:47] Load weight end. elapsed=0.45 s, type=Qwen3ForCausalLM, avail mem=286.05 GB, mem usage=1.21 GB.
[01:39:47] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1131687, K size: 60.44 GB, V size: 60.44 GB
[01:39:49] Scheduler hit an exception ...
Exception: Capture cuda graph failed:
  ROCm shim: sgl_kernel.rotary_embedding was really called - needs a native ROCm build
```

| Metric | Value |
|---|---|
| Cold start to failure | **20 s** (server never reaches `/health`) |
| Model-load VRAM | **1.21 GB** (Qwen3-Reranker-0.6B, bf16) |
| Weight-load time | 0.45 s |
| KV cache allocated | 121 GB (1,131,687 tokens @ `mem-fraction-static 0.5`) |
| Runner mode | `is_embedding=False` — correct decoder-only reranker path |
| Chat template | Loaded from `qwen3_reranker.jinja` without error |
| Endpoint reachable | **No** |
| Real rerank scores | **None** — no forward pass completes |

The configuration in question was therefore validated as far as the engine
allows: SGLang accepted the reranker **without** `--is-embedding`, took the jinja template,
and set up a causal-LM runner with radix caching off. Only the kernel call failed.

## Multi-GPU (TP=2) results (measured, gfx950)

```
[01:41:34 TP0] [AR] Using AiterCustomAllreduce (AMD default)
[01:41:34 TP1] [AR] Using AiterCustomAllreduce (AMD default)
[01:41:36 TP0] Load weight end. elapsed=0.33 s, type=Qwen3ForCausalLM, avail mem=285.65 GB, mem usage=0.61 GB.
[01:41:36 TP1] Load weight end. elapsed=0.33 s, type=Qwen3ForCausalLM, avail mem=285.65 GB, mem usage=0.61 GB.
Exception: Capture cuda graph failed:
  ROCm shim: sgl_kernel.rotary_embedding was really called - needs a native ROCm build
```

**Tensor-parallel sharding demonstrably works up to the kernel boundary:** per-rank weight
memory is **0.61 GB at TP=2 versus 1.21 GB at TP=1 — exactly half**, on both ranks, with
RCCL 2.27.7 forming the group and AMD's `AiterCustomAllreduce` selected by default. Cold
start to failure: 24 s. A matching `rocm-smi` sample from the sibling embedding TP=2 run
shows both owned cards holding memory at the same time:

```
card2,309220868096,1550635008 | card3,309220868096,1550630912    # ~1.44 GiB each, 288 GiB cards
```

**Is TP=2 worth it for a 0.6B reranker? No.** Sharding a 1.2 GB model across two 288 GB
cards buys nothing and adds an all-reduce per forward pass. In production run **two
independent single-GPU replicas** behind a load balancer and shard the *document set*, not
the model:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B --disable-radix-cache \
  --chat-template qwen3_reranker.jinja --port 8102 &
HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B --disable-radix-cache \
  --chat-template qwen3_reranker.jinja --port 8112 &
```

Neither replica can serve on this build, so the replica pattern could not be demonstrated
end to end — it is a recommendation, not a measurement.

## Arguments

Client (`inference_reranker_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8102` | Server port (8102 = SGLang reranker slot in the repo port map) |
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Model id as served; must match `--model-path` |
| `--query` | ROCm engines question | The rerank query |
| `--documents` | 3 sample documents | Candidates scored against the query |
| `--top_n` | `None` | If set, ask the server for only the top N |
| `--endpoint` | `/v1/rerank` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

Server flags that mattered:

| Flag | Value used | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3-Reranker-0.6B` | Decoder-only yes/no reranker |
| `--chat-template` | `qwen3_reranker.jinja` | **Required** — renders the Instruct/Query/Document prompt |
| `--disable-radix-cache` | on | Upstream reranker recipe; prefix reuse is useless across pairs |
| `--trust-remote-code` | on | Follows the upstream command |
| `--is-embedding` | **never** | Wrong for this model — it is not an embedding model |
| `--tp` | `1` / `2` | 1 is correct in production; 2 tested only to exercise multi-GPU |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang selects `aiter` |

## Output

No artifacts are written by the server. Validation logs are under
`/mnt/data_1.5t/outputs/inference_reranker_sglang/` (`rerank_tp1.log`, `rerank_tp2.log`).
Weights live in `$HF_HOME=/mnt/data_1.5t/hf_cache`. The client prints latency and ranked
scores to stdout only.

## Hardware support & evidence

| | NVIDIA | AMD |
|---|---|---|
| Status here | **Not tested** — no NVIDIA GPU on this host | **Tested, blocked** — 2×MI355X (gfx950), ROCm 7.2.4 |
| Upstream position | Native rerank support documented | Supported stack; exact target needs validation (this folder) |
| Install | `uv pip install sglang` | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Runner | decoder-only yes/no scoring | same — `is_embedding=False` confirmed on gfx950 |

Evidence gathered 2026-08-20:

- `sglang` 0.5.17 PyPI metadata lists CUDA-only packages as **base** dependencies; no
  `srt_hip` extra exists. `sglang-kernel` 0.4.5 ships only CUDA wheels
  (`manylinux2014_x86_64` 383 MB, `aarch64` 37 MB).
- `sglang/srt/layers/rotary_embedding/base.py:69-75,116-117` — under `if _is_hip:` SGLang
  imports `sgl_kernel` directly, so a real AITER build alone cannot substitute for it.
  AITER is opt-in besides: `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip`.
- `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` publishes no `aiter` or `sglang` wheel.
- `lmsysorg/sglang-rocm` publishes exact daily tags for this host, e.g.
  `v0.5.17-rocm720-mi35x-20260819` (23.4 GB compressed).
- `qwen3_reranker.jinja` fetched from the upstream `v0.5.17` tag and accepted by the server.
- Local run logs quoted above (TP=1 and TP=2 weight loads, allreduce selection, fail frame).

## Notes / quirks

- **`--is-embedding` remains the classic trap.** This folder
  respects it; SGLang started a causal-LM runner (`is_embedding=False`) as intended.
- **Template provenance matters.** `qwen3_reranker.jinja` renders the exact
  system + `<Instruct>/<Query>/<Document>` prompt whose `yes`/`no` logits define the score.
  A different template silently changes the scores rather than erroring.
- **KV cache is huge by default** — 121 GB on a 288 GB card for a 0.6B model. Set
  `--mem-fraction-static` or `--max-total-tokens` explicitly on a shared box.
- `Failed to import amdsmi` on every launch — harmless; install `amdsmi` for AMD telemetry.
- `Ignoring corrupted tree cache file ... Permission denied` — shared HF cache owned by
  another user; cosmetic.
- `--mem-fraction-static` is auto-reduced at TP>1 (0.4 → 0.34 at TP=2).
- **Ports** — 8102 is the SGLang reranker slot; 8100/8101 belong to sibling folders.
- Re-export `HIP_VISIBLE_DEVICES` **and** `CUDA_VISIBLE_DEVICES` after activating the venv;
  never set `CUDA_VISIBLE_DEVICES` empty on ROCm.

## Follow-ups

1. Re-run inside `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` when `/` has >150 GB
   free, with the canonical AMD docker flags (`--device /dev/kfd`,
   `--device /dev/dri/renderD144`, `--device /dev/dri/renderD152`, `--group-add video`,
   `--group-add render`, `--ipc=host`, `--cap-add=SYS_PTRACE`,
   `--security-opt seccomp=unconfined`, `--shm-size 64G`).
2. Then verify scoring correctness against the sentence-transformers `CrossEncoder`
   baseline on the same pairs, and measure rerank pairs/sec and p50/p95 across two replicas.
