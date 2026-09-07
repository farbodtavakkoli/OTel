# `inference/tei/embedding` — HF Text Embeddings Inference (EmbeddingGemma-300m)

> Stack overview, scope note, install/build route summary, and venv conventions:
> [`../README.md`](../README.md). Client deps: [`../requirements.txt`](../requirements.txt).

## Overview & when to use

Serve `google/embeddinggemma-300m` behind Hugging Face's **Text Embeddings Inference (TEI)**
router — a Rust HTTP server purpose-built for embedding and sequence-classification models —
and hit it with `embed_tei.py`, which embeds queries plus documents and prints the real
vector dimension and cosine scores.

Use this folder when embeddings are the *only* workload on the host and you want a small,
fast, dedicated server rather than a general LLM engine. TEI gives you `POST /embed` and an
OpenAI-compatible `POST /v1/embeddings`, batching, and Prometheus metrics, with a much
smaller footprint than vLLM or SGLang.

> **Tested topology:** 1×AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu,
> Python 3.12.3 host / 3.10.12 in container, TEI 1.9.3. This path works — see below.
> Upstream documents ROCm support as *experimental* and tested only on MI200/MI300, so
> **gfx950 is beyond its documented matrix.**

## Scope — why there is no LLM or reranker TEI folder

This is deliberate, not an omission — see [`../README.md`](../README.md) for the full
scope note.

| Missing folder | Why |
|---|---|
| `inference/tei/llm` | **TEI is not a generative server.** It serves embeddings and sequence classification only; there is no token-generation path. Use `../../vllm/llm/`, `../../sglang/llm/`, `../../llamacpp/llm/`, or `../../transformers/llm/`. |
| `inference/tei/reranker` | TEI *has* a `/rerank` API, but its supported reranker list targets **sequence-classification** architectures (XLM-RoBERTa, GTE, ModernBERT). `Qwen/Qwen3-Reranker-0.6B` is a **decoder-only yes/no** reranker and is not supported upstream (open model-support requests). Use [`../../llamacpp/reranker/`](../../llamacpp/reranker/) or [`../../vllm/reranker/`](../../vllm/reranker/). |

If you point TEI at a supported cross-encoder instead, the reranker path would be worth
re-testing on this hardware — the ROCm backend proven below is shared by both paths.

## Install (AMD / ROCm)

There is **no published AMD/ROCm container tag** for TEI — `ghcr.io/huggingface/text-embeddings-inference`
publishes CUDA tags (e.g. `cuda-1.9`), which are useless on AMD. The documented AMD
route is a **source build of the Rust router against a ROCm PyTorch base**.

Upstream's guide starts from `rocm/pytorch:latest`. The build below instead uses
`lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`, which already carries ROCm 7.2
PyTorch for gfx950 and so avoids a second large pull. Any ROCm PyTorch image with
a matching HIP works; nothing SGLang-specific is used.

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # server logs and result artifacts

# 1. Start a ROCm container with ONE GPU (e.g. physical GPU 6 -> renderD176; verify yours
#    with `ls /dev/dri/` and `rocm-smi --showbus`).
docker run -d --name tei_build \
  --device /dev/kfd --device /dev/dri/renderD176 \
  --group-add video --ipc=host --shm-size 16g \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -p 8301:8301 \
  -v $HF_HOME:/hf_cache \
  -v $OUTPUT_DIR/inference_embedding_tei:/outputs \
  lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819 sleep infinity

# 2. Toolchain: Rust + protoc are NOT in the base image.
docker exec tei_build bash -lc '
  cd /workspace
  curl -fsSL https://sh.rustup.rs -o rustup-init && sh rustup-init -y --no-modify-path
  . /workspace/rustup/env
  curl -fsSLO https://github.com/protocolbuffers/protobuf/releases/latest/download/protoc-*-linux-x86_64.zip
  unzip -q protoc-*.zip -d /workspace/protoc
'

# 3. Build the router with the Python (ROCm) backend.
docker exec tei_build bash -lc '
  . /workspace/rustup/env
  export PATH=/workspace/protoc/bin:$PATH
  cd /workspace
  git clone https://github.com/huggingface/text-embeddings-inference
  cd text-embeddings-inference
  pip install --no-deps -r backends/python/server/requirements-amd.txt
  cargo install --path router -F python -F http --no-default-features || \
    cargo build --release --bin text-embeddings-router -F python -F http --no-default-features
'
```

`cargo` reported **`Finished release profile ... in 3m 44s`**, producing a 28 MB
`target/release/text-embeddings-router`. **No source patches were needed for gfx950.**

### NVIDIA equivalent (generic sketch)

```bash
docker run --rm --gpus all -p 8080:80 -v "$PWD/data":/data \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id google/embeddinggemma-300m
```

> For **H100 / Hopper (cc 9.0)** the `cuda-1.9` tag above is the wrong architecture — use the
> **`hopper-1.9`** image and `--dtype float32`. See the verified
> **"Install & run (NVIDIA H100 / Hopper)"** section below for the exact, tested commands.

## Environment & secrets

`embed_tei.py` calls `load_dotenv("dev.env")`; `dev.env` is a symlink to the repo-root file
holding `HF_TOKEN`. EmbeddingGemma is a **gated** model, so the token is required for the
first download. Mount a pre-seeded HF cache — containers behind a corporate proxy often
cannot reach the HF CDN.

```bash
export HF_TOKEN=...        # from dev.env; never commit it
# in-container: HUGGINGFACE_HUB_CACHE=/hf_cache/hub
```

> TEI echoes its `Args { ... }` line at startup including a **partially-masked token**.
> Treat `tei_serve*.log` as sensitive and keep it out of git (write it under `$OUTPUT_DIR`).

## Run — single GPU

```bash
docker exec -d tei_build bash -lc '
  cd /workspace/text-embeddings-inference
  HUGGINGFACE_HUB_CACHE=/hf_cache/hub HF_TOKEN=$HF_TOKEN \
  ./target/release/text-embeddings-router \
    --model-id google/embeddinggemma-300m \
    --dtype bfloat16 \
    --port 8301 > /outputs/tei_server.log 2>&1
'

python embed_tei.py --host 127.0.0.1 --port 8301 --api both \
  --out $OUTPUT_DIR/inference_embedding_tei/tei_1gpu.json
```

The model card recommends BF16 or FP32 — do **not** force FP16.

## Single-GPU results

This path works unmodified. The ROCm backend is genuinely active — not a silent CPU
fallback, which is the classic false pass here:

```
python-backend: ROCm / HIP version: 7.2.26015-fc0010cf6a
python-backend: backend device: cuda                        <- ROCm reuses the CUDA API surface
Python backend ready in 6.754131585s
warmup_rocm{...}: finish rocm warmup for batch: 8, length: 2048   <- ROCm-specific warmup path
Starting HTTP server: 0.0.0.0:8301 / Ready
```

`warmup_rocm` is the decisive line: TEI's CPU path never emits it.

**Expected output** — served info and results (`tei_1gpu_meanpool.json`):

```
version 1.9.3 | model google/embeddinggemma-300m | model_dtype bfloat16
model_type embedding, pooling = mean | max_input_length 2048 | dim = 768
embed latency: queries 0.0465 s, documents 0.0435 s (batch)
per-request server timing: total ~19-27 ms, inference ~19-26 ms, tokenization ~0.13-0.23 ms
```

Cosine similarities — semantics are correct in both directions:

| Query | vLLM/ROCm doc | SGLang/ROCm doc | SQLite doc | Eiffel Tower doc |
|---|---|---|---|---|
| "What GPU runtimes support ROCm?" | **0.7170** | **0.6898** | 0.4175 | 0.3167 |
| "Which database is embedded and serverless?" | 0.4506 | 0.4347 | **0.7005** | 0.3714 |

Each query ranks its relevant document first by a wide margin
(`top1_doc_index_per_query : [0, 2]`).

With `--api both`, `/embed` returns 2 query + 4 document vectors at dim 768 with
`l2_norm[q0] = 1.000000` and latency around 0.022 s / 0.041 s; the OpenAI-compatible
`/v1/embeddings` returns the same dim at ~0.040 s. Run-to-run cosine values vary in the
4th decimal (bf16 nondeterminism).

## Install & run (NVIDIA H100 / Hopper)

**This path works from a prebuilt image, with no source build.** Unlike AMD, NVIDIA has a
**published Hopper container**, so there is nothing to compile — pull and run. Verified on
1×H100 80GB HBM3, driver 580.173.02, CUDA 13.0, Hopper cc(9,0), Python 3.12.3 host, TEI
**1.9.3** (same router version as the MI355X build), served on port **8090** pinned to a
single GPU.

### Image route

TEI ships GPU-architecture-specific CUDA images. For H100 (Hopper, **compute capability
9.0**) use the **Hopper** tag — the generic `cuda-*` tag targets older architectures and the
`turing`/`89` tags target Turing/Ada:

```
ghcr.io/huggingface/text-embeddings-inference:hopper-1.9   # digest sha256:e3009cd9… , 8.17 GB
```

Pull it with the corporate proxy **unset** (ghcr.io is proxy-blocked; `sudo docker` needed):

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull ghcr.io/huggingface/text-embeddings-inference:hopper-1.9
```

### Docker run (single GPU, pinned to one card)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
HF_TOKEN=$(grep -E '^HF_TOKEN=' ../dev.env | cut -d= -f2- | tr -d '"'"'"' ')   # gated model
sudo docker run -d --name tei_h100 \
  --gpus '"device=7"' \            # pin ONE physical GPU; use YOUR assigned index
  -p 8090:80 \                     # TEI listens on :80 in-container
  -v $HF_HOME:/data \
  -e HF_HOME=/data \               # so the router finds /data/hub (cache is at $HF_HOME/hub)
  -e HF_TOKEN="$HF_TOKEN" \
  ghcr.io/huggingface/text-embeddings-inference:hopper-1.9 \
  --model-id google/embeddinggemma-300m \
  --dtype float32 \                # see quirk below: Hopper image rejects bfloat16
  --port 80
```

> **`--dtype`: the Hopper (candle CUDA) backend accepts only `float16` or `float32`** — it
> hard-errors on `bfloat16` (`error: invalid value 'bfloat16' for '--dtype'`), unlike the
> ROCm source build which took `bfloat16`. The EmbeddingGemma model card recommends **BF16 or
> FP32 and warns against FP16**, so use **`float32`** to preserve numerical fidelity. The
> served `/info` confirms `model_dtype: float32`.

### Client

Client venv (the `.env_tei` convention fails on a CIFS/NFS share — pip bootstrap cannot
write symlinks there; build the venv on local or tmpfs disk instead):

```bash
python3 -m venv /tmp/tei_venv && /tmp/tei_venv/bin/pip install requests==2.34.2 python-dotenv==1.2.3
/tmp/tei_venv/bin/python embed_tei.py --host 127.0.0.1 --port 8090 --api both \
  --out /dev/shm/tei/tei_h100_1gpu.json
```

### Expected output (H100)

The candle **CUDA** backend is genuinely active — the H100 analogue of the MI355X
`warmup_rocm` line is the candle `Cuda(CudaDevice(...))` load line:

```
text_embeddings_backend_candle: Starting Gemma3 model on Cuda(CudaDevice(DeviceId(1)))
text_embeddings_backend_candle: Loading Dense module/s from path/s: ["2_Dense", "3_Dense"]
text_embeddings_router: Warming up model
text_embeddings_router::http::server: Ready
```

`embed_tei.py --api both` returns 768-dim vectors on both endpoints:

```
model_id : google/embeddinggemma-300m | dtype float32 | pooling mean | tei_version 1.9.3
--- POST /embed ---           n_query_vec 2  n_doc_vec 4  dim 768  l2_norm[q0] 1.000000
head[q0] : [-0.06663, -0.04311, -0.00338, -0.03156, 0.05392, 0.00015, -0.01757, 0.01226]
latency_s : queries 0.035 / documents 0.021
--- POST /v1/embeddings ---   n_vectors 2  dim 768  l2_norm[0] 1.000000  latency_s 0.019
```

Cosine similarities — semantics correct in both directions (related ≫ unrelated):

| Query | vLLM/ROCm doc | SGLang/ROCm doc | SQLite doc | Eiffel Tower doc |
|---|---|---|---|---|
| "What GPU runtimes support ROCm?" | **0.5747** | **0.5376** | 0.0979 | −0.0110 |
| "Which database is embedded and serverless?" | 0.2023 | 0.1902 | **0.5024** | 0.0830 |

`top1_doc_index_per_query : [0, 2]` — each query ranks its relevant document first.
(Cosine magnitudes differ from the MI355X bf16 run because this run is fp32 mean-pooled; the
*ordering* is identical, which is the sanity criterion.)

### GPU-residency proof (nvidia-smi by PID, sampled live)

```
$ sudo docker inspect -f '{{.State.Pid}}' tei_h100      -> <container-pid>
$ nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader -i 7
  7, GPU-<uuid>, 1847 MiB
$ nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep GPU-<uuid>
  GPU-<uuid>, <container-pid>, text-embeddings-router, 1838 MiB
```

The container's host PID (`text-embeddings-router`) should hold **~1.8 GB on the assigned
GPU's UUID and nowhere else** — decisive on-GPU residency, not a CPU fallback. Load
footprint is ~1.65 GB rising to ~1.85 GB after warmup. Matching by GPU UUID rather than
index is what keeps the attribution correct on a shared node.

### Multi-GPU (H100) — not covered here

Same story as MI355X: a 300M encoder does not shard and `text-embeddings-router` has **no
tensor-parallel flag**. The production pattern is N single-GPU replicas — one `docker run`
per card with a distinct `--gpus '"device=N"'` and `-p 809X:80`. Only a single-GPU replica
was measured here.

### H100 quirks (vs the MI355X route)

- **No source build.** The published `hopper-1.9` image replaces the entire MI355X Rust +
  protoc + `cargo build` step. Pull-and-run.
- **`bfloat16` unsupported by the Hopper image** → use `float32` (see dtype note above).
- **Cache mount differs.** Point `HF_HOME=/data` (cache lives at `/data/hub`); the MI355X
  route uses `HUGGINGFACE_HUB_CACHE=/hf_cache/hub`. TEI still logs a few "Downloading
  <config>.json" lines for tiny metadata files whose `.no_exist` markers are cached, then
  reads `model.safetensors` (1.2 GB) from the mount.
- **`Invalid hostname, defaulting to 0.0.0.0`** — same harmless warning as MI355X.
- **80 GB VRAM (vs 288)** is a non-issue for a 300M encoder (~1.8 GB resident).

### Platform notes (H100)

**Works unmodified from the prebuilt image.** TEI 1.9.3 serves EmbeddingGemma-300m on 1×H100
(Hopper cc 9.0) via the official `ghcr.io/huggingface/text-embeddings-inference:hopper-1.9`
container with correct 768-dim fp32 embeddings, correct semantic ranking, ~19–35 ms/request,
on its genuine candle CUDA backend resident on the assigned GPU. The only "changes" vs
MI355X are packaging (NVIDIA publishes the image, so no Rust build) and `--dtype float32`
(the Hopper image rejects bf16).

## Multi-GPU results

**A 300M encoder does not shard, and TEI has no tensor-parallel option** — `text-embeddings-router`
exposes no TP flag at all. The honest production pattern is **N independent single-GPU
replicas behind a load balancer**, one container per GPU.

This folder measured **one GPU**. A second replica on the same card, started on port
8302, served concurrently (`tei_serve2.log`, ~19.3-19.8 ms per request across 12+
consecutive successes), which proves replica-style concurrency but is **not** a two-GPU
measurement.

**Scaling to 8 GPUs is extrapolation, clearly labelled as such:** repeat the `docker run`
with a different `--device /dev/dri/renderD<N>` and `-p 830X:830X` per card. Nothing in the
ROCm path is shared between replicas, so linear scaling is expected — but it was not
measured here.

## Arguments

`embed_tei.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | TEI router host |
| `--port` | `8301` | TEI router port |
| `--model` | served model | Model name sent on the OpenAI-compatible call |
| `--api` | `embed` | `embed` (native), `openai` (`/v1/embeddings`), or `both` |
| `--queries` | 2 built-in ROCm/database queries | Query strings to embed |
| `--documents` | 4 built-in docs | Documents to embed and score |
| `--query_prompt_name` | — | TEI prompt template for queries |
| `--document_prompt_name` | — | TEI prompt template for documents |
| `--normalize` / `--no_normalize` | normalize | Ask TEI to L2-normalize |
| `--truncate` | off | Truncate inputs past `max_input_length` |
| `--timeout` | HTTP timeout (seconds) | |
| `--health_retries` | — | Retries while waiting for `/health` |
| `--reference` | `None` | Compare against a saved reference JSON |
| `--out` | `None` | Write a JSON artifact of the full result |

Key router flags: `--model-id`, `--dtype bfloat16`, `--port`, `--max-batch-tokens`,
`--max-client-batch-size`, `--auto-truncate`, `--pooling`.

## Output

A JSON artifact with the served `info` block, the query/document texts, `dim`, per-batch
latency, the full cosine matrix, and a truncated view of the raw vectors — e.g.
`$OUTPUT_DIR/inference_embedding_tei/tei_1gpu_meanpool.json`.

## Hardware support & evidence

| | Status |
|---|---|
| NVIDIA (generic) | Official — `ghcr.io/huggingface/text-embeddings-inference:cuda-1.9`. Architecture-specific tags apply; see below for Hopper. |
| **NVIDIA H100 (Hopper, cc 9.0), CUDA 13.0** | **Verified — works.** Requires the `hopper-1.9` image and `--dtype float32`; see "Install & run (NVIDIA H100 / Hopper)" above. |
| AMD ROCm | Upstream: **experimental**, documented on MI200/MI300 only, source build required. |
| **AMD MI355X (gfx950), ROCm 7.2.4** | **Verified — works.** Router built from source in 3m 44s with no patches; `warmup_rocm` + `ROCm / HIP version: 7.2.26015` confirm the ROCm backend; correct 768-dim embeddings at ~20 ms/request. |

Upstream validates nothing newer than MI300, so gfx950 support is **de facto, not
certified** — treat it as proven-for-this-model rather than production-blessed.

## Notes & quirks

- **No AMD container tag exists.** The Rust source build is the only ROCm route; budget for
  Rust + protoc, which a ROCm PyTorch base image does not ship.
- **`backend device: cuda` on AMD is correct**, not a misconfiguration — ROCm reuses the
  `torch.cuda` API surface. Confirm ROCm via the `warmup_rocm` line instead.
- **`Invalid hostname, defaulting to 0.0.0.0`** — TEI rejects the container hostname; harmless.
- **`Address already in use (os error 98)`** if a previous router still holds 8301; TEI does
  not fail over to another port.
- A one-off `Server error: transport error` can appear when the Python backend restarts
  under load; the router recovers on the next request.
- Startup echoes a partially-masked `HF_TOKEN` into the log — keep logs off git.

## Summary

**Works with changes (build-from-source required).** TEI 1.9.3 serves EmbeddingGemma-300m
on MI355X/gfx950 with correct 768-dim bf16 embeddings at ~20 ms per request, using its
genuine ROCm backend. The "changes" are entirely packaging: no AMD image is published, so
you must build the Rust router yourself inside a ROCm PyTorch container. Once built, no
source modification is needed for gfx950.

Not covered here: multi-GPU replica scaling (single card measured), the `/rerank` path with
a supported cross-encoder, ONNX/candle backends, and throughput benchmarking beyond
single-request latency.
