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

Built against **TEI 1.9.3** on **AMD MI355X (gfx950, ROCm 7.2)** via a source build and on
**NVIDIA H100 80GB (Hopper cc 9.0, CUDA 13)** via the prebuilt `hopper-1.9` image. Upstream
documents ROCm support as *experimental* and tested only on MI200/MI300, so gfx950 is beyond
its documented matrix.

## Scope — why there is no LLM or reranker TEI folder

| Missing folder | Why |
|---|---|
| `inference/tei/llm` | **TEI is not a generative server.** Use `../../vllm/llm/`, `../../sglang/llm/`, `../../llamacpp/llm/`, or `../../transformers/llm/`. |
| `inference/tei/reranker` | TEI's `/rerank` targets **sequence-classification** cross-encoders (XLM-RoBERTa, GTE, ModernBERT). `Qwen/Qwen3-Reranker-0.6B` is **decoder-only yes/no** scoring and is unsupported upstream. Use [`../../llamacpp/reranker/`](../../llamacpp/reranker/) or [`../../vllm/reranker/`](../../vllm/reranker/). |

## Install & run — AMD / ROCm (source build)

There is **no published AMD/ROCm container tag** for TEI — `ghcr.io/huggingface/text-embeddings-inference`
publishes CUDA tags only. The AMD route is a **source build of the Rust router against a ROCm
PyTorch base**.

The build below uses `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`, which already
carries ROCm 7.2 PyTorch for gfx950. Any ROCm PyTorch image with a matching HIP works;
nothing SGLang-specific is used.

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # server logs and result artifacts

# 1. Start a ROCm container with ONE GPU. Find your render node with
#    `ls /dev/dri/` and `rocm-smi --showbus`.
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

The build produces `target/release/text-embeddings-router`. **No source patches are needed
for gfx950.**

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

### Run — single GPU

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

### Expected output — AMD / ROCm

Check first that the ROCm backend is genuinely active, not a silent CPU fallback (the
classic false pass here):

```
python-backend: ROCm / HIP version: 7.2.26015-fc0010cf6a
python-backend: backend device: cuda                        <- ROCm reuses the CUDA API surface
Python backend ready in 6.754131585s
warmup_rocm{...}: finish rocm warmup for batch: 8, length: 2048   <- ROCm-specific warmup path
Starting HTTP server: 0.0.0.0:8301 / Ready
```

`warmup_rocm` is the decisive line: TEI's CPU path never emits it.

Then the client output:

```
version 1.9.3 | model google/embeddinggemma-300m | model_dtype bfloat16
model_type embedding, pooling = mean | max_input_length 2048 | dim = 768
```

Each query should rank its relevant document first (`top1_doc_index_per_query : [0, 2]`) —
that ordering, not the cosine magnitudes, is the sanity criterion. With `--api both`,
`/embed` returns 2 query + 4 document vectors at dim 768 with `l2_norm[q0] = 1.000000`; the
OpenAI-compatible `/v1/embeddings` returns the same dim. Run-to-run cosine values vary in the
4th decimal (bf16 nondeterminism).

## Install & run — NVIDIA / CUDA (prebuilt image)

**No source build.** NVIDIA has a published Hopper container — pull and run.

### Image

TEI ships GPU-architecture-specific CUDA images. For H100 (Hopper, **compute capability
9.0**) use the **Hopper** tag — the generic `cuda-*` tag targets older architectures and the
`turing`/`89` tags target Turing/Ada:

```
ghcr.io/huggingface/text-embeddings-inference:hopper-1.9
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
sudo docker run -d --name tei_cuda \
  --gpus '"device=0"' \            # pin ONE GPU; use your assigned index
  -p 8090:80 \                     # TEI listens on :80 in-container
  -v $HF_HOME:/data \
  -e HF_HOME=/data \               # so the router finds /data/hub (cache is at $HF_HOME/hub)
  -e HF_TOKEN="$HF_TOKEN" \
  ghcr.io/huggingface/text-embeddings-inference:hopper-1.9 \
  --model-id google/embeddinggemma-300m \
  --dtype float32 \                # see the dtype note: the Hopper image rejects bfloat16
  --port 80
```

> **`--dtype`: the Hopper (candle CUDA) backend accepts only `float16` or `float32`** — it
> hard-errors on `bfloat16` (`error: invalid value 'bfloat16' for '--dtype'`), unlike the
> ROCm source build which takes `bfloat16`. The EmbeddingGemma model card recommends **BF16 or
> FP32 and warns against FP16**, so use **`float32`** to preserve numerical fidelity. The
> served `/info` confirms `model_dtype: float32`.

### Client

```bash
python3 -m venv .env_tei && .env_tei/bin/pip install -r ../requirements.txt
.env_tei/bin/python embed_tei.py --host 127.0.0.1 --port 8090 --api both \
  --out $OUTPUT_DIR/inference_embedding_tei/tei_cuda_1gpu.json
```

> The venv convention fails on a CIFS/NFS share — pip's bootstrap cannot write symlinks
> there. Build the venv on local or tmpfs disk instead.

### Expected output — NVIDIA / CUDA

The candle **CUDA** backend is genuinely active — the analogue of ROCm's
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
--- POST /v1/embeddings ---   n_vectors 2  dim 768  l2_norm[0] 1.000000
```

`top1_doc_index_per_query : [0, 2]` — each query ranks its relevant document first. Cosine
magnitudes differ from the ROCm bf16 run because this one is fp32 mean-pooled; the *ordering*
is the sanity criterion.

### Confirm GPU residency

```bash
sudo docker inspect -f '{{.State.Pid}}' tei_cuda            # -> <container-pid>
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader | grep <the GPU's uuid>
#   GPU-<uuid>, <container-pid>, text-embeddings-router, 1838 MiB
```

The container's host PID (`text-embeddings-router`) should hold VRAM on the assigned GPU's
UUID and nowhere else. Match by GPU UUID rather than index — that keeps the attribution
correct on a shared node.

### Deviations from the ROCm route

- **No source build.** The published `hopper-1.9` image replaces the whole Rust + protoc +
  `cargo build` step.
- **`bfloat16` unsupported** → use `float32` (see the dtype note above).
- **Cache mount differs.** Point `HF_HOME=/data` (cache lives at `/data/hub`); the ROCm
  route uses `HUGGINGFACE_HUB_CACHE=/hf_cache/hub`. TEI still logs a few
  `Downloading <config>.json` lines for tiny metadata files whose `.no_exist` markers are
  cached, then reads `model.safetensors` (1.2 GB) from the mount.

## Multi-GPU

`text-embeddings-router` has **no tensor-parallel flag**. The pattern is **N independent
single-GPU replicas behind a load balancer**, one container per GPU: repeat the run with a
different `--device /dev/dri/renderD<N>` (ROCm) or `--gpus '"device=N"'` (CUDA) and a distinct
host port. Multiple replicas can also share one card.

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

## Hardware support

| | Status |
|---|---|
| NVIDIA (generic) | Official — `ghcr.io/huggingface/text-embeddings-inference:cuda-1.9`. Architecture-specific tags apply. |
| **NVIDIA H100 (Hopper, cc 9.0), CUDA 13** | **Works.** Requires the `hopper-1.9` image and `--dtype float32`. |
| AMD ROCm | Upstream: **experimental**, documented on MI200/MI300 only, source build required. |
| **AMD MI355X (gfx950), ROCm 7.2** | **Works.** Router builds from source with no patches; `warmup_rocm` confirms the ROCm backend. |

## Notes & quirks

- **No AMD container tag exists.** The Rust source build is the only ROCm route; budget for
  Rust + protoc, which a ROCm PyTorch base image does not ship.
- **`backend device: cuda` on AMD is correct**, not a misconfiguration — ROCm reuses the
  `torch.cuda` API surface. Confirm ROCm via the `warmup_rocm` line instead.
- **`Invalid hostname, defaulting to 0.0.0.0`** — TEI rejects the container hostname; harmless,
  and it appears on both backends.
- **`Address already in use (os error 98)`** if a previous router still holds the port; TEI
  does not fail over to another port.
- A one-off `Server error: transport error` can appear when the Python backend restarts
  under load; the router recovers on the next request.
- Startup echoes a partially-masked `HF_TOKEN` into the log — keep logs off git.

## Summary

- **AMD/ROCm:** no published image — build the Rust router inside a ROCm PyTorch container.
  No source modification needed for gfx950. Serves `bfloat16`.
- **NVIDIA/CUDA:** works unmodified from the prebuilt `hopper-1.9` image. Serve
  `--dtype float32` — the image rejects `bfloat16`.
