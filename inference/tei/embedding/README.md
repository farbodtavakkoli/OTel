# `inference/tei/embedding` — HF Text Embeddings Inference (EmbeddingGemma-300m)

Serves `google/embeddinggemma-300m` behind the TEI router and hits it with `embed_tei.py`,
which embeds queries plus documents and prints the real vector dimension and cosine scores.

Pick this folder when embeddings are the only workload on the host and you want a small,
dedicated server rather than a general LLM engine: `POST /embed`, an OpenAI-compatible
`POST /v1/embeddings`, batching and Prometheus metrics.

**Hardware:** TEI 1.9.3 on AMD MI355X (gfx950, ROCm 7.2) via a source build, and on NVIDIA
H100 80GB (Hopper cc 9.0, CUDA 13) via the prebuilt `hopper-1.9` image. Upstream documents
ROCm support as experimental and tested on MI200/MI300 only, so gfx950 is beyond its
documented matrix. Stack overview and scope: [`../README.md`](../README.md).

## Files

- `embed_tei.py` — embedding smoke client; exercises `/embed` and `/v1/embeddings`, prints
  the cosine matrix and top-1 document per query, optionally diffs a Transformers reference.

## Setup

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs     # server logs and result artifacts
export HF_TOKEN=...                    # from dev.env; EmbeddingGemma is gated
```

Create the shared `.env_tei` venv at the stack root per [`../README.md`](../README.md). Do
not install the ROCm python-backend requirements into it — they are container-only and pin
`python_version < "3.13"`.

Mount a pre-seeded HF cache if your host sits behind a proxy that cannot reach the HF CDN.
TEI echoes its `Args { ... }` line at startup including a partially-masked token, so write
server logs under `$OUTPUT_DIR` and keep them out of git.

### NVIDIA / CUDA — prebuilt image

Use the **Hopper** tag for cc 9.0; the generic `cuda-*` tag is the wrong architecture. ghcr.io
is often proxy-blocked:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull ghcr.io/huggingface/text-embeddings-inference:hopper-1.9
```

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
HF_TOKEN=$(grep -E '^HF_TOKEN=' ../dev.env | cut -d= -f2- | tr -d '"'"'"' ')   # gated model
sudo docker run -d --name tei_cuda \
  --gpus '"device=0"' \            # pin ONE GPU; use your assigned index
  -p 8090:80 \                     # TEI listens on :80 in-container
  -v $HF_HOME:/data \
  -e HF_HOME=/data \               # so the router finds /data/hub
  -e HF_TOKEN="$HF_TOKEN" \
  ghcr.io/huggingface/text-embeddings-inference:hopper-1.9 \
  --model-id google/embeddinggemma-300m \
  --dtype float32 \                # the Hopper backend rejects bfloat16
  --port 80
```

The candle CUDA backend accepts only `float16` or `float32` and hard-errors on `bfloat16`.
The EmbeddingGemma model card recommends BF16 or FP32 and warns against FP16, so use
`float32` here; `/info` confirms `model_dtype: float32`.

### AMD / ROCm — source build

There is no published AMD/ROCm container tag, so build the Rust router against a ROCm PyTorch
base. The example uses `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819`, which carries
ROCm 7.2 PyTorch for gfx950; any ROCm PyTorch image with a matching HIP works, and nothing
SGLang-specific is used.

```bash
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

The build produces `target/release/text-embeddings-router`. No source patches are needed for
gfx950.

## Run

### NVIDIA / CUDA

```bash
../.env_tei/bin/python embed_tei.py --host 127.0.0.1 --port 8090 --api both \
  --out $OUTPUT_DIR/inference_embedding_tei/tei_cuda_1gpu.json
```

### AMD / ROCm

```bash
docker exec -d tei_build bash -lc '
  cd /workspace/text-embeddings-inference
  HUGGINGFACE_HUB_CACHE=/hf_cache/hub HF_TOKEN=$HF_TOKEN \
  ./target/release/text-embeddings-router \
    --model-id google/embeddinggemma-300m \
    --dtype bfloat16 \
    --port 8301 > /outputs/tei_server.log 2>&1
'

../.env_tei/bin/python embed_tei.py --host 127.0.0.1 --port 8301 --api both \
  --out $OUTPUT_DIR/inference_embedding_tei/tei_1gpu.json
```

### Multi-GPU

`text-embeddings-router` has no tensor-parallel flag. Run N independent single-GPU replicas
behind a load balancer: repeat the run with a different `--device /dev/dri/renderD<N>` (ROCm)
or `--gpus '"device=N"'` (CUDA) and a distinct host port. Several replicas can also share one
card.

## Output

```text
model_id : google/embeddinggemma-300m | dtype float32 | pooling mean | tei_version 1.9.3
--- POST /embed ---           n_query_vec 2  n_doc_vec 4  dim 768  l2_norm[q0] 1.000000
head[q0] : [-0.06663, -0.04311, -0.00338, -0.03156, 0.05392, 0.00015, -0.01757, 0.01226]
--- POST /v1/embeddings ---   n_vectors 2  dim 768  l2_norm[0] 1.000000
```

`top1_doc_index_per_query : [0, 2]` — each query ranks its relevant document first. That
*ordering* is the sanity criterion, not the cosine magnitudes: they differ between the fp32
CUDA run and the bf16 ROCm run, and bf16 varies in the 4th decimal run to run.

Confirm the GPU backend is genuinely active rather than a silent CPU fallback. On ROCm the
decisive line is `warmup_rocm`, which TEI's CPU path never emits; on CUDA it is the candle
device line:

```text
# ROCm
python-backend: backend device: cuda                        <- ROCm reuses the CUDA API surface
warmup_rocm{...}: finish rocm warmup for batch: 8, length: 2048

# CUDA
text_embeddings_backend_candle: Starting Gemma3 model on Cuda(CudaDevice(DeviceId(1)))
text_embeddings_router::http::server: Ready
```

On CUDA, also check the router holds VRAM on the assigned card — match by GPU UUID rather
than index so attribution stays correct on a shared node:

```bash
sudo docker inspect -f '{{.State.Pid}}' tei_cuda            # -> <container-pid>
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader | grep <the GPU's uuid>
#   GPU-<uuid>, <container-pid>, text-embeddings-router, 1838 MiB
```

`--out` writes a JSON artifact with the served `info` block, the query/document texts, `dim`,
per-batch latency, the full cosine matrix and a truncated view of the raw vectors.

## Arguments

`embed_tei.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | TEI router host |
| `--port` | `8301` | TEI router port |
| `--model` | `google/embeddinggemma-300m` | Model name sent on the OpenAI-compatible call |
| `--api` | `both` | `embed` (native), `openai` (`/v1/embeddings`), or `both` |
| `--queries` | 2 built-in ROCm/database queries | Query strings to embed |
| `--documents` | 4 built-in docs | Documents to embed and score |
| `--query_prompt_name` | `query` | sentence-transformers prompt applied to queries; empty string disables |
| `--document_prompt_name` | `document` | sentence-transformers prompt applied to documents; empty string disables |
| `--normalize` / `--no_normalize` | normalize | Ask TEI to L2-normalize |
| `--truncate` | off | Let the router truncate inputs past `max_input_length` |
| `--timeout` | `300` | HTTP timeout (s) |
| `--health_retries` | `60` | `/health` polls before giving up |
| `--reference` | `None` | Compare against a saved Transformers reference JSON |
| `--out` | `None` | Write a JSON artifact of the full result |

Key router flags: `--model-id`, `--dtype`, `--port`, `--max-batch-tokens`,
`--max-client-batch-size`, `--auto-truncate`, `--pooling`.

## Notes

- `Invalid hostname, defaulting to 0.0.0.0` appears on both backends and is harmless.
- `Address already in use (os error 98)` means a previous router still holds the port — TEI
  does not fail over to another one.
- A one-off `Server error: transport error` can appear when the Python backend restarts under
  load; the router recovers on the next request.
- On the CUDA route the cache mount differs from ROCm's `HUGGINGFACE_HUB_CACHE=/hf_cache/hub`:
  set `HF_HOME=/data` and the cache is read from `/data/hub`. TEI still logs a few
  `Downloading <config>.json` lines for tiny metadata files before reading
  `model.safetensors` (1.2 GB) from the mount.
