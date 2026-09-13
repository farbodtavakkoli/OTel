# `embed_vllm.py` — EmbeddingGemma served by vLLM

## Overview & when to use

Serves **`google/embeddinggemma-300m`** through vLLM's OpenAI-compatible
`POST /v1/embeddings` endpoint, and `embed_vllm.py` is a small client that embeds a query
plus a few documents and prints the vector dimensions and the query↔document cosine
similarities.

Use this when you want **one serving stack for all three workloads** (LLM, embedding,
reranker) instead of running a dedicated embedding server. vLLM's `--runner pooling` mode is
what exposes `/v1/embeddings`. If embeddings are *all* you serve, a dedicated stack (TEI,
sentence-transformers) is lighter — see `training/embedding/sentence_transformers/`.

## Install

**There is no ROCm vLLM wheel.** PyPI `vllm` publishes CUDA-only wheels that hard-depend on
CUDA torch, and `repo.radeon.com` publishes none, so on ROCm the pip/venv route is
unavailable (a gfx950 source build works but takes hours) and the route that works is a
**container**. Known-working image:
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`, which ships a
working ROCm vLLM:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and run artifacts

docker run -d --name vllm_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  --network host \
  -v "$PWD":/workspace/repo -v "$HF_HOME":"$HF_HOME" \
  -e HF_HOME="$HF_HOME" -w /workspace/repo \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

Two things to note in that command:

- The image has **no `render` group**, so the canonical `--group-add render` fails with
  `unable to find group render`. Pass the host's **numeric** GIDs instead (typically
  `video`=44, `render`=993) — that is what the `getent` substitutions above do.
- Pinning GPUs by **render node** (`renderD128`, `renderD136` — match them to your own
  cards with `ls /dev/dri`) rather than by `HIP_VISIBLE_DEVICES` means the container sees
  exactly those two `gfx950` devices and cannot touch any other card on the box.

Versions inside that image:

| Component | Version |
|---|---|
| vLLM | `0.20.2rc1.dev253+g1ff9d3353` |
| torch | `2.9.1.dev20251204+rocm7.0.2.git351ff442` |
| `torch.version.hip` | `7.0.51831-7c9236b16` |
| transformers | `5.14.1` |
| ROCm (container) | 7.0.2 |
| ROCm (host) | 7.2.4 |
| GPU | AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, 309 GB |

The client itself needs only `python-dotenv` (everything else it uses is stdlib), so it
runs on the host outside the container:

```bash
pip install -r ../requirements.txt
```

## Environment & secrets

`dev.env` is symlinked to the repo-root file and supplies the HF token for the **gated**
`google/embeddinggemma-300m` repo:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded by `load_dotenv("dev.env")` in the client, and exported into the serving shell.
Never echo it. Model weights are kept off `/` by pointing `HF_HOME` at a data volume.

GPU pinning for the two cards in use:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides *every* GPU.

## Serve

Single GPU:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0
vllm serve google/embeddinggemma-300m \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name embeddinggemma
```

## Client / smoke command

```bash
python embed_vllm.py --port 8001 --model embeddinggemma
```

Raw equivalent:

```bash
curl -s http://localhost:8001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma","input":["vLLM supports AMD ROCm.","SQLite is an embedded database."]}'
```

## Single-GPU behaviour on gfx950

**This works, unmodified.** `vllm serve ... --runner pooling` comes up on one MI355X and
returns correct 768-dimension embeddings — no flags beyond the documented ones, no code
changes, no ROCm-specific workarounds.

Semantic check — query `"Which inference engines support AMD ROCm?"` against a ROCm document
vs a PostgreSQL one:

```
query vs ROCm-doc      : 0.6975
query vs postgres-doc  : 0.2216
ordering correct       : True
```

## Multi-GPU on gfx950 — TP is not available

**TP=2 is architecturally impossible for this model, and that is a property of
EmbeddingGemma, not of ROCm.** Attempting it fails fast at config validation:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
  Value error, Total number of attention heads (3) must be divisible by
  tensor parallel size (2).
```

EmbeddingGemma-300m has **3 attention heads**, and tensor parallelism shards the head
dimension, so only TP=1 or TP=3 are legal — TP=2 can never work, on either vendor.

**The production pattern for a model this size is horizontal replication — one
independent single-GPU server per GPU, behind a load balancer:**

```bash
# replica A — first GPU
HIP_VISIBLE_DEVICES=0 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8001 --served-model-name embeddinggemma &

# replica B — second GPU
HIP_VISIBLE_DEVICES=1 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8011 --served-model-name embeddinggemma &
```

Both come up and serve concurrently, each on its own card, and both answer the same request
with cosines agreeing to ~1e-3 (ordinary reduction-order variation, not a correctness
problem). Replicate rather than shard.

## H100 (NVIDIA)

**This works unmodified.** vLLM `0.27.1` (pip / CUDA 13.0) serves
`google/embeddinggemma-300m` on one H100 80GB with the documented `--runner pooling`
command — correct **768-dim** vectors and correct semantic ordering, no code changes and no
ROCm workarounds.

### Install (pip route — no container needed)

On **NVIDIA the pip wheel is native**. One shared venv at the stack root
(`inference/vllm/.env_vllm`) serves all three leaves:

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy      # -> the current CUDA 13 build (plain PyPI)
pip install vllm             # -> vllm 0.27.1; torch stays 2.13.0+cu130
pip install python-dotenv
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA) |
| torch | `2.13.0+cu130` |
| transformers | `5.15.1` |
| CUDA | 13.0, H100 80GB HBM3 |

### Serve (the H100 command — identical to ROCm minus the HIP var)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=<free-gpu>
vllm serve google/embeddinggemma-300m \
  --runner pooling \
  --host 0.0.0.0 --port 8500 \
  --served-model-name embeddinggemma
```

```bash
python embed_vllm.py --port 8500 --model embeddinggemma
```

### Expected output (H100)

```
[core.py:414] Resolved pooling config: pooling_type=MEAN(...), supported_tasks=('embed', 'token_embed')
[api_server.py:678] Supported tasks: ['token_embed', 'embed']
Route: /v1/embeddings, Methods: POST
INFO:     Application startup complete.
```

```
endpoint    : http://localhost:8500/v1/embeddings
model       : embeddinggemma
vectors     : 3
dimensions  : 768
query       : Which inference engines support AMD ROCm?
  first 5 dims: [-0.09573, -0.08041, 0.04351, -0.00888, 0.03014]

  cos=+0.7106  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
  cos=+0.2008  PostgreSQL is a relational database management system.
```

**Correct 768-dim vectors, correct ordering** — the ROCm document must outscore the
PostgreSQL one.

### GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

Only the selected GPU is touched.

### Quirks (H100)

- A **benign** `Operation not permitted` warning can appear while setting blob permissions
  on a network-mounted (CIFS/NFS) HF cache — non-fatal, the cached weights still load.
- vLLM suggests `pip install orjson` "to make the v1/embeddings API fast" — optional; the
  request still returns correct vectors without it.
- No AITER, no `quark_online_quant`, no GELU-approximation warning — those are all ROCm-only.

### Multi-GPU (architecturally capped)

Same as ROCm: **TP=2 is impossible for this model** (3 attention heads, not divisible by 2)
— an identical, correct refusal on NVIDIA. The right multi-GPU pattern is **N independent
single-GPU replicas** (one `vllm serve ... --port 85xx` per GPU).

## Arguments / flags

Serve-side flags:

| Flag | Value | Meaning |
|---|---|---|
| `--runner pooling` | required | Switches vLLM from generation to pooling/embedding mode; this is what exposes `/v1/embeddings` |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8001` | Benchmark-layout port for the embedding server |
| `--served-model-name` | `embeddinggemma` | Short alias clients pass as `model`, decoupling the API from the HF repo id |
| `--tensor-parallel-size` | *(not usable)* | Rejected — 3 attention heads not divisible by 2 |
| `--gpu-memory-utilization` | default `0.9` | Lower it (e.g. `0.15`) if you co-locate several servers on one GPU; vLLM preallocates its KV/activation pool up to this fraction |
| `--max-model-len` | default `2048` | Model default; raise only if you embed longer documents |

`embed_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8001` | Server port |
| `--model` | `embeddinggemma` | Served model name (must match `--served-model-name`) |
| `--query` | ROCm question | Query text; cosine is reported against each document |
| `--document` | 2 built-ins | Document text; repeat the flag for several documents |
| `--encoding_format` | `float` | `float` or `base64` |
| `--timeout` | `60.0` | HTTP timeout in seconds |
| `--show_dims` | `5` | How many leading vector components to print |

## Output

The client prints the endpoint, resolved model name, vector count, dimensionality, token
usage, the leading components of the query vector, and one cosine line per document:

**Expected output** for `python embed_vllm.py --port 8001`:

```
endpoint    : http://localhost:8001/v1/embeddings
model       : embeddinggemma
vectors     : 3
dimensions  : 768
usage       : {'prompt_tokens': 45, 'total_tokens': 45, 'completion_tokens': 0, 'prompt_tokens_details': None}

query       : Which inference engines support AMD ROCm?
  first 5 dims: [-0.09578, -0.08054, 0.04375, -0.00833, 0.03036]

  cos=+0.7105  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
  cos=+0.2011  PostgreSQL is a relational database management system.
```

Redirect server logs to a data volume, e.g. `$OUTPUT_DIR/inference_embedding_vllm/`.
Nothing large is written into the repo.

## Hardware support

- **AMD, working.** 1× and 2× (as replicas) AMD Instinct MI355X (`gfx950`,
  288 GB), host ROCm 7.2.4, container ROCm 7.0.2, vLLM `0.20.2rc1.dev253`, torch
  `2.9.1.dev+rocm7.0.2`. Correct 768-dim embeddings with correct semantic ordering.
- **NVIDIA, working** via the pip wheel — see the H100 section above. The same
  `vllm serve` command also applies with `vllm/vllm-openai:latest`.
- On AMD the *install* route is container-only — there is no ROCm vLLM wheel.

## Notes & quirks

- **`--runner pooling` is mandatory.** Without it vLLM loads the model as a generative
  decoder and `/v1/embeddings` is not routed.
- **The pooling server also exposes `/score`, `/rerank` and `/v2/rerank`.** They are
  registered on the same process; that is normal for a pooling runner and does not mean
  the embedding model is a reranker.
- **AITER JIT-builds on first launch.** The first `vllm serve` on this image prints
  `[aiter] start build [module_aiter_core]` and stalls while compiling; later launches reuse
  the build cache.
- **The `quark_online_quant` plugin fails to import** on every launch with a traceback.
  It is non-fatal and unrelated to this workload; the server starts normally. Do not
  chase it.
- **A ROCm-specific GELU warning is expected:** `[ROCm] PyTorch's native GELU with tanh
  approximation is unstable with torch.compile ... fallback to 'none' approximation.`
  vLLM handles it automatically and embedding quality is unaffected.
- **Default `--gpu-memory-utilization 0.9` inflates the VRAM figure in `rocm-smi`.** vLLM
  preallocates its pool, so a small model can show hundreds of GB "used". Use
  `--gpu-memory-utilization` to bound it when co-locating servers.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs
  rather than none.
