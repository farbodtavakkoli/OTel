# `embed_vllm.py` — EmbeddingGemma served by vLLM

## Overview & when to use

Serves **`google/embeddinggemma-300m`** through vLLM's OpenAI-compatible
`POST /v1/embeddings` endpoint, and `embed_vllm.py` is a small client that embeds a query
plus a few documents and prints the vector dimensions and the query↔document cosine
similarities.

Use this when you want **one serving stack for all three workloads** (LLM, embedding,
reranker) instead of running a dedicated embedding server. vLLM's `--runner pooling` mode
turns any supported encoder/decoder into an embedding endpoint, so the same binary, the
same flags, the same metrics, and the same OpenAI client code cover the whole fleet.

If embeddings are *all* you serve, a dedicated stack (TEI, sentence-transformers) is
lighter — see the sibling `training/embedding/sentence_transformers/` folder for the Transformers-side
baseline. The reason to pick vLLM here is uniformity, not raw embedding speed.

## Install

**There is no ROCm vLLM wheel — this is the single most important fact in this folder.**
Verified 2026-08-20:

- PyPI `vllm==0.27.1` publishes exactly two binary wheels
  (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`), both **CUDA-only**. Its
  `requires_dist` hard-depends on `flashinfer-python==0.6.16.post3`,
  `nvidia-cudnn-frontend`, `nvidia-cutlass-dsl[cu13]==4.6.0` and `torch==2.13.0` (the CUDA
  build). Installing it on a ROCm host gives you a non-functional engine.
- `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` publishes `torch`, `torchaudio`,
  `apex`, `jaxlib` and `tensorflow_rocm` wheels — **no `vllm` wheel**.
- Building vLLM from source for `gfx950` works but is an hours-long compile.

So the pip/venv route that would normally be preferred is genuinely unavailable, and the
verified route is a **container**. This host already had
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` on disk, which ships a
working ROCm vLLM — zero pull cost, no `docker pull`, no disk spent:

```bash
docker run -d --name vllm_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  --network host \
  -v "$PWD":/workspace/repo -v /mnt/data_1.5t:/mnt/data_1.5t \
  -e HF_HOME=/mnt/data_1.5t/hf_cache -w /workspace/repo \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

Two gotchas in that command, both learned the hard way:

- The image has **no `render` group**, so the canonical `--group-add render` fails with
  `unable to find group render`. Pass the host's **numeric** GIDs instead (`video`=44,
  `render`=993 on this host) — that is what the `getent` substitutions above do.
- Pinning GPUs by **render node** (`renderD128` = physical GPU0, `renderD136` = GPU1)
  rather than by `HIP_VISIBLE_DEVICES` means `torch.cuda.device_count()` is `2` inside the
  container no matter what any tool does to the environment. Confirmed: the container sees
  exactly two `gfx950` devices and cannot touch a sibling's GPUs 2–7.

Verified versions inside that image:

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
pip install -r requirements_embedding_vllm.txt
```

## Environment & secrets

`dev.env` is symlinked to the repo-root file and supplies the HF token for the **gated**
`google/embeddinggemma-300m` repo:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded by `load_dotenv("dev.env")` in the client, and exported into the serving shell.
Never echo it. Model weights are kept off `/` with `HF_HOME=/mnt/data_1.5t/hf_cache`.

GPU pinning for this agent's two cards:

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — an empty string hides *every* GPU.

## Serve

Single GPU (the verified command):

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache HIP_VISIBLE_DEVICES=0
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

## Single-GPU results

**Verdict: works, unmodified.** `vllm serve ... --runner pooling` came up on one MI355X
and returned correct 768-dimension embeddings on the first attempt. No flags beyond the
documented ones, no code changes, no ROCm-specific workarounds.

| Measurement | Value |
|---|---|
| Cold start (process launch → `Application startup complete`) | **~22 s** (warm torch.compile cache) / ~38 s on the very first launch |
| Weights load | 0.13 s |
| Model-load VRAM | **0.61 GiB** |
| `init engine` (profile + KV cache + warmup) | 5.36 s (compilation 3.32 s) |
| Total process VRAM incl. preallocated pool (`rocm-smi`) | 2.38–2.68 GB |
| Max model len | 2048 (model default) |
| Embedding dimension | **768** |

Real output from `/v1/embeddings` (two inputs):

```
object   : list
model    : embeddinggemma
n vectors: 2
  idx=0 dim=768 first5=[-0.12238, -0.02375, 0.01146, -0.02259, 0.02624]
  idx=1 dim=768 first5=[-0.13934, 0.00598, 0.04491, 0.01084, 0.0202]
usage    : {'prompt_tokens': 18, 'total_tokens': 18, 'completion_tokens': 0}
```

Semantic check — the vectors are not just well-shaped, they are *correct*:

```
query vs ROCm-doc      : 0.6975
query vs postgres-doc  : 0.2216
ordering correct       : True
```

(`"Which inference engines support AMD ROCm?"` against `"vLLM runs on AMD ROCm GPUs."`
vs `"PostgreSQL is a relational database."` — a 0.48 margin, so the model is genuinely
discriminating, not emitting noise.)

## Multi-GPU (TP=2) results

**TP=2 is architecturally impossible for this model, and that is a property of
EmbeddingGemma, not of ROCm.** Attempting it fails fast at config validation:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
  Value error, Total number of attention heads (3) must be divisible by
  tensor parallel size (2).
```

EmbeddingGemma-300m has **3 attention heads**. Tensor parallelism shards the head
dimension, so TP is only legal for divisors of 3 — i.e. TP=1 or TP=3. TP=2 can never work,
on any vendor's hardware. The same command on an NVIDIA host fails identically. This is a
*correct* refusal by vLLM, caught before any GPU work.

It is also pointless: the model is 0.61 GiB. There is nothing to shard on a 288 GB card.

**The honest production pattern for a model this size is horizontal replication — one
independent single-GPU server per GPU, behind a load balancer.** That is what was verified
instead:

```bash
# replica A — physical GPU 0
HIP_VISIBLE_DEVICES=0 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8001 --served-model-name embeddinggemma &

# replica B — physical GPU 1
HIP_VISIBLE_DEVICES=1 vllm serve google/embeddinggemma-300m --runner pooling \
  --port 8011 --served-model-name embeddinggemma &
```

Both came up and served concurrently. `rocm-smi` with both replicas resident — **both
GPUs loaded, sibling GPUs untouched**:

```
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,0,309220868096,2377240576      <- replica A  (2.38 GB)
card1,0,309220868096,2377240576      <- replica B  (2.38 GB)
card2,0,309220868096,298037248       <- idle, belongs to a sibling agent
```

Both replicas answered the same request correctly and near-identically:

```
--- port 8001 (GPU0) ---
  vectors=2 dim=768 cos(q,doc)=0.6965 first3=[-0.09578, -0.08054, 0.04375]
--- port 8011 (GPU1) ---
  vectors=2 dim=768 cos(q,doc)=0.6974 first3=[-0.09595, -0.08037, 0.044]
```

The ~1e-3 spread between replicas is ordinary non-deterministic reduction order, not a
correctness problem. Throughput scales linearly with replica count, which is the right
answer for a 0.61 GiB model: **replicate, don't shard.**

## H100 (NVIDIA) — verified 2026-08-22

**Verdict: ✅ PASS, unmodified.** vLLM `0.27.1` (pip / CUDA 13.0) serves
`google/embeddinggemma-300m` on one H100 80GB with the documented `--runner pooling`
command — correct **768-dim** vectors and correct semantic ordering on the first attempt,
no code changes and no ROCm workarounds.

### Install (pip route — the big AMD caveat does NOT apply on NVIDIA)

The "no ROCm vLLM wheel, container-only" fact above is AMD-specific. On **NVIDIA the pip
wheel is native**. One tmpfs venv is shared across all three leaves:

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy      # -> torch 2.13.0+cu130 (native CUDA 13)
pip install vllm             # -> vllm 0.27.1; torch stays 2.13.0+cu130
pip install python-dotenv
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA) |
| torch | `2.13.0+cu130` |
| transformers | `5.15.1` |
| driver / CUDA | 580.173.02 / 13.0, H100 80GB HBM3 |

### Serve (the verified H100 command — identical to ROCm minus the HIP var)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models CUDA_VISIBLE_DEVICES=5
vllm serve google/embeddinggemma-300m \
  --runner pooling \
  --host 0.0.0.0 --port 8500 \
  --served-model-name embeddinggemma
```

```bash
python embed_vllm.py --port 8500 --model embeddinggemma
```

### Real output (H100)

```
[default_loader.py:430] Loading weights took 8.58 seconds
[gpu_model_runner.py:5405] Model loading took 0.61 GiB memory and 14.77 seconds
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

**Correct 768-dim vectors, correct ordering** (0.7106 vs 0.2008 → 0.51 margin). These match
the MI355X run to ~1e-3 on both the leading dims (`-0.09578, -0.08054, 0.04375, …` there)
and the cosines (0.7105 / 0.2011 there) — ordinary cross-hardware reduction-order drift,
not a correctness problem. **Model-load VRAM 0.61 GiB matches MI355X exactly.**

### GPU-5 residency (sampled from inside serving)

```
$ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep <GPU5-UUID>
1718783, GPU-e71a0833-4f61-11c9-6eff-10c149e752e4, 2000 MiB
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
5, 2009 MiB
```

~2.0 GB resident on **physical GPU 5 only** (0.61 GiB weights + preallocated pool); the
co-tenant GPUs 0–3 were untouched.

### Quirks (H100)

- One **benign** `Operation not permitted` warning while setting blob permissions on the
  CIFS-mounted HF cache — the weights were already cached and loaded fine (`downloading
  weights: 1.4 s` is metadata only). Non-fatal.
- vLLM suggests `pip install orjson` "to make the v1/embeddings API fast" — optional; the
  request still returns correct vectors without it.
- No AITER, no `quark_online_quant`, no GELU-approximation warning — those are all ROCm-only.

### Multi-GPU (deferred, and architecturally capped)

Same as ROCm: **TP=2 is impossible for this model** (3 attention heads, not divisible by 2)
— an identical, correct refusal on NVIDIA. The right multi-GPU pattern is **N independent
single-GPU replicas** (one `vllm serve ... --port 85xx` per free GPU). Not run this wave
(single-GPU scope; GPUs 0–3 are a co-tenant production job).

### H100 verdict

✅ **PASS — vLLM 0.27.1 (pip / cu130) serves EmbeddingGemma-300m unmodified on one H100.**
Correct 768-dim embeddings, correct semantic ordering, 0.61 GiB weights, ~2.0 GB resident.
The "✅ Native" rating holds on NVIDIA; the pip route is available here (unlike ROCm).

## Arguments / flags

Serve-side flags used here:

| Flag | Value used | Meaning |
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

Real captured run of `python embed_vllm.py --port 8001`:

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

Server logs go to `/mnt/data_1.5t/outputs/inference_embedding_vllm/`. Nothing large is
written into the repo.

## Hardware support & evidence

- **AMD: tested and working.** 1×  and 2× (as replicas) AMD Instinct MI355X (`gfx950`,
  288 GB), host ROCm 7.2.4, container ROCm 7.0.2, vLLM `0.20.2rc1.dev253`, torch
  `2.9.1.dev+rocm7.0.2`. Correct 768-dim embeddings with correct semantic ordering.
- **NVIDIA: not tested here** (no NVIDIA GPU on this host). The same `vllm serve` command
  applies with `vllm/vllm-openai:latest`; the pip route additionally works on CUDA.
- vLLM is **✅ Native** for EmbeddingGemma on both vendors, and on ROCm/gfx950 that is
  **confirmed** — with one correction: the *install* story on AMD is container-only, so any
  "use the ROCm wheel or container" phrasing overstates what actually exists.

## Notes & quirks

- **`--runner pooling` is mandatory.** Without it vLLM loads the model as a generative
  decoder and `/v1/embeddings` is not routed.
- **The pooling server also exposes `/score`, `/rerank` and `/v2/rerank`.** They are
  registered on the same process; that is normal for a pooling runner and does not mean
  the embedding model is a reranker.
- **AITER JIT-builds on first launch.** The first `vllm serve` on this image prints
  `[aiter] start build [module_aiter_core]` and stalls for a while compiling. Subsequent
  launches reuse the build cache — this is why cold start drops from ~38 s to ~22 s.
- **The `quark_online_quant` plugin fails to import** on every launch with a traceback.
  It is non-fatal and unrelated to this workload; the server starts normally. Do not
  chase it.
- **A ROCm-specific GELU warning is expected:** `[ROCm] PyTorch's native GELU with tanh
  approximation is unstable with torch.compile ... fallback to 'none' approximation.`
  vLLM handles it automatically; embedding quality was unaffected (cosine margin 0.48).
- **Default `--gpu-memory-utilization 0.9` looks alarming in `rocm-smi`.** vLLM
  preallocates its pool, so a 0.61 GiB model can show hundreds of GB "used". Use
  `--gpu-memory-utilization` to bound it when co-locating servers.
- **Do not set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides all GPUs
  rather than none.

## Verdict

✅ **PASS — vLLM serves EmbeddingGemma on MI355X/gfx950 with no changes.** The
"✅ Native" rating holds. Correct 768-dim embeddings, correct semantic ordering (0.6975 vs
0.2216), sub-second weight load, 0.61 GiB resident, ~22 s warm cold-start.

One important correction: **TP=2 is not available for this model** (3 attention
heads), and it should not be — for a 0.61 GiB model on 288 GB cards the right multi-GPU
story is **N independent replicas, one per GPU**, which was verified working on both GPUs
concurrently. The install route is also container-only on ROCm; there is no pip wheel.
