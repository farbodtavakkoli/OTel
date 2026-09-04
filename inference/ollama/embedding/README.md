# `inference/ollama/embedding` — Ollama GGUF embedding serving (EmbeddingGemma-300M) on ROCm

## Overview & when to use

Serves **EmbeddingGemma-300M** as a GGUF through **Ollama**, running in the official
`ollama/ollama:rocm` container on **AMD Instinct MI355X (gfx950)** with **ROCm 7.2.4**.
The client `inference_embedding_ollama.py` hits `POST /api/embed` and the
OpenAI-compatible `POST /v1/embeddings`.

Use this folder when you want:

- **An embedding endpoint with essentially zero setup.** One `docker run`, a two-line
  `Modelfile`, and you have a managed HTTP embedding API. No build, no torch, no ROCm
  install on the host.
- **One endpoint for embeddings *and* generation.** The same daemon can hold this 300M
  embedder and the 27B LLM from `../../../inference/ollama/llm` at the same time, both
  GPU-resident, loading each on demand. That co-residency is Ollama's real strength.
- **A verified-correct embedder.** Cosine geometry matches the Transformers baseline to
  **worst |Δ| = 0.0034** (see the cross-check below).

Do **not** use this folder as a high-throughput embedding service:

> For high-throughput embedding services, vLLM or a dedicated serving implementation
> remains a better benchmark target.

Ollama loads this model with a fixed **2048-token context** and a single request slot by
default. For batch/production embedding on MI355X see `../../../inference/vllm/embedding` or
`../../../inference/tei/embedding`.

> **Critical format note (confirmed here):** Ollama needs a **GGUF**
> conversion, not the original `google/embeddinggemma-300m` safetensors. This folder
> serves `ggml-org/embeddinggemma-300M-GGUF:Q8_0`.

## Scope note — why there is no `inference/ollama/reranker`

There is deliberately **no reranker folder for Ollama**, and this is an evidenced scope
decision rather than an omission. The combination is **❌ Not suitable**:

> **Reranker: Qwen3-Reranker-0.6B — Status: ❌ not a preferred Ollama workload.**
> Ollama does not currently expose a first-class reranking API equivalent to llama.cpp's
> reranking endpoint.

This matters most *here*, because embedding and reranking are the two halves of a
retrieval stack and it is natural to assume that if Ollama does one it does the other.
It does not. Ollama's HTTP surface is `/api/generate`, `/api/chat`, `/api/embed` plus the
OpenAI shim (`/v1/chat/completions`, `/v1/embeddings`) — there is no rerank route in the
`ollama/ollama:rocm` 0.32.14 API used here. A reranker is a **cross-encoder**: it must
score a (query, document) *pair* jointly, which is architecturally different from
`/api/embed`'s bi-encoder "one vector per text". You cannot emulate it by embedding both
sides and taking a cosine — that is exactly the weaker signal a reranker exists to
replace.

**For reranking on this host, use instead:**

| Folder | Route |
|---|---|
| [`../../../inference/llamacpp/reranker`](../../../inference/llamacpp/reranker) | llama.cpp `llama-server --reranking`, `/v1/rerank` — the local/GGUF answer |
| [`../../../inference/vllm/reranker`](../../../inference/vllm/reranker) | vLLM scoring API — the production GPU-serving answer |

> **Tested topology:** 2x AMD Instinct MI355X (gfx950, 288 GB each), physical GPUs **2
> and 3**, ROCm 7.2.4, Ubuntu, Docker 29.7.2, Python 3.12.3. Verified **2026-08-20**.

## Install — the exact docker route that worked

Ollama is a static Go binary in a container; there is nothing to build.

```bash
docker pull ollama/ollama:rocm
```

**1.43 GB compressed / 4.55 GB on disk.** Verified at digest
`sha256:b5d0813b23540019b2a40d70af6531b1d76e5ebcbf17899e5922445c627b6063`
(image ID `b5d0813b2354`), **Ollama version 0.32.14**.

The image ships its **own ROCm 7.2 userspace** (`libdirs=ollama,rocm_v7_2`); the host
supplies only the amdgpu KFD driver. **No `HSA_OVERRIDE_GFX_VERSION` was needed** —
gfx950 was detected natively.

### Which render node is which GPU

The canonical run line passes all of `/dev/dri`, which would expose **all 8
GPUs**. Pin to your own instead:

```bash
rocm-smi --showbus          # GPU[2] -> 0000:A5:00.0 , GPU[3] -> 0000:DC:00.0
ls -l /dev/dri/by-path/     # pci-0000:a5:00.0-render -> ../renderD144
                            # pci-0000:dc:00.0-render -> ../renderD152
```

| Physical GPU | PCI bus | Render node |
|---|---|---|
| 2 | `0000:a5:00.0` | `/dev/dri/renderD144` |
| 3 | `0000:dc:00.0` | `/dev/dri/renderD152` |

### Serve — single GPU (physical GPU 2)

```bash
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  -v /mnt/data_450g/ollama:/root/.ollama \
  -v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_embed \
  ollama/ollama:rocm
```

Two deviations from the canonical run line, both deliberate:

- **`-v /mnt/data_450g/ollama:/root/.ollama` instead of the named volume `-v ollama:…`.**
  A docker named volume lives under `/var/lib/docker` on `/`, which has only **~99 GB
  free**. `ollama create` **copies** models into the store; shared with the 27B LLM the
  store reached **55 GB**, which must not land on `/`.
- **`-v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro`** exposes the already-downloaded
  GGUF so the Modelfile registers it with no download at all.

### Confirm the backend sees only your GPU

```bash
docker logs ollama_embed 2>&1 | grep "inference compute"
```

```text
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

Exactly one GPU, the right one. (`available` is ~75 GiB of 288 GiB because other users'
SGLang jobs held ~214 GiB on that card during this run.)

### NVIDIA variant

Not tested here — this host has no NVIDIA GPU. Only the image tag and device flags
change; the Modelfile and every API call are identical:

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env` (`ln -sf ../../../dev.env dev.env`) and
supplies `HF_TOKEN`. The client loads it via `load_dotenv("dev.env")`.

`google/embeddinggemma-300m` is a **gated** repo, so the token would be needed to pull
the original weights — but **not on this path**. The `ggml-org` GGUF conversion is
ungated and already cached, so no credential ever reaches the container. Nothing prints
or commits the token; the GGUF mount is read-only.

## Model registration — the Modelfile workflow

`Modelfile.embeddinggemma`:

```text
FROM /ggufs/models--ggml-org--embeddinggemma-300M-GGUF/snapshots/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf
```

```bash
docker cp Modelfile.embeddinggemma ollama_embed:/root/Modelfile.embeddinggemma
docker exec ollama_embed ollama create embeddinggemma -f /root/Modelfile.embeddinggemma
```

```text
parsing GGUF
verifying conversion
using existing layer sha256:b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63
writing manifest
success

real    0m1.364s
```

**1.4 seconds, no network.** The HF snapshot symlink (`../../blobs/<sha256>`) resolves
inside the container because the whole cache tree is mounted, not just the snapshot dir.

```bash
docker exec ollama_embed ollama list
```

```text
NAME                     ID              SIZE      MODIFIED
embeddinggemma:latest    b48ed6e89ad7    333 MB    About a minute ago
qwen3.8-27b-q8:latest    b0322c83ce26    29 GB     3 minutes ago
```

## The one correctness trap: prompt templates are NOT in the GGUF

**This is the single most important thing in this README.** EmbeddingGemma is trained
with asymmetric task prefixes. Sentence-Transformers applies them for you via
`encode_query()` / `encode_document()` — which is exactly what the
`../../../inference/transformers/embedding` baseline does. The **GGUF carries no template**,
and Ollama's `/api/embed` passes your input through verbatim. So the client must add
them:

```python
QUERY_TEMPLATE    = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: none | text: {text}"
```

Measured impact against the saved Transformers reference — same model, same texts, the
prefixes are the *only* difference:

| Client behaviour | worst \|Δ cosine\| vs baseline | Verdict |
|---|---|---|
| **Prefixes applied** (default) | **0.0034** | ✅ PASS |
| `--no_prompt_template` | **0.1751** | ❌ FAIL |

Raw text does not merely shift the scores, it compresses the whole similarity range and
destroys discrimination — the irrelevant Eiffel Tower document jumps from `-0.0125` to
`0.1389`. A silently un-prefixed client still returns 768 well-formed normalised vectors
and *looks* fine; this is the embedding equivalent of a silent CPU fallback.

## Client / smoke command

```bash
python3 -m venv .env_inference_embedding_ollama
.env_inference_embedding_ollama/bin/pip install -r requirements_embedding_ollama.txt

.env_inference_embedding_ollama/bin/python inference_embedding_ollama.py \
  --port 11434 \
  --reference /mnt/data_1.5t/outputs/inference_embedding_transformers/reference_embedding_fp32_1gpu.json \
  --out /mnt/data_1.5t/outputs/inference_embedding_ollama/embeddings_single_gpu.json
```

Equivalent raw curl (note the manual prefix):

```bash
curl -s http://127.0.0.1:11434/api/embed -d '{
  "model":"embeddinggemma",
  "input":["task: search result | query: What GPU runtimes support ROCm?"]}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["embeddings"][0]))'
```

## Results — single GPU (physical GPU 2, `renderD144`)

Real client output:

```text
endpoint        : http://127.0.0.1:11434/api/embed
model           : embeddinggemma
prompt_template : on (EmbeddingGemma task prefixes)
latency_s       : 0.324
n_vectors       : 6
dim             : 768
norm[q0]        : 1.0000
head[q0]        : [-0.0672, -0.04413, -0.00492, -0.0297, 0.05435, -0.00036, -0.01898, 0.01194]
--- cosine similarity (semantic sanity check) ---
q0 'What GPU runtimes support ROCm?'
    0.5726  'vLLM supports NVIDIA CUDA and AMD ROCm.'  <-- best
    0.5341  'SGLang provides a ROCm build for AMD Instinct accelerators.'
    0.0973  'SQLite is an embedded database.'
    -0.0125  'The Eiffel Tower is located in Paris, France.'
q1 'Which database is embedded and serverless?'
    0.2024  'vLLM supports NVIDIA CUDA and AMD ROCm.'
    0.1902  'SGLang provides a ROCm build for AMD Instinct accelerators.'
    0.5025  'SQLite is an embedded database.'  <-- best
    0.0829  'The Eiffel Tower is located in Paris, France.'
```

Vectors are **768-dimensional** (matching EmbeddingGemma's `n_embd = 768`) and arrive
**already L2-normalised** (`norm = 1.0000`) — Ollama applies pooling and normalisation
server-side. The geometry is *correct*, not merely well-shaped: each query's best match
is the right document, and the relevant/irrelevant gap is wide (0.5726 vs -0.0125).

Cold-start timing from `/api/embed` (`load_duration`):

```text
dim: 768
load_duration_s: 1.239
total_duration_s: 1.321
```

**1.24 s cold load**, ~0.08 s to embed after that. Warm calls: 0.32 s for 6 texts across
two round trips.

### GPU-residency proof — single GPU

A 300M model runs perfectly well on 256 CPU cores, so this is the check that matters.

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
```

**`100% GPU`** — not `CPU`. Corroborated by the runner's offload summary:

```text
llama_prepare_model_devices: using device ROCm0 (AMD Radeon Graphics) (0000:a5:00.0) - 76408 MiB free
load_tensors: offloading output layer to GPU
load_tensors: offloading 23 repeating layers to GPU
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
sched_reserve:      ROCm0 compute buffer size =    63.02 MiB
```

**25/25 layers on GPU.** And by `rocm-smi`, before vs after load:

```text
                       before                after              delta
GPU[2] VRAM used   230082174976 B      231924178944 B     +1842003968 B  (+1.72 GiB)
GPU[3] VRAM used   229335334912 B      229335334912 B                 0
```

**GPU[3] did not move** — `--device /dev/dri/renderD144` really does confine Ollama to
physical GPU 2.

## Results — multi-GPU (physical GPUs 2 and 3)

**Finding, stated plainly: a 393 MB embedder is never split across GPUs, and it should
not be.** Ollama spreads a model across devices only when it does not fit on one; this
one fits ~170x over. There is no tensor parallelism to demonstrate and claiming one would
be false.

The correct multi-GPU pattern for embedding is **one instance per GPU behind a load
balancer** — horizontal scale-out, which is what an embedding service actually needs.
That was measured:

```bash
# instance A — physical GPU 2, port 11434
docker run -d --device /dev/kfd --device /dev/dri/renderD144 \
  -v /mnt/data_450g/ollama:/root/.ollama \
  -v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro \
  -p 11434:11434 --name ollama_embed ollama/ollama:rocm

# instance B — physical GPU 3, port 11435
docker run -d --device /dev/kfd --device /dev/dri/renderD152 \
  -v /mnt/data_450g/ollama:/root/.ollama \
  -v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro \
  -p 11435:11434 --name ollama_embed_gpu3 ollama/ollama:rocm
```

Instance B's discovery confirms it landed on the *other* card:

```text
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 pci_id=0000:dc:00.0 type=discrete total="288.0 GiB" available="75.0 GiB"
```

Both daemons serving, each `100% GPU`:

```text
-- :11434 (GPU2) --
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
-- :11435 (GPU3) --
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
```

`rocm-smi` with both instances loaded, against the same pre-load baseline:

```text
                       before                after              delta
GPU[2] VRAM used   230082174976 B      231922626560 B     +1840451584 B  (+1.71 GiB)
GPU[3] VRAM used   229335334912 B      231175790592 B     +1840455680 B  (+1.71 GiB)
```

**Both GPUs loaded.** The GPU 3 instance produces byte-identical results and the same
baseline agreement over the OpenAI route:

```text
endpoint        : http://127.0.0.1:11435/v1/embeddings
dim             : 768
norm[q0]        : 1.0000
head[q0]        : [-0.0672, -0.04413, -0.00492, -0.0297, 0.05435, -0.00036, -0.01898, 0.01194]
worst_abs_delta : 0.0034  (tolerance 0.01)
agreement       : PASS
```

Note the two daemons share one read-mostly model store on `/mnt/data_450g/ollama`; that
worked cleanly here because both models were already created. Create models from a single
daemon, then scale out readers.

The other multi-model pattern — the 27B LLM and this embedder co-resident on the same
two-GPU daemon, both `100% GPU` — is documented in
[`../../../inference/ollama/llm/readme_llm_ollama.md`](../llm/README.md).

## Cross-check vs the Transformers baseline

`../../../inference/transformers/embedding` is the correctness baseline; its saved reference
vectors live at `/mnt/data_1.5t/outputs/inference_embedding_transformers/`. The client
compares against `reference_embedding_fp32_1gpu.json`
(`google/embeddinggemma-300m`, float32, `normalized: true`, `embedding_dim: 768`) using
the same two queries and four documents.

```text
--- cross-check vs google/embeddinggemma-300m (float32) ---
q0 ollama       : [0.5726, 0.5341, 0.0973, -0.0125]
q0 transformers : [0.5747, 0.5375, 0.0979, -0.0111]
q0 max_abs_delta: 0.0034
q1 ollama       : [0.2024, 0.1902, 0.5025, 0.0829]
q1 transformers : [0.2023, 0.1902, 0.5024, 0.083]
q1 max_abs_delta: 0.0001
worst_abs_delta : 0.0034  (tolerance 0.01)
agreement       : PASS
```

Direct vector-space agreement on the reference's saved first-16 dimensions:

| Query | cos(Ollama Q8_0, Transformers fp32) |
|---|---|
| q0 "What GPU runtimes support ROCm?" | **0.9992** |
| q1 "Which database is embedded and serverless?" | **0.9998** |

**Ollama's Q8_0 GGUF is numerically equivalent to the fp32 Transformers baseline for
retrieval purposes.** The residual ~0.003 is Q8_0 quantisation noise, an order of
magnitude below anything that changes a ranking — the document ordering is identical for
both queries. `embedding_dim`, normalisation, and ranking all match exactly.

## H100 (NVIDIA) — verified 2026-08-22

Single-GPU smoke on **NVIDIA H100 80GB HBM3** (Hopper cc 9.0, driver 580.173.02, CUDA 13.0),
physical **GPU 4 only** (shared node). Container/run-line details and the "GGUFs had to be
downloaded" finding are in [`../README.md`](../README.md).

**Model:** the **exact same** GGUF the Modelfile references — `ggml-org/embeddinggemma-300M-GGUF:Q8_0`
— but it is **not cached on this box** (the `FROM` path `/mnt/data_1.5t/hf_cache/llama_cpp/...`
is MI355X-era and absent), so it was downloaded fresh. The registered layer sha
`b5ce9d77a3fc…` is **byte-identical to the MI355X run**, so this is the same model, just fetched
rather than mounted from cache.

### Exact commands

```bash
# GGUF (proxy unset — HF is proxy-blocked here):
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download ggml-org/embeddinggemma-300M-GGUF embeddinggemma-300M-Q8_0.gguf \
  --local-dir $GGUF/embeddinggemma-300M-GGUF

# H100 Modelfile variant (CONTAINER path under the :ro /ggufs mount; MI355X Modelfile untouched):
#   FROM /ggufs/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
sudo docker cp Modelfile.embeddinggemma.h100 ollama_h100:/root/Modelfile.embeddinggemma
sudo docker exec ollama_h100 ollama create embeddinggemma -f /root/Modelfile.embeddinggemma

# client (venv on tmpfs; requests only). No Transformers reference JSON exists on this box,
# so --reference was omitted; correctness is shown by dim/norm/ranking + agreement with the
# MI355X numbers below:
python inference_embedding_ollama.py --port 11440 --model embeddinggemma --api native
```

### Real output — dim 768, L2-normalised, correct geometry

```text
endpoint        : http://127.0.0.1:11440/api/embed
model           : embeddinggemma
prompt_template : on (EmbeddingGemma task prefixes)
n_vectors       : 6
dim             : 768
norm[q0]        : 1.0000
head[q0]        : [-0.06693, -0.04403, -0.00289, -0.03042, 0.05471, 0.0003, -0.01863, 0.01056]
--- cosine similarity (semantic sanity check) ---
q0 'What GPU runtimes support ROCm?'
    0.5731  'vLLM supports NVIDIA CUDA and AMD ROCm.'  <-- best
    -0.0104  'The Eiffel Tower is located in Paris, France.'
q1 'Which database is embedded and serverless?'
    0.5050  'SQLite is an embedded database.'  <-- best
```

**dim = 768** (EmbeddingGemma's `n_embd`), vectors arrive **L2-normalised** (`norm = 1.0000`),
and the ranking is correct with a wide relevant/irrelevant gap. It also **cross-validates the
MI355X baseline without a reference file**: H100 q0 best `0.5731` vs MI355X `0.5726`, q1 best
`0.5050` vs `0.5025`, and `head[q0]` matches to ~3 decimals — well inside the `0.0034` worst-Δ
the MI355X README measured against the fp32 Transformers reference. Q8_0 on H100 is
numerically the same embedder. The client-side prompt-prefix requirement is unchanged (the
GGUF still carries no template — server log shows `chat_template=null`).

### GPU-residency proof — `100% GPU` + nvidia-smi on GPU 4

A 300M model runs fine on CPU, so this is the check that matters.

```text
# docker exec ollama_h100 ollama ps   (co-resident with the LLM, both on GPU 4)
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
qwen3-0.6b-q8:latest     605b58ae76ea    5.6 GB    100% GPU     40960      3 minutes from now
```

`100% GPU`. Offload summary (25/25 layers; the non-zero `CPU_Mapped` is the token-embedding
table — MI355X quirk #4 reproduces exactly):

```text
load_tensors: offloaded 25/25 layers to GPU
load_tensors:        CUDA0 model buffer size =   311.97 MiB
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
```

And by `nvidia-smi -i 4` — the embedder's runner PID holding VRAM on physical GPU 4, alongside
the LLM's:

```text
# nvidia-smi -i 4 --query-compute-apps=pid,process_name,used_memory --format=csv
1717079, /usr/lib/ollama/llama-server, 5942 MiB     # the LLM
1718422, /usr/lib/ollama/llama-server, 1030 MiB     # this embedder
```

**25/25 layers on GPU, 1030 MiB resident on GPU 4** — and it coexists with the 27B-class LLM on
the *same* card, both `100% GPU`, which is Ollama's real strength (one endpoint, embeddings +
generation, both GPU-resident). GPU 4 = UUID `GPU-e13d18b6-…`; GPUs 0–3 (production) untouched.

### Single vs multi-GPU on H100

Single-GPU only this wave (shared node; multi-GPU deferred). The MI355X finding is unchanged: a
393 MB embedder is **never** split and should not be — the right multi-GPU pattern is
**one instance per GPU behind a load balancer** (horizontal scale-out), which on H100 means a
second container with `--gpus '"device=5"'` on port 11441. Not launched here (GPUs 0–3 busy,
5/7 held by other agents).

### H100 verdict

**PASS.** `ollama/ollama` (0.32.15, CUDA-13 userspace) served EmbeddingGemma-300M on H100 out
of the box. `25/25` layers on GPU, `PROCESSOR: 100% GPU`, 1030 MiB on GPU 4 by `nvidia-smi`,
768-dim L2-normalised vectors with correct ranking that match the MI355X/fp32 numbers to ~3
decimals. Only deviations from the MI355X recipe: the container image tag + `--gpus` pinning,
and the GGUF was downloaded (not cached). The two caveats from MI355X still bite identically —
**apply the task prefixes client-side** (the GGUF has no template), and it is the low-friction
co-resident answer, **not** the high-throughput one (2048 context, one slot by default; use
`../../vllm/embedding` or `../../tei/embedding` for batch).

## Arguments

### Docker flags

| Flag | Value used | Why |
|---|---|---|
| `--device /dev/kfd` | required | ROCm compute node; without it there is no GPU at all |
| `--device /dev/dri/renderD144` | GPU 2 | per-GPU pinning; **use this instead of exposing the whole `/dev/dri`** |
| `--device /dev/dri/renderD152` | GPU 3 | second instance for scale-out |
| `-v /mnt/data_450g/ollama:/root/.ollama` | 387 GB free | model store; a named volume lands on `/` which has only ~99 GB |
| `-v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro` | read-only | reuse the cached GGUF, no download |
| `-p 11434:11434` / `-p 11435:11434` | default / +1 | Ollama's default port; a common benchmark layout suggests 8300 — this folder uses **11434** and **11435** |

### Server environment (`-e`)

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `1` | concurrent embedding request slots — raise for batch work |
| `OLLAMA_MAX_LOADED_MODELS` | `0` (auto) | how many models may be resident together |
| `OLLAMA_KEEP_ALIVE` | `5m0s` | idle time before the embedder is unloaded |
| `OLLAMA_CONTEXT_LENGTH` | `0` (auto) | ignored here — the embedder is pinned at its 2048 training context |

### Client (`inference_embedding_ollama.py`)

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Ollama server host |
| `--port` | `11434` | Ollama server port |
| `--model` | `embeddinggemma` | model name created by `ollama create` |
| `--api` | `native` | `native` → `/api/embed`, `openai` → `/v1/embeddings` |
| `--queries` | 2 retrieval queries | query strings to embed |
| `--documents` | 4 documents | document strings to embed |
| `--no_prompt_template` | off | send raw text with no task prefixes — **negative control, degrades agreement** |
| `--keep_alive` | `5m` | how long Ollama keeps the model resident |
| `--reference` | none | Transformers reference JSON to cross-check against |
| `--tolerance` | `0.01` | max allowed \|Δ cosine\| before the check FAILs |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/tags` probes before giving up |
| `--out` | none | write the results JSON |

## Output

Results JSON is written under `/mnt/data_1.5t/outputs/inference_embedding_ollama/`, not
into the repo, so `git status` stays clean:

| File | Contents |
|---|---|
| `embeddings_single_gpu.json` | GPU 2, `/api/embed`, with prefixes |
| `embeddings_gpu3_openai.json` | GPU 3, `/v1/embeddings`, with prefixes |

Each holds the model name, endpoint, whether prefixes were applied, `embedding_dim`, the
queries/documents, the full cosine matrix, the first 16 dims of each query vector, and
`worst_abs_delta_vs_reference` — the same shape as the Transformers reference files, so
the two are directly diffable.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| ROCm backend selected, gfx950 native | `library=ROCm compute=gfx950 libdirs=ollama,rocm_v7_2` |
| No `HSA_OVERRIDE_GFX_VERSION` needed | server config shows it empty; GPU still detected |
| Only the intended GPU visible | instance A lists only `pci_id=0000:a5:00.0`; instance B only `0000:dc:00.0` |
| Model is on the GPU, not the CPU | `ollama ps` → `PROCESSOR = 100% GPU`; `offloaded 25/25 layers to GPU` |
| Single-GPU placement measured | `rocm-smi` GPU[2] +1.84 GB, GPU[3] **unchanged** |
| Two-GPU scale-out measured | `rocm-smi` GPU[2] +1.84 GB **and** GPU[3] +1.84 GB |
| Output is numerically correct | worst \|Δ cosine\| **0.0034** vs Transformers fp32; first-16-dim cos 0.9992 / 0.9998 |
| Both API routes work | `/api/embed` and `/v1/embeddings` return identical vectors |

## Notes & quirks

1. **Prompt prefixes are the whole ballgame** — see the correctness section. Without
   them, worst |Δ| vs baseline goes 0.0034 → **0.1751**. The GGUF has no template and
   Ollama will not add one. You could bake them into the Modelfile with a `TEMPLATE`
   directive, but that cannot express *different* prefixes for queries vs documents, so
   client-side is the right layer.

2. **`/v1/embeddings` returns `"object": "list"` at the top level**, not the
   `"object": "embedding"` some OpenAI clients expect at that position; per-row objects
   are normal. It also reports `usage` with `prompt_tokens` only. Vectors are identical
   to `/api/embed`.

3. **Context is fixed at 2048** (`ollama ps` CONTEXT column) — EmbeddingGemma's training
   context. `OLLAMA_CONTEXT_LENGTH` does not raise it. Longer inputs are truncated, so
   chunk before embedding.

4. **`PROCESSOR: 100% GPU` coexists with a non-zero `CPU_Mapped` buffer**
   (`204.00 MiB` here — the token embedding table). All 25 *layers* are on GPU; the
   percentage refers to layer offload, not to every byte.

5. **`ollama create` copies the blob into the store.** Trivial for 319 MB, but the same
   store shared with the 27B LLM reached 55 GB — which is why it must not sit on `/`.

6. **The 5-minute `keep_alive` default will unload the model between batches**, giving a
   surprise 1.24 s cold load. Raise `OLLAMA_KEEP_ALIVE` (or pass `--keep_alive 30m`) for
   a steady service.

7. **Outbound calls to ollama.com fail on this host and are harmless** —
   `model show cloud cache hydration failed … context deadline exceeded` is the model
   recommendation refresh, not serving.

8. **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm.** Device selection here is done purely
   by which `renderD*` nodes are passed into the container; Ollama's config echo shows
   `CUDA_VISIBLE_DEVICES:` and `HIP_VISIBLE_DEVICES:` unset.

## VERDICT

**✅ PASS — Ollama on ROCm/MI355X serves EmbeddingGemma correctly, with proven GPU
residency and near-exact agreement with the Transformers baseline.**

- `ollama/ollama:rocm` (`0.32.14`) ran gfx950 **out of the box** — no build, no patch, no
  `HSA_OVERRIDE_GFX_VERSION`. Registration from the cached GGUF took **1.4 s** with zero
  network; cold load **1.24 s**.
- **Correctness is the headline:** worst |Δ cosine| **0.0034** vs the fp32 Transformers
  baseline, first-16-dim cosine **0.9992 / 0.9998**, identical document ranking. Q8_0
  costs nothing that matters for retrieval.
- **Single GPU:** `100% GPU`, 25/25 layers, +1.84 GB on GPU[2] and **nothing** on GPU[3].
- **Multi-GPU:** a 393 MB model is correctly **not** split — instead two instances, one
  per GPU, both `100% GPU`, both measured (+1.84 GB each). Scale-out, not tensor
  parallelism, and that is the right architecture for embedding.
- **One caveat that will bite you:** the GGUF has no prompt template. Apply
  EmbeddingGemma's task prefixes client-side or your retrieval quality silently degrades
  while every vector still looks perfectly valid.
- **Not the throughput answer.** 2048-token context, one request slot by default. For
  batch embedding on MI355X use `../../../inference/vllm/embedding` or
  `../../../inference/tei/embedding`. Ollama is the low-friction, co-resident-with-your-LLM
  answer — and at that it is excellent.
- **No reranker route exists** — see the scope note above.
