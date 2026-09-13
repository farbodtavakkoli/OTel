# `inference/ollama/embedding` — Ollama GGUF embedding serving (EmbeddingGemma-300M)

## Overview & when to use

Serves **EmbeddingGemma-300M** as a GGUF through **Ollama**, running in the official
`ollama/ollama:rocm` (AMD) or `ollama/ollama` (NVIDIA) container.
The client `inference_embedding_ollama.py` hits `POST /api/embed` and the
OpenAI-compatible `POST /v1/embeddings`.

The same daemon can hold this 300M embedder and the 27B LLM from
[`../llm`](../llm/README.md) at the same time, both GPU-resident, loading each on demand.

Ollama loads this model with a fixed **2048-token context** and a single request slot by
default. For batch/production embedding see [`../../vllm/embedding`](../../vllm/embedding)
or [`../../tei/embedding`](../../tei/embedding).

> **Critical format note:** Ollama needs a **GGUF** conversion, not the original
> `google/embeddinggemma-300m` safetensors. This folder serves
> `ggml-org/embeddinggemma-300M-GGUF:Q8_0`.

## Scope note — why there is no `inference/ollama/reranker`

Ollama's HTTP surface is `/api/generate`, `/api/chat`, `/api/embed` plus the OpenAI shim
(`/v1/chat/completions`, `/v1/embeddings`) — there is **no rerank route**. A reranker is a
cross-encoder that scores a (query, document) pair jointly; you cannot emulate it by
embedding both sides and taking a cosine.

**For reranking, use instead:**

| Folder | Route |
|---|---|
| [`../../llamacpp/reranker`](../../llamacpp/reranker) | llama.cpp `llama-server --reranking`, `/v1/rerank` — the local/GGUF answer |
| [`../../vllm/reranker`](../../vllm/reranker) | vLLM scoring API — the production GPU-serving answer |

## Install — the docker route

Ollama is a static Go binary in a container; there is nothing to build.

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
export GGUF=$LLAMA_CACHE                         # or a fresh dir if you download instead
export OUTPUT_DIR=/path/to/outputs               # inference artifacts
```

### AMD / ROCm

```bash
docker pull ollama/ollama:rocm
```

The image ships its **own ROCm 7.2 userspace** (`libdirs=ollama,rocm_v7_2`); the host
supplies only the amdgpu KFD driver. **No `HSA_OVERRIDE_GFX_VERSION` is needed** —
gfx950 is detected natively.

The canonical run line passes all of `/dev/dri`, which exposes **every GPU on the host**.
Pin to your own instead — map GPU → render node first:

```bash
rocm-smi --showbus          # GPU[N] -> 0000:A5:00.0
ls -l /dev/dri/by-path/     # pci-0000:a5:00.0-render -> ../renderD144
```

```bash
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_embed \
  ollama/ollama:rocm
```

Two deviations from the canonical run line, both deliberate:

- **`-v $DATA_DIR/ollama:/root/.ollama` instead of the named volume `-v ollama:…`.**
  A docker named volume lives under `/var/lib/docker` on the root filesystem.
  `ollama create` **copies** models into the store, which must not land on the root
  filesystem.
- **`-v $LLAMA_CACHE:/ggufs:ro`** exposes the already-downloaded
  GGUF so the Modelfile registers it with no download at all.

### NVIDIA / CUDA

Only the image tag and the device flags change; the Modelfile and every API call are
identical. Pin with `--gpus '"device=N"'` rather than `--gpus=all`:

```bash
docker pull ollama/ollama
sudo docker run -d --gpus '"device=0"' \
  -e OLLAMA_HOST=0.0.0.0:11440 \
  -v $GGUF:/ggufs:ro \
  -v /dev/shm/ollama/store:/root/.ollama \
  -p 127.0.0.1:11440:11440 \
  --name ollama_embed ollama/ollama
```

If the GGUF is not already cached under `$LLAMA_CACHE`, download it (unset the proxy if
Hugging Face is proxy-blocked on your host) and point the Modelfile's `FROM` at the
result — under the `:ro /ggufs` mount the container path is
`/ggufs/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download ggml-org/embeddinggemma-300M-GGUF embeddinggemma-300M-Q8_0.gguf \
  --local-dir $GGUF/embeddinggemma-300M-GGUF
```

### Confirm the backend sees only your GPU

```bash
docker logs ollama_embed 2>&1 | grep "inference compute"
```

```text
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

Exactly one GPU, the right one.

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
```

Registration needs **no network**. Mount the whole HF cache tree, not just the snapshot
dir — the snapshot symlink (`../../blobs/<sha256>`) must resolve inside the container.
`docker exec ollama_embed ollama list` then shows the registered model.

## The one correctness trap: prompt templates are not in the GGUF

EmbeddingGemma is trained with asymmetric task prefixes. Sentence-Transformers applies them
for you via `encode_query()` / `encode_document()`; the **GGUF carries no template**, and
Ollama's `/api/embed` passes your input through verbatim. So the client must add them:

```python
QUERY_TEMPLATE    = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: none | text: {text}"
```

Without the prefixes the similarity range collapses and ranking quality degrades, while
every vector still comes back 768-dim and normalised — it *looks* fine. This is the
embedding equivalent of a silent CPU fallback.

## Client / smoke command

```bash
cd .. && python3 -m venv .env_ollama && .env_ollama/bin/pip install -r requirements.txt && cd embedding

../.env_ollama/bin/python inference_embedding_ollama.py \
  --port 11434 \
  --reference $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_1gpu.json \
  --out $OUTPUT_DIR/inference_embedding_ollama/embeddings_single_gpu.json
```

Equivalent raw curl (note the manual prefix):

```bash
curl -s http://127.0.0.1:11434/api/embed -d '{
  "model":"embeddinggemma",
  "input":["task: search result | query: What GPU runtimes support ROCm?"]}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["embeddings"][0]))'
```

## Expected output

```text
endpoint        : http://127.0.0.1:11434/api/embed
model           : embeddinggemma
prompt_template : on (EmbeddingGemma task prefixes)
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

Vectors are **768-dimensional** and arrive **already L2-normalised** — Ollama applies
pooling and normalisation server-side. Each query's best match should be the right
document.

### Confirm GPU residency

A 300M model runs perfectly well on CPU, so this is the check that matters:

```bash
docker exec ollama_embed ollama ps
```

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
```

Look for **`100% GPU`** — not `CPU`. The server log carries the runner's offload summary
(`ROCm0` / `CUDA0` depending on the stack):

```text
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
```

## Multi-GPU — one instance per GPU

This model is small enough that Ollama never splits it across devices. The multi-GPU
pattern for embedding is one instance per GPU behind a load balancer:

```bash
# instance A — first GPU, port 11434
docker run -d --device /dev/kfd --device /dev/dri/renderD144 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11434:11434 --name ollama_embed ollama/ollama:rocm

# instance B — second GPU, port 11435
docker run -d --device /dev/kfd --device /dev/dri/renderD152 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11435:11434 --name ollama_embed_gpu2 ollama/ollama:rocm
```

On NVIDIA the same pattern is a second container with `--gpus '"device=<N+1>"'` on the
next port.

The two daemons share one read-mostly model store at `$DATA_DIR/ollama`. Create models
from a single daemon first, then scale out readers.

Running the 27B LLM and this embedder co-resident on one daemon is documented in
[`../llm/README.md`](../llm/README.md).

## Cross-check vs the Transformers baseline

[`../../transformers/embedding`](../../transformers/embedding) is the correctness baseline;
its saved reference vectors live at `$OUTPUT_DIR/inference_embedding_transformers/`. Pass
`--reference .../reference_embedding_fp32_1gpu.json` and the client prints a per-query
delta plus a PASS/FAIL against `--tolerance`.

## Hardware support

Works on **AMD MI355X (gfx950, ROCm 7.2)** — single-GPU and two-instance scale-out — and on
**NVIDIA H100 80GB (CUDA 13)**, including co-resident with an LLM on the same card. Only the
image tag, the device flag, and (optionally) downloading rather than mounting the GGUF
differ; every caveat below applies to both.

## Arguments

### Docker flags

| Flag | Value used | Why |
|---|---|---|
| `--device /dev/kfd` | required | ROCm compute node; without it there is no GPU at all |
| `--device /dev/dri/renderD<N>` | one render node | per-GPU pinning; **use this instead of exposing the whole `/dev/dri`** |
| `--gpus '"device=<N>"'` | NVIDIA equivalent | per-GPU pinning on CUDA; `--gpus=all` exposes every card |
| `-v $DATA_DIR/ollama:/root/.ollama` | a disk with room | model store; a named volume lands on the root filesystem instead |
| `-v $LLAMA_CACHE:/ggufs:ro` | read-only | reuse the cached GGUF, no download |
| `-p 11434:11434` / `-p 11435:11434` | default / +1 | Ollama's default port; a second instance takes the next port |

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

Results JSON is written under `$OUTPUT_DIR/inference_embedding_ollama/`, not
into the repo, so `git status` stays clean:

| File | Contents |
|---|---|
| `embeddings_single_gpu.json` | first instance, `/api/embed`, with prefixes |
| `embeddings_gpu3_openai.json` | second instance, `/v1/embeddings`, with prefixes |

Each holds the model name, endpoint, whether prefixes were applied, `embedding_dim`, the
queries/documents, the full cosine matrix, the first 16 dims of each query vector, and
`worst_abs_delta_vs_reference` — the same shape as the Transformers reference files, so
the two are directly diffable.

## Notes & quirks

1. **Apply the prompt prefixes client-side** — see the correctness section. The GGUF has no
   template and Ollama will not add one; a Modelfile `TEMPLATE` cannot express *different*
   prefixes for queries vs documents.

2. **`/v1/embeddings` returns `"object": "list"` at the top level**, not the
   `"object": "embedding"` some OpenAI clients expect at that position; per-row objects
   are normal. It also reports `usage` with `prompt_tokens` only. Vectors are identical
   to `/api/embed`.

3. **Context is fixed at 2048** (`ollama ps` CONTEXT column) — EmbeddingGemma's training
   context. `OLLAMA_CONTEXT_LENGTH` does not raise it. Longer inputs are truncated, so
   chunk before embedding.

4. **`PROCESSOR: 100% GPU` coexists with a non-zero `CPU_Mapped` buffer** (the token
   embedding table). All *layers* are on GPU; the percentage refers to layer offload, not
   to every byte.

5. **`ollama create` copies the blob into the store** — which is why the store must not sit
   on the root filesystem, especially when shared with the 27B LLM.

6. **The 5-minute `keep_alive` default will unload the model between batches**, giving a
   surprise cold load. Raise `OLLAMA_KEEP_ALIVE` (or pass `--keep_alive 30m`) for a steady
   service.

7. **Outbound calls to ollama.com may fail on an air-gapped or proxied host, and are
   harmless** — `model show cloud cache hydration failed … context deadline exceeded` is
   the model recommendation refresh, not serving.

8. **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm.** Device selection here is done purely
   by which `renderD*` nodes are passed into the container; Ollama's config echo shows
   `CUDA_VISIBLE_DEVICES:` and `HIP_VISIBLE_DEVICES:` unset.

