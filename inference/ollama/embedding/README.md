# `inference/ollama/embedding` — Ollama GGUF embedding serving (EmbeddingGemma-300M)

Serves `ggml-org/embeddinggemma-300M-GGUF:Q8_0` through Ollama in the official
`ollama/ollama:rocm` (AMD) or `ollama/ollama` (NVIDIA) container.
`inference_embedding_ollama.py` hits `POST /api/embed` and the OpenAI-compatible
`POST /v1/embeddings`.

The same daemon can hold this 300M embedder and the 27B LLM from
[`../llm/`](../llm/README.md) at once, both GPU-resident. Ollama loads this model with a
fixed 2048-token context and one request slot by default, so for batch/production embedding
see [`../../vllm/embedding/`](../../vllm/embedding/) or
[`../../tei/embedding/`](../../tei/embedding/). Ollama needs the GGUF conversion, not the
`google/embeddinggemma-300m` safetensors.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2), single-GPU and two-instance scale-out, and
NVIDIA H100 80GB (CUDA 13), including co-resident with an LLM on one card.

## Files

- `Modelfile.embeddinggemma` — registers the cached GGUF with Ollama, no download.
- `inference_embedding_ollama.py` — embedding smoke client; applies EmbeddingGemma's task
  prefixes, prints the cosine matrix, and optionally cross-checks a Transformers reference.

## Setup

Images, GPU-to-render-node mapping, run-line deviations and the shared `.env_ollama` venv:
[`../README.md`](../README.md). Then:

```bash
ln -sf ../../../dev.env dev.env                  # no token actually needed on this path
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
export GGUF=$LLAMA_CACHE                         # or a fresh dir if you download instead
export OUTPUT_DIR=/path/to/outputs               # inference artifacts
```

`google/embeddinggemma-300m` is gated, but the `ggml-org` GGUF conversion served here is not,
so no credential ever reaches the container.

## Run

### Serve — AMD / ROCm

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

The image ships its own ROCm 7.2 userspace (`libdirs=ollama,rocm_v7_2`) and detects gfx950
natively — no `HSA_OVERRIDE_GFX_VERSION`.

### Serve — NVIDIA / CUDA

Only the image tag and device flags change; the Modelfile and every API call are identical.

```bash
sudo docker run -d --gpus '"device=0"' \
  -e OLLAMA_HOST=0.0.0.0:11440 \
  -v $GGUF:/ggufs:ro \
  -v /dev/shm/ollama/store:/root/.ollama \
  -p 127.0.0.1:11440:11440 \
  --name ollama_embed ollama/ollama
```

If the GGUF is not already cached, download it and point the Modelfile's `FROM` at the
result — under the `:ro /ggufs` mount that is
`/ggufs/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
hf download ggml-org/embeddinggemma-300M-GGUF embeddinggemma-300M-Q8_0.gguf \
  --local-dir $GGUF/embeddinggemma-300M-GGUF
```

Verify the daemon sees only your GPU:

```bash
docker logs ollama_embed 2>&1 | grep "inference compute"
```

```text
... id=0 filter_id=0 library=ROCm compute=gfx950 name=ROCm0 libdirs=ollama,rocm_v7_2 pci_id=0000:a5:00.0 type=discrete total="288.0 GiB" available="74.8 GiB"
```

### Register the model

`Modelfile.embeddinggemma`:

```text
FROM /ggufs/models--ggml-org--embeddinggemma-300M-GGUF/snapshots/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf
```

```bash
docker cp Modelfile.embeddinggemma ollama_embed:/root/Modelfile.embeddinggemma
docker exec ollama_embed ollama create embeddinggemma -f /root/Modelfile.embeddinggemma
docker exec ollama_embed ollama list
```

Registration needs no network. Mount the whole HF cache tree, not just the snapshot
directory — the snapshot symlink (`../../blobs/<sha256>`) must resolve inside the container.

Do not add a `TEMPLATE` directive to this Modelfile: a template cannot express different
query and document prefixes, which is why the client applies them instead.

### Client

```bash
../.env_ollama/bin/python inference_embedding_ollama.py \
  --port 11434 \
  --reference $OUTPUT_DIR/inference_embedding_transformers/reference_embedding_fp32_1gpu.json \
  --out $OUTPUT_DIR/inference_embedding_ollama/embeddings_single_gpu.json
```

EmbeddingGemma is trained with asymmetric task prefixes and the GGUF carries no template, so
anything calling `/api/embed` directly must add them itself — without them similarity
collapses while vectors still come back 768-dim and normalised, so it looks fine:

```python
QUERY_TEMPLATE    = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: none | text: {text}"
```

```bash
curl -s http://127.0.0.1:11434/api/embed -d '{
  "model":"embeddinggemma",
  "input":["task: search result | query: What GPU runtimes support ROCm?"]}' \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["embeddings"][0]))'
```

### Scale out — one instance per GPU

This model is small enough that Ollama never splits it across devices. Run one container per
GPU behind a load balancer — a second container on the next render node and the next port:

```bash
docker run -d --device /dev/kfd --device /dev/dri/renderD152 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11435:11434 --name ollama_embed_gpu2 ollama/ollama:rocm
```

On NVIDIA it is `--gpus '"device=<N+1>"'` instead. Both daemons share one read-mostly model
store, so create models from a single daemon first, then scale out readers.

## Output

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
```

Vectors are 768-dimensional and arrive already L2-normalised; each query's best match must be
the right document — `q1` likewise ranks the SQLite document first, at 0.5025.

Results JSON lands under `$OUTPUT_DIR/inference_embedding_ollama/` in the same shape as the
Transformers reference files, so the two are directly diffable. Passing `--reference` against
[`../../transformers/embedding`](../../transformers/embedding)'s saved vectors prints a
per-query delta and a PASS/FAIL at `--tolerance`.

A 300M model runs fine on CPU, so confirm residency — look for `100% GPU`:

```bash
docker exec ollama_embed ollama ps
```

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
```

```text
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   204.00 MiB
load_tensors:        ROCm0 model buffer size =   311.97 MiB
```

## Arguments

Docker flags are the same as the LLM leaf — see the table in
[`../llm/README.md`](../llm/README.md). A second instance takes the next host port
(`-p 11435:11434`).

### Server environment (`-e`)

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `1` | concurrent embedding request slots — raise for batch work |
| `OLLAMA_MAX_LOADED_MODELS` | `0` (auto) | how many models may be resident together |
| `OLLAMA_KEEP_ALIVE` | `5m0s` | idle time before the embedder is unloaded |
| `OLLAMA_CONTEXT_LENGTH` | `0` (auto) | ignored here — the embedder is pinned at its 2048 training context |

### `inference_embedding_ollama.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Ollama server host |
| `--port` | `11434` | Ollama server port |
| `--model` | `embeddinggemma` | model name created by `ollama create` |
| `--api` | `native` | `native` → `/api/embed`, `openai` → `/v1/embeddings` |
| `--queries` | 2 retrieval queries | query strings to embed |
| `--documents` | 4 documents | document strings to embed |
| `--no_prompt_template` | off | send raw text with no task prefixes — negative control |
| `--keep_alive` | `5m` | how long Ollama keeps the model resident |
| `--reference` | none | Transformers reference JSON to cross-check against |
| `--tolerance` | `0.01` | max allowed \|Δ cosine\| before the check FAILs |
| `--timeout` | `600` | HTTP timeout (s) |
| `--health_retries` | `60` | `/api/tags` probes before giving up |
| `--out` | none | write the results JSON |

## Notes

- Context is fixed at 2048 (EmbeddingGemma's training context) and `OLLAMA_CONTEXT_LENGTH`
  does not raise it. Longer inputs are truncated, so chunk before embedding.
- Raise `OLLAMA_KEEP_ALIVE` (or pass `--keep_alive 30m`) for a steady service, or the
  5-minute default unloads the model between batches and the next one pays a cold load.
- `/v1/embeddings` returns `"object": "list"` at the top level, not the `"object":
  "embedding"` some OpenAI clients expect there, and reports `usage` with `prompt_tokens`
  only. Vectors are identical to `/api/embed`.
