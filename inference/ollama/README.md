# Ollama

**Ollama** is a model *appliance*, not just an engine: one daemon owns model management,
GPU discovery, automatic VRAM fitting, request queueing, and model lifetime
(auto-unload after `keep_alive`), and serves multiple models behind one endpoint — an LLM
and an embedder can run co-resident, both `100% GPU`. For datacenter-class serving use
`../vllm/` or `../sglang/`.

Ollama is a **GGUF path** — it does not consume HF FP8/safetensors checkpoints; both
leaves register already-cached GGUF conversions via `Modelfile` + `ollama create`.

## Hardware support

Both vendor images run out of the box — no build, no patch, and nothing hardware-specific
(on ROCm, not even `HSA_OVERRIDE_GFX_VERSION`). Each image bundles its own GPU userspace;
the host supplies only the kernel driver.

| Stack | Image |
|---|---|
| AMD ROCm (gfx950 / MI355X) | `ollama/ollama:rocm` |
| NVIDIA CUDA (H100, CUDA 13) | `ollama/ollama` |

Only the image tag and the device flags differ between the two; every `Modelfile`, API
call and client command is identical.

## Install — shared docker route (used by both leaves)

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data                # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
```

### AMD / ROCm

```bash
docker pull ollama/ollama:rocm
```

Pin the container to specific GPUs — the canonical run line passes all of `/dev/dri`,
which exposes **every GPU on the host**. Map GPU → render node first:

```bash
rocm-smi --showbus          # GPU[N] -> 0000:A5:00.0
ls -l /dev/dri/by-path/     # pci-0000:a5:00.0-render -> ../renderD144
```

```bash
# add a second --device /dev/dri/renderD<M> line for a two-GPU container
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  -v $DATA_DIR/ollama:/root/.ollama \
  -v $LLAMA_CACHE:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_llm \
  ollama/ollama:rocm
```

Two deliberate deviations from the canonical run line:

- **Bind-mount the model store to `$DATA_DIR/ollama`**, not a docker named volume —
  named volumes land under `/var/lib/docker` on the root filesystem, and `ollama create`
  **copies** GGUFs into the store.
- **Mount the HF GGUF cache read-only at `/ggufs`** so Modelfiles register cached files
  with zero network.

### NVIDIA / CUDA

```bash
docker pull ollama/ollama
```

If Docker Hub is proxy-blocked, pull with the proxy unset:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
sudo docker pull ollama/ollama
```

Pin with `--gpus '"device=N"'` — that alone confines Ollama to one card; it is the CUDA
analogue of ROCm's `--device /dev/dri/renderD*`:

```bash
GGUF=/path/to/ggufs               # where the downloaded GGUFs live
STORE=/dev/shm/ollama/store       # model store on tmpfs (NOT on /), ollama create COPIES
sudo docker run -d --gpus '"device=0"' \
  -e OLLAMA_HOST=0.0.0.0:11440 \
  -v $GGUF:/ggufs:ro \
  -v $STORE:/root/.ollama \
  -p 127.0.0.1:11440:11440 \
  --name ollama_llm ollama/ollama
```

Deviations from the ROCm run line:

- **A non-default port** (`OLLAMA_HOST=0.0.0.0:11440` + `-p 127.0.0.1:11440:11440`) avoids
  collisions with other tenants on a shared host; 11434 is the default.
- **Model store on `/dev/shm` tmpfs.** Same reason as ROCm (`ollama create` copies, keep it
  off the root filesystem); tmpfs also dodges the `Operation not permitted` atomic-rename
  failure some network filesystems throw at pip/venv.

### Populating the GGUF cache

The Modelfiles' `FROM` paths point into `$LLAMA_CACHE` / `/ggufs`; when that cache is empty,
fetch the GGUFs first (proxy unset) and point `FROM` at the result:

```bash
hf download <repo> <file.gguf> --local-dir <dir>
```

For a quick smoke test, a small GGUF of the same family works as a drop-in:
`unsloth/Qwen3-0.6B-GGUF:Q8_0` in place of `unsloth/Qwen3.8-27B-GGUF:Q8_0`, and
`ggml-org/embeddinggemma-300M-GGUF:Q8_0` for the embedder.

## Expected output

`ollama ps` with both models resident:

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
qwen3-0.6b-q8:latest     605b58ae76ea    5.6 GB    100% GPU     40960      3 minutes from now
```

## Environment & secrets

`dev.env` in each leaf is a symlink to the repo-root `dev.env` (from a leaf:
`ln -sf ../../../dev.env dev.env`) supplying `HF_TOKEN`. **Neither leaf actually needs
the token**: both register GGUFs already on disk, so no credential ever reaches the
container. The token would only matter for `ollama pull hf.co/<gated-repo>`.

## Scope note — deliberately no `reranker/` leaf

Ollama is ❌ for the reranker workload:
Ollama's HTTP surface is `/api/generate`, `/api/chat`, `/api/embed` plus the OpenAI shim —
**there is no rerank route**, and a cross-encoder cannot be emulated by embedding both
sides and taking a cosine. For reranking use
[`../llamacpp/reranker/`](../llamacpp/reranker/) (llama.cpp `/v1/rerank` — the local/GGUF
answer) or [`../vllm/reranker/`](../vllm/reranker/) (the production GPU-serving answer).

## Leaves

| Leaf | Model | Status |
|---|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | ✅ **works** — single- and multi-GPU, `100% GPU` residency. |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | ✅ **works** — task prefixes must be applied client-side (the one correctness trap). |

## Shared quirks (both leaves)

- **`ollama create` copies, it does not reference** — budget the store accordingly, and
  keep it off the root filesystem.
- **`PROCESSOR: 100% GPU` coexists with a non-zero `CPU_Mapped` buffer** — the percentage
  refers to layer offload, not every byte.
- **Outbound calls to ollama.com may fail on an air-gapped or proxied host, and are
  harmless** (model-recommendation refresh only).
- **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — device selection is done purely by
  which `renderD*` nodes are passed into the container.
- The 5-minute `keep_alive` default unloads idle models; raise `OLLAMA_KEEP_ALIVE` for a
  steady service.

## Requirements & venv

Client deps (both leaves are dependency-light — plain JSON over HTTP): see
[`requirements.txt`](requirements.txt). Venv convention: `python3 -m venv .env_ollama` at
this software root.
