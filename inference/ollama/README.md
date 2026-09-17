# `inference/ollama` — Ollama GGUF serving on ROCm and CUDA

Ollama is a model appliance rather than a bare engine: one daemon owns model management, GPU
discovery, VRAM fitting, request queueing and model lifetime, and serves several models
behind one endpoint — a 27B LLM and a 300M embedder can be co-resident, both at `100% GPU`.
For datacenter-class serving use [`../vllm/`](../vllm/) or [`../sglang/`](../sglang/).

Ollama is a **GGUF path**: it does not consume HF FP8/safetensors checkpoints. Both leaves
register an already-cached GGUF via `Modelfile` + `ollama create`.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2) and NVIDIA H100 80GB (CUDA 13). Both
vendor images run out of the box — no build, no patch, and on ROCm not even
`HSA_OVERRIDE_GFX_VERSION`. Each image bundles its own GPU userspace; the host supplies only
the kernel driver.

| Stack | Image |
|---|---|
| AMD ROCm (gfx950 / MI355X) | `ollama/ollama:rocm` |
| NVIDIA CUDA (H100, CUDA 13) | `ollama/ollama` |

Only the image tag and the device flags differ; every Modelfile, API call and client command
is identical.

## Leaves

| Leaf | Model |
|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` — single- and multi-GPU |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` — task prefixes must be applied client-side |

There is no `reranker/` leaf: Ollama's HTTP surface is `/api/generate`, `/api/chat`,
`/api/embed` plus the OpenAI shim, with no rerank route, and a cross-encoder cannot be
emulated by embedding both sides and taking a cosine. Use
[`../llamacpp/reranker/`](../llamacpp/reranker/) or [`../vllm/reranker/`](../vllm/reranker/).

## Setup

```bash
# Set these to suit your machine
export DATA_DIR=/path/to/data                    # Ollama model store (ollama create COPIES)
export LLAMA_CACHE=/path/to/hf_cache/llama_cpp   # existing GGUF cache, mounted read-only
```

### AMD / ROCm

```bash
docker pull ollama/ollama:rocm
```

Pin the container to specific GPUs — the canonical run line passes all of `/dev/dri`, which
exposes every GPU on the host. Map GPU to render node first:

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

Bind-mount the model store rather than using a docker named volume (those land under
`/var/lib/docker` on the root filesystem, and `ollama create` copies GGUFs in), and mount the
HF GGUF cache read-only at `/ggufs` so Modelfiles register cached files with zero network.

### NVIDIA / CUDA

```bash
docker pull ollama/ollama
# if Docker Hub is proxy-blocked:
#   unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy && sudo docker pull ollama/ollama
```

`--gpus '"device=N"'` alone confines Ollama to one card — the CUDA analogue of ROCm's
`--device /dev/dri/renderD*`:

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

The non-default port (11434 is the default) avoids collisions with other tenants on a shared
host; tmpfs for the store also dodges the `Operation not permitted` atomic-rename failure
some network filesystems throw.

### Populating the GGUF cache

The Modelfiles' `FROM` paths point into `$LLAMA_CACHE` / `/ggufs`. When that cache is empty,
fetch the GGUF first (proxy unset) and point `FROM` at the result:

```bash
hf download <repo> <file.gguf> --local-dir <dir>
```

For a smoke test, `unsloth/Qwen3-0.6B-GGUF:Q8_0` is a drop-in for
`unsloth/Qwen3.8-27B-GGUF:Q8_0`.

### Client venv

```bash
python3 -m venv .env_ollama
.env_ollama/bin/pip install -r requirements.txt
```

## Environment & secrets

Symlink the repo-root `dev.env` in each leaf (`ln -sf ../../../dev.env dev.env`) to supply
`HF_TOKEN`. Neither leaf needs it — both register GGUFs already on disk, so no credential
reaches the container. The token matters only for `ollama pull hf.co/<gated-repo>`.

## Notes

- `ollama create` copies, it does not reference — budget the store accordingly and keep it
  off the root filesystem.
- `PROCESSOR: 100% GPU` coexists with a non-zero `CPU_Mapped` buffer; the percentage refers
  to layer offload, not every byte.
- Raise `OLLAMA_KEEP_ALIVE` for a steady service — the 5-minute default unloads idle models.
- Device selection is done purely by which `renderD*` nodes (or `--gpus`) are passed into the
  container. Never set `CUDA_VISIBLE_DEVICES=""` on ROCm.
- Outbound calls to ollama.com fail harmlessly on an air-gapped or proxied host — that is the
  model-recommendation refresh, not serving.
