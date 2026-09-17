# `inference/llamacpp` — llama.cpp GGUF serving on ROCm/HIP and CUDA

`llama-server` is llama.cpp's OpenAI-compatible HTTP server: a single self-contained C++
binary, no python runtime and no torch in the serving path. **One build serves all three
leaves** — chat completions, `/v1/embeddings` and `/v1/rerank` are flag choices on the same
`build/bin/llama-server`.

llama.cpp consumes **GGUF** only, never HF safetensors, so every leaf serves a converted
GGUF repo (`unsloth/...-GGUF`, `ggml-org/...-GGUF`) rather than the official FP8/FP16 one.

**Hardware:** AMD Instinct MI355X (gfx950, ROCm 7.2) and NVIDIA H100 (Hopper cc 9.0,
CUDA 13), Python 3.12. Only the backend build flag changes; every serve and client command
is identical across the two.

## Leaves

| Leaf | Model | Port | Notes |
|---|---|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | 8200 | All layers offloaded. Multi-GPU layer split works. `unsloth/Qwen3-1.7B-GGUF:Q8_0` is a small stand-in for smoke tests |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | 8201 | 768-d L2-normalised vectors. Scale out one instance per GPU |
| [`reranker/`](reranker/README.md) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | 8202 | `/v1/rerank`, `/rerank`, `/v1/reranking` |

## Build

llama.cpp is a C++ build, not a pip package. Build it anywhere convenient — a git-ignored
directory under this folder works. Verified revisions: ROCm commit
`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd` (`llama-server` 0.1.2-dev, ggml 0.20.2); CUDA
commit `70adb1b` (`llama-server` 0.2.0-dev, ggml 0.21.0).

### Prerequisites

```bash
sudo apt-get install -y libssl-dev        # mandatory: the -hf downloader needs a TLS provider
# plus cmake 3.28.3, ninja 1.11.1, g++ 13.3.0, and ROCm 7.2.4 at /opt/rocm for the AMD build
```

If `cmake`/`ninja` are missing, pip-install them into a throwaway venv and put its `bin` on
`PATH`. If your model share is NFS and rejects renames, put the build tree and
`LLAMA_CACHE` on tmpfs (`/dev/shm/...`).

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
```

### AMD / ROCm (gfx950)

```bash
export PATH=/opt/rocm/bin:$PATH

cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=ON

cmake --build build --config Release -j 32
```

Use `GPU_TARGETS`, not `AMDGPU_TARGETS` — the latter is only forwarded to it, and omitting
both builds for every GPU present. No source patches are needed for gfx950.

### NVIDIA / CUDA (Hopper cc 9.0)

Only the backend flag changes. Keep `-DLLAMA_OPENSSL=ON` — it is vendor-neutral and still
required for `-hf`. cmake auto-detects `CMAKE_CUDA_ARCHITECTURES=90-real`; pass
`-DCMAKE_CUDA_ARCHITECTURES=90` to pin it.

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -S . -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_CURL=OFF          # optional: forces the OpenSSL/cpp-httplib -hf path

cmake --build build --config Release -j 32
```

The configure log must print `OpenSSL found: 3.0.13`.

### Verify the backend sees your GPUs

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
./build/bin/llama-server --list-devices
```

```text
Available devices:
  ROCm0: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
  ROCm1: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
```

Visible devices are renumbered from 0 (`ROCm0`/`CUDA0`) whatever physical index you select.

## Environment

```bash
ln -sf ../../dev.env dev.env                    # leaves use ../../../dev.env
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN

export HF_HOME=/path/to/hf_cache                # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp           # llama.cpp's own -hf cache
```

Set both: `LLAMA_CACHE` is what `-hf` actually writes to — llama.cpp ignores `HF_HOME` on
that path, and unset it drops weights in `~/.cache` on the root filesystem. None of the
three models here are gated, so no `HF_TOKEN` is needed. Never echo or commit it.

Never set `CUDA_VISIBLE_DEVICES=""` on either backend — an empty string hides every device
and the server silently runs on CPU.

## Client venv

The clients are dependency-light (`requests` + `python-dotenv`). One venv at this stack
root covers all three leaves:

```bash
python3 -m venv .env_llamacpp
.env_llamacpp/bin/pip install -r requirements.txt
```

## Notes

- **`-DLLAMA_OPENSSL=ON` + `libssl-dev` are mandatory for `-hf`.** Without them the build
  succeeds but every pull fails with `get_repo_commit: error: HTTPS is not supported`,
  followed by a misleading `failed to load model ''`. `libcurl4-openssl-dev` does not help.
- **Use `-lv 5` to prove GPU offload.** The default `-lv 3` hides the
  `offloaded N/N layers to GPU` line and per-device buffer sizes.
- **`--split-mode row` is unusable on CUDA and HIP** in these revisions — the backend
  registers no split buffer type. Only `layer` (the default) and `none` work.
- **Multi-GPU layer split is pipeline parallelism** — it buys capacity, not speed. For the
  small embedding and reranker models, run one instance per GPU behind a load balancer.
