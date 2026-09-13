# `inference/llamacpp` — llama.cpp GGUF serving on ROCm/HIP and CUDA

llama.cpp serves **GGUF** checkpoints through `llama-server`, its OpenAI-compatible
HTTP server, as a single self-contained C++ binary — no python runtime, no torch, no
vLLM/SGLang install surface in the serving path. **One HIP build serves all three
leaves in this tree** (LLM, embedding, reranker): the same `build/bin/llama-server`
binary runs chat completions, `/v1/embeddings`, and `/v1/rerank` depending on flags.

> **Format note:** llama.cpp does **not** consume HF safetensors. Every leaf serves a
> converted **GGUF** artifact (`unsloth/...-GGUF`, `ggml-org/...-GGUF`) — not the
> official FP8/FP16 repos.

> **Coverage:** all three leaves serve on AMD Instinct MI355X (gfx950, ROCm 7.2) and on
> NVIDIA H100 (Hopper cc 9.0, CUDA 13), Python 3.12. Only the backend build flag changes
> (`-DGGML_HIP=ON -DGPU_TARGETS=gfx950` vs `-DGGML_CUDA=ON`); every serve and client
> command is identical. See "NVIDIA (H100 / CUDA)" below and each leaf's H100 section.

## Leaves

| Leaf | Model | Notes |
|---|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | Works on ROCm and CUDA, all layers offloaded. Multi-GPU layer split works; `--split-mode row` is broken on HIP/CUDA. `unsloth/Qwen3-1.7B-GGUF:Q8_0` is a handy small stand-in for smoke tests. |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | Works on ROCm and CUDA; 768-d L2-normalised vectors. Scale out with one instance per GPU rather than splitting. |
| [`reranker/`](reranker/README.md) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Works on ROCm and CUDA with correct rankings. |

## Build — the ROCm/HIP commands

llama.cpp is a C++ build, not a pip package. Build it anywhere convenient (e.g. a
git-ignored directory under this folder); a cold build takes well under two minutes
with `-j 32`.

### 0. Build prerequisites (apt)

```bash
sudo apt-get install -y libssl-dev        # MANDATORY — see "OpenSSL" quirk below
# plus cmake 3.28.3, ninja 1.11.1, g++ 13.3.0 and ROCm 7.2.4 at /opt/rocm
```

### 1. Clone

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
```

Reference revision for the ROCm recipe: commit
**`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`**, `llama-server` **0.1.2-dev (build 1)**,
ggml **0.20.2**.

### 2. Configure + build (gfx950)

```bash
export PATH=/opt/rocm/bin:$PATH

cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=ON

cmake --build build --config Release -j 32
```

The build tree is ~514 MB. **No source patches are needed for gfx950** — the HIP
backend compiles clean.

### Arch flag

Use `GPU_TARGETS`. `AMDGPU_TARGETS` is only forwarded to it, and omitting both builds for
every GPU present — correct, but slower to compile.

### Confirm the backend sees your GPUs

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
./build/bin/llama-server --list-devices
```

```text
Available devices:
  ROCm0: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
  ROCm1: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
```

### NVIDIA (H100 / CUDA)

Supported on NVIDIA H100 (Hopper cc 9.0) with **CUDA 13.0** (`nvcc` at
`/usr/local/cuda`). **Only the backend flag changes** — swap `-DGGML_HIP=ON -DGPU_TARGETS=gfx950`
for **`-DGGML_CUDA=ON`**, drop the ROCm PATH, and use plain `CUDA_VISIBLE_DEVICES`. Every
serve/client command in the leaves is byte-for-byte identical. **Keep `-DLLAMA_OPENSSL=ON`
— it is vendor-neutral** and still required for the `-hf` HTTPS downloader (see NOTE 2 in
`requirements.txt`).

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -S . -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_CURL=OFF          # force the OpenSSL/cpp-httplib -hf path (optional)

cmake --build build --config Release -j 32
```

**No source patches** — cmake auto-detects `CMAKE_CUDA_ARCHITECTURES=90-real` for Hopper
(set `-DCMAKE_CUDA_ARCHITECTURES=90` to pin it) and the CUDA backend compiles clean. The
configure log must print `OpenSSL found: 3.0.13`, confirming the vendor-neutral OpenSSL
requirement is satisfied. Reference revision: `llama-server` **0.2.0-dev (build 1, commit
`70adb1b`)**, ggml **0.21.0**.

> **Two prerequisites a host may be missing (both fixable without touching the system):**
> 1. **`cmake` and `ninja` absent** (only `make`/`g++` 13.3.0). Install them via
>    `pip` into a throwaway venv — `python3 -m venv <venv> && <venv>/bin/python -m pip
>    install cmake ninja` (gives cmake 4.4.2 + ninja 1.13.0) — then put `<venv>/bin` on
>    `PATH`. A prebuilt CUDA `llama.cpp` release binary is the documented fallback if you
>    cannot get a toolchain, but the source build is quick enough to rarely need it.
> 2. **A network/NFS model share may reject pip installs and `-hf` downloads** with
>    `OSError: [Errno 1] Operation not permitted` on rename. Put both the toolchain venv
>    **and** `LLAMA_CACHE` on tmpfs (`/dev/shm/...`) — the build tree and model cache live
>    there, `HF_HOME` still points at the read-only model share.

**Confirm the backend sees your GPU.** Whichever physical index you select, the visible
device is renumbered to `CUDA0` — exactly as ROCm renumbers to `ROCm0`:

```bash
export CUDA_VISIBLE_DEVICES=0
./build/bin/llama-server --list-devices
```

```text
Available devices:
  CUDA0: NVIDIA H100 80GB HBM3 (81079 MiB, 80552 MiB free)
```

All three leaves serve on **port 8700** with all layers on the GPU
(`offloaded N/N layers to GPU`, weights in the `CUDA0` buffer, visible in `nvidia-smi`).
See each leaf's "H100 (NVIDIA, CUDA)" section for the expected output.

> **Do not set `CUDA_VISIBLE_DEVICES=""`** — like the ROCm caveat below, an empty string
> hides every device and the server silently runs on CPU.

## Environment & secrets

```bash
ln -sf ../../dev.env dev.env                    # at this software root (already done)
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

Leaves symlink the same file as `ln -sf ../../../dev.env dev.env`. None of the three
models served here are gated — no `HF_TOKEN` is needed. Never echo or commit
`HF_TOKEN`.

Point the model caches at a filesystem with room, not at the root filesystem:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache             # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp        # llama.cpp's own -hf cache
```

`LLAMA_CACHE` is the one that actually matters for `-hf`: llama.cpp keeps its own
download cache and ignores `HF_HOME` for this path. Set both.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Client dependencies

The clients are dependency-light (`requests` + `python-dotenv` — the
server speaks plain OpenAI-style REST). One shared venv at this software root covers
all three leaves:

```bash
python3 -m venv .env_llamacpp
.env_llamacpp/bin/pip install -r requirements.txt
```

## Shared quirks (all three leaves)

1. **`-DLLAMA_OPENSSL=ON` + `libssl-dev` are mandatory for `-hf`.** This revision
   replaced libcurl with bundled cpp-httplib, which needs a TLS provider. Without it
   the build succeeds but every `-hf` pull fails with
   `get_repo_commit: error: HTTPS is not supported`, followed by the misleading
   `failed to load model ''`. Installing `libcurl4-openssl-dev` does **not** help.
2. **Default log verbosity hides the GPU-offload proof.** At the default `-lv 3` you
   get "model loaded" and nothing else. Use `-lv 5` to see
   `offloaded N/N layers to GPU` and per-device buffer sizes, then drop it.
3. **`LLAMA_CACHE`, not `HF_HOME`, controls where `-hf` writes.** Set it explicitly
   or the weights land in `~/.cache` on the root filesystem.
4. **`--split-mode row` is broken on the CUDA/HIP backend** in this revision — the
   backend registers no `ggml_backend_split_buffer_type`. Only `layer` (the default)
   and `none` are usable. See the LLM leaf for the source trace.
5. **Multi-GPU layer split is pipeline parallelism** — use it for capacity. For small models
   (embedding, reranker), run one independent instance per GPU behind a load balancer
   instead of splitting.

## Hardware support

Covered by the recipes here: AMD Instinct MI355X (gfx950) on ROCm 7.2 / HIP, and NVIDIA
H100 (Hopper cc 9.0) on CUDA 13.
