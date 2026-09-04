# `inference/llamacpp` — llama.cpp GGUF serving on ROCm/HIP (MI355X)

llama.cpp serves **GGUF** checkpoints through `llama-server`, its OpenAI-compatible
HTTP server, as a single self-contained C++ binary — no python runtime, no torch, no
vLLM/SGLang install surface in the serving path. **One HIP build serves all three
leaves in this tree** (LLM, embedding, reranker): the same `build/bin/llama-server`
binary runs chat completions, `/v1/embeddings`, and `/v1/rerank` depending on flags.

> **Format note:** llama.cpp does **not** consume HF safetensors. Every leaf serves a
> converted **GGUF** artifact (`unsloth/...-GGUF`, `ggml-org/...-GGUF`) — not the
> official FP8/FP16 repos.

> **Tested topology:** 2xAMD Instinct MI355X (gfx950, 288 GB each), physical GPUs 6
> and 7, ROCm 7.2.4, Ubuntu, Python 3.12.3. Verified **2026-08-20**.
>
> **Also verified on NVIDIA H100** (1x H100 80GB HBM3, physical GPU 7, CUDA 13.0,
> driver 580.173.02, Hopper cc 9.0, Python 3.12.3) — single-GPU, **2026-08-22**. Only the
> backend build flag changes (`-DGGML_CUDA=ON`); every serve/client command is identical.
> See "NVIDIA (H100 / CUDA) — verified" below and each leaf's H100 section.

## Leaves

| Leaf | Model | MI355X verdict | H100 verdict |
|---|---|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | **PASS** — 66/66 layers on GPU, 66.7 tok/s single-GPU; layer split works (65.9 tok/s), `--split-mode row` broken on HIP | **PASS** — 29/29 layers on CUDA0 (verified with small `unsloth/Qwen3-1.7B-GGUF:Q8_0`, 370 tok/s); swap `-hf` back to the 27B for prod |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | **PASS** — 768-d L2-normalised vectors, ~50 ms/3-text batch; scale out with one instance per GPU, don't split | **PASS** — same GGUF, 25/25 layers on CUDA0 (311.97 MiB, identical buffer), 768-d, related 0.29 > unrelated 0.10 |
| [`reranker/`](reranker/README.md) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | **PASS** — correct rankings, 16 ms warm; one of the few working reranker paths on AMD (TEI lacks this model) | **PASS** — same GGUF, 29/29 layers on CUDA0 (603.87 MiB, identical buffer), correct rankings, 28 ms warm |

## Build — the exact ROCm/HIP commands that worked

llama.cpp is a C++ build, not a pip package. The campaign's build tree was removed
with the venvs in the 2026-08 reorg; rebuild it anywhere convenient (e.g. a
git-ignored directory under this folder) — the documented **40.6 s** cold rebuild
applies.

### 0. Build prerequisites (apt)

```bash
sudo apt-get install -y libssl-dev        # MANDATORY — see "OpenSSL" quirk below
# cmake 3.28.3, ninja 1.11.1, g++ 13.3.0 and ROCm 7.2.4 were already present
```

### 1. Clone

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
```

Verified at commit **`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`** (Wed Aug 19 2026),
`llama-server` version **0.1.2-dev (build 1)**, ggml version **0.20.2**.

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

**Build time: 40.6 s wall clock** for a genuinely cold build (681 ninja targets,
18 m 26 s of CPU time across `-j 32`; ccache reported 99.28 % misses). The build tree
is ~514 MB. **No source patches were needed for gfx950** — the HIP backend compiled
clean on a brand-new architecture, first try.

### Which arch flag?

`GPU_TARGETS` is the canonical variable in this revision; `AMDGPU_TARGETS` still
works but is merely forwarded (`ggml/src/ggml-hip/CMakeLists.txt:36`). Omitting both
also builds — ggml then targets every GPU present, which just makes the build slower.

### Confirm the backend sees your GPUs

```bash
export HIP_VISIBLE_DEVICES=6,7 CUDA_VISIBLE_DEVICES=6,7
./build/bin/llama-server --list-devices
```

```text
Available devices:
  ROCm0: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
  ROCm1: AMD Instinct MI355X (294896 MiB, 294310 MiB free)
```

### NVIDIA (H100 / CUDA) — verified 2026-08-22

Verified on **1x NVIDIA H100 80GB HBM3** (physical GPU 7, `CUDA_VISIBLE_DEVICES=7`),
**CUDA 13.0** (`nvcc` at `/usr/local/cuda`), driver **580.173.02**, Hopper cc 9.0,
Python 3.12.3. **Only the backend flag changes** — swap `-DGGML_HIP=ON -DGPU_TARGETS=gfx950`
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

**Build time: 100 s wall clock** for a cold build (691 ninja targets, `-j 32`; the
`fattn-*` CUDA flash-attention template instances dominate). **No source patches** —
cmake auto-detected `CMAKE_CUDA_ARCHITECTURES=90-real` (Hopper) and the CUDA backend
compiled clean, first try. Configure logs confirm the vendor-neutral OpenSSL fix took:
`OpenSSL found: 3.0.13`. Verified `llama-server` **0.2.0-dev (build 1, commit `70adb1b`)**,
ggml **0.21.0**.

> **Two prerequisites this box did NOT have (both fixed without touching the system):**
> 1. **`cmake` and `ninja` were absent** (only `make`/`g++` 13.3.0). Installed via
>    `pip` into a throwaway venv — `python3 -m venv <venv> && <venv>/bin/python -m pip
>    install cmake ninja` (gives cmake 4.4.2 + ninja 1.13.0) — then put `<venv>/bin` on
>    `PATH`. A prebuilt CUDA `llama.cpp` release binary is the documented fallback if you
>    cannot get a toolchain, but the source build here finished in 100 s so it was
>    unnecessary.
> 2. **The `/mnt/gsma` share rejects pip installs and `-hf` downloads** with
>    `OSError: [Errno 1] Operation not permitted` on rename. Put both the toolchain venv
>    **and** `LLAMA_CACHE` on tmpfs (`/dev/shm/...`) — the build tree and model cache live
>    there, `HF_HOME` still points at the read-only model share.

**Confirm the backend sees your GPU** (renumbered to `CUDA0` under
`CUDA_VISIBLE_DEVICES=7`, exactly as ROCm renumbers to `ROCm0`):

```bash
export CUDA_VISIBLE_DEVICES=7
./build/bin/llama-server --list-devices
```

```text
Available devices:
  CUDA0: NVIDIA H100 80GB HBM3 (81079 MiB, 80552 MiB free)
```

All three leaves were served on **port 8700** and passed with all layers on the GPU
(`offloaded N/N layers to GPU`, weights in the `CUDA0` buffer, and `nvidia-smi -i 7`
showing `llama-server` by PID). See each leaf's "H100 (NVIDIA, CUDA)" section for the
exact residency logs, the real output, and the per-leaf verdict.

> **Do not set `CUDA_VISIBLE_DEVICES=""`** — like the ROCm caveat below, an empty string
> hides every device and the server silently runs on CPU.

## Environment & secrets

```bash
ln -sf ../../dev.env dev.env                    # at this software root (already done)
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

Leaves symlink the same file as `ln -sf ../../../dev.env dev.env`. None of the three
models served here are gated — no `HF_TOKEN` was needed. Never echo or commit
`HF_TOKEN`.

Model weights must land on `/mnt`, not on `/`:

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache
export LLAMA_CACHE=/mnt/data_1.5t/hf_cache/llama_cpp   # llama.cpp's own -hf cache
```

`LLAMA_CACHE` is the one that actually matters for `-hf`: llama.cpp keeps its own
download cache and ignores `HF_HOME` for this path. Set both.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Client dependencies

The clients are deliberately dependency-light (`requests` + `python-dotenv` — the
server speaks plain OpenAI-style REST). One shared venv at this software root covers
all three leaves:

```bash
python3 -m venv .env_llamacpp
.env_llamacpp/bin/pip install -r requirements.txt
```

The per-leaf campaign venvs were removed in the 2026-08 reorg; rebuild from
`requirements.txt` here.

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
   or the weights land in `~/.cache` on `/`.
4. **`--split-mode row` is broken on the CUDA/HIP backend** in this revision — the
   backend registers no `ggml_backend_split_buffer_type`. Only `layer` (the default)
   and `none` are usable. See the LLM leaf for the source trace.
5. **Multi-GPU layer split is pipeline parallelism** — it buys capacity, not speed.
   For small models (embedding, reranker), run one independent instance per GPU
   behind a load balancer instead of splitting.

## Hardware support

Verified here: AMD Instinct MI355X (gfx950), ROCm 7.2.4 / HIP.

Other hardware (upstream claims — not verified here): NVIDIA CUDA, Apple Metal
(Apple Silicon first-class), Vulkan (cross-vendor GPU), SYCL (Intel GPU), OpenCL
(Qualcomm Adreno), Moore Threads MUSA, Huawei Ascend CANN, IBM zDNN, plus plain CPU
(AVX/NEON) and CPU+GPU hybrid offload.
