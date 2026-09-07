# `inference/lemonade` — Lemonade Server on AMD MI355X

One Lemonade Server process serves **all three workloads on one port (8350)** — the only
local stack in this repo that covers completion, embeddings, and reranking behind a
single API. Leaves: [`llm/`](llm/) · [`embedding/`](embedding/) · [`reranker/`](reranker/).

| Leaf | Model | Status on MI355X (gfx950, ROCm 7.2.4) |
|---|---|---|
| [`llm/`](llm/) | `Qwen3.8-27B-Q8_0` GGUF | **works** — 143 tok/s single GPU with MTP speculative decoding |
| [`embedding/`](embedding/) | `embeddinggemma-300M` GGUF | **works** — dim 768, cosine 0.9996+ vs the Transformers baseline |
| [`reranker/`](reranker/) | `Qwen3-Reranker-0.6B` GGUF | **works** — `/v1/reranking`, ~28,000× score separation |

**NVIDIA H100 (Hopper, sm_90) — single-GPU smoke.** Same story, mirror
image: the accelerated backend that resolves this GPU is **`llamacpp:cuda`** (the ROCm
backend reports `Unsupported GPU`, exactly as vLLM does on gfx950). All three workloads run
on **one Lemonade Server on port 8350**, GPU-resident, via the CUDA llama.cpp backend. Small
GGUFs are enough for a fast smoke test:

| Leaf | Model (H100 smoke) | Status on H100 (sm_90, CUDA 13 host / bundled CUDA 12.9) |
|---|---|---|
| [`llm/`](llm/) | `Qwen3-0.6B-Q8_0` GGUF | **works** — 584 tok/s, coherent output, 1.67 GB on GPU 6 |
| [`embedding/`](embedding/) | `embeddinggemma-300M` GGUF | **works** — dim 768, related 0.29 > unrelated 0.10, 0.98 GB on GPU 6 |
| [`reranker/`](reranker/) | `Qwen3-Reranker-0.6B` GGUF | **works** — `/v1/reranking`, ~21,000× separation, 1.95 GB on GPU 6 |

> The H100 smoke deliberately uses **small** GGUFs (0.6B llm, 300M embed, 0.6B rerank), not
> the 27 GB Qwen3.8-27B, because the point is to prove the backend/GPU path, not raw
> throughput. The install and serving path is identical to the
> 27B; only the checkpoint size differs. See each leaf's **H100** section.

## Install (AMD / ROCm — the working route, and the trap)

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

**Lemonade is two different products sharing a name.** The pip package (`lemonade-sdk`,
v9.x) has no `backends` subcommand and its ROCm backend resolution **fails on gfx950**
(`ValueError: ... no compatible ROCm target architecture found`). The route that works is
the **C++ Lemonade Server 11.7.0 embeddable tarball**:

```bash
# 1. Download lemonade-embeddable-<ver>-ubuntu-x64.tar.gz from lemonade-sdk releases
#    (the .deb is Debian-13-only and does not load on Ubuntu 24.04).
# 2. Install the ROCm backend — it downloads an ARCH-MATCHED gfx950 wheel, no compiler:
lemonade --no-discovery backends install llamacpp:rocm    # ~27 s; rocm_sdk_device_gfx950
# 3. Register models (reuses a pre-seeded HF cache — see leaves) and serve on port 8350.
```

Always pass `--no-discovery` on CLI calls (otherwise it broadcasts UDP looking for
beacons and hangs). The experimental `vllm:rocm` backend **refuses gfx950**
(`Unsupported GPU: gfx950`) even though its own version pins list one — documented in
[`llm/README.md`](llm/README.md).

## Install (NVIDIA / CUDA — H100)

The install route is **identical** to AMD; only the backend name changes. The same C++
embeddable tarball, the same `backends install` command, `llamacpp:cuda` instead of
`llamacpp:rocm`:

```bash
# 1. Same embeddable tarball (the .deb still does not load on Ubuntu 24.04):
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
LEM=$PWD/emb/lemonade-embeddable-11.7.0-ubuntu-x64

# 2. Start the server (proxy UNSET so it can reach github/HF for backend+models):
export CUDA_VISIBLE_DEVICES=6   # HF_HOME as exported above
$LEM/lemond /dev/shm/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast &

# 3. Install the CUDA backend — arch-matched sm_90 prebuilt, no compiler, ~41 s:
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:cuda
#   [1/1] llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz   ->   Backend installed successfully
```

**How Lemonade selects the backend.** On startup `lemond` probes the machine and logs its
backend availability. On an H100 host:

```text
[Info] (ModelManager) Backend availability:
[Info] (ModelManager)   - NVIDIA GPU: NVIDIA H100 80GB HBM3 (compute 9.0, sm_90)
...
[Info] (LlamaCpp) Using LlamaCpp Backend: cuda
[Info] (LlamaCpp) Respecting existing CUDA_VISIBLE_DEVICES=6
```

`lemonade backends --all` then shows the per-recipe support matrix. The relevant rows on
H100 (the mirror image of the gfx950 matrix):

```text
Recipe     Backend  Status        Message/Version
llamacpp   cuda     installable   Backend is supported but not installed.   <- the GPU path
llamacpp   rocm     unsupported   Unsupported GPU                           <- (AMD-only)
llamacpp   vulkan   installable   Backend is supported but not installed.
llamacpp   cpu      installable   Backend is supported but not installed.
vllm       rocm     unsupported   Unsupported GPU                           <- NO cuda row
```

**Is an NVIDIA/CUDA GGUF path supported? Yes — via `llamacpp:cuda`, and it is a
first-class, versioned backend.** `resources/backend_versions.json` pins
`"llamacpp": {"cuda":"b10397", "rocm-stable":"b10470", "vulkan":..., "cpu":...}`. The CUDA
llama.cpp is a prebuilt `sm_90` (Hopper) tarball that **bundles its own CUDA 12.9 runtime**
(`libcublas.so.12.9.2.10`, `libcudart.so.12.9.79`) — it does not use the host's CUDA 13,
the same "brings its own runtime" pattern as the ROCm backend.

**The nuance / honest caveat.** Lemonade is an AMD-focused product (its accelerated tiers
are Ryzen AI NPU, ROCm, and Vulkan). On NVIDIA, its **only** GPU path is llama.cpp/GGUF —
there is **no `vllm:cuda`** backend at all (`vllm` lists only a `rocm` row, which itself
reports `Unsupported GPU` here). So on H100 Lemonade is a **CUDA llama.cpp front-end
only**: exactly symmetric to MI355X, where it is a ROCm llama.cpp front-end only and the
FP8/vLLM path is refused. GGUF completion/embedding/reranking all work and hit the GPU;
there is no native-FP8 / high-throughput-engine path through Lemonade on either vendor.

## Environment & secrets

Clients load `dev.env` from their own folder (leaves symlink the repo root:
`ln -sf ../../../dev.env dev.env`). Client venv:

```bash
cd inference/lemonade
python3 -m venv .env_lemonade && source .env_lemonade/bin/activate
pip install -r requirements.txt
```

## Shared quirks

- Pre-seeding the HF cache (`refs/main` + `snapshots/<sha>/` + registry json) makes
  `lemonade pull` skip re-downloads — avoids a 28 GB pull.
- `--label embeddings` / `--label reranking` are mandatory at registration; they drive
  `--embeddings` / `--reranking` on the wrapped llama-server.
- The `user.` prefix is registration-only — the API model id omits it.
- Reasoning models + small `max_tokens` return empty `content` (output goes to
  `reasoning_content`); budget ≥900 tokens or disable thinking.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4): verified** — all three workloads on one process.
- **NVIDIA:** not yet verified in this repo (upstream backend: `llamacpp:cuda`).
- **Other hardware (upstream claims — not verified here):** AMD Ryzen AI NPU and
  iGPU/APU targets, and Vulkan as a broad-compatibility GPU fallback — Lemonade's
  backend list; it is purpose-built for heterogeneous AMD client hardware.
