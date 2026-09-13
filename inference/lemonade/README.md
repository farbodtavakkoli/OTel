# `inference/lemonade` — Lemonade Server on AMD MI355X and NVIDIA H100

One Lemonade Server process serves **all three workloads on one port (8350)** — the only
local stack in this repo that covers completion, embeddings, and reranking behind a
single API. Leaves: [`llm/`](llm/) · [`embedding/`](embedding/) · [`reranker/`](reranker/).

The two vendors are mirror images: the accelerated backend is `llamacpp:rocm` on AMD and
`llamacpp:cuda` on NVIDIA (each reports `Unsupported GPU` on the other vendor). Everything
else — install route, registration, serving, clients — is identical.

| Leaf | Model | MI355X (gfx950, ROCm 7.2) | H100 (sm_90, CUDA 13 host / bundled CUDA 12.9) |
|---|---|---|---|
| [`llm/`](llm/) | `Qwen3.8-27B-Q8_0` GGUF | **works** — MTP speculative decoding supported | **works** |
| [`embedding/`](embedding/) | `embeddinggemma-300M` GGUF | **works** — dim 768 | **works** — dim 768 |
| [`reranker/`](reranker/) | `Qwen3-Reranker-0.6B` GGUF | **works** — `/v1/reranking` | **works** — `/v1/reranking` |

> The NVIDIA examples in the leaves use **small** GGUFs (`Qwen3-0.6B-Q8_0` for the LLM, the
> same 300M embed and 0.6B rerank models) so the GPU path can be exercised quickly. The
> install and serving path is identical for the 27B; only the checkpoint size differs. See
> each leaf's **H100** section.

## Install (AMD / ROCm — the working route)

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
lemonade --no-discovery backends install llamacpp:rocm    # pulls rocm_sdk_device_gfx950
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

# 2. Start the server (any proxy UNSET so it can reach github/HF for backend+models):
export CUDA_VISIBLE_DEVICES=0   # HF_HOME as exported above
$LEM/lemond /dev/shm/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast &

# 3. Install the CUDA backend — arch-matched sm_90 prebuilt, no compiler needed:
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:cuda
#   [1/1] llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz   ->   Backend installed successfully
```

Confirm the backend choice in the `lemond` startup log (`lemonade backends --all` lists the
whole matrix):

```text
[Info] (LlamaCpp) Using LlamaCpp Backend: cuda
[Info] (LlamaCpp) Respecting existing CUDA_VISIBLE_DEVICES=0
```

The CUDA llama.cpp is a prebuilt `sm_90` tarball that **bundles its own CUDA 12.9 runtime**,
so it does not use the host's CUDA 13 — the same pattern as the ROCm backend.

**Note.** On NVIDIA Lemonade's **only** GPU path is llama.cpp/GGUF — there is no
`vllm:cuda` backend. GGUF completion/embedding/reranking all work and hit the GPU, but there
is no native-FP8 / high-throughput-engine path through Lemonade on either vendor.

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

- **AMD MI355X (gfx950, ROCm 7.2.4)** — all three workloads on one process, via
  `llamacpp:rocm`.
- **NVIDIA H100 (Hopper sm_90, CUDA 13 host)** — all three workloads on one process, via
  `llamacpp:cuda`.
