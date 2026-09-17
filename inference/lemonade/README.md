# `inference/lemonade` — one local server for LLM, embeddings and reranking

One Lemonade Server process serves all three workloads on **one port (8350)** with one model
registry — the only local stack here that covers completion, embeddings and reranking behind
a single API, with no build step. Leaves: [`llm/`](llm/) · [`embedding/`](embedding/) ·
[`reranker/`](reranker/).

The two vendors are mirror images: the backend is `llamacpp:rocm` on AMD and `llamacpp:cuda`
on NVIDIA (each reports `Unsupported GPU` on the other vendor). Install, registration,
serving and clients are otherwise identical. llama.cpp/GGUF is Lemonade's **only** GPU path:
`vllm:rocm` refuses gfx950 and there is no `vllm:cuda` backend, so use
[`../vllm`](../vllm) or [`../sglang`](../sglang) for a native-FP8 checkpoint.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4 host) · NVIDIA H100 (Hopper sm_90, CUDA 13
host). Lemonade brings its own runtime — ROCm 7.14 wheels or a bundled CUDA 12.9 — and
ignores the host's.

## Workloads

| Leaf | Model (GGUF) | Endpoint | Registration label |
|---|---|---|---|
| [`llm/`](llm/) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | `/v1/chat/completions` | none |
| [`embedding/`](embedding/) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | `/v1/embeddings` | `embeddings` |
| [`reranker/`](reranker/) | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | `/v1/reranking` | `reranking` |

## Install the server

Use the **C++ Lemonade Server 11.7.0 embeddable tarball**. The pip `lemonade-sdk` package
(9.x) is a different product with no `backends install` subcommand, and its ROCm backend
cannot resolve gfx950. Do not use the `.deb` either — it is built for Debian 13 and fails on
Ubuntu 24.04 with missing `libmbedcrypto.so.16` / `libcpp-httplib.so.0.41`.

```bash
export HF_HOME=/path/to/hf_cache       # model weights land in $HF_HOME/hub/
export DATA_DIR=/path/to/data          # Lemonade tarball, binaries and backend cache (~4.8 GB)
export OUTPUT_DIR=/path/to/outputs     # client artifacts

cd $DATA_DIR/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
```

Start the server, then install the backend for your vendor (arch-matched prebuilts, no
compiler; the CLI needs the server running):

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0      # set on lemond, it forks llama-server
$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast &

$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm    # AMD: pulls rocm_sdk_device_gfx950
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:cuda    # NVIDIA: llama-b10397-ubuntu-cuda-sm_90-x64
```

Confirm the choice in the `lemond` log (`Using LlamaCpp Backend: cuda` / `rocm`);
`lemonade backends --all` prints the whole support matrix.

Pass `--no-discovery` on every CLI call — without it the client broadcasts UDP looking for
beacons and can hang past 60 s on a shared host. Never set `CUDA_VISIBLE_DEVICES=""` on
ROCm; an empty string hides every device and the stack silently falls back to CPU.

## Client venv

```bash
cd inference/lemonade
python3 -m venv .env_lemonade && source .env_lemonade/bin/activate
pip install -r requirements.txt
```

This venv is only the HTTP clients; the runtime is the C++ server and its own managed
backend venv inside the Lemonade cache. Each leaf symlinks the repo-root secrets file
(`ln -sf ../../../dev.env dev.env`); `HF_TOKEN` is only needed for a gated checkpoint, and
all three models above are ungated. Build the venv on local disk or tmpfs — `python3 -m venv`
on an NFS mount does not create pip/console scripts reliably.

## Shared behaviour

- **The `user.` prefix is registration-only.** You register `user.Qwen38-Q8`; the API answers
  to `Qwen38-Q8`. Sending the prefixed id is a 404.
- **`--label embeddings` / `--label reranking` are mandatory at registration.** They drive
  `--embeddings` / `--reranking` on the wrapped `llama-server` and create the route.
- **Lemonade builds the `llama-server` command line itself** — no `-ngl`, `--tensor-split`
  or `--ctx-size` control. `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` on the `lemond`
  process is your only device control, so run one server per GPU rather than splitting.
- **Reasoning models plus a small `max_tokens` return empty `content`** — output goes to
  `reasoning_content`. Budget at least 900 tokens.
- **Startup warnings are harmless** for the embeddable build: `Could not load
  architecture_defaults.json`, `Web app directory not found`. Only the web UI is affected.
