# Ollama on MI355X — works out of the box (`ollama/ollama:rocm`)

**Ollama** is a model *appliance*, not just an engine: one daemon owns model management,
GPU discovery, automatic VRAM fitting, request queueing, and model lifetime
(auto-unload after `keep_alive`), and serves multiple models behind one endpoint —
the 27B LLM and the 300M embedder here run co-resident, both `100% GPU`. It is the
least-friction local serving route in this repo; it is **not** the datacenter-throughput
route (single-host by design, layer splitting only — capacity, not speed). For
MI355X-class throughput use `../vllm/` or `../sglang/`.

**The MI355X route that worked:** the official `ollama/ollama:rocm` image (0.32.14,
digest `sha256:b5d0813b2354...`, 1.43 GB pull / 4.55 GB on disk) ran gfx950 **out of the
box** — no build, no patch, no `HSA_OVERRIDE_GFX_VERSION`. The image bundles its own
ROCm 7.2 userspace (`libdirs=ollama,rocm_v7_2`); the host supplies only the amdgpu KFD
driver. Ollama is a **GGUF path** — it does not consume HF FP8/safetensors checkpoints;
both leaves register already-cached GGUF conversions via `Modelfile` + `ollama create`.

## Install — shared docker route (hoisted from both leaves)

```bash
docker pull ollama/ollama:rocm
```

Pin containers to specific GPUs — the canonical run line passes all of `/dev/dri`,
which would expose **all 8 GPUs** and trample other users. Map physical GPU → render node:

```bash
rocm-smi --showbus          # GPU[2] -> 0000:A5:00.0 , GPU[3] -> 0000:DC:00.0
ls -l /dev/dri/by-path/     # pci-0000:a5:00.0-render -> ../renderD144
                            # pci-0000:dc:00.0-render -> ../renderD152
```

```bash
# add a second --device /dev/dri/renderD152 line for a two-GPU container
docker run -d \
  --device /dev/kfd \
  --device /dev/dri/renderD144 \
  -v /mnt/data_450g/ollama:/root/.ollama \
  -v /mnt/data_1.5t/hf_cache/llama_cpp:/ggufs:ro \
  -p 11434:11434 \
  --name ollama_llm \
  ollama/ollama:rocm
```

Two deliberate deviations from the canonical run line:

- **Bind-mount the model store to `/mnt/data_450g/ollama`**, not a docker named volume —
  named volumes land under `/var/lib/docker` on `/` (~99 GB free), and `ollama create`
  **copies** GGUFs into the store (55 GB after both models).
- **Mount the HF GGUF cache read-only at `/ggufs`** so Modelfiles register cached files
  with zero network.

NVIDIA variant (not tested here — no NVIDIA GPU): only the image tag and device flags
change (`--gpus=all`, image `ollama/ollama`); every Modelfile and API command is identical.

---

## H100 (NVIDIA) — verified 2026-08-22 (`ollama/ollama` CUDA container)

Verified the same route on **NVIDIA H100 80GB HBM3** (Hopper cc 9.0, driver **580.173.02**,
CUDA 13.0). The MI355X claim above — "NVIDIA variant: only the image tag and device flags
change; every Modelfile and API command is identical" — **holds exactly.** The `ollama/ollama`
CUDA image ran H100 out of the box: no build, no patch, and (unlike ROCm's optional
`HSA_OVERRIDE_GFX_VERSION`) nothing hardware-specific at all. GPU discovery reports
`library=CUDA compute=9.0 name=CUDA0 ... libdirs=ollama,cuda_v13 driver=13.0` — a native
CUDA-13 userspace bundled in the image, host supplies only the NVIDIA driver.

**Container:** `ollama/ollama` (Docker Hub `:latest`), digest
`sha256:57d60e686821ea81a7748a3ec8141308c8b8f95b27105713954abf7a6529e700`, 8.43 GB on disk,
**Ollama 0.32.15**. Pull with the proxy unset (Docker Hub is proxy-blocked on this box):
`unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy && sudo docker pull ollama/ollama`.

**GGUFs had to be downloaded — nothing was cached.** The Modelfiles' `FROM` paths point at
`/mnt/data_1.5t/hf_cache/llama_cpp/...`, an **MI355X-era path that does not exist on this box**,
and no `*-GGUF` repos are in `HF_HOME`. So the GGUFs were fetched fresh (proxy unset,
`hf download <repo> <file.gguf> --local-dir <dir>`). For a fast single-GPU smoke this wave
used **small** GGUFs rather than the 29 GB Qwen3.8-27B:

| Leaf | GGUF used on H100 | Size | Why |
|---|---|---|---|
| `llm/` | `unsloth/Qwen3-0.6B-GGUF:Q8_0` (`Qwen3-0.6B-Q8_0.gguf`) | 639 MB | same Qwen3 family + Q8_0 as the 27B, fast pull/load for a smoke |
| `embedding/` | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | 334 MB | the **exact** repo the Modelfile references (layer sha `b5ce9d77…` identical) |

The 29 GB `unsloth/Qwen3.8-27B-GGUF:Q8_0` also fits an 80 GB H100 (weights ~26 GB + KV);
it was skipped only to keep this wave's pull short, not for any capacity reason.

**Shared H100 run line** (per-leaf `ollama create`/client commands in the leaf READMEs). Pin
`--gpus '"device=N"'` — that alone confines Ollama to one card (discovery then lists exactly
one GPU), the CUDA analogue of ROCm's `--device /dev/dri/renderD*`:

```bash
GGUF=/dev/shm/h100/out/ollama/ggufs        # where the downloaded GGUFs live
STORE=/dev/shm/h100/out/ollama/store       # model store on tmpfs (NOT on /), ollama create COPIES
sudo docker run -d --gpus '"device=4"' \
  -e OLLAMA_HOST=0.0.0.0:11440 \
  -v $GGUF:/ggufs:ro \
  -v $STORE:/root/.ollama \
  -p 127.0.0.1:11440:11440 \
  --name ollama_h100 ollama/ollama
```

Deviations from the MI355X run line, all H100/shared-node driven:

- **`--gpus '"device=4"'`** replaces `--device /dev/kfd --device /dev/dri/renderD*`. This box
  is shared: GPUs 0–3 ran a production job and 5/7 other agents — only physical **GPU 4** was
  used. Discovery confirmed a single `CUDA0` = H100, and `nvidia-smi -i 4` showed the VRAM.
- **Port 11440**, not the default 11434, to avoid collisions with co-tenant agents
  (`OLLAMA_HOST=0.0.0.0:11440` + `-p 127.0.0.1:11440:11440`).
- **Model store on `/dev/shm` tmpfs.** Same reason as MI355X (`ollama create` copies, keep it
  off `/`); here tmpfs also dodged an `Operation not permitted` atomic-rename failure that the
  `/mnt/gsma` network filesystem throws at pip/venv.

**Both leaves PASS with proven `100% GPU` residency**, and the co-residency headline reproduces:
both models resident on one card at once, each `100% GPU`:

```text
NAME                     ID              SIZE      PROCESSOR    CONTEXT    UNTIL
embeddinggemma:latest    b48ed6e89ad7    393 MB    100% GPU     2048       4 minutes from now
qwen3-0.6b-q8:latest     605b58ae76ea    5.6 GB    100% GPU     40960      3 minutes from now
```

```text
# nvidia-smi -i 4 --query-compute-apps=pid,process_name,used_memory --format=csv
1717079, /usr/lib/ollama/llama-server, 5942 MiB     # the LLM
1718422, /usr/lib/ollama/llama-server, 1030 MiB     # the embedder
```

Per-leaf evidence (offload logs, tok/s, embedding correctness) is in the leaf READMEs. The
NVIDIA reranker gap is identical to ROCm — Ollama's 0.32.15 API still has no rerank route, so
there is still deliberately no `reranker/` leaf here (use `../llamacpp/reranker/` or
`../vllm/reranker/`).

## Environment & secrets

`dev.env` in each leaf is a symlink to the repo-root `dev.env` (from a leaf:
`ln -sf ../../../dev.env dev.env`) supplying `HF_TOKEN`. **Neither leaf actually needs
the token**: both register GGUFs already on disk, so no credential ever reaches the
container. The token would only matter for `ollama pull hf.co/<gated-repo>`.

## Scope note — deliberately no `reranker/` leaf

Ollama is ❌ for the reranker workload:
Ollama's HTTP surface is `/api/generate`, `/api/chat`, `/api/embed` plus the OpenAI shim —
**there is no rerank route** (confirmed on the 0.32.14 API used here), and a cross-encoder
cannot be emulated by embedding both sides and taking a cosine. For reranking use
[`../llamacpp/reranker/`](../llamacpp/reranker/) (llama.cpp `/v1/rerank` — the local/GGUF
answer) or [`../vllm/reranker/`](../vllm/reranker/) (the production GPU-serving answer).

## Leaves

| Leaf | Model | Verdict |
|---|---|---|
| [`llm/`](llm/README.md) | `unsloth/Qwen3.8-27B-GGUF:Q8_0` | ✅ **PASS** — single- and multi-GPU, proven `100% GPU` residency, ~76 tok/s decode; second GPU adds capacity, not speed. |
| [`embedding/`](embedding/README.md) | `ggml-org/embeddinggemma-300M-GGUF:Q8_0` | ✅ **PASS** — worst \|Δ cosine\| 0.0034 vs Transformers fp32 baseline; task prefixes must be applied client-side (the one correctness trap). |

## Shared quirks (both leaves)

- **`ollama create` copies, it does not reference** — budget the store accordingly, off `/`.
- **`PROCESSOR: 100% GPU` coexists with a non-zero `CPU_Mapped` buffer** — the percentage
  refers to layer offload, not every byte.
- **Outbound calls to ollama.com fail on this host and are harmless** (model-recommendation
  refresh only).
- **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — device selection is done purely by
  which `renderD*` nodes are passed into the container.
- The 5-minute `keep_alive` default unloads idle models; raise `OLLAMA_KEEP_ALIVE` for a
  steady service.

## Requirements & venv

Client deps (both leaves are dependency-light — plain JSON over HTTP): see
[`requirements.txt`](requirements.txt). Venv convention: `python3 -m venv .env_ollama` at
this software root. Per-campaign venvs were removed in the 2026-08 reorg.

## Other hardware (upstream claims — not verified here)

Upstream supports Apple Silicon/macOS via Metal (with an MLX engine option on arm64),
NVIDIA CUDA on Linux/Windows, AMD ROCm v7 drivers on Linux/Windows, Vulkan on
Windows/Linux for GPUs CUDA/ROCm don't cover, and full CPU fallback (x86-64 and ARM64) —
none verified here.
