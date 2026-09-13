# `inference/lemonade/llm` — Lemonade Server GGUF LLM serving (Qwen3.8-27B-Q8_0)

## Overview & when to use

Serves **Qwen3.8-27B** as **Q8_0 GGUF** (27.1 GB) through **Lemonade Server**, which
manages a `llama-server` subprocess built against **ROCm**. The client
`inference_llm_lemonade.py` hits `POST /v1/chat/completions` on the Lemonade port
(**8350** here).

Two Lemonade LLM paths exist. Only one of them works on this GPU:

| Path | Command | Result on MI355X |
|---|---|---|
| **GGUF via llama.cpp** | `lemonade pull user.Qwen38-Q8 … --recipe llamacpp` | **works** — MTP speculative decoding |
| **FP8 via experimental vLLM ROCm** | `lemonade pull user.Qwen38-FP8 … --recipe vllm` | **blocked** — `Unsupported GPU: gfx950` |

This folder's server is the *same process* as `inference/lemonade/embedding` and
`inference/lemonade/reranker` — one port, one model registry, three workloads, and no build
step. Use `inference/vllm/llm` or `inference/sglang/llm` for the native FP8 checkpoint, or
raw `llama.cpp` when you need server flags (`-ngl`, `--tensor-split`, `-c`) that Lemonade
does not expose.

## Install — the route that works

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache (weights land here)
export DATA_DIR=/path/to/data          # Lemonade tarball, binaries and backend cache
export OUTPUT_DIR=/path/to/outputs     # inference artifacts
```

### Two different Lemonades share the name

| What | Command it gives you | Has `backends install`? | Notes |
|---|---|---|---|
| pip `lemonade-sdk==9.1.4` | `lemonade`, `lemonade-server-dev` | **No** | Deprecated python server — **its ROCm backend rejects gfx950** (see below) |
| C++ Lemonade Server **11.7.0** (GitHub releases) | `lemonade`, `lemond` | **Yes** | **This is the documented server — use this one** |

`lemonade backends install llamacpp:rocm` **only exists in the C++ server**, which ships as
a GitHub release artifact, not on PyPI. The pip server also cannot serve on this GPU — its
first inference request dies with
`ValueError: ROCm backend selected but no compatible ROCm target architecture found.`

On MI355X, **the C++ server 11.7.0 is the only Lemonade that resolves gfx950.**

```bash
# (a) python venv — shared across the three inference_*_lemonade folders (see note)
python3 -m venv .env_lemonade
export PIP_CACHE_DIR=$DATA_DIR/pip_cache
.env_lemonade/bin/pip install lemonade-sdk        # 9.1.4

# (b) the C++ server that actually implements `backends install`
cd $DATA_DIR/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
```

> **Do not use the `.deb`.** `lemonade-server_11.7.0-debian13_amd64.deb` is built for
> **Debian 13** and fails on Ubuntu 24.04 with missing `libmbedcrypto.so.16` /
> `libcpp-httplib.so.0.41`. Use the **`lemonade-embeddable-…-ubuntu-x64.tar.gz`** artifact.

### Shared venv

**This folder owns the real venv**; the other two `inference_*_lemonade` folders symlink
to it:

```bash
# in inference/lemonade/embedding/ and inference/lemonade/reranker/
ln -sfn ../../../inference/lemonade/llm/.env_lemonade .env_lemonade
```

The python side is only the HTTP client; the runtime is the C++ server plus its own managed
ROCm venv inside the Lemonade cache.

### Backend install

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

```text
Installing backend: llamacpp:rocm
[1/2] llama-b10470-bin-ubuntu-rocm-7.14-x64.tar.gz
[2/2] ROCm runtime (preparing venv)   # incl. rocm_sdk_device_gfx950 — the arch-matched wheel
Backend installed successfully: llamacpp:rocm
```

Lemonade resolves `gfx950` itself and fetches the matching device package. It does **not**
use the host's ROCm 7.2.4 — it brings its own **ROCm 7.14** runtime in a private venv
(~4.8 GB of cache).

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed for the GGUF path — `unsloth/Qwen3.8-27B-GGUF` is ungated and registers
with no `HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0       # single-GPU runs
# HF_HOME (exported above) is where model weights land
```

- **`HF_HOME` controls the weights** (`$HF_HOME/hub/`). 27.1 GB for this model.
- **The `cache_dir` positional controls the binaries** — `$DATA_DIR/lemonade/cache`,
  **4.8 GB**. Neither should land on the root filesystem.
- Set these on the **`lemond`** process, not on the CLI client: `lemond` forks
  `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

### Reusing a 28 GB GGUF you already have

Lemonade downloads into its own HF-hub-shaped tree, so a GGUF sitting in a `LLAMA_CACHE`
directory is invisible to it. You can pre-seed it and skip the re-download entirely:

```bash
SRC=$HF_HOME/llama_cpp/models--unsloth--Qwen3.8-27B-GGUF
DST=$HF_HOME/hub/models--unsloth--Qwen3.8-27B-GGUF
SHA=27af057ecb382ddfea5d12837360a8980560e3ed

mkdir -p $DST/refs $DST/snapshots/$SHA
echo -n $SHA > $DST/refs/main
ln -sf $SRC/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf $DST/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf
cat > $DST/.lemonade_registry.json <<EOF
{"processed_models":{"user.Qwen38-Q8":{"selection":"llamacpp\nmain=unsloth/Qwen3.8-27B-GGUF:Q8_0","snapshot_id":"$SHA"}},
 "repo_id":"unsloth/Qwen3.8-27B-GGUF","revision":"$SHA","snapshot_id":"$SHA","source":"huggingface"}
EOF
```

`pull` then reports `(already downloaded)` for the 27.7 GB file, saving the re-download.
Symlinks are followed fine.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=$DATA_DIR/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0

$LEM/lemond $DATA_DIR/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register the Q8 GGUF (the recommended path)

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen38-Q8 \
  --checkpoint main unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  --recipe llamacpp
```

```text
Pulling model: user.Qwen38-Q8
[1/2] Qwen3.8-27B-Q8_0.gguf   (already downloaded)
Model pulled successfully: user.Qwen38-Q8
```

### 3. Load and prompt it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.Qwen38-Q8

.env_lemonade/bin/python inference_llm_lemonade.py \
  --port 8350 --max_tokens 900 \
  --out $OUTPUT_DIR/inference_llm_lemonade/chat_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s http://127.0.0.1:8350/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen38-Q8","messages":[{"role":"user","content":"Explain an OpenTelemetry span."}],
       "max_tokens":900,"temperature":0.0}' | jq -r '.choices[0].message.content'
```

> The registered name is `user.Qwen38-Q8`; the id the **API** answers to is `Qwen38-Q8`
> (the `user.` prefix is stripped in `/v1/models`).

## Expected output — single GPU

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen38-Q8
fingerprint: b10469-666f8898a
finish     : stop
prompt     : In exactly two sentences, explain what an OpenTelemetry span is.
--- generated text ---
An OpenTelemetry span represents a single unit of work in a distributed trace, such as a
request, function call, or database query. It records timing, attributes, events, and
status to help visualize and diagnose the flow of operations across services.
```

### Speculative decoding is enabled automatically

Lemonade builds this command line by itself:

```text
llama-server -m …/Qwen3.8-27B-Q8_0.gguf --ctx-size 262144 --port 8003 \
  --jinja --metrics --reasoning-format auto --spec-type draft-mtp --no-ui
```

**`--spec-type draft-mtp`** — Lemonade recognises that this checkpoint carries an MTP
(multi-token-prediction) head and enables speculative decoding automatically; non-zero
`draft_n` in the client output confirms it is live. It also picks `--ctx-size 262144`, the
model's full context, which is a large KV allocation.

## GPU-residency check

Lemonade launches `llama-server` **without an explicit `-ngl`**, so verify the offload
rather than assuming it. `rocm-smi --showpids` reads the kernel's KFD accounting,
which a CPU-only process cannot appear in at all:

```bash
rocm-smi --showpids
cat /sys/class/kfd/kfd/proc/<pid>/vram_*     # per-GPU bytes for that PID
```

The weights plus KV cache, the MTP draft model and the HIP context should be attributed to
the `llama-server` PID on exactly one GPU, with every other card at its idle baseline. The
server log independently names the ROCm device:

```text
llama_sampler_backend_support: device 'ROCm0' does not have support for op TOP_K …
```

## Multi-GPU

With two GPUs visible (`HIP_VISIBLE_DEVICES=0,1`), llama.cpp's default `--split-mode layer`
splits the 27B's layers across both cards and produces identical output.

**Multi-GPU is worth it here only when the model does not fit on one card**; otherwise run
one server per GPU. Lemonade exposes no `--tensor-split` or `--split-mode` knob, so
`HIP_VISIBLE_DEVICES` is the only control you have.

## The experimental vLLM ROCm path is blocked on gfx950

The documented FP8 path is `lemonade backends install vllm:rocm` +
`lemonade pull user.Qwen38-FP8 --checkpoint Qwen/Qwen3.8-27B-FP8 --recipe vllm`. Both
steps are refused on this machine:

```text
$ lemonade backends install vllm:rocm
Installing backend: vllm:rocm
Error: Cannot install vllm:rocm on this system: Unsupported GPU: gfx950

$ lemonade pull user.Qwen38-FP8 --checkpoint main Qwen/Qwen3.8-27B-FP8 --recipe vllm
Pulling model: user.Qwen38-FP8
Error pulling model: Model 'user.Qwen38-FP8' cannot be used on this system
  (recipe: vllm): Unsupported GPU: gfx950
```

`lemonade backends --all` reports the same before any install attempt (`vllm rocm unsupported`).
This is a hard gate in the runtime, so the `Qwen/Qwen3.8-27B-FP8` checkpoint **cannot be
served through Lemonade on gfx950 at all**, even when it is already present in
`$HF_HOME/hub/`. Use `inference/vllm/llm` or `inference/sglang/llm` for that checkpoint.

## Arguments

### Lemonade Server / CLI

| Argument | Value | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `$DATA_DIR/lemonade/cache` | Binaries + backend venv. Keep off the root filesystem |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs |
| `backends install llamacpp:rocm` | the working backend | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `backends install vllm:rocm` | **refused** | `Unsupported GPU: gfx950` |
| `pull --checkpoint TYPE REPO:QUANT` | `main unsloth/Qwen3.8-27B-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` (`vllm` refused) | Backend family |
| `pull --label` | not needed for chat | Only embeddings/reranking need a label |
| `load <name>` | optional | Start/warm the subprocess before the first request |
| `list` / `backends --all` | — | Registry + per-arch backend support matrix |

### `inference_llm_lemonade.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Lemonade Server host |
| `--port` | `8350` | Lemonade Server port |
| `--model` | `Qwen38-Q8` | API model id (no `user.` prefix) |
| `--endpoint` | `/v1/chat/completions` | Also `/api/v1/chat/completions`, `/v1/completions` |
| `--prompt` | OpenTelemetry span question | User message |
| `--system` | `None` | Optional system message |
| `--max_tokens` | `256` | Generation cap — **raise it, see quirk 6** |
| `--temperature` | `0.0` | Sampling temperature |
| `--timeout` | `900` | HTTP timeout (s) |
| `--health_retries` | `120` | `/api/v1/health` polls before giving up |
| `--load_model` | off | `POST /api/v1/load` first |
| `--show_reasoning` | off | Also print `reasoning_content` |
| `--out` | `None` | Write the raw JSON response here |

## Output

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_llm_lemonade/` — never the root filesystem. Weights (27.1 GB) live in
`$HF_HOME/hub/`; Lemonade binaries + ROCm wheels (4.8 GB) in `$DATA_DIR/lemonade/cache/`.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2.4 host):** works via `llamacpp:rocm` (Lemonade brings its
  own ROCm 7.14 and an arch-matched gfx950 device wheel). `vllm:rocm` is **refused**
  (`Unsupported GPU: gfx950`), so the FP8 checkpoint is unreachable here.
- **NVIDIA H100 (Hopper sm_90):** works via `llamacpp:cuda` — see the H100 section below.
  There is no `vllm:cuda` backend.

## Notes & quirks

1. **Two different products share the name.** pip `lemonade-sdk` != the C++ Lemonade
   Server. Only the latter has `backends install`. See "Install".
2. **The Linux `.deb` is Debian-13-only** (`libmbedcrypto.so.16`,
   `libcpp-httplib.so.0.41`). Use the `embeddable` tarball on Ubuntu 24.04.
3. **`--no-discovery` on every CLI call.** Without it the client broadcasts UDP looking
   for servers and can hang well past 60 s on a multi-tenant host.
4. **The `user.` prefix is registration-only.** `/v1/models` reports `Qwen38-Q8`; sending
   `"model":"user.Qwen38-Q8"` is a 404.
5. **Lemonade brings its own ROCm.** It installs ROCm **7.14** wheels into a private venv
   and ignores the host's ROCm 7.2.4. Good for reproducibility, but 4.8 GB of cache.
6. **Qwen3.8 is a reasoning model and `--max_tokens` must account for it.** With
   `--reasoning-format auto`, thinking tokens land in `reasoning_content` and `content`
   stays **empty** until thinking finishes. A 320-token cap on a code prompt returns
   `finish_reason: length` with an **empty `content`** and 320 tokens of reasoning; 900
   works. Pass `--show_reasoning` to print the reasoning output.
7. **No `-ngl`, no `--tensor-split`, no `-c`.** Lemonade builds the `llama-server` command
   line itself. `HIP_VISIBLE_DEVICES` on the `lemond` process is your only device control.
8. **`--ctx-size 262144` is chosen for you** — the model's full context, which is a large KV
   allocation.
9. **`--spec-type draft-mtp` is enabled automatically** for this checkpoint.
10. **Harmless `ROCm0 does not have support for op TOP_K` warnings.** The sampler falls
    back to CPU for `top-k`; the model itself is fully on GPU. Output is unaffected.
11. **Harmless startup warnings** for the embeddable build: `Could not load
    architecture_defaults.json`, `Web app directory not found`. Only the web UI is
    affected; the API is complete.
12. **Pre-seed the HF cache to skip a 28 GB re-download** — see "Environment & secrets".

## H100 (NVIDIA, Hopper sm_90)

Single-GPU, host CUDA 13.0, Python 3.12.

**Install is the AMD route with one word changed: `llamacpp:cuda` instead of
`llamacpp:rocm`.** Same C++ embeddable tarball, same `backends install`. It pulls the
arch-matched Hopper prebuilt `llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` with no compiler,
and bundles its own CUDA 12.9 runtime, ignoring the host CUDA 13. Confirm with
`Using LlamaCpp Backend: cuda` in the `lemond` log.

**Model.** The example below uses a small same-family GGUF —
**`unsloth/Qwen3-0.6B-GGUF:Q8_0`** (610 MB) — to exercise the backend/GPU path quickly when
no large GGUF is cached locally. The 27B registers identically:

```bash
LEM=/dev/shm/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=0   # HF_HOME as exported above
$LEM/lemond /dev/shm/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast &
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:cuda
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-06B-Q8 \
  --checkpoint main unsloth/Qwen3-0.6B-GGUF:Q8_0 --recipe llamacpp
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-06B-Q8
```

**Smoke command and expected output:**

```bash
.env_lemonade/bin/python inference_llm_lemonade.py --port 8350 --model Qwen3-06B-Q8 \
  --max_tokens 900 --out $OUTPUT_DIR/lemonade/chat_single_gpu.json
```

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen3-06B-Q8
fingerprint: b10394-680a9ae63
finish     : stop
--- generated text ---
An OpenTelemetry span is a way to track the flow of data and operations within a
distributed system, providing detailed information about each step in the process. It
enables monitoring and logging of application performance, helping in debugging and
optimizing the system.
```

**No `draft_n`** — unlike the 27B, the 0.6B has no MTP head, so Lemonade does not enable
`--spec-type draft-mtp` here. The wrapped command line Lemonade built (note: no `-ngl`, same
as MI355X, and `--ctx-size 4096` chosen for this small model):

```text
/dev/shm/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--unsloth--Qwen3-0.6B-GGUF/…/Qwen3-0.6B-Q8_0.gguf \
  --ctx-size 4096 --port 8001 --jinja --metrics --reasoning-format auto --no-ui
```

**GPU-residency check.** `nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid`
must attribute the `llama-server` PID's memory to the selected GPU; a CPU-only run shows
0 MiB.

**Multi-GPU.** Same as MI355X: Lemonade exposes no `--tensor-split`/`-ngl`, so
`CUDA_VISIBLE_DEVICES` on the `lemond` process is the only device control (llama.cpp default
`--split-mode layer`). For a model that fits on one card, run one server per GPU.

**Two NVIDIA-side notes:** the 0.6B has no MTP head, so no spec-decode is enabled; and
build the client venv on a local disk or tmpfs — `python3 -m venv` on an NFS mount does not
create console scripts / pip reliably (use `python -m pip` there).
