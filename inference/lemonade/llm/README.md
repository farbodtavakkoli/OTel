# `inference/lemonade/llm` — Lemonade Server GGUF LLM serving (Qwen3.8-27B-Q8_0) on ROCm/gfx950

## Overview & when to use

Serves **Qwen3.8-27B** as **Q8_0 GGUF** (27.1 GB) through **Lemonade Server**, which
manages a `llama-server` subprocess built against **ROCm**. The client
`inference_llm_lemonade.py` hits `POST /v1/chat/completions` on the Lemonade port
(**8350** here).

Two Lemonade LLM paths exist. Only one of them works on this GPU:

| Path | Command | Result on MI355X |
|---|---|---|
| **GGUF via llama.cpp** | `lemonade pull user.Qwen38-Q8 … --recipe llamacpp` | **PASS** — 143 tok/s, 50.7 GB VRAM |
| **FP8 via experimental vLLM ROCm** | `lemonade pull user.Qwen38-FP8 … --recipe vllm` | **BLOCKED** — `Unsupported GPU: gfx950` |

Use this folder when you want:

- **One local OpenAI-compatible service for all three workloads** — this folder's server
  is the *same process* as `inference/lemonade/embedding` and
  `inference/lemonade/reranker`. One port, one model registry, three workloads.
- **Zero-build ROCm deployment of a 27B model** — no `cmake`, no `-DGPU_TARGETS`, no
  `libssl-dev`. 27 seconds from nothing to a working ROCm backend.
- **Speculative decoding you did not have to configure** — Lemonade turned on
  `--spec-type draft-mtp` by itself (see Results).

Prefer **vLLM** or **SGLang** for multi-user throughput and for the native FP8 checkpoint;
prefer raw `llama.cpp` when you need server flags (`-ngl`, `--tensor-split`, `-c`) that
Lemonade does not expose per model.

> **Tested topology:** 2x AMD Instinct MI355X (gfx950, 288 GB each), physical GPUs
> **4 and 5**, ROCm 7.2.4 host, Ubuntu 24.04, Python 3.12.3. Verified **2026-08-20**.

## Install — the exact route that worked

### The package-name trap: two different Lemonades

| What | Command it gives you | Has `backends install`? | Verdict here |
|---|---|---|---|
| pip `lemonade-sdk==9.1.4` | `lemonade`, `lemonade-server-dev` | **No** | Deprecated python server — **its ROCm backend rejects gfx950** (tested, see below) |
| C++ Lemonade Server **11.7.0** (GitHub releases) | `lemonade`, `lemond` | **Yes** | **This is the documented server — used here** |

`pip install lemonade-sdk` installs the SDK and a **deprecated** Python server whose
backend choice is a serve-time flag (`lemonade-server-dev serve --llamacpp rocm`):

```text
DEPRECATION NOTICE
The Python-based 'lemonade-server-dev' command is deprecated.
Please use the C++ Lemonade Server instead …
```

The documented `lemonade backends install llamacpp:rocm` **only exists in the C++ server**,
which ships as a release artifact, not on PyPI.

**And the pip server cannot serve on this GPU at all.** Tested end to end: it starts,
`/api/v1/health` returns 200, `/api/v1/pull` succeeds and even reuses a local GGUF — then
the first inference request dies while installing its llama.cpp ROCm backend:

```text
$ lemonade-server-dev serve --llamacpp rocm --port 8351
Lemonade Server v9.1.4 Ready!
INFO:  127.0.0.1 - "GET /api/v1/health HTTP/1.1" 200 OK
INFO:  Resolved local GGUF model: …/embeddinggemma-300M-Q8_0.gguf
INFO:  Model already exists locally, skipping download
INFO:  127.0.0.1 - "POST /api/v1/pull HTTP/1.1" 200 OK

  File ".../lemonade/tools/llamacpp/utils.py", line 375, in install_llamacpp
    raise ValueError(
ValueError: ROCm backend selected but no compatible ROCm target architecture found.
INFO:  127.0.0.1 - "POST /api/v1/embeddings HTTP/1.1" 422 Unprocessable Entity
```

So on MI355X the choice is not stylistic: **the C++ server 11.7.0 is the only Lemonade
that resolves gfx950.** Transcript at
`/mnt/data_450g/outputs/pip_lemonade_server_rocm_gfx950_unsupported.txt`.

```bash
# (a) python venv — shared across the three inference_*_lemonade folders (see note)
python3 -m venv .env_inference_llm_lemonade
export PIP_CACHE_DIR=/mnt/data_1.5t/pip_cache
.env_inference_llm_lemonade/bin/pip install lemonade-sdk        # 9.1.4, 8.8 s

# (b) the C++ server that actually implements `backends install`
cd /mnt/data_450g/lemonade
curl -sL -O https://github.com/lemonade-sdk/lemonade/releases/download/v11.7.0/lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz
mkdir -p emb && tar xzf lemonade-embeddable-11.7.0-ubuntu-x64.tar.gz -C emb
```

> **Do not use the `.deb`.** `lemonade-server_11.7.0-debian13_amd64.deb` is the only Linux
> package offered and it is built for **Debian 13**. Extracted on Ubuntu 24.04:
>
> ```text
> ./cxx/usr/bin/lemonade: error while loading shared libraries:
>   libmbedcrypto.so.16: cannot open shared object file: No such file or directory
> # also missing: libcpp-httplib.so.0.41
> ```
>
> The **`lemonade-embeddable-…-ubuntu-x64.tar.gz`** artifact is `ldd`-clean on this host.

### Shared venv

**This folder owns the real venv**; the other two `inference_*_lemonade` folders symlink
to it:

```bash
# in inference/lemonade/embedding/ and inference/lemonade/reranker/
ln -sfn ../../../inference/lemonade/llm/.env_inference_llm_lemonade .env_inference_<name>_lemonade
```

Deliberate, and a sibling folder set does the same. The python side is only the HTTP
client; the runtime is the C++ server plus its own managed ROCm venv inside the Lemonade
cache. Sharing one venv saves ~500 MB and keeps a single `lemonade-sdk` version in play.

### Backend install — and what it chose

```bash
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:rocm
```

```text
Installing backend: llamacpp:rocm
[1/2] llama-b10470-bin-ubuntu-rocm-7.14-x64.tar.gz

[2/2] ROCm runtime (preparing venv)
      rocm-7.14.0.tar.gz
      rocm_sdk_core-7.14.0-py3-none-linux_x86_64.whl          (414.6 MB)
      rocm_sdk_libraries-7.14.0-py3-none-linux_x86_64.whl     (557.6 MB)
      rocm_sdk_device_gfx950-7.14.0-py3-none-linux_x86_64.whl (1240.1 MB)
Backend installed successfully: llamacpp:rocm      # real 0m27.3s
```

**Selected backend: `llamacpp:rocm`, build `b10470`** (binaries labelled `b10469`,
`system_fingerprint` `b10469-666f8898a`). The `rocm_sdk_device_gfx950` wheel is the
headline — Lemonade probed the GPU, resolved `gfx950`, and fetched the matching TheRock
device package. It does **not** use the host's ROCm 7.2.4; it brings its own **ROCm 7.14**
runtime in a private venv.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                       # already done in this folder
export $(grep -v '^#' ../dev.env | xargs)       # only when a gated checkpoint needs HF_TOKEN
```

Not needed for the GGUF path — `unsloth/Qwen3.8-27B-GGUF` is ungated and was registered
with no `HF_TOKEN` exported. Never echo or commit the token.

```bash
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4       # single-GPU runs
export HF_HOME=/mnt/data_1.5t/hf_cache                    # where model weights land
```

- **`HF_HOME` controls the weights** (`$HF_HOME/hub/`). 27.1 GB for this model.
- **The `cache_dir` positional controls the binaries** — `/mnt/data_450g/lemonade/cache`,
  **4.8 GB**. Neither may land on `/` (99 GB free).
- Set these on the **`lemond`** process, not on the CLI client: `lemond` forks
  `llama-server`, so it is what propagates `HIP_VISIBLE_DEVICES`.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device and
> the stack silently falls back to CPU.

### Reusing a 28 GB GGUF you already have

Lemonade downloads into its own HF-hub-shaped tree, so a GGUF sitting in a `LLAMA_CACHE`
directory is invisible to it. You can pre-seed it and skip the re-download entirely:

```bash
SRC=/mnt/data_1.5t/hf_cache/llama_cpp/models--unsloth--Qwen3.8-27B-GGUF
DST=/mnt/data_1.5t/hf_cache/hub/models--unsloth--Qwen3.8-27B-GGUF
SHA=27af057ecb382ddfea5d12837360a8980560e3ed

mkdir -p $DST/refs $DST/snapshots/$SHA
echo -n $SHA > $DST/refs/main
ln -sf $SRC/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf $DST/snapshots/$SHA/Qwen3.8-27B-Q8_0.gguf
cat > $DST/.lemonade_registry.json <<EOF
{"processed_models":{"user.Qwen38-Q8":{"selection":"llamacpp\nmain=unsloth/Qwen3.8-27B-GGUF:Q8_0","snapshot_id":"$SHA"}},
 "repo_id":"unsloth/Qwen3.8-27B-GGUF","revision":"$SHA","snapshot_id":"$SHA","source":"huggingface"}
EOF
```

`pull` then reports `(already downloaded)` for the 27.7 GB file — **28 GB and ~12 minutes
saved**. Symlinks are followed fine.

## Exact commands

### 1. Start the server on port 8350

```bash
LEM=/mnt/data_450g/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4
export HF_HOME=/mnt/data_1.5t/hf_cache

$LEM/lemond /mnt/data_450g/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast
```

### 2. Register the Q8 GGUF (the recommended path)

```bash
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen38-Q8 \
  --checkpoint main unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  --recipe llamacpp
```

```text
Pulling model: user.Qwen38-Q8
Total: 27.1 GB, 2 files
[1/2] Qwen3.8-27B-Q8_0.gguf (27701.5 MB)   (already downloaded)
[2/2] config.json (0.0 MB)
Model pulled successfully: user.Qwen38-Q8
```

### 3. Load and prompt it

```bash
$LEM/lemonade --port 8350 --no-discovery load user.Qwen38-Q8     # 3.9 s warm page cache

.env_inference_llm_lemonade/bin/python inference_llm_lemonade.py \
  --port 8350 --max_tokens 900 \
  --out /mnt/data_450g/outputs/inference_llm_lemonade/chat_single_gpu.json
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

## Results — single GPU (physical GPU 4)

Real client output:

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen38-Q8
fingerprint: b10469-666f8898a
latency_s  : 1.190
finish     : stop
prompt     : In exactly two sentences, explain what an OpenTelemetry span is.
--- generated text ---
An OpenTelemetry span represents a single unit of work in a distributed trace, such as a
request, function call, or database query. It records timing, attributes, events, and
status to help visualize and diagnose the flow of operations across services.
--- llama.cpp timings ---
prompt_n            : 66
predicted_n         : 148
predicted_per_second: 143.34 tok/s
draft_n / accepted  : 141 / 101 (71.6% accepted)
```

And a longer code-generation probe, to show it is not just producing plausible prose:

```text
latency_s  : 5.175
finish     : stop
predicted_per_second: 131.77 tok/s
draft_n / accepted  : 675 / 428 (63.4% accepted)
--- generated text ---
```python
import os


def rocm_visible():
    """
    Return the visible ROCm device indices from HIP_VISIBLE_DEVICES.

    The HIP_VISIBLE_DEVICES environment variable is expected to contain a
    comma-separated list of integer device indices, for example "0,1,2".

    If HIP_VISIBLE_DEVICES is unset, this function returns [0].

    Returns:
        list[int]: A list of integer device indices.
    """
    value = os.environ.get("HIP_VISIBLE_DEVICES")

    if value is None:
        return [0]

    return [int(part.strip()) for part in value.split(",") if part.strip()]
```
```

653 tokens of correct, runnable Python at **131.8 tok/s**.

### Speculative decoding, unasked-for

Lemonade built this command line by itself:

```text
llama-server -m …/Qwen3.8-27B-Q8_0.gguf --ctx-size 262144 --port 8003 \
  --jinja --metrics --reasoning-format auto --spec-type draft-mtp --no-ui
```

**`--spec-type draft-mtp`** — Lemonade recognised that this checkpoint carries an MTP
(multi-token-prediction) head and enabled speculative decoding without being asked. The
`draft_n / draft_n_accepted` fields prove it is live and paying off: **62-72 % of drafted
tokens accepted** across runs. It also picked `--ctx-size 262144`, the model's full
context, because the card has room.

## GPU-residency proof

Lemonade launches `llama-server` **without an explicit `-ngl`**, so offload must be proven,
not assumed. `rocm-smi --showpids` reads the kernel's KFD accounting, which a CPU-only
process cannot appear in at all:

```text
PID      PROCESS NAME    GPU(s)  VRAM USED     SDMA USED       CU OCCUPANCY
1969631  llama-server    1       50692304896   2760503915888   0
```

**50.69 GB attributed to our PID on exactly one GPU.** Per-GPU breakdown from
`/sys/class/kfd/kfd/proc/1969631/`:

```text
vram_51023 = 50692304896 bytes   -> gpu_id 51023 = unique_id 0xf743d583ac01fcfd = card4
(every other vram_* entry for this PID is 0)
7 live HSA compute queues
```

Card-level `rocm-smi`, with GPU 5 sitting at its idle baseline for contrast:

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card4,309220868096,60627656704      <-- 60.6 GB, this server
card5,309220868096,298688512        <-- 0.30 GB idle baseline
```

27.1 GB of Q8_0 weights + KV cache + the MTP draft model + HIP context ≈ 50.7 GB. A CPU
fallback would show `0` here and would not be running at 143 tok/s.

The server log independently names the ROCm device:

```text
llama_sampler_backend_support: device 'ROCm0' does not have support for op TOP_K …
```

## Results — multi-GPU (physical GPUs 4 and 5)

With `HIP_VISIBLE_DEVICES=4,5`, llama.cpp's default `--split-mode layer` performs a
**genuine layer split of the 27B across both cards** — this is real sharding, not a
replica pattern:

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen38-Q8
latency_s  : 1.234
finish     : stop
--- generated text ---
An OpenTelemetry span represents a single unit of work in a distributed trace, such as a
request, function call, or database query. It records timing, attributes, events, and
status to help visualize and diagnose the flow of operations across services.
--- llama.cpp timings ---
prompt_n            : 4
predicted_n         : 148
predicted_per_second: 140.78 tok/s
draft_n / accepted  : 141 / 101 (71.6% accepted)
```

Per-GPU VRAM proves the split is real:

```text
PID      PROCESS NAME    GPU(s)  VRAM USED
1919684  llama-server    2       55216840704

/sys/class/kfd/kfd/proc/1919684/vram_51023 = 26437902336   -> card4, 26.4 GB
/sys/class/kfd/kfd/proc/1919684/vram_17306 = 28778938368   -> card5, 28.8 GB

device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card4,309220868096,33091641344
card5,309220868096,36038447104
```

### Single vs multi-GPU: the honest comparison

| | Single GPU (4) | Two GPUs (4+5) |
|---|---|---|
| Output text | identical | identical |
| Tokens/s | **143.34** | 140.78 |
| Draft acceptance | 71.6 % | 71.6 % |
| VRAM (our PID) | 50.7 GB on 1 card | 55.2 GB across 2 cards |

**Two GPUs are slightly slower.** That is the expected and correct result: a 27.1 GB model
fits comfortably in one 288 GB MI355X, so a layer split buys no capacity and only adds a
device-to-device transfer at the split boundary. Identical output text and identical draft
acceptance confirm the split is numerically transparent — it changes where the weights
live, not what the model says.

**Multi-GPU is worth it here only when the model does not fit on one card.** For this
checkpoint on this hardware, the right answer is one GPU per server and **N independent
servers** for throughput. Lemonade exposes no `--tensor-split` or `--split-mode` knob, so
`HIP_VISIBLE_DEVICES` is the only control you have.

## The experimental vLLM ROCm path: BLOCKED on gfx950

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

`lemonade backends --all` says the same thing before you even try:

```text
Recipe    Backend  Status       Message/Version
vllm      rocm     unsupported  Unsupported GPU: gfx950
llamacpp  rocm     installable  Backend is supported but not installed.
```

**This is a hard gate in the runtime, not a missing wheel.** The interesting detail is that
Lemonade's own version table already anticipates this GPU —
`resources/backend_versions.json`:

```json
"vllm": {
  "comment": "… gfx950 (CDNA4) rides the same CDNA vLLM/ROCm line as gfx942 (CDNA3) …",
  "rocm": "vllm0.20.1-rocm7.12.0",
  "rocm_arch_overrides": { "gfx942": "vllm0.19.1-rocm7.13.0",
                           "gfx950": "vllm0.19.1-rocm7.13.0" }
}
```

So a gfx950 vLLM build is **pinned and named**, but the compiled support check in
`lemond`/`lemonade` 11.7.0 still rejects the arch. The documented warning that MI300X support
was "staged/manual rather than the normal one-click path" understates it for **MI355X**:
on gfx950 the one-click path is not merely staged, it is **actively refused**.

Consequence: the exact `Qwen/Qwen3.8-27B-FP8` checkpoint (present on disk at
`/mnt/data_1.5t/hf_cache/hub/models--Qwen--Qwen3.8-27B-FP8`) **cannot be served through
Lemonade on this host at all**. Use `inference/vllm/llm` or `inference/sglang/llm` for
that checkpoint.

## Arguments

### Lemonade Server / CLI

| Argument | Used | Meaning |
|---|---|---|
| `lemond <cache_dir>` | `/mnt/data_450g/lemonade/cache` | Binaries + backend venv. Keep off `/` |
| `--port` / `--host` | `8350` / `127.0.0.1` | Bind address (suggested port) |
| `--no-broadcast` | on | Disable the UDP discovery beacon |
| `--no-discovery` (client) | on | **Required on a shared host** — without it the CLI hangs |
| `backends install llamacpp:rocm` | used | Prebuilt ROCm llama.cpp + arch-matched ROCm wheels |
| `backends install vllm:rocm` | **refused** | `Unsupported GPU: gfx950` |
| `pull --checkpoint TYPE REPO:QUANT` | `main unsloth/Qwen3.8-27B-GGUF:Q8_0` | Register a `user.*` model |
| `pull --recipe` | `llamacpp` (`vllm` refused) | Backend family |
| `pull --label` | not needed for chat | Only embeddings/reranking need a label |
| `load <name>` | used | Start/warm the subprocess before timing |
| `list` / `backends --all` | used | Registry + per-arch backend support matrix |

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

Artifacts go to `/mnt/data_450g/outputs/inference_llm_lemonade/`, never to `/`:

```text
chat_single_gpu.json           raw /v1/chat/completions response, GPU 4
client_single_gpu.txt          client stdout, GPU 4
chat_single_gpu_code.json      653-token code-generation probe
client_single_gpu_code.txt     client stdout for the code probe
chat_two_gpu.json              raw response, GPUs 4+5
client_two_gpu.txt             client stdout, GPUs 4+5
server_cmdline_single_gpu.txt  the wrapped llama-server command Lemonade built
rocm_smi_pids_single_gpu.txt   KFD process attribution
kfd_vram_single_gpu.txt        per-GPU VRAM for our PID
rocm_smi_single_gpu.csv        per-card VRAM (card5 idle baseline for contrast)
```

Shared with the other two folders: `/mnt/data_450g/outputs/vllm_rocm_unsupported_gfx950.txt`
(the full vLLM refusal transcript) and `lemonade_backends_before_install.txt`.

Weights (27.1 GB) live in `/mnt/data_1.5t/hf_cache/hub/`; Lemonade binaries + ROCm wheels
(4.8 GB) in `/mnt/data_450g/lemonade/cache/`.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| Lemonade detects gfx950 | Per-recipe `Unsupported GPU: gfx950` messages -> the arch was resolved, not guessed |
| `llamacpp:rocm` supported on gfx950 | Listed `installable`, then `installed b10470` |
| **Backend really carries gfx950 code** | Install pulled `rocm_sdk_device_gfx950-7.14.0…whl` (1240 MB); `libggml-hip.so` (1.26 GB fat binary) contains a `gfx950` code object alongside gfx900/906/908/942/10xx/11xx/12xx |
| No `hipErrorNoBinaryForGpu` / `HSA_STATUS_ERROR_INVALID_ISA` | 27B loaded and generated; zero HSA/ISA errors in the server log |
| ROCm backend is the live device | Server log names `device 'ROCm0'` |
| Model really on GPU | `rocm-smi --showpids`: PID 1969631, **1 GPU, 50.69 GB**, 7 live KFD queues; card5 idle at 0.30 GB |
| Generation is real | 148- and 653-token completions, correct prose *and* runnable Python |
| Speculative decoding live | `draft_n 675 / accepted 428` (63.4 %) |
| Real multi-GPU sharding | 26.4 GB card4 + 28.8 GB card5, same output text |
| **`vllm:rocm` NOT supported** | `Cannot install vllm:rocm on this system: Unsupported GPU: gfx950` |
| FP8 checkpoint unusable via Lemonade | `pull … --recipe vllm` -> same refusal |
| Ungated download | Registered with no `HF_TOKEN` exported |

## Notes & quirks

1. **Two different products share the name.** pip `lemonade-sdk` != the C++ Lemonade
   Server. Only the latter has `backends install`. See "Install".
2. **The Linux `.deb` is Debian-13-only** (`libmbedcrypto.so.16`,
   `libcpp-httplib.so.0.41`). Use the `embeddable` tarball on Ubuntu 24.04.
3. **`--no-discovery` on every CLI call.** Without it the client broadcasts UDP looking
   for servers and hangs well past 60 s on this multi-tenant host. This cost a wasted
   timeout before it was diagnosed.
4. **The `user.` prefix is registration-only.** `/v1/models` reports `Qwen38-Q8`; sending
   `"model":"user.Qwen38-Q8"` is a 404.
5. **Lemonade brings its own ROCm.** It installs ROCm **7.14** wheels into a private venv
   and ignores the host's ROCm 7.2.4. Good for reproducibility, but 4.8 GB of cache.
6. **Qwen3.8 is a reasoning model and `--max_tokens` must account for it.** With
   `--reasoning-format auto`, thinking tokens land in `reasoning_content` and `content`
   stays **empty** until thinking finishes. A 320-token cap on the code prompt returned
   `finish_reason: length` with an **empty `content`** and 320 tokens of reasoning. 900
   worked. Use `--show_reasoning` to see what it was doing.
7. **No `-ngl`, no `--tensor-split`, no `-c`.** Lemonade builds the `llama-server` command
   line itself. `HIP_VISIBLE_DEVICES` on the `lemond` process is your only device control.
8. **`--ctx-size 262144` is chosen for you** — the model's full context, because the card
   has room. That is a large KV allocation; it is part of the 50.7 GB.
9. **`--spec-type draft-mtp` is enabled automatically** for this checkpoint. Nice, but it
   means your tok/s depends on draft acceptance and will vary with the prompt.
10. **Harmless `ROCm0 does not have support for op TOP_K` warnings.** The sampler falls
    back to CPU for `top-k`; the model itself is fully on GPU. Output is unaffected.
11. **Harmless startup warnings** for the embeddable build: `Could not load
    architecture_defaults.json`, `Web app directory not found`. Only the web UI is
    affected; the API is complete.
12. **Pre-seed the HF cache to skip a 28 GB re-download** — see "Environment & secrets".

## H100 (NVIDIA, Hopper sm_90) — verified 2026-08-22

Single-GPU smoke on **physical GPU 6** (`CUDA_VISIBLE_DEVICES=6`), driver 580.173.02,
host CUDA 13.0, Python 3.12.3. Multi-GPU deferred (production job on GPUs 0–3).

**Install is the AMD route with one word changed: `llamacpp:cuda` instead of
`llamacpp:rocm`.** Same C++ embeddable tarball, same `backends install`. It pulled the
arch-matched Hopper prebuilt `llama-b10397-ubuntu-cuda-sm_90-x64.tar.xz` (build **b10397**,
`system_fingerprint b10394-680a9ae63`) in **~41 s, no compiler**. It bundles its own CUDA
12.9 runtime (`libcublas.so.12.9.2.10`, `libcudart.so.12.9.79`), ignoring the host CUDA 13
— the NVIDIA analogue of the ROCm backend bringing its own ROCm 7.14. Lemonade auto-detected
`NVIDIA H100 80GB HBM3 (compute 9.0, sm_90)` and logged `Using LlamaCpp Backend: cuda` /
`Respecting existing CUDA_VISIBLE_DEVICES=6`.

**Model.** No GGUFs were cached on this box (`/mnt/data_1.5t` absent), so instead of the
27 GB Qwen3.8-27B-Q8_0 the smoke used a small same-family GGUF — **`unsloth/Qwen3-0.6B-GGUF:Q8_0`**
(610 MB) — to prove the backend/GPU path fast. Registered identically:

```bash
LEM=/dev/shm/h100/lemonade/emb/lemonade-embeddable-11.7.0-ubuntu-x64
export CUDA_VISIBLE_DEVICES=6 HF_HOME=/mnt/gsma/gsma/gsma/models
$LEM/lemond /dev/shm/h100/lemonade/cache --port 8350 --host 127.0.0.1 --no-broadcast &
$LEM/lemonade --port 8350 --no-discovery backends install llamacpp:cuda
$LEM/lemonade --port 8350 --no-discovery pull user.Qwen3-06B-Q8 \
  --checkpoint main unsloth/Qwen3-0.6B-GGUF:Q8_0 --recipe llamacpp
$LEM/lemonade --port 8350 --no-discovery load user.Qwen3-06B-Q8
```

**Exact smoke command + real output:**

```bash
.env_lemonade/bin/python inference_llm_lemonade.py --port 8350 --model Qwen3-06B-Q8 \
  --max_tokens 900 --out /dev/shm/h100/out/lemonade/chat_single_gpu.json
```

```text
endpoint   : http://127.0.0.1:8350/v1/chat/completions
model      : Qwen3-06B-Q8
fingerprint: b10394-680a9ae63
latency_s  : 0.391
finish     : stop
--- generated text ---
An OpenTelemetry span is a way to track the flow of data and operations within a
distributed system, providing detailed information about each step in the process. It
enables monitoring and logging of application performance, helping in debugging and
optimizing the system.
--- llama.cpp timings ---
prompt_n            : 22
predicted_n         : 212
predicted_per_second: 584.20 tok/s
```

Coherent, correct, `finish_reason=stop`. **No `draft_n`** — unlike the 27B, the 0.6B has no
MTP head, so Lemonade does not enable `--spec-type draft-mtp` here (correctly). The wrapped
command line Lemonade built (note: no `-ngl`, same as MI355X, and `--ctx-size 4096` chosen
for this small model):

```text
/dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server \
  -m …/models--unsloth--Qwen3-0.6B-GGUF/…/Qwen3-0.6B-Q8_0.gguf \
  --ctx-size 4096 --port 8001 --jinja --metrics --reasoning-format auto --no-ui
```

**GPU-residency proof.** `nvidia-smi` VRAM-by-PID, filtered to my llama-server on the GPU-6
UUID (`GPU-e4fe48bc` = physical index 6):

```text
1720513, /dev/shm/h100/lemonade/cache/bin/llamacpp/cuda/llama-server, 1674 MiB, GPU-e4fe48bc-…
6, GPU-e4fe48bc-0c29-f21e-6523-759f964bf823, 1683 MiB
```

**1.67 GB attributed to our PID on GPU 6.** 610 MB Q8_0 weights + KV cache + CUDA context.
A CPU-only run would show 0 MiB here and would not sustain 584 tok/s for a Q8 0.6B
(CPU would be ~1–2 orders of magnitude slower). Backend, VRAM, and throughput all agree:
the model is on the GPU.

**Multi-GPU (deferred).** Same as MI355X: Lemonade exposes no `--tensor-split`/`-ngl`, so
`CUDA_VISIBLE_DEVICES` is the only device control. A multi-GPU pass would set e.g.
`CUDA_VISIBLE_DEVICES=6,7` on the `lemond` process (llama.cpp default `--split-mode layer`)
— but for a model that fits on one 80 GB card that only adds a device-to-device hop, so the
right pattern here is one server per GPU. Not launched (production on GPUs 0–3).

**Deviations from the MI355X recipe:** (1) backend `llamacpp:cuda` (b10397) instead of
`llamacpp:rocm` (b10470); (2) smaller GGUF (0.6B vs 27B) for a fast smoke, no GGUF cached
on this box; (3) no MTP/spec-decode (0.6B has no MTP head); (4) client venv built in tmpfs
(`/dev/shm`) — a `python3 -m venv` on the `/mnt/gsma` NFS mount did not create console
scripts / pip reliably, so use `python -m pip` or a tmpfs venv there.

**H100 VERDICT: PASS on the GGUF/llama.cpp CUDA path.** Zero-build, arch-matched sm_90
backend in ~41 s, 584 tok/s, coherent output, 1.67 GB resident on GPU 6. **No `vllm:cuda`
backend exists in Lemonade** (`vllm` lists only a `rocm` row), so — exactly as FP8/vLLM was
unreachable on gfx950 — there is no high-throughput-engine / native-FP8 path through
Lemonade on NVIDIA either. On both vendors Lemonade is a **llama.cpp front-end**.

## VERDICT (MI355X)

**PASS on the GGUF/llama.cpp path. FAIL — evidenced and hard-gated — on the experimental
vLLM ROCm path.**

`lemonade backends install llamacpp:rocm` auto-detected `gfx950`, fetched an arch-matched
`rocm_sdk_device_gfx950` wheel plus a prebuilt ROCm `llama.cpp`, and was ready in **27
seconds** with **no compiler, no cmake, no arch flags**. The specific risk flagged going in
— a prebuilt ROCm backend built only for gfx90a/gfx942 dying with `hipErrorNoBinaryForGpu`
— **did not materialise**: the fat `libggml-hip.so` carries a gfx950 code object, and the
27B ran clean.

Qwen3.8-27B-Q8_0 serves at **143.3 tok/s single-GPU** with **50.69 GB of VRAM and 7 live
KFD compute queues on GPU 4**, producing correct prose and correct runnable Python, with
**MTP speculative decoding enabled automatically** (63-72 % draft acceptance). Multi-GPU
layer split works and is numerically transparent but is **1.8 % slower** — correct, since
a 27 GB model has no reason to leave one 288 GB card.

The experimental **`vllm:rocm` backend is refused outright**: `Unsupported GPU: gfx950`,
at both `backends install` and `pull --recipe vllm`. The FP8 checkpoint is therefore
**unreachable through Lemonade on MI355X**, even though Lemonade's own
`backend_versions.json` already pins a gfx950 vLLM build. On this hardware Lemonade is a
**llama.cpp front-end only**.

Net: an outstanding zero-friction local-serving story on MI355X for GGUF, and no path to
FP8.

## Follow-ups

- Benchmark concurrent throughput; 143 tok/s is single-stream and Lemonade fixes
  `llama-server`'s parallelism settings.
- Quantify what `--spec-type draft-mtp` is actually worth here by comparing against raw
  `llama.cpp` without it (`inference/llamacpp/llm` has no MTP head configured).
- Re-test `vllm:rocm` on a later Lemonade release — the version pin exists, so the runtime
  gate looks like the only blocker. Watch `lemonade-sdk/vllm-rocm` for a gfx950 portable
  build.
- Ask upstream for a per-model `--ctx-size` / `-ngl` override; 262144 tokens of KV is a
  lot to spend by default on a smaller card.
- Watch for a Ubuntu-24.04 `.deb`; the `embeddable` tarball has no systemd unit, so the
  server here is a bare background process.
