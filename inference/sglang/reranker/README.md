# `inference/sglang/reranker` — SGLang reranker serving (Qwen3-Reranker-0.6B)

## Overview & when to use

Serve `Qwen/Qwen3-Reranker-0.6B` behind `POST /v1/rerank` with **SGLang**
(`python -m sglang.launch_server`), and score query/document pairs with
`inference_reranker_sglang.py`, which prints the real relevance scores in ranked order.

Qwen3-Reranker is **not** a cross-encoder with a classification head: it is a
**decoder-only yes/no reranker**, scored from the logits of the `yes`/`no` tokens under a
specific prompt. Two consequences drive this whole folder:

- **Never launch it with `--is-embedding`.** That flag is for true embedding models
  (see the sibling EmbeddingGemma folder); using it here mis-configures the runner.
- **It needs the `qwen3_reranker` jinja chat template**, shipped here as
  `qwen3_reranker.jinja`, which renders the Instruct/Query/Document prompt the model was
  trained on.

Use this folder when SGLang already serves your generation traffic and you want reranking
in the same engine. On **AMD gfx950 it does not serve from a pip install** — use the
container route below, the sentence-transformers `CrossEncoder`, or the vLLM reranker route.

## H100 (NVIDIA) — the pip route

**On NVIDIA the pip route works** — `sgl_kernel` ships as a CUDA wheel, so
`pip install "sglang[all]"` serves `/v1/rerank` directly, **no container**. The reranker
configuration rules are unchanged: no `--is-embedding`, the `qwen3_reranker.jinja` template,
`--disable-radix-cache`. The commands below are single-GPU.

### H100 install (pip route)

Same shared venv as the LLM/embedding leaves (full block in `llm/README.md` H100 section):

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and run artifacts
```

```bash
python3 -m venv .env_sglang && source .env_sglang/bin/activate
pip install -U pip && pip install torch numpy            # the current CUDA 13 build
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
pip install "sglang[all]"                                 # sglang 0.5.18, sglang-kernel 0.4.6.post1, flashinfer 0.6.17
python -c "import sgl_kernel; print('sgl_kernel OK')"     # OK — impossible on ROCm
```

### H100 serve — single GPU

Same flags as the ROCm container command, minus the HIP env: **no `--is-embedding`**, the folder's
jinja template, `--disable-radix-cache`.

```bash
source .env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1   # plain CUDA — no HIP_VISIBLE_DEVICES; HF_HOME set above
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --chat-template ./qwen3_reranker.jinja \
  --disable-radix-cache --mem-fraction-static 0.3 \
  --host 127.0.0.1 --port 8600
```

On the pip route `--chat-template` takes a **host** path (`./qwen3_reranker.jinja`) — there is no
container bind-mount, so the `/work/...` path the ROCm section uses does not apply here.

**Expected output** (server log):

```
Loading chat template from argument: ./qwen3_reranker.jinja
The server is fired up and ready to roll!
```

`/get_model_info` confirms the reranker is served as a causal LM (not a pooling encoder):

```json
{"model_type": "qwen3", "is_generation": true, "architectures": ["Qwen3ForCausalLM"]}
```

### H100 client / smoke command

```bash
python inference_reranker_sglang.py --port 8600
```

**Expected output** (correct ordering):

```
[health] server ready after 1.0s
[latency] 0.06s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.777300  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.148047  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.000033  PostgreSQL is a relational database.
```

The relevant document must rank first and the database sentence last.

### H100 GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader
```

Only the card named by `CUDA_VISIBLE_DEVICES` is touched. The VRAM it holds is the
`--mem-fraction-static 0.3` KV pool, not real demand.

### H100 quirks / notes

- Same benign `torchcodec`/`libavutil` startup traceback as the other leaves — ignore it.
- **`--disable-radix-cache` is still correct** — every rerank pair is a distinct query+document
  prompt with no shared prefix, so prefix caching only wastes memory.
- Scale with independent single-GPU replicas (one `CUDA_VISIBLE_DEVICES` each), not `--tp`.

## Container route — `lmsysorg/sglang-rocm`

**This is the route that works on gfx950.**

Image: `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260819` — **89.9 GB on disk** (allow
>150 GB free on the filesystem holding it).

The container is started once and reused (see [`../embedding/README.md`](../embedding/README.md)
for the full `docker run`); the two `--device /dev/dri/renderD*` nodes you pass map the target
GPUs to `cuda:0`/`cuda:1` inside.

### Serve — single GPU

Note the flags: **no `--is-embedding`**, the folder's jinja template, and `--disable-radix-cache`.

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8102 \
     > $OUTPUT_DIR/inference_reranker_sglang/rerank_single.log 2>&1"
```

**Expected output** (`rerank_single.log`):

```
INFO:     Uvicorn running on http://0.0.0.0:8102 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

**`GET /get_model_info` confirms the "no `--is-embedding`" rule:**

```json
{"model_path": "Qwen/Qwen3-Reranker-0.6B", "is_generation": true,
 "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
```

`/v1/rerank` renders each (query, document) pair through `qwen3_reranker.jinja` and reads the
`yes` logit as the score. Adding `--is-embedding` would switch the model into a pooling
encoder, destroy the LM head the score comes from, and break `/v1/rerank`.

### Client / smoke command

```bash
docker exec -w /work/inference/sglang/reranker sglang_bringup \
  python inference_reranker_sglang.py --port 8102
```

**Expected output:**

```
[health] server ready after 1.0s
[latency] 0.14s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.777300  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.140336  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.000024  PostgreSQL is a relational database.
```

The relevant document must rank first, the unrelated database sentence last.

### Two concurrent single-GPU replicas (instead of TP=2)

For a 0.6B reranker scale with replicas, not tensor parallelism — one server per GPU:

```bash
# replica A -> first GPU, port 8102   (command above)
# replica B -> second GPU, port 8112
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8112 \
     > $OUTPUT_DIR/inference_reranker_sglang/rerank_rep_b.log 2>&1"
```

Hit both replicas concurrently (ports 8102 and 8112); the scores are identical across the two
GPUs, so a replica pool can be load-balanced without score drift reshuffling result ordering.

### Container-route quirks

- **`--chat-template` takes a container-visible path.** `/work` is the bind-mounted repo, so
  `/work/inference/sglang/reranker/qwen3_reranker.jinja` resolves. A host path will not.
- **`--disable-radix-cache` is correct here.** Every rerank pair is a distinct
  query+document prompt with essentially no shared prefix, so prefix caching only wastes memory.
- **AITER JIT probes a compiler flag and fails loudly at startup — this is benign:**
  ```
  clang (LLVM option parsing): Unknown command line argument '-amdgpu-coerce-illegal-types=1'.
  [aiter] -mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc.
  ```
  AITER drops the flag and rebuilds without it; startup continues. Do not chase this error.
- **AITER JIT-compiles kernels into `/root/.aiter/build/` on first launch.** Keep one
  long-lived container so that cache is reused.
- `Ignore import error when loading sglang.srt.models.inkling: No module named 'cutlass'` —
  benign, a CUDA-only model class failing to register.

## Install

Python 3.12; the shared [`../requirements.txt`](../requirements.txt) is the pinned set. Same
stack as the other two SGLang leaves, so one venv can be shared.

### AMD / ROCm

```bash
python3 -m venv .env_sglang
source .env_sglang/bin/activate
pip install -U pip
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install --no-deps sglang==0.5.17
pip install -r ../requirements.txt
python -c "import torch; print(torch.__version__, torch.version.hip)"   # 2.11.0+rocm7.2 7.2.26015
```

`--no-deps` is mandatory: sglang 0.5.17 declares CUDA packages and the PyPI CUDA
`torch==2.11.0` as **base** dependencies, so a plain install replaces ROCm torch with a CUDA
wheel. Recovery:

```bash
pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Never `pip install flash-attn` (CUDA-only). Never `pip install aiter` — that PyPI name is
an unrelated async-iterator library, not AMD's AITER.

The three SGLang leaves share one identical environment: build a single `.env_sglang` at the
software root (`inference/sglang/`) and activate it from `llm/`, `embedding/` and `reranker/`.

This venv cannot serve on ROCm (`aiter` and `sgl_kernel` have no ROCm wheel), so use it only
for the client; serve from the container.

### NVIDIA / CUDA (see the H100 section above for the full variant)

```bash
pip install --upgrade pip && pip install uv
uv pip install sglang==0.5.17
```

## Environment & secrets

`dev.env` is symlinked to the repo root (`ln -sf ../../../dev.env dev.env`):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`Qwen/Qwen3-Reranker-0.6B` is ungated, so the token is only needed if you swap in a gated
model. The client loads it via `load_dotenv("dev.env")`; run from inside this folder. Never
print or commit the token.

```bash
export HF_HOME=/path/to/hf_cache           # weights must not land on /
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1            # never empty on ROCm
```

## Run

### Launch — single GPU

```bash
source .env_sglang/bin/activate
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HF_HOME=/path/to/hf_cache

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --chat-template qwen3_reranker.jinja \
  --mem-fraction-static 0.5 \
  --host 0.0.0.0 --port 8102
```

Note what is **absent**: no `--is-embedding` (wrong for a decoder-only yes/no reranker) and
no `--attention-backend` (on ROCm SGLang auto-selects `aiter`; `flashinfer` is NVIDIA-only).
`--disable-radix-cache` is right here — prefix reuse across unrelated query/document pairs is
not useful.

### Launch — multi-GPU, TP=2

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
python -m sglang.launch_server --model-path Qwen/Qwen3-Reranker-0.6B --trust-remote-code \
  --disable-radix-cache --chat-template qwen3_reranker.jinja \
  --tp 2 --mem-fraction-static 0.4 --host 0.0.0.0 --port 8102
```

### Client / smoke command

```bash
python inference_reranker_sglang.py --port 8102 \
  --query "Which inference engines support AMD ROCm?" \
  --documents "vLLM and SGLang both support AMD ROCm GPUs." \
              "PostgreSQL is a relational database." \
              "ROCm is AMD's open compute platform for Instinct accelerators."
```

Expected on a working build — the database sentence must rank last:

```
[health] server ready after <N>s
[latency] 0.0x s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.9xxxxx  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.xxxxxx  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.0xxxxx  PostgreSQL is a relational database.
```

## Where the ROCm pip route stops

Under pip on gfx950 the engine accepts the reranker configuration in full, loads the weights
and allocates the KV pool, then dies in CUDA-graph capture at `sgl_kernel.rotary_embedding`,
which has no ROCm build; the endpoint never reaches `/health`. TP=2 hits the same wall. For
production scaling run two independent single-GPU replicas (see the container route above),
not TP.

## Arguments

Client (`inference_reranker_sglang.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | SGLang server host |
| `--port` | `8102` | Server port (8102 = SGLang reranker slot in the repo port map) |
| `--model` | `Qwen/Qwen3-Reranker-0.6B` | Model id as served; must match `--model-path` |
| `--query` | ROCm engines question | The rerank query |
| `--documents` | 3 sample documents | Candidates scored against the query |
| `--top_n` | `None` | If set, ask the server for only the top N |
| `--endpoint` | `/v1/rerank` | Endpoint path |
| `--wait` | `600` | Seconds to wait for `/health` |
| `--timeout` | `120` | Per-request timeout |

Server flags that matter:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3-Reranker-0.6B` | Decoder-only yes/no reranker |
| `--chat-template` | `qwen3_reranker.jinja` | **Required** — renders the Instruct/Query/Document prompt |
| `--disable-radix-cache` | on | Upstream reranker recipe; prefix reuse is useless across pairs |
| `--trust-remote-code` | on | Follows the upstream command |
| `--is-embedding` | **never** | Wrong for this model — it is not an embedding model |
| `--tp` | `1` | Correct for a 0.6B model; scale with replicas, not TP |
| `--attention-backend` | *unset* | **Leave unset on ROCm** — SGLang selects `aiter` |

## Output

No artifacts are written by the server. Logs land wherever you redirect them, e.g.
`$OUTPUT_DIR/inference_reranker_sglang/`. Weights live in `$HF_HOME`. The client prints
latency and ranked scores to stdout only.

## Hardware support

| | NVIDIA | AMD |
|---|---|---|
| Status | **Works** — H100 80GB, CUDA 13.0, pip route (see the H100 section above) | **pip route blocked** — MI355X (gfx950), ROCm 7.2.4; container route works |
| Install | `uv pip install sglang` | pip route unusable; needs `lmsysorg/sglang-rocm` or a hipcc source build |
| Runner | decoder-only yes/no scoring | same — `is_embedding=False` on gfx950 |

## Notes / quirks

- **Do not pass `--is-embedding`.** Leave it off; SGLang then starts a causal-LM
  runner (`is_embedding=False`), which is what `/v1/rerank` needs.
- **Template provenance matters.** `qwen3_reranker.jinja` renders the exact
  system + `<Instruct>/<Query>/<Document>` prompt whose `yes`/`no` logits define the score.
  A different template silently changes the scores rather than erroring.
- **KV cache is large by default** even for a 0.6B model. Set `--mem-fraction-static` or
  `--max-total-tokens` explicitly on a shared box.
- `Failed to import amdsmi` on every launch — harmless; install `amdsmi` for AMD telemetry.
- `Ignoring corrupted tree cache file ... Permission denied` — shared HF cache owned by
  another user; cosmetic.
- `--mem-fraction-static` is auto-reduced at TP>1 (0.4 → 0.34 at TP=2).
- **Ports** — 8102 is the SGLang reranker slot; 8100/8101 belong to sibling folders.
- Re-export `HIP_VISIBLE_DEVICES` **and** `CUDA_VISIBLE_DEVICES` after activating the venv;
  never set `CUDA_VISIBLE_DEVICES` empty on ROCm.
- **Fullest AMD container flag set**, if the minimal one hits a permission or memory error:
  `--device /dev/kfd --device /dev/dri/renderD<a> --device /dev/dri/renderD<b> --group-add
  video --group-add render --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
  --shm-size 64G`.
