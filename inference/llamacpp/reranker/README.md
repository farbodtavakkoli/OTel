# `inference/llamacpp/reranker` — llama.cpp GGUF reranker serving (Qwen3-Reranker-0.6B)

## Overview & when to use

Serves **Qwen3-Reranker-0.6B** as GGUF through `llama-server` with `--reranking`, built
against **ROCm 7.2 / HIP** (AMD Instinct MI355X, gfx950) or **CUDA 13** (NVIDIA H100). The
client
`inference_reranker_llamacpp.py` hits `POST /v1/rerank` and verifies that relevant
documents actually outrank irrelevant ones.

Use this folder for a cheap second-stage reranker in a retrieval pipeline, served by the
**same binary** as the LLM and embedding leaves. HF TEI does not support this exact Qwen3
reranker, so llama.cpp (or vLLM/SGLang) is the route.

> **Format note:** llama.cpp needs **GGUF**, not the HF safetensors. This folder serves
> `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0`, the converted artifact of
> `Qwen/Qwen3-Reranker-0.6B`. The repo is ungated — downloaded here with no `HF_TOKEN`.

> **Architecture note:** Qwen3-Reranker is a **decoder-style yes/no reranker**, not a
> classic cross-encoder with a regression head. llama.cpp implements the yes/no
> logit-comparison internally and exposes it as a plain `relevance_score` in `[0, 1]`,
> so the client never has to build the yes/no prompt itself.

## Build — see [`../README.md`](../README.md)

One llama.cpp HIP build serves all three `inference/llamacpp/*` leaves — the full
ROCm build recipe (prerequisites, cmake flags, the mandatory `-DLLAMA_OPENSSL=ON`,
NVIDIA variant) lives in [`../README.md`](../README.md).

Reference revision for the ROCm recipe: commit
**`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`**, `llama-server` **0.1.2-dev (build 1)**,
ggml **0.20.2**.

## Environment & secrets

```bash
ln -sf ../../../dev.env dev.env                 # already done in this folder
export $(grep -v '^#' dev.env | xargs)          # only when a gated repo needs HF_TOKEN
```

Not required for this model. Never echo or commit `HF_TOKEN`.

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache             # Hugging Face model cache
export LLAMA_CACHE=$HF_HOME/llama_cpp        # llama.cpp's own -hf cache
export OUTPUT_DIR=/path/to/outputs           # inference artifacts

export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0      # the GPU this server uses
```

`LLAMA_CACHE` is what actually controls where `-hf` writes; point it at a filesystem
with room rather than the root filesystem. This model is 610 MB on disk.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Serve

### Single GPU

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
# HF_HOME / LLAMA_CACHE as exported above

./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --reranking \
  -ngl 99 \
  -lv 5 \
  --host 127.0.0.1 --port 8202 \
  --alias qwen3-reranker-0.6b
```

`--reranking` is required — without it the rerank routes are not served.
The first run downloads 610 MB; cached restarts take under a second.

### Two concurrent instances, one per GPU (the production pattern)

```bash
# instance A — first GPU, port 8212
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --reranking -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8212 --alias qwen3-reranker-0.6b-gpu0 &

# instance B — second GPU, port 8202
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --reranking -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8202 --alias qwen3-reranker-0.6b &
```

## Client / smoke command

One shared venv at the software root serves all three leaves — build it from
`../requirements.txt`:

```bash
cd .. && python3 -m venv .env_llamacpp && .env_llamacpp/bin/pip install -r requirements.txt && cd reranker

../.env_llamacpp/bin/python inference_reranker_llamacpp.py \
  --port 8202 \
  --out $OUTPUT_DIR/inference_reranker_llamacpp/rerank_single_gpu.json
```

Equivalent raw curl:

```bash
curl -s -X POST http://127.0.0.1:8202/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker-0.6b",
       "query":"What is an OpenTelemetry span?",
       "documents":["A span is a unit of work in a trace.","Bake bread at 200C."]}' \
  | jq -c '.results'
```

## Expected output

Two relevant documents interleaved with two irrelevant ones, shown after the client
re-sorts by score:

```text
endpoint  : http://127.0.0.1:8202/v1/rerank
model     : qwen3-reranker-0.6b
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=+0.99966  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99639  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00004  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00002  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

Both OpenTelemetry documents must outrank both distractors. The input order was
relevant/irrelevant/relevant/irrelevant, so the model reorders rather than echoing input
order. The **first request is slow** (prompt-cache and graph warm-up) — send a throwaway
request before putting an instance into rotation.

**GPU-residency check** — `-ngl 99` can silently fall back to CPU, so check the
offload summary at `-lv 5`:

```text
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   157.37 MiB
load_tensors:        ROCm0 model buffer size =   603.87 MiB
```

Confirm with `rocm-smi` and with `rocm-smi --showpids`, which attributes the allocation to
the `llama-server` PID. All 29 layers are on the GPU; the `CPU_Mapped` buffer is the
token-embedding table, kept host-side by design.

The alternate route `/rerank` (`/v1/reranking` is also listed) returns the same
scores:

```text
[{"index":0,"relevance_score":0.9925786852836609},
 {"index":1,"relevance_score":0.00003648855636129156}]
```

## Multi-GPU

**Do not split a 0.6B reranker across two GPUs** — a layer split only adds a
device-to-device transfer per forward pass, and `--split-mode row` is unavailable on this
backend anyway (see the LLM leaf).

The pattern is **one independent server instance per GPU**, load-balanced (commands above).
Concurrent instances produce identical relevance scores, so a load balancer can route to
either without changing ranking.

## H100 (NVIDIA, CUDA)

Mirror of the MI355X setup on **NVIDIA H100 80GB HBM3**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12. Same binary, same **exact model**
(`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` — only 610 MB, no substitution needed),
same client, same `/v1/rerank` path. Only the backend build flag changes:
`-DGGML_CUDA=ON` instead of `-DGGML_HIP=ON` (full recipe in
[`../README.md`](../README.md); `-DLLAMA_OPENSSL=ON` is kept — vendor-neutral, required
for the `-hf` HTTPS pull). Reference revision: `llama-server` **0.2.0-dev (build 1, commit
`70adb1b`)**, ggml **0.21.0**.

```bash
cd /dev/shm/llamacpp/llama.cpp                   # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=0
# HF_HOME as exported above
export LLAMA_CACHE=/dev/shm/llamacpp/model_cache   # tmpfs, see quirks below

./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 \
  --reranking -ngl 999 -lv 5 \
  --host 127.0.0.1 --port 8700 --alias qwen3-reranker-0.6b
```

Client (served here on port 8700):

```bash
../.env_llamacpp/bin/python inference_reranker_llamacpp.py --port 8700 --model qwen3-reranker-0.6b
```

**GPU-residency check** (`-ngl` can silently fall back to CPU). Startup log at `-lv 5`:

```text
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   157.37 MiB
load_tensors:        CUDA0 model buffer size =   603.87 MiB
```

All **29/29 layers on the GPU**, with the same `CUDA0` buffer size as the MI355X `ROCm0` one
(same GGUF). The `CPU_Mapped` buffer is the token-embedding table, host-side by design.
Confirm with
`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`, which should
attribute the server's VRAM to the `llama-server` PID on the selected card.

**Reranking is correct** — client output (2 relevant docs interleaved with 2
irrelevant, re-sorted by the client):

```text
1. score=+0.99968  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99656  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00005  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00003  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

Both relevant docs must outrank both distractors, as on MI355X, and the input order is
interleaved so this is a real reordering. The alternate `/rerank` route returns the same
scores.

**Quirks (shared with MI355X):** the harmless minja
`Callee is not a function: got Undefined (hint: 'lstrip')` chat-template error prints at
load — reranking does not go through the chat template, so scoring is unaffected (proven
by the correct rankings). `--reranking` is mandatory or the rerank routes 404. This GGUF
is ungated and pulls with no `HF_TOKEN`; if the host has no cmake/ninja, install them
into a throwaway venv (see `../README.md`) and pull the weights to a tmpfs
`LLAMA_CACHE`.

**Multi-GPU:** the commands above are single-GPU. As on MI355X, do **not** split this
reranker — run one instance per GPU behind a load balancer.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Value | Meaning |
|---|---|---|
| `-hf <repo>:<quant>` | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` | Pull GGUF from the Hub. Needs an SSL-enabled build |
| `--reranking` | on | **Required** to expose `/v1/rerank`, `/rerank`, `/v1/reranking` |
| `-ngl N` | `99` | Layers offloaded to GPU. 99 = "all"; verify via the offload summary |
| `--host` / `--port` | `127.0.0.1` / `8202` | Bind address |
| `--alias` | `qwen3-reranker-0.6b` | Name reported in the API `model` field |
| `-lv N` | `5` | Log verbosity. **Default 3 hides the offload summary** |
| `-c N` | model default | Context length; Qwen3-Reranker supports 32K |

### `inference_reranker_llamacpp.py`

| Argument | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | llama-server host |
| `--port` | `8202` | llama-server port |
| `--model` | `qwen3-reranker-0.6b` | Model name echoed in the request body |
| `--query` | OTel span question | Query to rank documents against |
| `--documents` | 2 relevant + 2 irrelevant | Candidate documents |
| `--top_n` | `None` | Return only the top N documents |
| `--endpoint` | `/v1/rerank` | Route: `/v1/rerank`, `/rerank`, or `/v1/reranking` |
| `--timeout` | `300` | HTTP timeout (s) |
| `--health_retries` | `60` | `/health` polls before giving up |
| `--out` | `None` | Write the raw JSON response here |

## Output

The client writes the raw JSON response wherever `--out` points, e.g.
`$OUTPUT_DIR/inference_reranker_llamacpp/`; redirect the server log there too. Model
weights (610 MB) live in `$LLAMA_CACHE/`, never on the root filesystem.

## Hardware support

- **AMD Instinct MI355X (gfx950, ROCm 7.2):** HIP build works with no source patches;
  scale out with one instance per GPU.
- **NVIDIA H100 (Hopper cc 9.0, CUDA 13):** CUDA build works with no source patches.
- All three routes work on both: `/v1/rerank`, `/rerank`, `/v1/reranking`.

## Notes & quirks

1. **`--reranking` is mandatory.** Without it the server starts and the rerank routes
   404.
2. **The standard build recipe is incomplete for `-hf`.** `cmake -B build -DGGML_HIP=ON`
   compiles but cannot download: this revision replaced libcurl with bundled
   cpp-httplib, which needs TLS. Symptom is
   `get_repo_commit: error: HTTPS is not supported`, followed by the misleading
   `failed to load model ''`. Fix: `apt install libssl-dev` + `-DLLAMA_OPENSSL=ON`.
3. **A chat-template Jinja error is printed at startup and is harmless:**

   ```text
   Error: Callee is not a function: got Undefined (hint: 'lstrip')
   Error: Callee is not a function: got Undefined (hint: 'lstrip')
   ```

   llama.cpp's minja engine cannot fully parse the GGUF's embedded chat template
   (it uses a filter minja does not implement). Reranking does **not** go through the
   chat template, so scoring is unaffected — verified by the correct rankings above.
   It would matter only if you tried to use this model for chat.
4. **The first request is much slower than subsequent ones** because of prompt-cache and
   graph warm-up. Send one throwaway request at startup before putting an instance into
   rotation.
5. **Scores are absolute, not relative.** They come from a yes/no logit comparison, so
   they are comparable across queries — you can threshold on them (e.g. drop anything
   below 0.5) rather than only ranking within a candidate set.
6. **Default log verbosity hides the GPU-offload proof.** Use `-lv 5` when you need to
   prove residency.
7. **`LLAMA_CACHE`, not `HF_HOME`, controls where `-hf` writes.**
8. **Do not split this model across GPUs.** Run one instance per GPU instead.
