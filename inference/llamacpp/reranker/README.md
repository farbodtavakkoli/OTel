# `inference/llamacpp/reranker` — llama.cpp GGUF reranker serving (Qwen3-Reranker-0.6B) on ROCm/HIP

## Overview & when to use

Serves **Qwen3-Reranker-0.6B** as GGUF through `llama-server` with `--reranking`, built
against **ROCm 7.2.4 / HIP** for **AMD Instinct MI355X (gfx950)**. The client
`inference_reranker_llamacpp.py` hits `POST /v1/rerank` and verifies that relevant
documents actually outrank irrelevant ones.

Use this folder when you want:

- **A cheap second-stage reranker** in a retrieval pipeline — 604 MiB on the GPU,
  **~16 ms** to score 4 documents against a query.
- **The same binary as your LLM and embedding services** — one llama.cpp build covers
  all three workloads.
- **A reranker path that actually exists on AMD.** HF TEI does **not**
  currently support this exact Qwen3 reranker, so llama.cpp (or vLLM/SGLang) is the
  route.

> **Format note:** llama.cpp needs **GGUF**, not the HF safetensors. This folder serves
> `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0`, the converted artifact of
> `Qwen/Qwen3-Reranker-0.6B`. The repo is ungated — downloaded here with no `HF_TOKEN`.

> **Architecture note:** Qwen3-Reranker is a **decoder-style yes/no reranker**, not a
> classic cross-encoder with a regression head. llama.cpp implements the yes/no
> logit-comparison internally and exposes it as a plain `relevance_score` in `[0, 1]`,
> so the client never has to build the yes/no prompt itself.

> **Tested topology:** 2xAMD Instinct MI355X (gfx950, 288 GB each), physical GPUs 6
> and 7, ROCm 7.2.4, Ubuntu, Python 3.12.3.

## Build — see [`../README.md`](../README.md)

One llama.cpp HIP build serves all three `inference/llamacpp/*` leaves — the full
ROCm build recipe (prerequisites, cmake flags, the mandatory `-DLLAMA_OPENSSL=ON`,
NVIDIA variant) lives in [`../README.md`](../README.md); a cold build takes about
40.6 s.

Verified at commit **`d59d455fd8ea09e5a2e87ce2a9d668267ffb5ccd`** (Wed Aug 19 2026),
`llama-server` **0.1.2-dev (build 1)**, ggml **0.20.2**.

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

export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7      # this folder's test GPU
```

`LLAMA_CACHE` is what actually controls where `-hf` writes; point it at a filesystem
with room rather than the root filesystem. This model is 610 MB on disk.

> **Never set `CUDA_VISIBLE_DEVICES=""` on ROCm** — an empty string hides every device
> and the server silently falls back to CPU.

## Serve

### Single GPU (physical GPU 7)

```bash
cd <your llama.cpp checkout>          # built per ../README.md
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7
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
First run downloads 610 MB and is ready in ~8 s; cached restarts take under 1 s.

### Two concurrent instances, one per GPU (the production pattern)

```bash
# instance A — physical GPU 6, port 8212
HIP_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=6 ./build/bin/llama-server \
  -hf ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0 --reranking -ngl 99 -lv 5 \
  --host 127.0.0.1 --port 8212 --alias qwen3-reranker-0.6b-gpu6 &

# instance B — physical GPU 7, port 8202
HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 ./build/bin/llama-server \
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

## Results — single GPU

**Expected client output** — **2 relevant documents interleaved with 2 irrelevant ones**,
shown after the client re-sorts by score:

```text
endpoint  : http://127.0.0.1:8202/v1/rerank
model     : qwen3-reranker-0.6b
latency_s : 0.942
query     : What is an OpenTelemetry span?
--- ranked documents (best first) ---
1. score=+0.99966  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99639  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00004  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00002  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

The ranking is **correct and decisively separated**: both OpenTelemetry documents score
above **0.996**, both distractors below **0.00004** — a gap of roughly **4 orders of
magnitude**. Note the input order was relevant/irrelevant/relevant/irrelevant, so the
model reordered rather than echoing input order.

`0.942 s` is the cold first request (prompt-cache warm-up). **Warm requests are 16 ms:**

```text
latency_s : 0.016     # port 8202, GPU 7
latency_s : 0.025     # port 8212, GPU 6
```

**GPU-residency proof** — `-ngl 99` can silently fall back to CPU, so here is the
offload summary at `-lv 5`:

```text
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   157.37 MiB
load_tensors:        ROCm0 model buffer size =   603.87 MiB
```

Confirmed by `rocm-smi` — card7 held 1.92 GB against a 0.30 GB idle baseline on the
other cards while this server ran, and `rocm-smi --showpids` attributed the allocation
to the `llama-server` PID. All 29 layers are on the GPU; the 157 MiB `CPU_Mapped`
buffer is the token-embedding table, kept host-side by design.

The alternate route `/rerank` (`/v1/reranking` is also listed) returns the same
scores:

```text
[{"index":0,"relevance_score":0.9925786852836609},
 {"index":1,"relevance_score":0.00003648855636129156}]
```

## Results — multi-GPU

**Splitting a 0.6B reranker across two GPUs is pointless, and this README will not
pretend otherwise.** The model is 604 MiB of GPU buffer on a card with 288 GB of VRAM —
0.2 % occupancy. A layer split would put ~15 layers on each GPU and add a
device-to-device transfer to every forward pass, making a 16 ms request *slower* with
zero capacity benefit. `--split-mode row` is unavailable on this backend regardless
(see the LLM folder's README for the root-cause trace).

The honest production pattern is **one independent server instance per GPU**,
load-balanced. Demonstrated with two concurrent instances:

```text
### instance A (GPU 6, port 8212)
1. score=+0.99966  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99639  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00004  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00002  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True

### instance B (GPU 7, port 8202)
1. score=+0.99966  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99639  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00004  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00002  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

The two GPUs produce **bit-identical relevance scores**, so a load balancer can route to
either without changing ranking. Both instances reported
`offloaded 29/29 layers to GPU` with `ROCm0 model buffer size = 603.87 MiB`, and both
cards show live residency:

```text
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card6,309220868096,9601277952     <-- 9.60 GB
card7,309220868096,9601253376     <-- 9.60 GB
```

(Each card also runs the embedding folder's server; the per-card total is dominated by
KV-cache and HIP context allocation rather than the 604 MiB of weights.)

## H100 (NVIDIA, CUDA)

Mirror of the MI355X run above, on **1x NVIDIA H100 80GB HBM3** (physical GPU 7,
`CUDA_VISIBLE_DEVICES=7`), driver **580.173.02**, **CUDA 13.0**, Hopper cc 9.0,
Python 3.12.3. Same binary, same **exact model**
(`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` — only 610 MB, no substitution needed),
same client, same `/v1/rerank` path. Only the backend build flag changes:
`-DGGML_CUDA=ON` instead of `-DGGML_HIP=ON` (full recipe in
[`../README.md`](../README.md); `-DLLAMA_OPENSSL=ON` is kept — vendor-neutral, required
for the `-hf` HTTPS pull). Verified `llama-server` **0.2.0-dev (build 1, commit
`70adb1b`)**, ggml **0.21.0**.

```bash
cd /dev/shm/llamacpp/llama.cpp                   # CUDA build per ../README.md
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # HF pull
export CUDA_VISIBLE_DEVICES=7
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

**GPU-residency proof** (`-ngl` can silently fall back to CPU). Startup log at `-lv 5`:

```text
load_tensors: offloaded 29/29 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   157.37 MiB
load_tensors:        CUDA0 model buffer size =   603.87 MiB
```

All **29/29 layers on the GPU**; the **CUDA0 buffer is 603.87 MiB — bit-identical to the
MI355X ROCm0 buffer** (same GGUF). The 157 MiB `CPU_Mapped` buffer is the token-embedding
table, host-side by design. Confirmed by `nvidia-smi` filtered to GPU 7, by PID:

```text
$ nvidia-smi -i 7 --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
<pid>, ./build/bin/llama-server, 6010 MiB
```

**Reranking is correct** — client output (2 relevant docs interleaved with 2
irrelevant, re-sorted by the client):

```text
latency_s : 0.134
1. score=+0.99968  idx=2  OpenTelemetry spans nest inside a trace to describe the path of a request
2. score=+0.99656  idx=0  A span represents a single unit of work in a distributed trace and carries a start time, d
3. score=+0.00005  idx=1  To bake sourdough, feed the starter twelve hours before mixing the dough.
4. score=+0.00003  idx=3  The 1998 football World Cup final was played in Saint-Denis.
top_hit_is_relevant : True
```

Both relevant docs at **0.99968 / 0.99656**, both distractors at **0.00005 / 0.00003** —
a ~4-order-of-magnitude gap, matching MI355X (0.99966 / 0.99639 / 0.00004 / 0.00002).
Input order is interleaved, so the model reorders rather than echoing input order. The
alternate `/rerank` route returns the same scores
(`[{"index":0,"relevance_score":0.9925...},{"index":1,"relevance_score":3.36e-05}]`).
**Warm latency 28 ms** for a 4-document request (vs 16 ms on MI355X — same order).

**Quirks (shared with MI355X):** the harmless minja
`Callee is not a function: got Undefined (hint: 'lstrip')` chat-template error prints at
load — reranking does not go through the chat template, so scoring is unaffected (proven
by the correct rankings). `--reranking` is mandatory or the rerank routes 404. This GGUF
is ungated and pulls with no `HF_TOKEN`; if the host has no cmake/ninja, install them
into a throwaway venv (see `../README.md`) and pull the weights to a tmpfs
`LLAMA_CACHE`.

**Multi-GPU:** only GPU 7 is used in this single-GPU smoke. As on MI355X, do **not**
split a 604 MiB reranker — run one instance per GPU behind a load balancer.

**On H100 this path works.** All 29/29 layers on the H100 (CUDA0 603.87 MiB + `nvidia-smi` by
PID), correct rankings with a decisive margin, 28 ms warm. Both `/v1/rerank` and `/rerank`
routes work.

## Arguments

### `llama-server` (the ones that matter here)

| Argument | Used | Meaning |
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

Artifacts go to `$OUTPUT_DIR/inference_reranker_llamacpp/`, never to the root
filesystem:

```text
server_single_gpu.log        offload summary + load trace, GPU 7
server_instance_gpu6.log     second concurrent instance, GPU 6
rerank_single_gpu.json       raw /v1/rerank response
client_single_gpu.txt        client stdout
rocm_smi_two_instances.csv   per-card VRAM with both instances live
```

Model weights (610 MB) live in `$LLAMA_CACHE/`, never on the root filesystem.

## Hardware support & evidence

| Claim | Evidence |
|---|---|
| HIP build works on gfx950 | 681/681 ninja targets, EXIT=0, no patches |
| GPUs detected | `--list-devices` → `ROCm0`/`ROCm1` MI355X, 294896 MiB each |
| Model really on GPU | `offloaded 29/29 layers to GPU`, `ROCm0 model buffer size = 603.87 MiB` |
| VRAM occupied | `rocm-smi`: card7 1.92 GB vs 0.30 GB idle baseline |
| Reranking is real | Relevant docs 0.99966 / 0.99639, distractors 0.00004 / 0.00002 |
| Ranking is *correct* | Input was interleaved; both relevant docs were promoted to top-2 |
| Warm latency | 16 ms (GPU 7) / 25 ms (GPU 6) for a 4-document request |
| Two-GPU scale-out works | Both cards loaded, bit-identical scores from each |
| Endpoint aliases work | `/rerank` returns the same scores as `/v1/rerank` |

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
4. **First request is ~60x slower than subsequent ones** (0.942 s vs 0.016 s) because of
   prompt-cache and graph warm-up. Send one throwaway request at startup before putting
   an instance into rotation.
5. **Scores are absolute, not relative.** They come from a yes/no logit comparison, so
   they are comparable across queries — you can threshold on them (e.g. drop anything
   below 0.5) rather than only ranking within a candidate set.
6. **Default log verbosity hides the GPU-offload proof.** Use `-lv 5` when you need to
   prove residency.
7. **`LLAMA_CACHE`, not `HF_HOME`, controls where `-hf` writes.**
8. **Do not split this model across GPUs.** Run one instance per GPU instead.

## Summary

**Fully working on MI355X (gfx950).**

The ROCm/HIP build serves `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0` with **all
29/29 layers on the GPU** (603.87 MiB ROCm0 buffer, 1.92 GB VRAM in `rocm-smi`) and
ranks correctly with a decisive margin — relevant documents at **0.99966 / 0.99639**,
distractors at **0.00004 / 0.00002** — in **16 ms** warm for a 4-document request.
All three documented routes (`/v1/rerank`, `/rerank`, `/v1/reranking`) work.

This is notable because **HF TEI does not support this exact Qwen3
reranker**, so llama.cpp is one of the few working reranker paths on AMD hardware.

Multi-GPU splitting is **deliberately not used** — 604 MiB on a 288 GB card gains
nothing from a layer split. The demonstrated scale-out pattern is **two concurrent
single-GPU instances**, which produce **bit-identical scores** on both GPUs.

The main gotcha is the build-time one shared by all three folders:
**`-DLLAMA_OPENSSL=ON` + `libssl-dev`** are required for the `-hf` downloader.

## Follow-ups

- Measure throughput with realistic candidate-set sizes (top-50 / top-100 from a
  retriever) rather than 4 documents; per-document cost is what matters in production.
- Quantify Q8_0 drift against the FP16 `Qwen/Qwen3-Reranker-0.6B` on a labelled set
  (nDCG@10) — the score separation here is large enough that Q8_0 looks safe, but that
  is one query, not a benchmark.
- Compare against the cross-encoder trained in `training/reranker/sentence_transformers/` on the same
  OTel data to decide whether a fine-tune beats the zero-shot Qwen3 reranker.
- Chase the minja `lstrip` chat-template error upstream if this model is ever needed for
  generation rather than scoring.
