# `inference/sglang/reranker` — SGLang reranker serving (Qwen3-Reranker-0.6B)

Serves `Qwen/Qwen3-Reranker-0.6B` behind `POST /v1/rerank` with SGLang
(`python -m sglang.launch_server`). `inference_reranker_sglang.py` scores query/document
pairs and prints them in ranked order.

Qwen3-Reranker is not a cross-encoder with a classification head: it is a decoder-only
yes/no reranker, scored from the logits of the `yes`/`no` tokens under a specific prompt.
Two consequences drive this folder: **never pass `--is-embedding`**, and always pass the
`qwen3_reranker.jinja` chat template. Use this folder when SGLang already serves your
generation traffic and you want reranking in the same engine.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2.4) · NVIDIA H100 80GB (CUDA 13.0)

## Files

- `inference_reranker_sglang.py` — scores documents against a query, prints the ranking.
- `qwen3_reranker.jinja` — renders the system + `<Instruct>/<Query>/<Document>` prompt
  whose `yes`/`no` logits define the score. Required by the serve command.

## Setup

Engine install (NVIDIA pip or the ROCm container) is in [`../README.md`](../README.md).
Then, in this folder:

```bash
ln -sf ../../../dev.env dev.env    # HF_TOKEN — only needed if you swap in a gated model
export HF_HOME=/path/to/hf_cache
export OUTPUT_DIR=/path/to/outputs
```

`Qwen/Qwen3-Reranker-0.6B` is ungated.

## Serve — NVIDIA H100

`--chat-template` takes a host path here; there is no container bind-mount.

```bash
source ../.env_sglang/bin/activate
export CUDA_VISIBLE_DEVICES=<free-gpu> HF_HUB_OFFLINE=1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --chat-template ./qwen3_reranker.jinja \
  --disable-radix-cache --mem-fraction-static 0.3 \
  --host 127.0.0.1 --port 8600
```

Server log on success:

```
Loading chat template from argument: ./qwen3_reranker.jinja
The server is fired up and ready to roll!
```

## Serve — AMD / ROCm

From the long-lived container. `--chat-template` takes a container-visible path — `/work`
is the bind-mounted repo, and a host path will not resolve.

```bash
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8102 \
     > $OUTPUT_DIR/inference_reranker_sglang/rerank_single.log 2>&1"
```

`GET /get_model_info` confirms the runner is a causal LM, not a pooling encoder:

```json
{"model_path": "Qwen/Qwen3-Reranker-0.6B", "is_generation": true,
 "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
```

## Scaling — replicate, do not shard

For a 0.6B reranker run one independent server per GPU rather than `--tp 2`:

```bash
# replica B -> second GPU, port 8112
docker exec -d sglang_bringup bash -lc \
  "HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-Reranker-0.6B \
     --chat-template /work/inference/sglang/reranker/qwen3_reranker.jinja \
     --disable-radix-cache --mem-fraction-static 0.3 \
     --host 0.0.0.0 --port 8112 \
     > $OUTPUT_DIR/inference_reranker_sglang/rerank_rep_b.log 2>&1"
```

Scores are identical across the two GPUs, so a replica pool load-balances without score
drift reshuffling the ordering.

## Client

```bash
python inference_reranker_sglang.py --port 8102 \
  --query "Which inference engines support AMD ROCm?" \
  --documents "vLLM and SGLang both support AMD ROCm GPUs." \
              "PostgreSQL is a relational database." \
              "ROCm is AMD's open compute platform for Instinct accelerators."
```

Use `--port 8600` against the H100 server. From inside the container:

```bash
docker exec -w /work/inference/sglang/reranker sglang_bringup \
  python inference_reranker_sglang.py --port 8102
```

## Arguments

Server flags that matter:

| Flag | Value | Why |
|---|---|---|
| `--model-path` | `Qwen/Qwen3-Reranker-0.6B` | Decoder-only yes/no reranker |
| `--chat-template` | `qwen3_reranker.jinja` | **Required** — renders the Instruct/Query/Document prompt |
| `--disable-radix-cache` | on | Prefix reuse is useless across unrelated query/document pairs |
| `--mem-fraction-static` | `0.3` | KV/static pool fraction; auto-reduced at TP>1 |
| `--is-embedding` | **never** | Wrong for this model — it is not an embedding model |
| `--tp` | `1` | Correct for a 0.6B model; scale with replicas, not TP |
| `--attention-backend` | *unset* | Leave unset — `aiter` on gfx950, CUDA path on Hopper |

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

## Output

The relevant document must rank first and the database sentence last:

```
[health] server ready after 1.0s
[latency] 0.14s | pairs=3 | query='Which inference engines support AMD ROCm?'
[rank 1] score=0.777300  vLLM and SGLang both support AMD ROCm GPUs.
[rank 2] score=0.140336  ROCm is AMD's open compute platform for Instinct accelerators.
[rank 3] score=0.000024  PostgreSQL is a relational database.
```

The server writes no artifacts; redirect logs where you want them, e.g.
`$OUTPUT_DIR/inference_reranker_sglang/`. Weights live in `$HF_HOME`.

## Notes

- **Do not pass `--is-embedding`.** It switches the model into a pooling encoder, destroys
  the LM head the score comes from, and breaks `/v1/rerank`.
- **Template provenance matters.** A different chat template silently changes the scores
  rather than erroring.
- **KV cache is large by default** even for a 0.6B model. Set `--mem-fraction-static` or
  `--max-total-tokens` explicitly on a shared box.
- Benign startup noise: `Failed to import amdsmi` (install `amdsmi` for AMD telemetry),
  `Ignoring corrupted tree cache file ... Permission denied` on a shared `$HF_HOME`,
  `torchcodec`/`libavutil` import errors, `Ignore import error when loading
  sglang.srt.models.inkling`, and AITER's
  `-amdgpu-coerce-illegal-types=1 is not supported by hipcc` probe, after which it rebuilds
  without the flag.
