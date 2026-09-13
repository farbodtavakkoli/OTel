# `rerank_vllm.py` — Qwen3-Reranker-0.6B served by vLLM

## Overview & when to use

Serves **`Qwen/Qwen3-Reranker-0.6B`** through vLLM's OpenAI-compatible
`POST /v1/rerank` endpoint. `rerank_vllm.py` sends one query plus a list of candidate
documents and prints them re-ordered with relevance scores.

Qwen3-Reranker is natively a **decoder-only yes/no-token scorer**: the original design
asks the model "does this document satisfy the query?" and reads the probability mass on
the `yes` vs `no` tokens. vLLM converts that into an efficient **sequence-classification**
path, which is why the serve command needs the `--hf_overrides` JSON and the jinja chat
template — both are required: they are what performs the conversion.

Use this stage as the precision step after a cheap recall step: embed-and-retrieve top-50
with `inference/vllm/embedding`, then rerank down to top-5 here.

## Install

**There is no ROCm vLLM wheel.** PyPI `vllm` publishes CUDA-only wheels that hard-depend on
CUDA torch, and `repo.radeon.com` publishes none, so the pip/venv route is unavailable on
ROCm (a gfx950 source build works but takes hours) and the route that works is a
**container**. Known-working image:
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2`:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache        # Hugging Face model cache
export OUTPUT_DIR=/path/to/outputs      # server logs and run artifacts

docker run -d --name vllm_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  --network host \
  -v "$PWD":/workspace/repo -v "$HF_HOME":"$HF_HOME" \
  -e HF_HOME="$HF_HOME" -w /workspace/repo \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

Two things to note:

- The image has **no `render` group** — the canonical `--group-add render` fails with
  `unable to find group render`. Use the host's numeric GIDs (typically `video`=44,
  `render`=993), as the `getent` substitutions above do.
- Pinning by **render node** (`renderD128`, `renderD136` — match them to your own cards with
  `ls /dev/dri`) instead of `HIP_VISIBLE_DEVICES` makes the isolation structural: the
  container sees exactly those two cards.

Versions inside that image:

| Component | Version |
|---|---|
| vLLM | `0.20.2rc1.dev253+g1ff9d3353` |
| torch | `2.9.1.dev20251204+rocm7.0.2.git351ff442` |
| `torch.version.hip` | `7.0.51831-7c9236b16` |
| transformers | `5.14.1` |
| ROCm (container / host) | 7.0.2 / 7.2.4 |
| GPU | AMD Instinct MI355X, `gfx950:sramecc+:xnack-`, 309 GB |

Client side (host, outside the container) needs only `python-dotenv`:

```bash
pip install -r ../requirements.txt
```

## Environment & secrets

`dev.env` is symlinked to the repo-root file:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`; exported into the serving shell for the model pull.
Never echo it. Weights live under `$HF_HOME`, off `/`.

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — it hides every GPU.

## Serve

Single GPU. `qwen3_reranker.jinja` ships in this folder, copied
verbatim from `/workspace/vllm/examples/pooling/score/template/qwen3_reranker.jinja` in
the image, so the command is self-contained:

```bash
export HF_HOME=/path/to/hf_cache HIP_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 \
  --port 8002
```

Multi-GPU (TP=2) — add one flag:

```bash
export HIP_VISIBLE_DEVICES=0,1
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --tensor-parallel-size 2 \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 \
  --port 8002
```

## Client / smoke command

```bash
python rerank_vllm.py --port 8002 --model qwen3-reranker
```

Raw equivalent:

```bash
curl -s http://localhost:8002/v1/rerank -H 'Content-Type: application/json' -d '{
 "model":"qwen3-reranker",
 "query":"Which inference engines support AMD ROCm?",
 "documents":[
  "vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.",
  "PostgreSQL is a relational database management system.",
  "SGLang also provides a ROCm build for AMD GPUs.",
  "The Eiffel Tower is located in Paris, France."
 ]}'
```

## Single-GPU behaviour on gfx950

**This works, unmodified.** The documented command — `--hf_overrides` JSON plus
jinja template — works as written on one MI355X.

**Expected output** from `/v1/rerank`:

```
model: qwen3-reranker
  rank idx=0 score=0.999503  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
  rank idx=2 score=0.992390  SGLang also provides a ROCm build for AMD GPUs.
  rank idx=1 score=0.000128  PostgreSQL is a relational database management system.
  rank idx=3 score=0.000027  The Eiffel Tower is located in Paris, France.
usage: {'prompt_tokens': 372, 'total_tokens': 372}
```

Both ROCm-related documents must rank above both irrelevant ones.

## Multi-GPU (TP=2) on gfx950

**TP=2 works.** Unlike EmbeddingGemma (3 attention heads, TP=2 rejected),
Qwen3-Reranker-0.6B has a head count divisible by 2, so tensor parallelism is legal and
vLLM shards it cleanly across both MI355X cards, on RCCL, with no extra flags. Ranking is
identical to TP=1 (scores agree to ~1e-5).

At this model size prefer **two independent single-GPU replicas** (as shown in
[`../embedding/README.md`](../embedding/README.md)) over TP.

## H100 (NVIDIA)

**This works unmodified.** vLLM `0.27.1` (pip / CUDA 13.0) serves
`Qwen/Qwen3-Reranker-0.6B` on one H100 80GB with the documented `--hf_overrides` JSON plus
`qwen3_reranker.jinja` — both **mandatory** — giving correct ranking, no code changes.

### Install (pip route — no container needed)

On **NVIDIA the pip wheel is native**. One shared venv at the stack root
(`inference/vllm/.env_vllm`) serves all three leaves:

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy      # -> the current CUDA 13 build (plain PyPI)
pip install vllm             # -> vllm 0.27.1; torch stays 2.13.0+cu130
pip install python-dotenv
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA) |
| torch | `2.13.0+cu130` |
| transformers | `5.15.1` |
| CUDA | 13.0, H100 80GB HBM3 |

### Serve (the H100 command — identical to ROCm minus the HIP var)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/path/to/hf_cache CUDA_VISIBLE_DEVICES=<free-gpu>
vllm serve Qwen/Qwen3-Reranker-0.6B \
  --runner pooling \
  --hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template qwen3_reranker.jinja \
  --served-model-name qwen3-reranker \
  --host 0.0.0.0 --port 8500
```

```bash
python rerank_vllm.py --port 8500 --model qwen3-reranker
```

### Expected output (H100)

The `--hf_overrides` conversion is confirmed — vLLM loads the base `Qwen3ForCausalLM`
checkpoint **as a sequence classifier**:

```
[model.py:645] Resolved architecture: Qwen3ForSequenceClassification
[api_server.py:678] Supported tasks: ['classify', 'token_classify']
Route: /score / /v1/score / /rerank / /v1/rerank
INFO:     Application startup complete.
```

```
endpoint    : http://localhost:8500/v1/rerank
model       : qwen3-reranker
query       : Which inference engines support AMD ROCm?

rank  index  score       document
   1      0  0.999427  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
   2      2  0.991585  SGLang also provides a ROCm build for AMD GPUs.
   3      1  0.000130  PostgreSQL is a relational database management system.
   4      3  0.000027  The Eiffel Tower is located in Paris, France.
```

Both ROCm docs must rank above both irrelevant ones. Reranker score *scale* is
template-driven, so the check is the ranking, not the absolute value.

Swapping the query flips the order, which confirms the scoring is query-conditioned:

```
query : Where is the Eiffel Tower located?
   1      1  0.998697  The Eiffel Tower is located in Paris, France.   <- jumps to #1
   2      2  0.000048  PostgreSQL is a relational database management system.
   3      0  0.000012  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
```

### GPU residency check

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

Most of the resident VRAM on the selected GPU is the default `--gpu-memory-utilization 0.9`
KV pool, **not** the weights — read `Model loading took N GiB` from the server log for the
true weight footprint. No other card is touched.

## Arguments / flags

Serve-side:

| Flag | Value | Meaning |
|---|---|---|
| `--runner pooling` | required | Pooling/scoring mode; exposes `/v1/rerank`, `/score`, `/v1/score` |
| `--hf_overrides` | JSON below | **Required.** Rewrites the loaded architecture — see breakdown |
| `--chat-template` | `qwen3_reranker.jinja` | **Required.** Formats query+document into the Instruct/Query/Document prompt the model was trained on |
| `--tensor-parallel-size` | `1` / `2` | Shards the model across GPUs; both work, but prefer replication at this model size |
| `--served-model-name` | `qwen3-reranker` | Client-facing alias |
| `--host` / `--port` | `0.0.0.0` / `8002` | Bind address and benchmark-layout reranker port |
| `--gpu-memory-utilization` | default `0.9` | Lower when co-locating servers on one card |

The `--hf_overrides` JSON, field by field:

| Key | Value | Why |
|---|---|---|
| `architectures` | `["Qwen3ForSequenceClassification"]` | Loads the model as a sequence classifier instead of a causal LM |
| `classifier_from_token` | `["no","yes"]` | Builds the 2-logit classifier head from the existing `no`/`yes` token embeddings — this is what preserves the original scoring semantics |
| `is_original_qwen3_reranker` | `true` | Tells vLLM this is the original Qwen reranker layout so the conversion is applied correctly |

`rerank_vllm.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `localhost` | Server host |
| `--port` | `8002` | Server port |
| `--model` | `qwen3-reranker` | Served model name (matches `--served-model-name`) |
| `--query` | ROCm question | Query to rank documents against |
| `--document` | 4 built-ins | Candidate document; repeat the flag for several |
| `--top_n` | `None` | Return only the top N results (default: all) |
| `--timeout` | `120.0` | HTTP timeout in seconds |

## Output

The client prints endpoint, model, query, usage, then a ranked table. **Expected output**
for `python rerank_vllm.py --port 8002`:

```
endpoint    : http://localhost:8002/v1/rerank
model       : qwen3-reranker
query       : Which inference engines support AMD ROCm?
usage       : {'prompt_tokens': 372, 'total_tokens': 372}

rank  index  score       document
   1      0  0.999505  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
   2      2  0.992223  SGLang also provides a ROCm build for AMD GPUs.
   3      1  0.000128  PostgreSQL is a relational database management system.
   4      3  0.000028  The Eiffel Tower is located in Paris, France.
```

Redirect server logs to a data volume, e.g. `$OUTPUT_DIR/inference_reranker_vllm/`.
Nothing large lands in the repo.

## Hardware support

- **AMD, working in both topologies.** 1× and 2× (TP=2) AMD Instinct MI355X
  (`gfx950`, 288 GB), host ROCm 7.2.4, container ROCm 7.0.2, vLLM `0.20.2rc1.dev253`,
  torch `2.9.1.dev+rocm7.0.2`.
- **NVIDIA, working** via the pip wheel — see the H100 section above. The same
  command also applies with `vllm/vllm-openai:latest`.
- On AMD the *install* route is container-only — there is no ROCm vLLM wheel.

## Notes & quirks

- **Both `--hf_overrides` and `--chat-template` are mandatory.** Drop the overrides and
  the model loads as a causal LM with no `/v1/rerank` scoring head. Drop the template and
  the prompt no longer matches the Instruct/Query/Document format the model was trained
  on, so scores degrade silently while the endpoint still answers.
- **The template is shipped in this folder** (`qwen3_reranker.jinja`) so the serve command
  does not depend on a path inside the image. It emits an empty `<think></think>` block
  before the assistant turn — that is intentional for this reasoning-capable base model.
- **The pooling server also exposes `/v1/embeddings`.** Registered on the same process;
  it does not mean the reranker is an embedding model. Use the right endpoint.
- **AITER JIT-builds on first launch** (`[aiter] start build [module_aiter_core]`),
  stalling the first startup. Later launches reuse the cache.
- **The `quark_online_quant` plugin fails to import** on every launch with a traceback.
  Non-fatal, unrelated; the server starts normally.
- **Default `--gpu-memory-utilization 0.9` makes `rocm-smi` read most of the card** for a
  0.6B model — that is the preallocated KV pool, not the weights. Read
  `Model loading took N GiB` from the server log for the true weight footprint.
- **`--runner pooling`, not `--is-embedding`.** The equivalent mistake in vLLM is omitting
  the `--hf_overrides` conversion.
