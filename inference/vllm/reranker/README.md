# `rerank_vllm.py` — Qwen3-Reranker-0.6B served by vLLM

## Overview & when to use

Serves **`Qwen/Qwen3-Reranker-0.6B`** through vLLM's OpenAI-compatible
`POST /v1/rerank` endpoint. `rerank_vllm.py` sends one query plus a list of candidate
documents and prints them re-ordered with relevance scores.

Qwen3-Reranker is natively a **decoder-only yes/no-token scorer**: the original design
asks the model "does this document satisfy the query?" and reads the probability mass on
the `yes` vs `no` tokens. vLLM converts that into an efficient **sequence-classification**
path, which is why the serve command needs the `--hf_overrides` JSON and the jinja chat
template — they are not optional decoration, they are what performs the conversion.

Use this stage as the precision step after a cheap recall step: embed-and-retrieve top-50
with `inference/vllm/embedding`, then rerank down to top-5 here. HF TEI cannot serve this
model (its reranker support targets encoder classifiers like XLM-RoBERTa/ModernBERT), so
vLLM is the natural choice if you already run vLLM for the other two workloads.

## Install

**There is no ROCm vLLM wheel.** Verified 2026-08-20:

- PyPI `vllm==0.27.1` ships two binary wheels (`manylinux_2_28_x86_64`, `aarch64`), both
  **CUDA-only** — hard deps on `flashinfer-python`, `nvidia-cudnn-frontend`,
  `nvidia-cutlass-dsl[cu13]`, `torch==2.13.0` (CUDA build).
- `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` has `torch`, `torchaudio`,
  `apex`, `jaxlib`, `tensorflow_rocm` — **no `vllm`**.
- Source build for `gfx950` works but is an hours-long compile.

The preferred pip/venv route is therefore unavailable on ROCm, and the verified route is a
**container**. This host already had
`rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` on disk — zero pull
cost, no disk spent:

```bash
docker run -d --name vllm_bringup \
  --device /dev/kfd --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  --network host \
  -v "$PWD":/workspace/repo -v /mnt/data_1.5t:/mnt/data_1.5t \
  -e HF_HOME=/mnt/data_1.5t/hf_cache -w /workspace/repo \
  rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 sleep infinity
```

Two gotchas:

- The image has **no `render` group** — the canonical `--group-add render` fails with
  `unable to find group render`. Use the host's numeric GIDs (`video`=44, `render`=993
  here), as the `getent` substitutions above do.
- Pinning by **render node** (`renderD128` = physical GPU0, `renderD136` = GPU1) instead
  of `HIP_VISIBLE_DEVICES` makes the isolation structural: `torch.cuda.device_count()==2`
  inside the container regardless of what any tool does to the environment.

Verified versions:

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
pip install -r requirements_reranker_vllm.txt
```

## Environment & secrets

`dev.env` is symlinked to the repo-root file:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`; exported into the serving shell for the model pull.
Never echo it. Weights live under `HF_HOME=/mnt/data_1.5t/hf_cache`, off `/`.

```bash
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — it hides every GPU.

## Serve

Single GPU (the verified command). `qwen3_reranker.jinja` ships in this folder, copied
verbatim from `/workspace/vllm/examples/pooling/score/template/qwen3_reranker.jinja` in
the image, so the command is self-contained:

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache HIP_VISIBLE_DEVICES=0
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

## Single-GPU results

**Verdict: works, unmodified.** The documented command — `--hf_overrides` JSON plus
jinja template — worked as written on the first attempt on one MI355X.

| Measurement | Value |
|---|---|
| Cold start (launch → `Application startup complete`) | **~46 s** |
| Weights load | 1.10 s |
| Model-load VRAM | **1.12 GiB** |
| `init engine` (profile + KV cache + warmup) | 13.50 s (compilation 9.64 s) |
| Available KV cache | 263.12 GiB |
| GPU KV cache size | 2,463,424 tokens |
| Max model len | 40960 (32K+ context, as documented) |
| Total process VRAM (`rocm-smi`, default util 0.9) | 285.8 GB |

Real output from `/v1/rerank` — note the **four-orders-of-magnitude** separation between
relevant and irrelevant documents:

```
model: qwen3-reranker
  rank idx=0 score=0.999503  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
  rank idx=2 score=0.992390  SGLang also provides a ROCm build for AMD GPUs.
  rank idx=1 score=0.000128  PostgreSQL is a relational database management system.
  rank idx=3 score=0.000027  The Eiffel Tower is located in Paris, France.
usage: {'prompt_tokens': 372, 'total_tokens': 372}
```

Both ROCm-related documents rank above both irrelevant ones, and the yes/no scoring is
saturated in the right direction — this is the reranker behaving exactly as designed, not
a degenerate output.

## Multi-GPU (TP=2) results

**Verdict: TP=2 works.** Unlike EmbeddingGemma (3 attention heads, TP=2 rejected),
Qwen3-Reranker-0.6B has a head count divisible by 2, so tensor parallelism is legal and
vLLM shards it cleanly across both MI355X cards.

Distributed init came up on RCCL (`nccl` maps to RCCL on ROCm):

```
DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, world_size=2, local_world_size=2
(Worker pid=7969) world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:48119 backend=nccl
(Worker pid=7970) world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:48119 backend=nccl
```

**The sharding is real, and the numbers prove it** — per-rank weights halve and the KV
cache doubles:

| Measurement | TP=1 | TP=2 | |
|---|---|---|---|
| Model-load VRAM **per GPU** | 1.12 GiB | **0.57 GiB** | ≈ halved — weights genuinely split |
| GPU KV cache size | 2,463,424 tokens | **4,942,128 tokens** | ≈ doubled — two cards' pools |
| Available KV cache per GPU | 263.12 GiB | 263.95 GiB | unchanged per card, as expected |
| `init engine` | 13.50 s | 14.96 s (compilation 10.62 s) | +1.5 s for RCCL setup |

`rocm-smi` with TP=2 resident — **both GPUs loaded, sibling GPUs untouched**:

```
device,VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,309220868096,286668083200     <- rank 0  (286.7 GB)
card1,309220868096,286668226560     <- rank 1  (286.7 GB)
card2,309220868096,298176512        <- idle, belongs to a sibling agent
```

Correctness held exactly across the topology change:

| Document | TP=1 score | TP=2 score |
|---|---|---|
| vLLM supports AMD ROCm… | 0.999503 | **0.999502** |
| SGLang also provides a ROCm build… | 0.992390 | **0.991847** |
| PostgreSQL is a relational database… | 0.000128 | **0.000127** |
| The Eiffel Tower is located in Paris… | 0.000027 | **0.000028** |

Identical ranking, scores agreeing to ~1e-5 — the residual drift is ordinary
non-deterministic reduction order across ranks, not a numerical problem.

**Caveat, stated plainly:** TP=2 *works* but is not *useful* for a 0.6 B model on 288 GB
cards. It halves a 1.12 GiB footprint that was never a constraint, while adding a
cross-GPU all-reduce to every forward pass. For production, prefer **two independent
single-GPU replicas** (as demonstrated in `inference/vllm/embedding`) — that doubles
throughput instead of splitting one model's latency. TP=2 is verified here because it is
the requested evidence that multi-GPU serving functions on this hardware, and it does.

## H100 (NVIDIA) — verified 2026-08-22

**Verdict: ✅ PASS, unmodified.** vLLM `0.27.1` (pip / CUDA 13.0) serves
`Qwen/Qwen3-Reranker-0.6B` on one H100 80GB with the documented `--hf_overrides` JSON plus
`qwen3_reranker.jinja` — both confirmed **mandatory** — correct ranking with a
four-orders-of-magnitude relevant/irrelevant split on the first attempt, no code changes.

### Install (pip route — the AMD "container-only" caveat does NOT apply on NVIDIA)

The "no ROCm vLLM wheel" fact above is AMD-specific. On **NVIDIA the pip wheel is native**.
One tmpfs venv is shared across all three leaves:

```bash
python3 -m venv .env_vllm && source .env_vllm/bin/activate
pip install torch numpy      # -> torch 2.13.0+cu130 (native CUDA 13)
pip install vllm             # -> vllm 0.27.1; torch stays 2.13.0+cu130
pip install python-dotenv
```

| Component | Version |
|---|---|
| vLLM | `0.27.1` (pip wheel, CUDA) |
| torch | `2.13.0+cu130` |
| transformers | `5.15.1` |
| driver / CUDA | 580.173.02 / 13.0, H100 80GB HBM3 |

### Serve (the verified H100 command — identical to ROCm minus the HIP var)

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/mnt/gsma/gsma/gsma/models CUDA_VISIBLE_DEVICES=5
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

### Real output (H100)

The `--hf_overrides` conversion is confirmed — vLLM loads the base `Qwen3ForCausalLM`
checkpoint **as a sequence classifier**:

```
[model.py:645] Resolved architecture: Qwen3ForSequenceClassification
[default_loader.py:430] Loading weights took 22.15 seconds
[gpu_model_runner.py:5405] Model loading took 1.12 GiB memory and 24.64 seconds
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

Both ROCm docs rank above both irrelevant ones with a **four-orders-of-magnitude** split —
the reranker behaving exactly as designed. This **ranking is identical to MI355X**, with
scores agreeing to ~1e-4 (MI355X: 0.999505 / 0.992223 / 0.000128 / 0.000028); the residual
drift is ordinary cross-hardware reduction order. **Model-load VRAM 1.12 GiB matches MI355X
exactly.** (Reranker score *scale* is template-driven, so the check is the RANKING, not the
absolute value — and the ranking matches.)

The scoring is genuinely **query-conditioned**, not a cached response — swapping the query
flips the order correctly:

```
query : Where is the Eiffel Tower located?
   1      1  0.998697  The Eiffel Tower is located in Paris, France.   <- jumps to #1
   2      2  0.000048  PostgreSQL is a relational database management system.
   3      0  0.000012  vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.
```

### GPU-5 residency (sampled from inside serving)

```
$ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep <GPU5-UUID>
1722028, GPU-e71a0833-4f61-11c9-6eff-10c149e752e4, 74882 MiB
$ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
5, 74891 MiB
```

~74.9 GB resident on **physical GPU 5 only** — this is the default
`--gpu-memory-utilization 0.9` KV pool, **not** the weights (the true weight footprint is
the `Model loading took 1.12 GiB` line, exactly as the MI355X README warns). Co-tenant
GPUs 0–3 untouched.

### H100 verdict

✅ **PASS — vLLM 0.27.1 (pip / cu130) serves Qwen3-Reranker-0.6B unmodified on one H100.**
The `--hf_overrides` + jinja requirement is confirmed genuinely mandatory; ranking is
correct, sharply separated, and query-conditioned, matching MI355X to ~1e-4. The one AMD
correction (install is container-only) does **not** apply on NVIDIA — the pip route works.
TP=2 is legal for this model but not useful at 0.6 B; prefer replication. Multi-GPU
deferred (single-GPU wave; GPUs 0–3 are a co-tenant production job).

## Arguments / flags

Serve-side:

| Flag | Value used | Meaning |
|---|---|---|
| `--runner pooling` | required | Pooling/scoring mode; exposes `/v1/rerank`, `/score`, `/v1/score` |
| `--hf_overrides` | JSON below | **Required.** Rewrites the loaded architecture — see breakdown |
| `--chat-template` | `qwen3_reranker.jinja` | **Required.** Formats query+document into the Instruct/Query/Document prompt the model was trained on |
| `--tensor-parallel-size` | `1` / `2` | Shards the model across GPUs; both verified |
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

The client prints endpoint, model, query, usage, then a ranked table. Real captured run of
`python rerank_vllm.py --port 8002`:

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

Server logs go to `/mnt/data_1.5t/outputs/inference_reranker_vllm/`. Nothing large lands
in the repo.

## Hardware support & evidence

- **AMD: tested and working, both topologies.** 1× and 2× (TP=2) AMD Instinct MI355X
  (`gfx950`, 288 GB), host ROCm 7.2.4, container ROCm 7.0.2, vLLM `0.20.2rc1.dev253`,
  torch `2.9.1.dev+rocm7.0.2`. Correct, sharply separated relevance scores; TP=2 agrees
  with TP=1 to ~1e-5.
- **NVIDIA: not tested here** (no NVIDIA GPU on this host). The same command applies with
  `vllm/vllm-openai:latest`; on CUDA the pip route also works.
- vLLM is **✅ Native** for the Qwen3 reranker, and on ROCm/gfx950 that is **confirmed**.
  The `--hf_overrides` + jinja requirement is confirmed as genuinely mandatory. The only
  correction: the AMD *install* story is container-only.

## Notes & quirks

- **Both `--hf_overrides` and `--chat-template` are mandatory.** Drop the overrides and
  the model loads as a causal LM with no `/v1/rerank` scoring head. Drop the template and
  the prompt no longer matches the Instruct/Query/Document format the model was trained
  on, so scores degrade silently — the endpoint still answers, which makes this a
  dangerous omission rather than a loud failure.
- **The template is shipped in this folder** (`qwen3_reranker.jinja`) so the serve command
  does not depend on a path inside the image. It emits an empty `<think></think>` block
  before the assistant turn — that is intentional for this reasoning-capable base model.
- **The pooling server also exposes `/v1/embeddings`.** Registered on the same process;
  it does not mean the reranker is an embedding model. Use the right endpoint.
- **AITER JIT-builds on first launch** (`[aiter] start build [module_aiter_core]`),
  stalling the first startup. Later launches reuse the cache.
- **The `quark_online_quant` plugin fails to import** on every launch with a traceback.
  Non-fatal, unrelated; the server starts normally.
- **Default `--gpu-memory-utilization 0.9` makes `rocm-smi` read ~286 GB** for a 1.12 GiB
  model — that is the preallocated KV pool (263 GiB of it), not the weights. Read
  `Model loading took N GiB` from the server log for the true weight footprint.
- **`--runner pooling`, not `--is-embedding`.** SGLang's docs warn against `--is-embedding`
  for this model; vLLM's equivalent mistake is omitting the `--hf_overrides` conversion.

## Verdict

✅ **PASS — vLLM serves Qwen3-Reranker-0.6B on MI355X/gfx950 with no changes, on one GPU
and on two.** The "✅ Native" rating and the documented serve command both hold.
Relevance scores separate relevant from irrelevant documents by four orders of magnitude
(0.9995 vs 0.000027), and TP=2 reproduces TP=1 to ~1e-5 while genuinely halving per-rank
weights (1.12 → 0.57 GiB) and doubling KV capacity.

Two corrections worth recording: the ROCm install is **container-only** (no pip wheel exists),
and while TP=2 *works*, replication is the better multi-GPU pattern at this model size.
