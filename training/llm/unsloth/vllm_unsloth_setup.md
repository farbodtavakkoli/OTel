# GRPO env (`.grpo_env`) — Unsloth + vLLM, reproducible setup

This is the **isolated** environment for GRPO experiments that need vLLM (fast rollouts).
It is **separate** from the main training venv (`/home/adminuser/.venv`, torch 2.11) and does
not touch it.

> ⚠️ **READ THIS FIRST — the env evolved.** The original build below targeted vLLM 0.10.2
> (torch 2.8) to match TRL 0.24's server mode. **That vLLM is too old to serve gemma-4**
> (no `Gemma4ForConditionalGeneration`, no `rope_parameters` schema). For **inference** we
> upgraded the env to **vLLM 0.26.0 + torch 2.11 + transformers 5.5.4**, which serves the
> bnb-4bit gemma-4 model successfully — see "**VERIFIED: vLLM inference on bnb-4bit gemma-4**"
> below. The torch-2.8 sections are kept for history / the (still-unfinished) TRL-server path.

---

## ✅ VERIFIED: vLLM inference on bnb-4bit gemma-4 (2026-08-17)

Standalone `vllm serve` of `gemma-4-31b-it-unsloth-bnb-4bit` **works** and generates
correctly (`COMPLETION: 'hello from gemma4'`, finish=stop). This is the proven recipe.

### Current env state for inference
```
vllm==0.26.0   torch==2.11.0+cu130   transformers==5.5.4   xformers==0.0.33.post1
```
(These superseded the torch-2.8 pins. `transformers 5.5.4` is the key: ≥5.5.3 that vLLM 0.26
requires, but not the 5.15.x that introduced the per-layer `head_dim` regression.)

### The three blockers and their fixes (in order hit)
1. **vLLM 0.10.2 can't serve gemma-4** (arch not registered; `rope_scaling should have a
   'rope_type' key`). → **Upgrade to vLLM 0.26.0** (`pip install vllm==0.26.0`; pulls
   torch 2.11, transformers 5.15). Confirm: `Gemma4ForConditionalGeneration` in
   `ModelRegistry.get_supported_archs()` → True.
2. **transformers 5.15 `head_dim` regression** — `AmbiguousGlobalPerLayerAttributeError:
   'head_dim' is a per-layer attribute`. vLLM pins `transformers>=5.5.3`, so →
   **`pip install transformers==5.5.4`** (the newest that satisfies vLLM but predates the
   regression). NOTE: 5.5.4 *still* guards the global access for this config, so also →
3. **Config overlay to allow the global `head_dim` read.** gemma-4's `text_config` has both
   `head_dim=256` and `global_head_dim=512` (heterogeneous), which transformers refuses to
   read globally. Build a thin overlay dir: symlink every model file, but replace
   `config.json` with a real copy that adds `allow_global_per_layer_attribute_access: true`
   (at top level AND under `text_config`). vLLM then reports *"heterogeneous head dimensions
   … Using FA4 for all layers"* and proceeds.

### Build the config overlay
```bash
SNAP=/mnt/gsma/gsma/gsma/models/hub/models--unsloth--gemma-4-31b-it-unsloth-bnb-4bit/snapshots/8e256fc6d63003fc0ca8c91b976e6dcc38433385
OVER=/tmp/gemma4bnb_cfgfix
rm -rf "$OVER"; mkdir -p "$OVER"
for f in "$SNAP"/*; do ln -s "$f" "$OVER/$(basename "$f")"; done
rm "$OVER/config.json"      # replace the symlink with a patched real file
python - <<PY
import json
c=json.load(open("$SNAP/config.json"))
c["allow_global_per_layer_attribute_access"]=True
c.setdefault("text_config",{})["allow_global_per_layer_attribute_access"]=True
json.dump(c, open("$OVER/config.json","w"), indent=2)
PY
```

### Serve it (GPU 0, text-only, offline)
```bash
source /home/adminuser/.grpo_env/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PATH="/home/adminuser/.grpo_env/bin:/usr/local/cuda/bin:$PATH"
vllm serve /tmp/gemma4bnb_cfgfix \
  --served-model-name gemma4bnb \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --host 127.0.0.1 --port 8000
```
Wait ~5–7 min (weight load ~4 min at ~65 s/shard + CUDA-graph capture). Healthy signals:
`load_format=bitsandbytes`, `GPU KV cache size: 58,770 tokens`, `Maximum concurrency 14.35x`,
`Application startup complete`. ~23 GB on GPU 0.

### Smoke test
```bash
curl -s --noproxy '*' http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4bnb","messages":[{"role":"user","content":"Reply with exactly: hello from gemma4"}],"max_tokens":32,"temperature":0}'
```

> **Scope note:** this proves *inference* (standalone serving, e.g. for offline eval of a
> finished adapter — the deployment-guide use case). It is **NOT** yet wired to GRPO training.
> The GRPO fast-path (TRL live weight-sync to the server) needs a matched TRL version — see
> the "GRPO fast-loop: status & remaining tests" section at the bottom.

---

## (Historical) torch-2.8 build for TRL-0.24 server mode

- **Path:** `/home/adminuser/.grpo_env`  (`source /home/adminuser/.grpo_env/bin/activate`)
- **Python:** 3.12.3
- **Why a separate env:** vLLM ≤0.11 hard-pins **torch 2.8.0**, while Unsloth's main pipeline
  runs torch 2.11. Unsloth 2026.8.9 *also* supports torch 2.8 (declares `torch>=2.4,<2.12`
  and ships an `xformers==0.0.32.post2` build for torch 2.8), so a single torch-2.8 env can
  host **both** the vLLM server and Unsloth training — this is "Option B".

## Verified working (import + CUDA level)

- `torch 2.8.0+cu128`, sees **8 GPUs** ✅
- `unsloth.FastModel` imports (gemma4 patches load) ✅
- `trl.GRPOTrainer` + `GRPOConfig` import; server-mode fields present
  (`vllm_mode`, `vllm_server_host`, `vllm_server_port`) ✅
- vLLM server launcher runs via `python -m trl.scripts.vllm_serve --help` ✅
  (the `trl vllm-serve` CLI wrapper does NOT work — see fixes below; use `python -m`)

## Key versions

| package | version |
|---|---|
| torch | 2.8.0+cu128 |
| torchvision / torchaudio | 0.23.0 / 2.8.0 |
| vllm | 0.10.2 |
| xformers | 0.0.32.post1 |
| triton | 3.4.0 |
| unsloth | 2026.8.9 |
| unsloth_zoo | 2026.8.6 |
| trl | 0.24.0 |
| transformers | 5.5.0 |
| peft | 0.20.0 |
| accelerate | 1.6.0 |
| datasets | 4.3.0 |
| bitsandbytes | 0.50.1 |
| pydantic | 2.11.7 (core 2.33.2) |
| tyro | 1.0.15 |
| structlog | 26.1.0 |

## Build order (what was installed, in sequence)

```bash
python3.12 -m venv /home/adminuser/.grpo_env
source /home/adminuser/.grpo_env/bin/activate

# 1) vLLM first — it pins the exact torch 2.8.0 / torchvision / torchaudio (cu128) stack.
pip install "vllm==0.10.2"

# 2) HF + RL stack.
pip install "transformers==5.5.0" "trl==0.24.0" peft accelerate datasets bitsandbytes python-dotenv

# 3) Unsloth (no-deps so it cannot move torch/transformers) + torch-2.8 xformers.
pip install "xformers==0.0.32.post1"
pip install --no-deps "unsloth==2026.8.9" "unsloth_zoo==2026.8.6"
```

## Fixes applied (non-obvious — needed to make the stack import)

These are the gotchas discovered while stabilizing the env. Re-apply if rebuilding:

1. **pydantic pinned to 2.11.7.** vLLM 0.10.2 needs `pydantic>=2.11.7`, but Unsloth 2026.8.9
   breaks on pydantic 2.13.x (`PydanticSchemaGenerationError` on `torch.Tensor`). 2.11.7 is
   the only version satisfying both. Install *with deps* so `pydantic-core` (2.33.2) matches:
   `pip install "pydantic==2.11.7"`.
2. **Do NOT install mergekit.** It's pydantic-incompatible and, when present, crashes TRL's
   GRPO import chain (`callbacks.py → mergekit_utils → mergekit`). TRL does not need mergekit
   for GRPO. `pip uninstall -y mergekit` if it got pulled in.
3. **Add `tyro` and `structlog`** — Unsloth deps that `--no-deps` skipped:
   `pip install tyro structlog`.
4. **Stub `vllm_ascend`.** vLLM 0.10.2 makes `importlib.util.find_spec("vllm_ascend")`
   succeed (phantom plugin spec), so TRL's `vllm_serve.py` tries to import the Ascend/NPU
   `PyHcclCommunicator` even on a CUDA box, and fails with `ModuleNotFoundError: vllm_ascend`.
   It's only referenced under the never-executed NPU branch. Fix with a stub package under
   site-packages:
   ```
   vllm_ascend/__init__.py
   vllm_ascend/distributed/__init__.py
   vllm_ascend/distributed/device_communicators/__init__.py
   vllm_ascend/distributed/device_communicators/pyhccl.py   # defines class PyHcclCommunicator (raises if instantiated)
   ```
5. **Launch the server with `python -m trl.scripts.vllm_serve`, not `trl vllm-serve`.** The
   `trl` CLI aggregator hard-imports `dpo_trainer → mergekit` and fails; the `-m` module path
   bypasses it.

## Usage sketch (GRPO with a vLLM server)

Server (dedicated GPU, e.g. GPU 0):
```bash
source /home/adminuser/.grpo_env/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
    --model <bnb-4bit snapshot path> --port 8000 --gpu_memory_utilization 0.9
```

Training (remaining GPUs) connects with `train_llm_unsloth.py --vllm_server` (added):
`--train_mode grpo --vllm_server --vllm_server_host 127.0.0.1 --vllm_server_port 8000`,
launched via `torchrun --nproc_per_node 7` on `CUDA_VISIBLE_DEVICES=1,...,7`.

> Still to verify at runtime: that vLLM 0.10.2 actually **serves the bnb-4bit gemma-4
> checkpoint** and that TRL's per-step LoRA weight-sync to the server works. Import-level
> checks pass; the model-load + weight-sync smoke test is the next step.

## Full `pip freeze`

<details><summary>166 packages</summary>

```
accelerate==1.6.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
astor==0.8.1
attrs==26.1.0
bitsandbytes==0.50.1
blake3==1.0.9
cachetools==7.1.7
cbor2==6.1.4
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.1
click==8.2.1
cloudpickle==3.1.2
compressed-tensors==0.11.0
cuda-pathfinder==1.6.0
cupy-cuda12x==14.1.1
datasets==4.3.0
depyf==0.19.0
detect-installer==0.1.0
dill==0.4.1
diskcache==5.6.3
distro==1.9.0
dnspython==2.8.0
docstring_parser==0.18.0
einops==0.8.2
email-validator==2.3.0
fastapi==0.141.1
fastapi-cli==0.0.32
fastapi-cloud-cli==0.23.0
fastar==0.11.0
filelock==3.32.3
frozendict==2.4.7
frozenlist==1.8.0
fsspec==2026.6.0
gguf==0.19.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.10.0
httptools==0.8.0
httpx==0.28.1
httpx2==2.10.0
huggingface_hub==1.16.1
idna==3.18
immutables==0.21
interegular==0.3.3
Jinja2==3.1.6
jiter==0.16.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
lark==1.2.2
llguidance==0.7.30
llvmlite==0.44.0
lm-format-enforcer==0.11.3
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mistral_common==1.11.7
mpmath==1.3.0
msgpack==1.2.1
msgspec==0.21.1
multidict==6.7.1
multiprocess==0.70.19
networkx==3.6.1
ninja==1.13.0
numba==0.61.2
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
openai==3.1.0
openai-harmony==0.0.8
opencv-python-headless==5.0.0.93
outlines_core==0.2.11
packaging==26.3
pandas==3.0.5
partial-json-parser==0.2.1.1.post7
peft==0.20.0
pillow==12.3.0
prometheus-fastapi-instrumentator==8.1.0
prometheus_client==0.26.0
propcache==0.5.2
protobuf==7.35.1
psutil==7.2.2
py-cpuinfo==9.0.0
pyarrow==25.0.1
pybase64==1.5.0
pycountry==26.2.16
pycparser==3.0
pydantic==2.11.7
pydantic-extra-types==2.11.1
pydantic-settings==2.15.0
pydantic_core==2.33.2
Pygments==2.21.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.3
python-json-logger==4.2.0
python-multipart==0.0.32
PyYAML==6.0.3
pyzmq==27.1.0
ray==2.57.0
referencing==0.37.0
regex==2026.7.19
requests==2.34.2
rich==15.0.0
rich-toolkit==0.20.3
rignore==0.8.1
rpds-py==2026.6.3
safetensors==0.5.3
scipy==1.18.0
sentencepiece==0.2.2
sentry-sdk==2.68.0
setproctitle==1.3.7
setuptools==79.0.1
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
soundfile==0.14.0
soxr==1.1.0
starlette==1.6.0
structlog==26.1.0
sympy==1.14.0
tiktoken==0.13.0
tokenizers==0.22.2
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==5.5.0
triton==3.4.0
trl==0.24.0
truststore==0.10.4
typeguard==4.6.0
typer==0.27.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tyro==1.0.15
unsloth==2026.8.9
unsloth_zoo==2026.8.6
urllib3==2.7.0
uvicorn==0.52.3
uvloop==0.22.1
vllm==0.10.2
watchfiles==1.2.0
websockets==17.0.1
wheel==0.48.0
xformers==0.0.32.post1
xgrammar==0.1.23
xxhash==4.0.1
yarl==1.24.5
```

</details>

---

## GRPO fast-loop: status & remaining tests

**Goal:** a fast GRPO loop = train the LoRA on GPUs 1–7 (Unsloth) while a vLLM server on
GPU 0 does the rollouts, with TRL syncing the updated adapter to the server every step.

### ✅✅ FULL FAST LOOP VERIFIED END-TO-END (2026-08-17)
A 2-GPU GRPO smoke run against the separate vLLM server completed a full step with correct
rewards. Server access log proves every protocol call succeeded:
```
init_communicator     200   (NCCL weight-sync handshake)
update_named_param  ×1126 200  (entire LoRA adapter synced to the server)
generate               200   (rollouts returned; was 500 before the patch below)
reset_prefix_cache     200   (cache invalidated after the weight update)
```
Training step 1 logged: `reward=4.5 (reward_correct=4.0, reward_format=0.5)`,
`completions/mean_length≈191`, `clipped_ratio=0`, `step_time≈393s`. The fast loop works:
rollout on the server GPU → reward → advantage → LoRA update on the training GPUs → sync back.

**Winning combo:** vLLM 0.26.0 + **TRL 0.29.1** + Unsloth 2026.8.9 + transformers 5.5.4 +
torch 2.11, all in `.grpo_env`. Server on GPU 0 (`python -m trl.scripts.vllm_serve` against
the config-overlay), training on the rest via `train_llm_unsloth.py --vllm_server`.

Two code fixes were required and are IN PLACE:
1. **`vllm_serve.py` SamplingParams shim** — TRL 0.29 passes `truncate_prompt_tokens` to
   `SamplingParams`, which vLLM 0.26 removed → server 500 on `/generate/`. Patched both
   `SamplingParams(**generation_kwargs)` call sites to filter kwargs to the accepted set
   (`.bak` saved alongside). *This edits the installed TRL file — re-apply if TRL is reinstalled.*
2. **`train_llm_unsloth.py` typed-content normalization** — TRL 0.29's `apply_chat_template` needs
   message `content` as typed parts (`[{"type":"text","text":...}]`), not a plain string
   (crashed `TypeError: string indices must be integers`). `run_grpo` now converts prompts.
   Also made `GRPOConfig(max_prompt_length=...)` conditional (dropped in TRL 0.29).
3. **Removed the `vllm_ascend` stub** — needed for the old TRL 0.24 import, but with it present
   TRL 0.29 selected the Ascend NPU communicator at runtime and crashed `init_communicator`.
   Deleting it lets TRL use the real CUDA `PyNcclCommunicator`.

### Earlier verified pieces (subsumed by the above)
- ✅ vLLM serves the bnb-4bit gemma-4 model and generates.
- ✅ `train_llm_unsloth.py --vllm_server` sets `vllm_mode="server"` + host/port on `GRPOConfig`
  (bypasses Unsloth's colocated allowlist that blocks gemma-4).

### The core conflict (version matrix) — RESOLVED by TRL 0.29.1
| | vLLM 0.10.2 | vLLM 0.26.0 |
|---|---|---|
| TRL 0.24 `vllm_serve.py` imports | ✅ | ❌ (`GuidedDecodingParams` removed) |
| TRL **0.29.1** `vllm_serve.py` imports | ✅ | ✅ (version-gates on `vllm.__version__`) |
| Serves gemma-4 | ❌ (arch not registered) | ✅ |

**Fix (verified 2026-08-17):** `pip install --no-deps trl==0.29.1`. TRL 0.29.1's
`vllm_serve.py` branches on `Version(vllm.__version__)` to pick `StructuredOutputsParams`
and `vllm.utils.network_utils.get_open_port` for vLLM 0.26. **Import-verified with both:**
- `trl.scripts.vllm_serve` imports against vLLM 0.26 ✅
- `unsloth.FastModel` + `trl.GRPOTrainer` import together ✅ — Unsloth 2026.8.9's RL patcher
  is version-aware (handles "TRL 0.26+"); its `<=0.24` metadata pin is stale, not a code limit.

Current `.grpo_env` for the fast loop: **vLLM 0.26.0 + TRL 0.29.1 + Unsloth 2026.8.9 +
transformers 5.5.4 + torch 2.11**. Still to prove at RUNTIME: the weight-sync handshake
(`VLLMClient.init_communicator` + per-step `update_named_param`) and Unsloth's GRPO patch
mid-training under TRL 0.29 (imports pass; a 2-step run is the real test).

### Remaining tests to get the fast loop working (in order)
1. **Find a TRL version whose `vllm_serve.py` targets vLLM 0.26** (i.e. uses
   `StructuredOutputsParams`, the current distributed/util APIs). Candidates: TRL ≥ 0.25.
   Check its `trl/scripts/vllm_serve.py` imports against vLLM 0.26.
2. **Verify that TRL version still works with Unsloth 2026.8.9** (Unsloth pins `trl<=0.24`).
   If it doesn't, the training side may need to drop the Unsloth GRPO patches, or Unsloth
   must be upgraded. This is the crux risk.
3. **Start the TRL vLLM server** (`python -m trl.scripts.vllm_serve` — the `trl` CLI wrapper
   hard-imports mergekit and fails; use `-m`) against the **config-overlay** dir on GPU 0,
   and confirm it reaches "server ready" AND accepts a weight-sync init
   (`vllm_client.init_communicator`). The NCCL weight-sync handshake is the least-tested part.
4. **Launch `train_llm_unsloth.py --train_mode grpo --vllm_server`** on GPUs 1–7 (torchrun
   `--nproc_per_node 7`, `CUDA_VISIBLE_DEVICES=1..7`). Confirm: rollouts come back from the
   server, `reward` logs, and per-step `update_named_param` succeeds (adapter reaches the
   server — watch that reward actually moves, proving the server sees the *updated* policy).
5. **Batching invariant** on 7 GPUs: `batch_size * grad_acc * 7` must be divisible by
   `num_generations` (e.g. bs1 × grad_acc 8 × 7 = 56, divisible by 8 ✓).

### Fallback if the TRL/vLLM pairing can't be reconciled
On-device GRPO (no vLLM) already works in the **main venv** — slow (~300–465 s/step) but
correct. Use a bounded probe (`--max_steps 150`, `--max_completion_length 768`) to answer
"does GRPO help GSMA" without the fast loop, and revisit the vLLM path after a TRL/Unsloth
upgrade.

### Also useful now (independent of the fast loop)
The verified inference server is exactly what the **offline eval / deployment** path needs
(serve a *finished* merged adapter, drive with the OpenAI API). See
`others/VLLM_DEPLOYMENT_GUIDE_gemma4.md`.
