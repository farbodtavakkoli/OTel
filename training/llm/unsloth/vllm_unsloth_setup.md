# GRPO env (`.grpo_env`) — Unsloth + vLLM, reproducible setup

This is the **isolated** environment for GRPO experiments that need vLLM (fast rollouts).
It is **separate** from the main training venv (see `requirements_unsloth.txt`) and does not
touch it: vLLM, TRL and Unsloth pin against each other in ways that conflict with the main
venv's pins. Do not upgrade the main venv to match — transformers ≥ 5.15 breaks the gemma-4
load (per-layer `head_dim` regression).

Working combination:

```
vllm==0.26.0   trl==0.29.1   unsloth==2026.8.9   transformers==5.5.4
torch==2.11.0+cu130   xformers==0.0.33.post1
```

`transformers 5.5.4` is load-bearing: it is ≥ 5.5.3 as vLLM 0.26 requires, but predates the
5.15.x per-layer `head_dim` regression.

---

## Serving bnb-4bit gemma-4

Standalone `vllm serve` of `gemma-4-31b-it-unsloth-bnb-4bit` works and generates correctly.

### Three blockers, in the order you hit them

1. **vLLM < 0.26 cannot serve gemma-4** — the arch is not registered, and the config trips
   `rope_scaling should have a 'rope_type' key`. Use **vLLM 0.26.0**; confirm with
   `Gemma4ForConditionalGeneration` in `ModelRegistry.get_supported_archs()` -> True.
2. **transformers 5.15 `head_dim` regression** — `AmbiguousGlobalPerLayerAttributeError:
   'head_dim' is a per-layer attribute`. vLLM pins `transformers>=5.5.3`, so install
   **`transformers==5.5.4`** (the newest that satisfies vLLM but predates the regression).
   5.5.4 *still* guards the global access for this config, so you also need:
3. **Config overlay to allow the global `head_dim` read.** gemma-4's `text_config` has both
   `head_dim=256` and `global_head_dim=512` (heterogeneous), which transformers refuses to
   read globally. Build a thin overlay dir: symlink every model file, but replace
   `config.json` with a real copy that adds `allow_global_per_layer_attribute_access: true`
   (at top level AND under `text_config`). vLLM then reports *"heterogeneous head dimensions
   … Using FA4 for all layers"* and proceeds.

### Build the config overlay
```bash
# $HF_HOME is your Hugging Face cache root
SNAP=$HF_HOME/hub/models--unsloth--gemma-4-31b-it-unsloth-bnb-4bit/snapshots/<snapshot-hash>
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
source ~/.grpo_env/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PATH="$HOME/.grpo_env/bin:/usr/local/cuda/bin:$PATH"
vllm serve /tmp/gemma4bnb_cfgfix \
  --served-model-name gemma4bnb \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --host 127.0.0.1 --port 8000
```
Startup takes several minutes. Healthy signals: `load_format=bitsandbytes`, a reported
`GPU KV cache size`, then `Application startup complete`.

### Smoke test
```bash
curl -s --noproxy '*' http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4bnb","messages":[{"role":"user","content":"Reply with exactly: hello from gemma4"}],"max_tokens":32,"temperature":0}'
```

> **Scope note:** this covers standalone serving — e.g. offline eval of a finished adapter.
> Wiring the server into GRPO training (TRL live weight-sync) needs a matched TRL version;
> see [GRPO fast loop](#grpo-fast-loop-training--separate-vllm-server) below.

---

## Environment build

```bash
python3.12 -m venv ~/.grpo_env
source ~/.grpo_env/bin/activate

# 1) vLLM first — it pins the torch/torchvision/torchaudio stack.
pip install "vllm==0.26.0"
# 2) The transformers version vLLM 0.26 needs, without the 5.15 head_dim regression.
pip install "transformers==5.5.4"
# 3) HF + RL stack. TRL 0.29.1 is the version whose vllm_serve.py supports vLLM 0.26.
pip install --no-deps "trl==0.29.1"
pip install peft accelerate datasets bitsandbytes python-dotenv
# 4) Unsloth with --no-deps so it cannot move torch/transformers, plus its skipped deps.
pip install --no-deps "unsloth==2026.8.9" "unsloth_zoo==2026.8.6"
pip install tyro structlog
```

### Non-obvious fixes needed to make the stack import

1. **Do not install mergekit.** It is pydantic-incompatible and, when present, crashes TRL's
   GRPO import chain (`callbacks.py → mergekit_utils → mergekit`). TRL does not need mergekit
   for GRPO — `pip uninstall -y mergekit` if something pulled it in.
2. **Launch the server with `python -m trl.scripts.vllm_serve`, not `trl vllm-serve`.** The
   `trl` CLI aggregator hard-imports `dpo_trainer → mergekit` and fails; the `-m` module path
   bypasses it.
3. **Do not create a `vllm_ascend` stub package.** Older vLLM made
   `importlib.util.find_spec("vllm_ascend")` succeed (a phantom plugin spec), so TRL 0.24's
   `vllm_serve.py` needed a stub to import on a CUDA box. With TRL 0.29 the opposite is true:
   a stub present makes TRL select the Ascend NPU communicator at runtime and crash
   `init_communicator`. Leave it absent so TRL uses the real CUDA `PyNcclCommunicator`.
4. **If you must build against vLLM ≤ 0.11 (torch 2.8) instead,** two extra pins apply:
   `pydantic==2.11.7` installed *with* deps (vLLM 0.10.2 needs `pydantic>=2.11.7` while
   Unsloth 2026.8.9 breaks on pydantic 2.13.x with `PydanticSchemaGenerationError` on
   `torch.Tensor`; 2.11.7 is the only version satisfying both), and the torch-2.8 xformers
   build `xformers==0.0.32.post1`. That stack cannot serve gemma-4, though — prefer
   vLLM 0.26.


---

## GRPO fast loop (training + separate vLLM server)

A fast GRPO loop trains the LoRA on most GPUs (Unsloth) while a vLLM server on a dedicated
GPU does the rollouts, with TRL syncing the updated adapter to the server every step. This
works end to end with the version combo above.

Server on GPU 0, training on the rest:

```bash
# server
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
    --model /tmp/gemma4bnb_cfgfix --port 8000 --gpu_memory_utilization 0.9

# training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 train_llm_unsloth.py \
    --train_mode grpo --vllm_server --vllm_server_host 127.0.0.1 --vllm_server_port 8000
```

`--vllm_server` sets `vllm_mode="server"` plus host/port on `GRPOConfig`, which bypasses
Unsloth's colocated allowlist that blocks gemma-4.

A healthy server access log shows every protocol call succeeding:

```
init_communicator     200   (NCCL weight-sync handshake)
update_named_param    200   (the LoRA adapter synced to the server, once per tensor)
generate              200   (rollouts returned)
reset_prefix_cache    200   (cache invalidated after the weight update)
```

**Batching invariant:** `batch_size × grad_acc_steps × world_size` must be divisible by
`num_generations` (e.g. bs 1 × grad_acc 8 × 7 ranks = 56, divisible by 8).

### Two code fixes the fast loop needs

1. **`vllm_serve.py` SamplingParams shim.** TRL 0.29 passes `truncate_prompt_tokens` to
   `SamplingParams`, which vLLM 0.26 removed → the server returns 500 on `/generate/`. Filter
   the kwargs to the accepted set at both `SamplingParams(**generation_kwargs)` call sites.
   *This edits the installed TRL file — re-apply it if TRL is reinstalled.*
2. **Typed-content normalization** (already in `train_llm_unsloth.py`). TRL 0.29's
   `apply_chat_template` needs message `content` as typed parts
   (`[{"type":"text","text":...}]`), not a plain string, or it raises
   `TypeError: string indices must be integers`. `run_grpo` converts prompts, and
   `GRPOConfig(max_prompt_length=...)` is set conditionally because TRL 0.29 dropped it.

### Fallback

On-device GRPO (no vLLM) works in the **main venv** — slow but correct. Use a bounded probe
(`--max_steps 150`, `--max_completion_length 768`) to answer "does GRPO help on my task"
without the fast loop.

The verified inference server is also what an offline-eval / deployment path needs: serve a
*finished* merged adapter and drive it with the OpenAI API.