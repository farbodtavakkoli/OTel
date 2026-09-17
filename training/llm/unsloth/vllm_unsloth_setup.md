# `.grpo_env` — Unsloth + vLLM environment for GRPO rollouts

The isolated venv for GRPO experiments that need vLLM rollouts, and for standalone serving of a
gemma-4 bnb-4bit checkpoint. It is **separate** from the main training venv and does not touch
it — vLLM, TRL, and Unsloth pin against each other in ways that conflict with
`requirements_unsloth.txt`. Everything else (SFT, on-device GRPO, arguments, data schema) lives
in [`readme_unsloth.md`](readme_unsloth.md).

Working combination:

```
vllm==0.26.0   trl==0.29.1   unsloth==2026.8.9   transformers==5.5.4
torch==2.11.0+cu130   xformers==0.0.33.post1
```

`transformers==5.5.4` is load-bearing: it satisfies vLLM 0.26's `transformers>=5.5.3` while
predating the 5.15.x per-layer `head_dim` regression that breaks the gemma-4 load.

## Build the environment

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

- **Do not install mergekit.** It is pydantic-incompatible and crashes TRL's GRPO import chain
  (`callbacks.py -> mergekit_utils -> mergekit`). TRL does not need it for GRPO;
  `pip uninstall -y mergekit` if something pulled it in.
- **Do not create a `vllm_ascend` stub package.** With TRL 0.29 a stub makes TRL select the
  Ascend NPU communicator and crash `init_communicator`; leave it absent so TRL uses the real
  CUDA `PyNcclCommunicator`.
- Launch the server with `python -m trl.scripts.vllm_serve`, never `trl vllm-serve` — the CLI
  aggregator hard-imports `dpo_trainer -> mergekit` and fails.

## Config overlay for bnb-4bit gemma-4

gemma-4's `text_config` carries both `head_dim=256` and `global_head_dim=512`, which
transformers refuses to read globally. Build a thin overlay directory: symlink every model
file, then replace `config.json` with a patched copy.

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

vLLM then reports *"heterogeneous head dimensions ... Using FA4 for all layers"* and proceeds.
Confirm the arch is registered with `Gemma4ForConditionalGeneration` in
`ModelRegistry.get_supported_archs()`.

## Serve it

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

```bash
curl -s --noproxy '*' http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma4bnb","messages":[{"role":"user","content":"Reply with exactly: hello from gemma4"}],"max_tokens":32,"temperature":0}'
```

This standalone server is also the offline-eval / deployment path: serve a finished merged
adapter and drive it with the OpenAI API.

## GRPO fast loop

Train the LoRA on most GPUs while a vLLM server on a dedicated GPU does the rollouts, with TRL
syncing the updated adapter to the server every step. `--vllm_server` sets `vllm_mode="server"`
plus host/port on `GRPOConfig`, which bypasses Unsloth's colocated allowlist that blocks
gemma-4.

```bash
# server
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
    --model /tmp/gemma4bnb_cfgfix --port 8000 --gpu_memory_utilization 0.9

# training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node 7 train_llm_unsloth.py \
    --train_mode grpo --vllm_server --vllm_server_host 127.0.0.1 --vllm_server_port 8000
```

`batch_size x grad_acc_steps x world_size` must be divisible by `num_generations` (e.g.
1 x 8 x 7 = 56, divisible by 8).

A healthy server access log shows every protocol call succeeding:

```
init_communicator     200   (NCCL weight-sync handshake)
update_named_param    200   (the LoRA adapter synced to the server, once per tensor)
generate              200   (rollouts returned)
reset_prefix_cache    200   (cache invalidated after the weight update)
```

**One patch the fast loop needs:** TRL 0.29 passes `truncate_prompt_tokens` to
`SamplingParams`, which vLLM 0.26 removed, so the server returns 500 on `/generate/`. Filter the
kwargs to the accepted set at both `SamplingParams(**generation_kwargs)` call sites in the
installed `vllm_serve.py`, and re-apply after any TRL reinstall.

If you do not need the fast loop, on-device GRPO in the main venv is slower but correct — use a
bounded probe (`--max_steps 150`, `--max_completion_length 768`).
