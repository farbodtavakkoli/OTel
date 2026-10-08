# Red Hat training hub GRPO

[training_hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) `lora_grpo(backend="verl")` is a thin wrapper over verl without override passthrough, so the launcher intercepts the verl subprocess, injects what this environment needs and starts its own Ray head. The resolved command lands in `command.txt`. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build -f training/llm/redhat/grpo/setup/Dockerfile.grpo_mi355 \
  -t grpo-traininghub:mi355 training/llm/redhat/grpo/setup

# H100 (CUDA) venv
cd training/llm/redhat/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

On ROCm start the container with `--network=host`, only the allowed render nodes, `$GRPO_SHARED_DIR` mounted at the same path inside and out, and `$REPO_ROOT` mounted at `/workspace/handoff`. Then run from `/workspace/handoff/training/llm/redhat/grpo` with `docker exec <container> python` in place of the venv python.

## Run

```bash
cd $REPO_ROOT/training/llm/redhat/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_redhat_traininghub/bin/python train_llm_grpo_redhat.py --model <hf-dir> \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl --output-dir $GRPO_SHARED_DIR/runs/redhat_traininghub --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` / `--data` / `--output-dir` | required | checkpoint dir, prompt JSONL, run dir |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--max-tokens` / `--max-prompt-length` | 1024 / 3072 | completion cap; prompt cap (keep it set) |
| `--n-gpus` / `--gpu-memory-utilization` | 2 / 0.5 | GPU count; vLLM share of each GPU |
| `--lora-regex` | derived | verl `target_modules`; multimodal models adapt only the text tower |
| `--override` | none | any verl `key=value`, repeatable |
| `--verify` / `--export-only` / `--dry-run` | off | reload under PEFT and generate; export from existing checkpoints; print and exit |

## Notes

1. Do not `pip install training-hub[grpo]`; the extras replace torch, vLLM and transformers. verl and training_hub go in from pinned commits `--no-deps`.
2. `verl_scalarlm_vllm_fork_lora_redhat.patch` is required, otherwise the vLLM fork loads the adapter from disk and the rollout policy stays frozen.
3. `accelerate_init_empty_weights_hf_initialized.patch` fixes text only checkpoints failing with `_is_hf_initialized` (accelerate 1.9.0 with transformers 5.12).
4. The ROCm image relabels vLLM version metadata, because verl refuses the untagged `0.1.dev` build as `vllm < 0.18`.
5. `vllm==0.28.0`: 0.29 renamed `WeightsMapper.get_unstacked_mapper` and LoRA hot load fails.
6. `tensordict==0.13.0`: 0.14 breaks the `["mfu"]` metrics lookup with `TypeError`.
7. `trl==1.12.0`: verl imports `AutoModelForCausalLMWithValueHead`, which 1.13 dropped.
8. GPUs: on CUDA use `CUDA_VISIBLE_DEVICES`; on ROCm use device passthrough and set no visibility variable (Ray rewrites them). Keep the judge GPU out.
9. Keep `--max-prompt-length` set; the training_hub default of 16384 pads every sequence and runs out of memory.
10. An empty `adapter_checkpoints: []` is expected; the launcher exports the adapter from the shards.
11. CUDA: `GRPO_RAY_OBJECT_STORE_GB` (default 32) caps the Ray object store so a large `/dev/shm` does not kill the raylet.
12. A `DataLoader worker killed by signal` traceback at exit is benign.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
