# Unsloth GRPO

[Unsloth](https://github.com/unslothai/unsloth) runs TRL `GRPOTrainer` on its own kernels with an in process vLLM engine colocated with the trainer on a single GPU (`fast_inference=True`). The engine shares the model weights, so there is no weight sync and no rollout server. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build -f training/llm/unsloth/grpo/setup/Dockerfile.grpo_mi355 \
  -t grpo-unsloth:mi355 training/llm/unsloth/grpo/setup

# H100 (CUDA) venv
cd training/llm/unsloth/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

On ROCm start the container with exactly one render node, `--network host`, `$GRPO_SHARED_DIR` mounted at the same path, `$REPO_ROOT` mounted at `/work` and working directory `/work/training/llm/unsloth/grpo`.

## Run

```bash
cd $REPO_ROOT/training/llm/unsloth/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_unsloth/bin/python train_llm_grpo_unsloth.py --model <hf-dir> \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl --output-dir $GRPO_SHARED_DIR/runs/unsloth --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` / `--data` | staged 8B model / `grpo_prompts.jsonl` under `$GRPO_SHARED_DIR` | policy checkpoint, prompt JSONL |
| `--output-dir` | required | run dir |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--per-device-bs` | 2 | micro batch; the generation batch must be a multiple of it |
| `--max-tokens` / `--max-seq-length` | 1024 / 4096 | completion cap; prompt plus completion cap |
| `--gpu-memory-utilization` | 0.5 | permanent KV cache share of the colocated engine |
| `--logprob-chunks` | generation batch / 8 | pins `unsloth_grpo_mini_batch`; autotune runs out of memory on large policies |
| `--allow-gemma4-vllm` | off | force the gemma4 fast inference path (see Notes) |
| `--verify` / `--verify-only` / `--dry-run` | off | reload adapter in a fresh process and generate; check only; print and exit |

## Notes

1. Pins: `unsloth==2026.9.2`, `unsloth_zoo==2026.9.1`, `trl==1.12.0`, CUDA `vllm==0.27.1`. Unsloth goes in `--no-deps` because its metadata pins stale torch, transformers and TRL.
2. `setup/unsloth_zoo_vllm_utils_gemma4_31b.patch` guards the vLLM state dict conversion against `None` Gemma 4 fields; the file is CRLF normalised first.
3. Unsloth fast inference does not support the `gemma4` architecture and produces corrupted generations on large multimodal policies; a text only 8B policy is fine.
4. ROCm: set `UNSLOTH_GRPO_SEQ_PACKING=0` (the image does) or the backward fails with `CheckpointError`.
5. ROCm: `utils.patch_vllm_fork_lora_manager()` routes the vLLM fork LoRA manager back to Unsloth's in memory one.
6. Do not set `TORCH_COMPILE_DISABLE=1`; Unsloth needs a compiled RMSNorm and fails with `found no compiled frames`.
7. Exactly one GPU is used. On ROCm set one visibility variable only; combining `ROCR_`, `HIP_` and `CUDA_VISIBLE_DEVICES` on one index yields zero devices. Keep the judge GPU out.
8. The policy plus KV cache must fit one GPU; rollouts cannot be sharded.
9. `unsloth_compiled_cache/` appears next to the launcher on first run and is safe to delete.

Benchmarked on MI355X with an 8B policy, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
