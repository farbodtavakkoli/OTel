# TRL + DeepSpeed GRPO

[TRL](https://github.com/huggingface/trl) `GRPOTrainer` with a PEFT LoRA policy under DeepSpeed ZeRO, in split placement: `trl.scripts.vllm_serve` runs on its own GPU and the trainer reaches it over HTTP. Each step TRL 1.12 merges the adapter, streams the whole model over NCCL/RCCL, then unmerges. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build -f training/llm/deepspeed/grpo/setup/Dockerfile.grpo_mi355 \
  -t grpo-trl:mi355 training/llm/deepspeed/grpo/setup

# H100 (CUDA) venv; run with --server-mode process
cd training/llm/deepspeed/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

## Run

```bash
cd $REPO_ROOT/training/llm/deepspeed/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_trl_deepspeed/bin/python train_llm_grpo_deepspeed.py --server-mode process \
  --model <hf-dir> --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl \
  --output-dir $GRPO_SHARED_DIR/runs/trl_deepspeed --server-gpu 0 --train-gpu 1 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` / `--data` / `--output-dir` | required | checkpoint dir, prompt JSONL, run dir |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--per-device-batch` | 8 | micro batch and the main memory knob |
| `--max-tokens` / `--max-model-len` | 1024 / 8192 | completion cap; server context, must cover prompt plus completion |
| `--zero-stage` | 2 | DeepSpeed ZeRO stage via `accelerate launch` (`0` is the plain Trainer) |
| `--server-mode` | `docker` | `docker`, `process` or `external` (server already running) |
| `--server-gpu` / `--train-gpu` | 0 / 1 | indices into the visible GPU set, not physical ids |
| `--server-timeout` | 900 | seconds to wait for server `/health` |
| `--verify` / `--verify-only` / `--dry-run` | off | reload adapter and generate; reload only; print and exit |

## Notes

1. Pins: `trl==1.12.0`, `deepspeed==0.19.6`, all `--no-deps`; a resolving install pulls a CUDA torch over the ROCm one.
2. Server and trainer must run the same TRL version; the weight transfer protocol is version specific.
3. vLLM must be 0.27.1 or older: TRL 1.12 imports `NCCLTrainerSendWeightsArgs`, removed in 0.28.
4. `msgpack` (imported by DeepSpeed) and `ninja` (JIT build of `fused_adam`, must be on the trainer `PATH`) are required.
5. No source patches; `utils.patch_trl_vllm_client()` fixes at run time TRL's version gate that rejects `/update_weights` on an untagged vLLM build.
6. Both sides must see both GPUs: the per step push is a cross process collective and fails at `ncclCommInitRank` otherwise. Keep the judge GPU out.
7. ROCm: set only `ROCR_VISIBLE_DEVICES`; adding `HIP_VISIBLE_DEVICES` can leave zero devices.
8. Prompts are not truncated (TRL 1.12 has no `max_prompt_length`), so `--max-model-len` must cover the longest prompt.
9. A large checkpoint takes minutes to become healthy; raise `--server-timeout` rather than assume a hang.
10. TRL `completions/clipped_ratio` reads 1.0 on Gemma style checkpoints; use the reward `finish_length_frac` instead.
11. The server container is always named `grpo-trl-vllm`, so two runs of this stack cannot share a host.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
