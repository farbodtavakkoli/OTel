# verl GRPO

[verl](https://github.com/volcengine/verl) runs vLLM rollout through Ray against an FSDP2 trainer. Every GPU holds one FSDP2 rank and one colocated vLLM replica, and weight sync is adapter only. `utils.BatchedJudgeAgentLoopManager` makes one judge call per step, so runs use the classic trainer (`trainer.use_v1=False`). Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build -f training/llm/verl/grpo/setup/Dockerfile.grpo_mi355 -t grpo-verl:mi355 training/llm/verl/grpo/setup

# H100 (CUDA) venv
cd training/llm/verl/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

On ROCm start the container with `--network=host`, the allowed render nodes, `$GRPO_SHARED_DIR` at the same path, and these fixed mounts: `-v $REPO_ROOT/training/llm/verl/grpo:/workspace/verl_folder -v $REPO_ROOT/training/llm/grpo_common:/workspace/grpo`.

## Run

```bash
cd $REPO_ROOT/training/llm/verl/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_verl/bin/python train_llm_grpo_verl.py --model <hf-dir> --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl \
  --output-dir <run> --steps 2 --prompt-batch-size 8 --group-size 4 --max-tokens 256 --verify
```

In the ROCm container run `docker exec verl_grpo python3 /workspace/verl_folder/train_llm_grpo_verl.py ...`.

| Flag | Default | Meaning |
|---|---|---|
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--max-prompt-length` / `--max-tokens` | 3072 / 1024 | their sum is `max_model_len` |
| `--n-gpus` / `--rollout-tp` / `--gpu-memory-utilization` | 2 / 1 / 0.5 | colocated ranks; raise `--rollout-tp` if the policy does not fit one GPU |
| `--ppo-micro-bs` / `--logprob-micro-bs` | 2 / 4 | fixed micro batches; the fp32 log prob pass is the memory peak |
| `--weight-bucket-mb` | 8192 | keep large (see Notes) |
| `--remove-padding` | off | on: packed sequences with `flash_attention_2`; off: padded with `sdpa` |
| `--lora-adapter-path` | none | start from an SFT adapter |
| `--override` | none | any verl Hydra `key=value` |
| `--verify` / `--export-only` / `--dry-run` | off | reload and generate; export from checkpoints; print and exit |

## Notes

1. verl is pinned to commit `d040717b`, installed `--no-deps`; a plain install pulls a CUDA torch.
2. `vllm==0.28.0` exactly: 0.29 renamed `WeightsMapper.get_unstacked_mapper` and LoRA hot load fails.
3. `tensordict==0.13.0`: 0.14 breaks the `["mfu"]` metrics lookup with `TypeError`.
4. `verl_scalarlm_vllm_fork_lora.patch` is required, otherwise the adapter is dropped and the rollout policy never updates. `verl_classic_trainer_filter_groups.patch` adds DAPO dynamic sampling (off by default).
5. ROCm image relabels vLLM version metadata (verl rejects `vllm < 0.18`) and applies `accelerate_init_empty_weights_hf_initialized.patch` for text only checkpoints.
6. GPUs: on ROCm use device passthrough and set no visibility variable; on CUDA use `CUDA_VISIBLE_DEVICES`. Keep the judge GPU out.
7. A small `--weight-bucket-mb` sends LoRA tensors through CUDA IPC and fails with `IndexError` in `rebuild_ipc`.
8. Do not enable dynamic token batching in padded mode; it runs out of memory.
9. `run_sft_gemma4_verl.sh` is an optional LoRA SFT warm up; pass its exported adapter as `--lora-adapter-path`. Keep `data.pad_mode=no_padding` and `utils.GrpoSFTDataset`.
10. The Hydra `outputs/` tree is root owned in the container; remove it from inside the container.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
