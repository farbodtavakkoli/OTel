# Axolotl GRPO

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) runs GRPO from one YAML (`rl: grpo`) on top of TRL, with `axolotl vllm-serve` as a LoRA aware rollout server. Weight sync is a filesystem hand off of the adapter, so the server and the trainer each need one GPU. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. Build contexts and `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build --network host -f training/llm/axolotl/grpo/setup/Dockerfile.grpo_mi355 \
  -t grpo-axolotl:mi355 training/llm/axolotl/grpo/setup

# H100 (CUDA) venv; run with --server-mode process
cd training/llm/axolotl/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

## Run

Needs two GPUs besides the judge. `--config` is resolved relative to the working directory. Swap in `grpo_bench_llama31_8b.yaml` (8B) or `grpo_bench.yaml` (31B) for the full run.

```bash
cd $REPO_ROOT/training/llm/axolotl/grpo
../../grpo_common/judge_server.sh start
python3 train_llm_grpo_axolotl.py --config grpo_smoke_llama31_8b.yaml \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl \
  --output-dir $GRPO_SHARED_DIR/runs/axolotl/smoke --server-gpu 0 --train-gpu 1 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | required | GRPO YAML; `--model`, `--data`, `--output-dir`, `--steps` etc. override its keys |
| `--set KEY=VALUE` | none | override any other YAML key (dotted, repeatable) |
| `--server-mode` | `docker` | `docker` (one container per side from `--image`) or `process` (current env) |
| `--server-gpu` / `--train-gpu` | `0` / `1` | GPU index, or `/dev/dri/renderD<N>` on ROCm in `docker` mode |
| `--image` / `--sync-dir` | `GRPO_IMAGE` / `$GRPO_ENV_DIR/env_axolotl/lora_sync` | container image; shared adapter hand off directory |
| `--server-timeout` | `1200` | seconds to wait for the server `/health/` |
| `--verify` / `--export-only` / `--dry-run` | off | reload adapter and generate; check an existing adapter only; print commands |

## Notes

1. Everything is installed `--no-deps`; a resolving install replaces the torch vLLM was built against.
2. `vllm==0.27.1`: TRL imports a vLLM NCCL symbol removed in 0.28, so `GRPOTrainer` fails to import on newer versions.
3. `trl==1.8.0` (Axolotl 0.18.0 pin): server and trainer must share one TRL; TRL 1.12 removed the in process server.
4. `liger-kernel==0.8.0` is required: Axolotl always patches Gemma 4 attention with a Liger kernel.
5. `accelerate==1.13.0`, `bitsandbytes==0.50.0`, `evaluate==0.4.6` are import time requirements of `axolotl.cli.train`.
6. On CUDA the venv `bin/` must lead `PATH`, because Axolotl re execs `accelerate` and `axolotl` by bare name.
7. Axolotl kills every `VLLM::EngineCore` process on the host at start and exit, the judge included; use `docker` mode or a dedicated host.
8. `trl.use_data_producer: true` is required; without it the adapter sync key is dropped and sync falls back to merge plus NCCL.
9. `--sync-dir` must be the same local path on both sides; in `process` mode keep it short (it becomes `TMPDIR` with `AF_UNIX` sockets).
10. `micro_batch_size` is the memory lever; prompts times group size must be a multiple of it.
11. `utils.main` relabels `vllm.__version__` (`GRPO_VLLM_VERSION_RELABEL`), else an untagged dev build returns HTTP 500 on every request.
12. ROCm: pin with `ROCR_VISIBLE_DEVICES` only; adding `HIP_VISIBLE_DEVICES` can yield zero devices. Exclude the judge GPU.
13. A `PermissionError` on the adapter check after a good run means a root owned adapter; re check in container with `--export-only --verify --server-mode process`.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
