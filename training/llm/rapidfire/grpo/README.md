# RapidFire AI GRPO

[RapidFire AI](https://github.com/RapidFireAI/rapidfireai) wraps TRL `GRPOTrainer` (`RFGRPOConfig`) and runs it in a spawned worker per visible GPU. Placement is split: a `trl vllm-serve` rollout server on one GPU, the trainer on another. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. Build contexts and `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container; the three wheels it copies are not in the repo
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
python3 -m pip download --no-deps rapidfireai==0.16.1 trl==1.12.0 bitsandbytes==0.50.2 \
  -d training/llm/rapidfire/grpo/setup
docker build -f training/llm/rapidfire/grpo/setup/Dockerfile.rapidfire_mi355 \
  -t grpo-rapidfire:mi355 training/llm/rapidfire/grpo/setup

# H100 (CUDA) venv; run with --server-mode process
cd training/llm/rapidfire/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

## Run

```bash
cd $REPO_ROOT/training/llm/rapidfire/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_rapidfire/bin/python train_llm_grpo_rapidfire.py --model $GRPO_SHARED_DIR/models/<policy> \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl --output-dir $GRPO_SHARED_DIR/runs/rapidfire/smoke \
  --server-mode process --server-gpu 0 --train-gpu 1 \
  --steps 2 --warmup-steps 0 --prompt-batch-size 8 --group-size 4 --max-tokens 256 --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` / `--data` / `--output-dir` | required | absolute checkpoint dir, prompt JSONL, run dir |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--per-device-batch` / `--max-tokens` | 8 / 1024 | completions per micro batch; completion cap |
| `--server-mode` | `docker` | `docker` (two containers) or `process` (server as child process) |
| `--server-gpu` / `--train-gpu` | 0 / 1 | indices into the GPUs each side can see |
| `--experiment-name` | `rf-grpo-<output dir name>` | must be unique per RapidFire DB; set it on a re run |
| `--shm-size` / `--image` | `100g` / `GRPO_IMAGE` | `docker` mode trainer `/dev/shm` and image |
| `--verify` / `--verify-only` / `--dry-run` | off | reload adapter and generate; check only; print commands |

## Notes

1. `rapidfireai==0.16.1` goes in `--no-deps`; its own pins would replace TRL and torch. Do not run `rapidfireai init --train`.
2. `trl==1.12.0` must match on server and trainer; `vllm==0.27.1` because TRL 1.12 imports a symbol removed in 0.28.
3. `mlflow>=3.16.0` (older caps `pandas<3`), `bitsandbytes` (imported by the shm transfer), `dill<0.3.9` and `ipython` are required; `kernels` must be absent.
4. CUDA `setup.sh` patches starlette `wsgi.py`, otherwise every MLflow request returns 500.
5. `/dev/shm` must hold a bf16 copy of the base model plus about 1 GiB, and `nofile` must be high (1048576). `docker` mode sets both; in `process` mode the host must.
6. The trainer must see exactly one GPU; a second visible GPU spawns a second worker, not a sharded trainer.
7. In `docker` mode both containers get both GPUs (the weight push opens the peer GPU); the visibility variable narrows each side. Exclude the judge GPU.
8. ROCm: pin with `ROCR_VISIBLE_DEVICES` only; `HIP_VISIBLE_DEVICES` must stay unset.
9. With `--num-chunks` above 1 and few steps a run can end without a `final_checkpoint`.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
