# OpenRLHF GRPO

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) runs GRPO as `openrlhf.cli.train_ppo_ray` with the `group_norm` estimator. Ray places a DeepSpeed ZeRO 2 LoRA actor on one GPU and a vLLM rollout engine on another; weight sync is a merged broadcast. Data contract, judge and environment variables: [`../../grpo_common/README.md`](../../grpo_common/README.md).

## Setup

Commands run from `REPO_ROOT`. `GRPO_ENV_DIR` must be on a local disk without `noexec`.

```bash
# MI355X (ROCm) container
training/llm/grpo_common/build_base_image.sh          # once per host, builds grpo-base:mi355
docker build -f training/llm/openrlhf/grpo/setup/Dockerfile.grpo_mi355 \
  -t grpo-openrlhf:mi355 training/llm/openrlhf/grpo/setup

# H100 (CUDA) venv; compiles flash_attn from source, budget hours
cd training/llm/openrlhf/grpo && export GRPO_ENV_DIR=/path/to/envs && ./setup/setup.sh
```

## Run

On ROCm run inside the container with `--network host`, `/dev/kfd` and the two render nodes.

```bash
cd $REPO_ROOT/training/llm/openrlhf/grpo
../../grpo_common/judge_server.sh start
$GRPO_ENV_DIR/env_openrlhf/bin/python train_llm_grpo_openrlhf.py --model $GRPO_SHARED_DIR/models/<policy> \
  --data $GRPO_SHARED_DIR/data/grpo_prompts.jsonl --output-dir $GRPO_SHARED_DIR/runs/openrlhf/smoke \
  --steps 2 --prompt-batch-size 8 --group-size 4 --max-tokens 256 --enforce-eager --verify
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` / `--data` / `--output-dir` | required | checkpoint dir, prompt JSONL, run dir |
| `--steps` / `--prompt-batch-size` / `--group-size` | 22 / 32 / 8 | steps, prompts per step, completions per prompt |
| `--max-tokens` / `--max-len` | 1024 / 4096 | completion cap; prompt plus response cap |
| `--actor-gpus` / `--vllm-engines` / `--vllm-tp` | 1 / 1 / 1 | placement; Ray starts with actor + engines × tp GPUs |
| `--micro-train-bs` / `--micro-rollout-bs` | 4 / 8 | update micro batch; old log prob micro batch (the memory peak) |
| `--gpu-memory-utilization` / `--enforce-eager` | 0.85 / off | vLLM memory share; skip graph capture when short of memory |
| `--override` | none | any OpenRLHF option, repeatable (`~key` removes) |
| `--verify` / `--export-only` / `--dry-run` | off | generate after training; report an existing run; print command |

## Notes

1. Stock OpenRLHF 0.11.0 never syncs the LoRA adapter to vLLM, so nothing learns; `setup/openrlhf_0.11.0_lora_vllm_sync.patch` is mandatory.
2. The patch broadcasts merged weights under HF names, scopes LoRA targets with one `re:` regex, and accepts an untagged `0.1.dev` vLLM; `patch_openrlhf_lora_sync.py --check` verifies it.
3. Confirm sync: the log must show `syncing N merged LoRA module weights to vLLM` once per step.
4. Pins: `openrlhf==0.11.0` (patch anchors), `deepspeed==0.19.5`, `ray[default]==2.55.0`, all `--no-deps` so torch is not replaced.
5. `msgpack`, `hjson`, `py-cpuinfo` and other pure python imports are installed explicitly because `--no-deps` skips them.
6. `flash_attn` is imported at module scope even for `sdpa`, so CUDA compiles it from source.
7. Merged sync works within one node only and rejects ZeRO 3.
8. Raise the open file limit (`ulimit -n 262144`) or Ray aborts with `Too many open files`.
9. GPU count is derived; which GPUs comes from the visibility variable or container devices. Pinning inside the run has no effect (Ray rewrites it). Keep the judge GPU out.
10. Rows are consumed in file order; keep the data interleaved by source or groups have zero gradient while the run exits 0.

Benchmarked on MI355X, smoke validated on H100. See [findings](../../../../docs/grpo_findings.md) and [porting](../../../../docs/grpo_porting.md).
