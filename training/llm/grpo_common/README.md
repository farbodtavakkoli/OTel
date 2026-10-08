# GRPO common harness

Shared pieces for the GRPO recipes under `training/llm/<framework>/grpo` and `training/llm/scalarlm`. Every stack uses the same policy, prompts, judge and reward, so two runs differ only by the stack.

| File | Purpose |
|---|---|
| `judge_server.sh` | starts, stops or checks the shared vLLM judge on its own GPU |
| `judge_reward.py` | the reward every stack scores through; run it directly for a self check (10 / 0 / 0) |
| `split_eval.py` | optional seeded train and eval split per source |
| `setup_common.sh` | CUDA venv helpers sourced by each `setup/setup.sh` |
| `build_base_image.sh`, `Dockerfile.base_mi355` | the ROCm base image `grpo-base:mi355` every MI355X stack image builds on |
| `data/grpo_prompts.jsonl` | 70 telecom QA prompts, 10 from each of seven sources |
| `.env.example` | paths every stack reads |

## Quick start

```bash
cp training/llm/grpo_common/.env.example .env.grpo && $EDITOR .env.grpo && source .env.grpo
export GRPO_PLATFORM=rocm
mkdir -p $GRPO_SHARED_DIR/data && cp training/llm/grpo_common/data/grpo_prompts.jsonl $GRPO_SHARED_DIR/data/
training/llm/grpo_common/build_base_image.sh
JUDGE_GPU=0 training/llm/grpo_common/judge_server.sh start
training/llm/grpo_common/judge_server.sh health
```

Set `GRPO_PLATFORM=cuda` on NVIDIA; there each stack builds a venv with `setup/setup.sh` instead of an image. Then follow the README of the chosen stack.

## Environment

| Variable | Meaning |
|---|---|
| `GRPO_SHARED_DIR` | holds `models/<org>--<name>/`, `data/`, `runs/<stack>/` |
| `GRPO_ENV_DIR` | per stack venvs; must be a local disk without `noexec` |
| `HF_CACHE_DIR` | mounted into containers as `/hf_cache` |
| `GRPO_PLATFORM` | `rocm` (default) or `cuda`; only changes GPU selection |
| `GRPO_JUDGE_URL`, `GRPO_JUDGE_MODEL`, `GRPO_JUDGE_CONC`, `GRPO_JUDGE_RUBRIC` | how stacks reach and score with the judge |
| `JUDGE_GPU`, `JUDGE_PORT`, `JUDGE_MODEL_DIR`, `JUDGE_IMAGE` | judge server placement and model |
| `GRPO_COMMON_DIR` | overrides where stacks find this folder |

## Data contract

One JSON object per line. `ground_truth` is a JSON string, not an object.

```json
{"messages": [{"role": "user", "content": "..."}],
 "ground_truth": "{\"question\": \"...\", \"reference\": \"...\", \"source\": \"...\", \"id\": \"...\"}"}
```

`messages` must not contain the answer. `reference` is what the judge grades against. Keep rows interleaved by source: grouped rows make every completion in a step score alike, which gives zero gradient while the run still exits 0.

## Further reading

[Method](../../../docs/grpo_method.md), [findings and stack choice](../../../docs/grpo_findings.md), [ROCm and CUDA porting](../../../docs/grpo_porting.md).
