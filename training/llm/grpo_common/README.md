# GRPO common harness

Shared pieces for the GRPO recipes under `training/llm/<framework>/grpo` and `training/llm/scalarlm`. Stacks share the local judge policy by default; ScalarLM can explicitly select an API judge with a different rubric.

| File | Purpose |
|---|---|
| `judge_server.sh` | starts, stops or checks the shared vLLM judge on its own GPU |
| `grpo_judge_local.py` | shared local telecom reward; run directly for a self check (10 / 0 / 0) |
| `grpo_judge_api.py` | configurable API correctness and helpfulness reward |
| `tests/grpo_judge_test.py` | automated judge tests without live API calls |
| `judge_reward.py` | compatibility alias for `grpo_judge_local` used by other stacks |
| `haiku_reward.py` | compatibility alias for `grpo_judge_api` |
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
| `GRPO_JUDGE_URL`, `GRPO_JUDGE_CONC`, `GRPO_JUDGE_RUBRIC` | how stacks reach and score with the local judge |
| `GRPO_JUDGE_MODULE` | ScalarLM reward module: `grpo_judge_local` (default) or `grpo_judge_api` |
| `GRPO_JUDGE_MODEL` | local served model name, or required API model name |
| `GRPO_JUDGE_BASE_URL` | required API endpoint; legacy `LITELLM_BASE_URL` is supported |
| `GRPO_JUDGE_API_KEY` | API credential; legacy `LITELLM_KEY` or `LITELLM_API_KEY` is supported |
| `GRPO_JUDGE_MAX_RETRIES` | API SDK retry count, default 2 |
| `JUDGE_GPU`, `JUDGE_PORT`, `JUDGE_MODEL_DIR`, `JUDGE_IMAGE` | judge server placement and model |
| `GRPO_COMMON_DIR` | overrides where stacks find this folder |

The API judge has no default model or endpoint. Set its model, endpoint and credential before use.

## Tests

From the OTel repository root, target only `grpo_common/tests` so pytest loads its configuration:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider training/llm/grpo_common/tests
```

For combined suites from a common ancestor, explicitly select that configuration so `grpo_judge_test.py` is collected:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -c training/llm/grpo_common/tests/pytest.ini training/llm/grpo_common/tests training/llm/scalarlm/tests
```

## Data contract

One JSON object per line. `ground_truth` is a JSON string, not an object.

```json
{"messages": [{"role": "user", "content": "..."}],
 "ground_truth": "{\"question\": \"...\", \"reference\": \"...\", \"source\": \"...\", \"id\": \"...\"}"}
```

`messages` must not contain the answer. `reference` is what the judge grades against. Keep rows interleaved by source: grouped rows make every completion in a step score alike, which gives zero gradient while the run still exits 0.

## Further reading

[Method](../../../docs/grpo_method.md), [findings and stack choice](../../../docs/grpo_findings.md), [ROCm and CUDA porting](../../../docs/grpo_porting.md).
