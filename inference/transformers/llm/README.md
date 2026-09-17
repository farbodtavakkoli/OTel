# `inference/transformers/llm` — reference LLM generation

Reference generation / prompted classification for `Qwen/Qwen3.8-27B-FP8` on plain
PyTorch + Transformers. The completions this leaf writes are the correctness baseline that
vLLM, SGLang and llama.cpp are diffed against. For throughput use vLLM.

**Hardware:** AMD MI355X (gfx950, ROCm 7.2) — FP8 `float8_e4m3fn` loads natively, 1 and 2
GPUs · NVIDIA H100 80GB (CUDA 13) — single-GPU, pass `--attn_impl sdpa`.

## Files

- `infer_llm_transformers.py` — loads the model, applies the chat template, generates
  greedily, prints completions, writes the JSON reference artifact.

## Setup

See [`../README.md`](../README.md) — one venv at `inference/transformers/` serves all three
leaves. Do not install `flash-attn`.

```bash
export HF_HOME=/path/to/hf_cache       # weights are ~30 GB; keep off /
export OUTPUT_DIR=/path/to/outputs
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # the GPUs you want to use
```

`dev.env` (symlinked to the repo-root file) supplies `HF_TOKEN`. Never set
`CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--device_map` indexes *within*
the visible set, so `cuda:0` is the first card listed above.

## Run

Smoke test the harness on the small BF16 model:

```bash
source ../.env_transformers/bin/activate
python infer_llm_transformers.py --model Qwen/Qwen3-4B --device_map cuda:0
```

Reference run, single GPU (the 30 GB FP8 checkpoint fits on one 80 GB card):

```bash
python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map cuda:0 \
  --max_new_tokens 96 \
  --output $OUTPUT_DIR/inference_llm_transformers/reference_llm_qwen3.8-27b-fp8_1gpu.json
```

Sharded across both cards (`--device_map auto`); completions are byte-identical to the
single-GPU run under greedy decoding, so shard only when one card is too small:

```bash
python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map auto \
  --max_new_tokens 96 \
  --output $OUTPUT_DIR/inference_llm_transformers/reference_llm_qwen3.8-27b-fp8_2gpu.json
```

Prompted classification against your own prompts:

```bash
python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map auto \
  --system "Reply with exactly one word: positive or negative." \
  --prompts_file my_prompts.json --max_new_tokens 8
```

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B` | Model id or path; use `Qwen/Qwen3.8-27B-FP8` for the reference run |
| `--prompts_file` | `None` | JSON list / JSONL / plain-text prompts; omit for the built-in set |
| `--system` | `None` | System message prepended to every prompt |
| `--dtype` | `auto` | `auto` honours the checkpoint dtype; or `bfloat16`/`float32`/`float16` |
| `--device_map` | `cuda:0` | `cuda:N` pins one GPU; `auto`/`balanced` shards across all visible GPUs |
| `--max_new_tokens` | `64` | Tokens generated per prompt |
| `--do_sample` | off | Sample instead of greedy; leave off for reproducible reference runs |
| `--temperature` | `0.7` | Only with `--do_sample` |
| `--top_p` | `0.8` | Only with `--do_sample` |
| `--attn_impl` | `None` | Auto-selects `sdpa` on ROCm, `eager` elsewhere; pass `sdpa` on CUDA |
| `--dequantize_fp8` | off | Dequantize FP8 to bf16 at load. Do not use on this checkpoint — it drops projection weights and emits garbage |
| `--no_deepgemm` / `--allow_deepgemm` | on | Forces the Triton FP8 kernel. Pass `--allow_deepgemm` on NVIDIA Hopper SM90+ / CUDA 12.3+ for the faster DeepGEMM path |
| `--fix_fp8_gate_proj` / `--no_fix_fp8_gate_proj` | on | Drops `mlp.gate*` skip-entries that corrupt `gate_proj`. Pass `--no_fix_fp8_gate_proj` only for a genuine MoE FP8 checkpoint, whose router those entries protect |
| `--trust_remote_code` | on | Allow checkpoint-provided modeling code |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

stdout carries the model/arch/quant/attention/device line, load time with parameter dtypes
and devices, per-GPU VRAM, each prompt with its completion, latency and tok/s, and peak
VRAM. `Paris` and `negative` on the built-in prompts are the check that matters — a
corrupted FP8 load emits fluent nonsense with no error.

`--output` writes the artifact other stacks diff against:

```
$OUTPUT_DIR/inference_llm_transformers/
  reference_llm_qwen3.8-27b-fp8_1gpu.json   <- canonical FP8 reference
  reference_llm_qwen3.8-27b-fp8_2gpu.json   <- byte-identical completions
  reference_llm_qwen3-4b_1gpu.json          <- BF16 control run
```

Each JSON records model, architecture, quant method, attention implementation, device map,
per-parameter devices and dtypes, sampling settings, seed, torch/HIP versions, GPU arch,
and per prompt the completion text and token counts. To compare another engine, run the
same prompts greedily and diff `completion` — with `do_sample=False` any divergence is
real, not sampling noise.

## Notes

- `Qwen3.8-27B` is a reasoning model: it emits a `<think>` monologue before the answer.
  Budget `--max_new_tokens` accordingly — at 8 tokens you capture only reasoning and never
  reach the label. For strict classification, parse after the last `</think>`.
  `Qwen/Qwen3-4B` behaves the same way.
- `weight_scale_inv | UNEXPECTED` in the load report is a red flag, not noise — a dropped
  scale silently corrupts that layer. Check `gate_proj.weight` is `float8_e4m3fn`, not
  `bfloat16`.
- `The fast path is not available` at load is expected on ROCm (`flash-linear-attention`
  and `causal-conv1d` are CUDA-only). Do not pip-install those on ROCm.
