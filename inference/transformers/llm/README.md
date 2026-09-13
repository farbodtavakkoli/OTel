# `infer_llm_transformers.py`

## Overview & when to use

**Reference LLM generation / prompted classification** for `Qwen/Qwen3.8-27B-FP8` on the
Transformers + PyTorch stack. This is the **correctness baseline**: the text this folder
generates is what vLLM, SGLang and llama.cpp get diffed against. For throughput use vLLM.

The script loads the model, applies the chat template to each prompt, generates greedily
(deterministic by default), prints the completions, and writes a JSON reference artifact.

## FP8 status — it loads, and it works

FP8 loads natively on gfx950; no BF16 fallback is needed. The checkpoint is blockwise FP8
(`quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`,
`weight_block_size: [128,128]`), and **gfx950 uses OCP E4M3FN, not FNUZ** — the same format
the checkpoint stores — so weights load as `torch.float8_e4m3fn` with `float32`
`weight_scale_inv` scales.

Two fixes are required, both **on by default**. Do not turn them off:

1. **`--no_deepgemm`** — transformers dispatches `128×128` blockwise FP8 to DeepGEMM and
   only catches `ImportError` from the probe. On gfx950 the probe raises
   `OSError: libcudart.so: cannot open shared object file` instead (ROCm reports capability
   9.5 and slips past the Hopper gate), so generation **crashes**. The flag pre-marks
   DeepGEMM unavailable and execution falls through to the Triton `finegrained-fp8` kernel.
   This needs **`kernels>=0.12.0,<0.13`**: without the package you get
   `AttributeError: 'NoneType' object has no attribute 'w8a8_fp8_matmul'`, and with
   `kernels` 0.16.0 transformers 5.5.0 fails at import
   (`ValueError: Either a revision or a version must be specified`).
2. **`--fix_fp8_gate_proj`** — the checkpoint's `modules_to_not_convert` carries defensive
   MoE entries named `...mlp.gate`, which transformers prefix-matches against
   `...mlp.gate_proj`. `gate_proj` then loads as plain `bfloat16` with its scales dropped
   (`...mlp.gate_proj.weight_scale_inv | UNEXPECTED` in the load report) and the model emits
   **fluent garbage** — no error, no crash. The flag drops those skip entries.
   Dropping them is safe here only because **this checkpoint is dense** — it has no real MoE
   router, so the `mlp.gate*` entries guard nothing. If you point the script at a genuine MoE
   FP8 checkpoint, those same entries are load-bearing: pass `--no_fix_fp8_gate_proj` there,
   or you will quantize a router the checkpoint deliberately kept out of FP8.

**Do not pass `--dequantize_fp8` on this checkpoint** — the dequantize path drops nearly
every projection weight and also produces garbage.

## Install

Python 3.12, in its own venv.

One venv at `inference/transformers/` serves all three leaves.

### AMD / ROCm

```bash
cd inference/transformers
python3 -m venv .env_transformers
source .env_transformers/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements.txt
```

**Pin `torchvision`.** A bare `pip install torchvision` resolves to the newest wheel and
silently upgrades torch to `2.13.0+rocm7.2`, breaking the `2.11.0` pin this repo
standardises on. `torchvision` and `pillow` are needed because the Qwen3.8 checkpoint is
multimodal and `AutoProcessor` imports the image processor even for text-only use.

Do **not** install `flash-attn`; it is a CUDA build and the script selects `sdpa`.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# True 7.2.x
```

### NVIDIA / CUDA 13

Default PyPI wheels are CUDA-enabled; install torch first, then the rest:

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch==2.11.0        # CUDA 13 build from PyPI; no --index-url needed
pip install -r requirements.txt  # the torch pin is already satisfied
```

The PyPI `torch==2.11.0` wheel is a CUDA 13 build, so the `requirements.txt` pin holds on
CUDA as it does on ROCm. Add `--index-url https://download.pytorch.org/whl/cu130` if you
want the `+cu130` local version tag.

On CUDA the script auto-selects `eager` (`torch.version.hip` is `None`), so **pass
`--attn_impl sdpa` explicitly**. `flash-attn` has no prebuilt cu130 wheel and does not build
in reasonable time; `sdpa` picks an efficient Hopper kernel, so there is no correctness cost.

On NVIDIA you can pass `--allow_deepgemm` to re-enable the (3–6× faster) DeepGEMM FP8
path, which needs Hopper SM90+ and CUDA runtime 12.3+.

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`. Weights are ~30 GB, so keep the cache off `/`:

```bash
# Set these to suit your machine
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache (large volume)
export OUTPUT_DIR=/path/to/outputs     # inference artifacts

export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1   # the GPUs you want to use
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--device_map` is
indexed **within** the visible set, so `cuda:0` is the first card listed above.

## Run

Single GPU (the 30 GB FP8 checkpoint fits comfortably on one 80 GB+ card):

```bash
source .env_transformers/bin/activate
export HIP_VISIBLE_DEVICES=0,1 CUDA_VISIBLE_DEVICES=0,1

python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map cuda:0 \
  --max_new_tokens 96 \
  --output $OUTPUT_DIR/inference_llm_transformers/reference_llm_qwen3.8-27b-fp8_1gpu.json
```

Multi-GPU, sharded across both cards:

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

BF16 fallback model (fast smoke test of the harness itself):

```bash
python infer_llm_transformers.py --model Qwen/Qwen3-4B --device_map cuda:0
```

## Expected output

`Qwen/Qwen3.8-27B-FP8 --device_map cuda:0`:

```
fp8 fix: dropped 130 'mlp.gate*' entries from modules_to_not_convert
model=Qwen/Qwen3.8-27B-FP8 arch=Qwen3_5ForConditionalGeneration quant=fp8 attn=sdpa device_map=cuda:0
load: 31.3s | param dtypes=['torch.bfloat16', 'torch.float32', 'torch.float8_e4m3fn'] | devices=['cuda:0']
VRAM(GiB)=[28.38, 0.0]
DeepGEMM kernel is not available or compatible, falling back to Triton finegrained-fp8 kernel.

[PROMPT] What is the capital of France? Answer in one word.
OUTPUT:  'User asks: "What is the capital of France? Answer in one word." Need one word
          answer. Capital is Paris. Ensure one word.\n</think>\n\nParis'

[PROMPT] Classify the sentiment as exactly one word, positive or negative: 'This product broke after two days.'
OUTPUT:  'We need to answer user\'s request: ... Need output exactly one word. Sentiment
          negative. Final: negative.\n</think>\n\nnegative'

peak VRAM(GiB)=[28.71, 0.0]
```

`Paris` and `negative` are the check that matters — a corrupted FP8 load emits fluent
nonsense instead.

## Single vs multi-GPU

`--device_map auto` shards the model across all visible GPUs via accelerate. Single-GPU and
2-GPU completions are byte-identical under greedy decoding, so shard only when one card is
too small; otherwise run one model per GPU.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B` | Model id or path. Use `Qwen/Qwen3.8-27B-FP8` for the reference run |
| `--prompts_file` | `None` | JSON list / JSONL / plain-text file of prompts; omit for the built-in set |
| `--system` | `None` | System message prepended to every prompt |
| `--dtype` | `auto` | `auto` honours the checkpoint dtype; or `bfloat16`/`float32`/`float16` |
| `--device_map` | `cuda:0` | `cuda:N` pins one GPU; `auto`/`balanced` shards across all visible GPUs |
| `--max_new_tokens` | `64` | Tokens to generate per prompt |
| `--do_sample` | off | Sample instead of greedy. **Greedy is reproducible — leave off for reference runs** |
| `--temperature` | `0.7` | Sampling temperature (only with `--do_sample`) |
| `--top_p` | `0.8` | Nucleus sampling p (only with `--do_sample`) |
| `--attn_impl` | `None` | Attention implementation; auto-selects `sdpa` on ROCm |
| `--dequantize_fp8` | off | Dequantize FP8 → bf16 at load. **Broken for this checkpoint — see FP8 status** |
| `--no_deepgemm` / `--allow_deepgemm` | on | Force the Triton FP8 kernel. Required on ROCm; disable on NVIDIA Hopper+ |
| `--fix_fp8_gate_proj` / `--no_fix_fp8_gate_proj` | on | Drop `mlp.gate*` skip-entries that corrupt `gate_proj`. **Required for correct output** |
| `--trust_remote_code` | on | Allow checkpoint-provided modeling code |
| `--seed` | `42` | Torch seed |
| `--output` | `None` | Path for the reference-output JSON artifact |
| `--hf_home` | `None` | `HF_HOME` model-cache override |

## Output

Printed to stdout: the fp8-fix line, a model/arch/quant/attention/device_map line, load
time with the full set of parameter dtypes and devices, per-GPU VRAM, then each prompt
with its completion, latency and tok/s, and finally peak VRAM.

`--output` writes the **reference artifact** other stacks are diffed against:

```
$OUTPUT_DIR/inference_llm_transformers/
  reference_llm_qwen3.8-27b-fp8_1gpu.json   <- canonical FP8 reference
  reference_llm_qwen3.8-27b-fp8_2gpu.json   <- byte-identical completions
  reference_llm_qwen3-4b_1gpu.json          <- BF16 control run
  reference_llm_lfm2.5-350m_1gpu.json       <- small-model harness smoke
```

Each JSON records model, architecture, quant method, attention implementation, device map,
per-parameter devices and dtypes, sampling settings, seed, torch/HIP versions, GPU arch, and
for every prompt the completion text and token counts. To compare another engine, run the
same prompts greedily and diff `completion` strings — with `do_sample=False` any divergence
is a real difference, not sampling noise.

## Hardware support

- **AMD MI355X (gfx950, ROCm 7.2)**: works, 1 and 2 GPUs, with FP8 (`float8_e4m3fn`)
  inference. Requires the two fixes above, both applied automatically.
- **NVIDIA H100 80GB (CUDA 13)**: works with `torch 2.13.0+cu130` and `--attn_impl sdpa`.
  Pass `--allow_deepgemm` for the faster FP8 path. The `--no_deepgemm` crash cannot trigger
  on CUDA, but the `gate_proj` corruption is vendor-independent — keep
  `--fix_fp8_gate_proj` on.

## Notes & quirks

- **`Qwen3.8-27B` is a reasoning model.** It emits a `<think>` monologue before the
  answer, terminated by `</think>`. Budget `--max_new_tokens` accordingly — at 8 tokens
  you will capture only reasoning and never reach the label. For strict prompted
  classification, parse after the last `</think>`. `Qwen/Qwen3-4B` behaves the same way.
- **`--dequantize_fp8` is broken for this checkpoint** — it drops nearly every projection
  weight and produces garbage. Documented above; left in place for other FP8 models.
- **`weight_scale_inv` UNEXPECTED in the load report is a red flag, not noise.** For a
  quantized checkpoint a dropped scale silently corrupts that layer. With the fix applied
  the load report is clean — check `gate_proj.weight` is `float8_e4m3fn`, not `bfloat16`.
- **`The fast path is not available`** at load is expected on ROCm: `flash-linear-attention`
  and `causal-conv1d` are CUDA-only, so Gated DeltaNet runs the torch fallback. Do not try
  to pip-install those on ROCm.
- **Shared HF cache permission warnings.** If `$HF_HOME` was populated by
  another user you may see `Ignoring corrupted tree cache file ... Permission denied`.
  Cache-metadata only; results unaffected.

## Recommended reference command

`--model Qwen/Qwen3.8-27B-FP8 --device_map cuda:0 --max_new_tokens 96`
(the checkpoint loads as `Qwen3_5ForConditionalGeneration` with `trust_remote_code=True`).
