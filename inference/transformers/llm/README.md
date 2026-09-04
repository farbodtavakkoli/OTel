# `infer_llm_transformers.py`

## Overview & when to use

**Reference LLM generation / prompted classification** for `Qwen/Qwen3.8-27B-FP8` on the
Transformers + PyTorch stack. This is the
**correctness baseline**: the text
this folder generates is what vLLM, SGLang and llama.cpp get diffed against. It is
optimised for *correct, reproducible output*, not throughput — if you want tokens/sec,
use vLLM.

Use it when you need to:

- Verify a checkpoint runs *correctly* on a new GPU/ROCm/driver combination before
  trusting a faster engine's output.
- Produce a golden generation artifact other engines are compared to.
- Do prompted classification (the recommended approach for this model — it is not
  a classification-head model, so you ask it to emit one allowed label).

The script loads the model, applies the chat template to each prompt, generates greedily
(deterministic by default), prints the completions with timings, and writes a JSON
reference artifact.

**Headline result: FP8 works natively on gfx950 — but only after fixing two real bugs.**
See *FP8 status* below. Both fixes are applied automatically by the script.

Design notes:

- **Architecture auto-resolve** — `load_model()` reads `config.architectures[0]` and looks
  the class up on `transformers`, so `Qwen3_5ForConditionalGeneration` is selected for the
  Qwen3.8 checkpoint and `AutoModelForCausalLM` covers everything else. One code path
  serves the FP8 model and the BF16 fallbacks.
- **Greedy by default** — `--do_sample` is off, so re-running the same command reproduces
  byte-identical text (verified: single-GPU and 2-GPU completions match exactly).
- **ROCm attention switch** — `sdpa` is selected when `torch.version.hip` is set;
  `flash-attn` is a CUDA-only build and must never be installed here.
- **Multimodal-aware tokenizer** — the Qwen3.8 checkpoint is natively multimodal, so
  `AutoTokenizer` can fail on it; `load_tokenizer()` falls back to
  `AutoProcessor(...).tokenizer`.

## FP8 status — it loads, and it works

**Verdict: FP8 loaded successfully. No fallback to BF16 was needed.** Falling back to a
BF16 checkpoint is sanctioned if FP8 is problematic; that turned out to be unnecessary
on gfx950, but getting there required diagnosing two genuine defects.

The checkpoint is blockwise FP8: `quant_method: fp8`, `fmt: e4m3`,
`activation_scheme: dynamic`, `weight_block_size: [128,128]`. Critically, **gfx950 uses
OCP E4M3FN, not FNUZ** — the same format the checkpoint stores — so the storage dtype was
never the problem. Weights load as `torch.float8_e4m3fn` with `float32`
`weight_scale_inv` blockwise scales.

### Bug 1 — the DeepGEMM probe crashes on ROCm

`transformers/integrations/finegrained_fp8.py` dispatches `128×128` blockwise FP8 to
DeepGEMM and wraps the probe in `except ImportError:`, intending to fall back to a Triton
kernel. On gfx950 all three gates misfire:

| check | ROCm/gfx950 result | consequence |
|---|---|---|
| `torch.cuda.is_available()` | `True` (ROCm maps onto `torch.cuda`) | passes |
| `torch.cuda.get_device_capability()[0]` | `9` (gfx950 reports 9.5) | passes the "Hopper SM90+" gate |
| `get_cuda_runtime_version()` | `ctypes.CDLL("libcudart.so")` → **`OSError`** | not an `ImportError` — **not caught** |

```
OSError: libcudart.so: cannot open shared object file: No such file or directory
```

gfx950 is uniquely bitten here: it is the ROCm part that *looks* like Hopper to that
check. Fix (`--no_deepgemm`, on by default) pre-marks DeepGEMM unavailable so the probe
raises `ImportError`, which *is* caught, and execution falls through to the Triton
`finegrained-fp8` kernel — which has a genuine `build/torch-rocm` variant on the Hub.

This also requires `kernels>=0.12.0,<0.13`. Without the package,
`lazy_load_kernel` returns `None` and you get
`AttributeError: 'NoneType' object has no attribute 'w8a8_fp8_matmul'`. With the *latest*
`kernels` (0.16.0) transformers 5.5.0 fails at import with
`ValueError: Either a revision or a version must be specified`. Respect the declared bound.

### Bug 2 — `mlp.gate` prefix-matches `mlp.gate_proj`, silently corrupting the model

With Bug 1 worked around, the model generated **fluent-looking garbage**:

```
'althocie不成这儿ardy…融合融合融合融合融合'
```

Root cause, confirmed by probing layer 0 of the loaded model:

| module | weight dtype | `weight_scale_inv` |
|---|---|---|
| `up_proj` | `torch.float8_e4m3fn` | present, `(136,40)` f32, mean 1.02e-4 |
| `down_proj` | `torch.float8_e4m3fn` | present, `(40,136)` f32, mean 1.46e-4 |
| **`gate_proj`** | **`torch.bfloat16`** | **MISSING** |

The checkpoint's `quantization_config.modules_to_not_convert` contains defensive MoE
entries named `...layers.N.mlp.gate`. Transformers matches that as a **prefix** of
`...layers.N.mlp.gate_proj`, so `gate_proj` is excluded from FP8 conversion, becomes a
plain bf16 `Linear`, and its scales are discarded (visible in the load report as
`model.layers.{0...63}.mlp.gate_proj.weight_scale_inv | UNEXPECTED` — the scales *are* in
the checkpoint; all 65 of them). Scales are ~1e-4, so `gate_proj` output lands ~4 orders
of magnitude off, the SwiGLU gate is meaningless, and the model emits noise.

This 27B is **dense** — layer-0 MLP keys are only `gate_proj`/`up_proj`/`down_proj`, with
no `mlp.gate.weight` and no experts — so those skip entries match nothing legitimate.
`--fix_fp8_gate_proj` (on by default) drops the 130 `.mlp.gate` / `.mlp.shared_expert_gate`
entries, after which `gate_proj` loads as `float8_e4m3fn` with its scales and the model
generates correctly.

> This is worth reporting upstream: the skip-list should match on module-path boundaries,
> not raw prefixes. Any dense checkpoint whose config carries defensive MoE names is
> exposed to the same silent corruption — and it is *silent*, which is the dangerous part.

### Controls that make the diagnosis airtight

- `Qwen/Qwen3-4B` (BF16, `Qwen3ForCausalLM`) in the **same venv, same script, same sdpa,
  same ROCm torch** generates perfectly coherent text at 11.8–57 tok/s. The environment
  was never at fault.
- `FineGrainedFP8Config(dequantize=True)` on the same checkpoint *also* produced garbage,
  with an even larger UNEXPECTED list (all `q/k/v/o_proj`, `gate/up/down_proj`,
  `linear_attn.in_proj_*`) — a second, independent key-mapping failure in the dequantize
  path. **Do not use `--dequantize_fp8` on this checkpoint**; it is kept as an option for
  other FP8 models but is broken for this architecture.

## Install

Python 3.12, in its own venv.

### AMD (ROCm) — the route that was verified

```bash
python3 -m venv .env_inference_llm_transformers
source .env_inference_llm_transformers/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r requirements_inference_llm_transformers.txt
```

**Pin `torchvision`.** A bare `pip install torchvision` resolves to the newest wheel and
silently upgrades torch to `2.13.0+rocm7.2`, breaking the `2.11.0` pin this repo
standardises on. `torchvision` and `pillow` are needed because the Qwen3.8 checkpoint is
multimodal and `AutoProcessor` imports the image processor even for text-only use.

Do **not** install `flash-attn`; it is a CUDA build and the script selects `sdpa`.

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# True 7.2.26015
```

### NVIDIA (CUDA)

Default PyPI wheels are CUDA-enabled:

```bash
pip install -r requirements_inference_llm_transformers.txt   # omit the ROCm index-url step
```

On NVIDIA you can pass `--allow_deepgemm` to re-enable the (3–6× faster) DeepGEMM FP8
path, which needs Hopper SM90+ and CUDA runtime 12.3+.

## Environment & secrets

`dev.env` is symlinked to the repo-root `dev.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Loaded via `load_dotenv("dev.env")`. Weights are ~30 GB, so keep the cache off `/`:

```bash
export HF_HOME=/mnt/data_1.5t/hf_cache
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5
```

Never set `CUDA_VISIBLE_DEVICES=""` on ROCm — use explicit indices. `--device_map` is
indexed **within** the visible set, so `cuda:0` is physical card 4 above.

## Run

Single GPU (the 30 GB FP8 checkpoint fits on one 288 GB MI355X with room to spare):

```bash
source .env_inference_llm_transformers/bin/activate
export HIP_VISIBLE_DEVICES=4,5 CUDA_VISIBLE_DEVICES=4,5 HF_HOME=/mnt/data_1.5t/hf_cache

python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map cuda:0 \
  --max_new_tokens 96 \
  --output /mnt/data_1.5t/outputs/inference_llm_transformers/reference_llm_qwen3.8-27b-fp8_1gpu.json
```

Multi-GPU, sharded across both cards:

```bash
python infer_llm_transformers.py --model Qwen/Qwen3.8-27B-FP8 --device_map auto \
  --max_new_tokens 96 \
  --output /mnt/data_1.5t/outputs/inference_llm_transformers/reference_llm_qwen3.8-27b-fp8_2gpu.json
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

## Single-GPU results

`Qwen/Qwen3.8-27B-FP8 --device_map cuda:0` (1× MI355X, physical card 4), 2026-08-20:

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

`rocm-smi` confirmed **card4=29717 MiB, card5=284 MiB** — the whole model on one card.
Both answers are correct (`Paris`, `negative`), which is the real result: this is the same
model that emitted `融合融合融合` before the gate_proj fix.

| Metric | value |
|---|---|
| Model-load VRAM (torch allocated) | 28.38 GiB |
| Peak VRAM | 28.71 GiB |
| `rocm-smi` card 4 | 29717 MiB |
| Load time | 31.3 s |
| Decode | 1.18 / 3.38 / 10.45 tok/s (prompts 1/2/3) |

## H100 results (NVIDIA, CUDA 13 — verified 2026-08-22)

**Model swap: `LiquidAI/LFM2.5-350M`, not `Qwen/Qwen3.8-27B-FP8`.** The FP8 27B checkpoint
is *not* present in this H100 node's offline cache (only a bare Qwen3 tokenizer dir is), and
Hub access is 403-blocked. `LiquidAI/LFM2.5-350M` is fully cached with a chat template, so it
is the H100 harness-correctness model. This means the two FP8 fixes (`--no_deepgemm`,
`--fix_fp8_gate_proj`) are **not exercised on H100** — LFM2.5-350M is a dense bf16
`Lfm2ForCausalLM`, `quant=None`. Bug 2 (`gate_proj`) remains vendor-independent and would
still need `--fix_fp8_gate_proj` for the FP8 27B on H100; that is inherited from the MI355X
run, not re-verified here.

Install (shared stack venv — see the stack-root README):

```bash
cd inference/transformers
python3 -m venv .env_transformers && source .env_transformers/bin/activate
pip install torch numpy            # -> torch 2.13.0+cu130 (CUDA 13.0)
pip install -r requirements.txt
```

Key versions: `torch 2.13.0+cu130`, `transformers 5.5.0`, `accelerate 1.14.0`,
`kernels 0.12.3`, driver 580.173.02, Python 3.12.3. `flash-attn` has no prebuilt cu130 wheel
(source build exceeded the time budget) → ran `--attn_impl sdpa`. On CUDA the script would
otherwise auto-select `eager` (`torch.version.hip` is `None`); pass `sdpa` explicitly.

Exact smoke command (physical GPU 5):

```bash
export HF_HOME=/mnt/gsma/gsma/gsma/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=5
python infer_llm_transformers.py --model LiquidAI/LFM2.5-350M --device_map cuda:0 \
  --attn_impl sdpa --max_new_tokens 96 \
  --output /dev/shm/h100/out/transformers/llm/reference_llm_lfm2.5-350m_1gpu.json
```

Real output (2026-08-22):

```
model=LiquidAI/LFM2.5-350M arch=Lfm2ForCausalLM quant=None attn=sdpa device_map=cuda:0
load: 13.6s | param dtypes=['torch.bfloat16'] | devices=['cuda:0']
VRAM(GiB)=[0.66]

[1.6s, 1.28 tok/s] PROMPT: What is the capital of France? Answer in one word.
OUTPUT: 'Paris'
[0.2s, 9.05 tok/s] PROMPT: Classify the sentiment ... 'This product broke after two days.'
OUTPUT: 'negative'

peak VRAM(GiB)=[0.69]
```

GPU-5 residency, sampled by PID from inside the run (`nvidia-smi --query-compute-apps` on `-i 5`):

```
[smi] GPU5 pid=1444042  1196 MiB
[smi] GPU5 pid=1444042  1198 MiB
```

**Verdict (H100): WORKS** for harness correctness. `Paris` and `negative` are both correct
(prompt 3 is a small-350M-model hallucination — real ROCm engines are not listed — and is
not a harness fault). The `gcnArchName` field reads `NVIDIA H100 80GB HBM3` and `hip=None` in
the artifact, confirming the CUDA build. tf32: these inference paths run bf16, so the fp32
tf32 guard is not on the hot path (`torch.backends.cuda.matmul.allow_tf32` stays `False` by
default in this torch; irrelevant to a bf16 generate). The **FP8 27B reference remains
MI355X-only** on this node until the checkpoint is cached. Multi-GPU deferred (production job
on GPUs 0–3).

## Multi-GPU results

`--device_map auto` shards the model across both visible GPUs via accelerate. Verified
2026-08-20:

```
model=Qwen/Qwen3.8-27B-FP8 arch=Qwen3_5ForConditionalGeneration quant=fp8 attn=sdpa device_map=auto
load: 32.3s | param dtypes=['torch.bfloat16', 'torch.float32', 'torch.float8_e4m3fn']
             | devices=['cuda:0', 'cuda:1']
VRAM(GiB)=[12.17, 16.28]
peak VRAM(GiB)=[12.46, 16.37]
```

`rocm-smi` sampled every 5 s, physical cards 4 and 5 — **weights resident on BOTH**:

```
card4=284MiB    card5=284MiB     t=30s   (idle, before load)
card4=13355MiB  card5=17545MiB   t=35s   (sharded load complete — both cards)
card4=15721MiB  card5=17896MiB   t=50s   (generating)
card4=16404MiB  card5=19015MiB   t=80s   (generating)
```

**Correctness: single-GPU and 2-GPU completions are byte-identical.** All three prompts
produced exactly the same text in both configurations — greedy decoding plus a
deterministic shard boundary means device placement does not perturb the output at all.
That is the strongest possible result for a correctness baseline.

| | 1 GPU (`cuda:0`) | 2 GPUs (`auto`) |
|---|---|---|
| VRAM | 28.38 GiB on card 4 | 12.17 + 16.28 GiB |
| Load time | 31.3 s | 32.3 s |
| Decode (prompt 1 / 2 / 3) | 1.18 / 3.38 / 10.45 tok/s | 0.98 / 3.19 / 10.51 tok/s |
| Completions | — | **byte-identical to 1 GPU** |

**Multi-GPU buys nothing here, and that is the honest answer.** The model fits in 28 GB on
a single 288 GB card, so `device_map="auto"` is pure pipeline parallelism: layers are
split, only one card computes at a time, and there is cross-device transfer per token.
Throughput is fractionally *worse*. Shard this model across GPUs only if your cards are
small; on MI355X, run one model per GPU instead.

**The tok/s numbers are low and expected to be.** Two multipliers are missing: the Triton
FP8 kernel is the fallback path (DeepGEMM is 3–6× faster but CUDA-only), and the hybrid
Gated DeltaNet layers run the pure-torch fallback because `flash-linear-attention` and
`causal-conv1d` are CUDA-only builds — transformers logs `The fast path is not available`
at load. 48 of 64 layers are `linear_attention`, so that fallback dominates. This folder
is the correctness reference; use vLLM (which merged Qwen3.8 ROCm support
in PR #50068) for throughput.

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
/mnt/data_1.5t/outputs/inference_llm_transformers/
  reference_llm_qwen3.8-27b-fp8_1gpu.json   <- canonical FP8 reference
  reference_llm_qwen3.8-27b-fp8_2gpu.json   <- byte-identical completions
  reference_llm_qwen3-4b_1gpu.json          <- BF16 control run
```

Each JSON records model, architecture, quant method, attention implementation, device map,
per-parameter devices and dtypes, sampling settings, seed, torch/HIP versions, GPU arch,
load time, per-GPU VRAM, and for every prompt the completion text, token counts, latency
and tok/s. To compare another engine, run the same prompts greedily and diff `completion`
strings — with `do_sample=False` any divergence is a real difference, not sampling noise.

## Hardware support & evidence

- **AMD: tested.** 1× and 2× MI355X (gfx950, 288 GB), ROCm 7.2.4, Python 3.12.3,
  `torch==2.11.0+rocm7.2`, `torchvision==0.26.0+rocm7.2`, `transformers==5.5.0`,
  `accelerate==1.14.0`, `kernels==0.12.3`. **FP8 (`float8_e4m3fn`) inference verified
  working** on one card and sharded across two, with `rocm-smi` confirming residency, and
  byte-identical completions between the two configurations. Requires the two fixes above,
  both applied automatically.
- **NVIDIA H100: tested 2026-08-22 with `LiquidAI/LFM2.5-350M`** (the FP8 27B is not in this
  node's cache), `torch 2.13.0+cu130`, CUDA 13.0, driver 580.173.02, attn `sdpa` — see "H100
  results" above. Correct `Paris`/`negative` output, GPU-5 residency ~1.2 GiB. **The FP8 path
  and its two fixes were NOT exercised on H100** (LFM2.5-350M is dense bf16, `quant=None`). The
  script is vendor-neutral; on CUDA with the FP8 27B pass `--allow_deepgemm` for the faster FP8
  path. Both bugs above are ROCm- or checkpoint-specific — Bug 1 cannot trigger on CUDA, but
  **Bug 2 (`gate_proj`) is vendor-independent** and would still corrupt the FP8 checkpoint on
  NVIDIA (inherited from MI355X, not re-verified on H100).
- **The FP8 path does work.** It is often treated as a compatibility/debug route, with a
  BF16 fallback recommended for this checkpoint; that fallback proved unnecessary here — the
  caution is well-earned, but the path works.

## Notes & quirks

- **`Qwen3.8-27B` is a reasoning model.** It emits a `<think>` monologue before the
  answer, terminated by `</think>`. Budget `--max_new_tokens` accordingly — at 8 tokens
  you will capture only reasoning and never reach the label. For strict prompted
  classification, parse after the last `</think>`. `Qwen/Qwen3-4B` behaves the same way.
- **`--dequantize_fp8` is broken for this checkpoint** — it drops nearly every projection
  weight and produces garbage. Documented above; left in place for other FP8 models.
- **`weight_scale_inv` UNEXPECTED in the load report is a red flag, not noise.** The report
  calls it ignorable ("can be ignored when loading from different task/architecture"). For
  a quantized checkpoint it is not — a dropped scale silently corrupts that layer. With the
  fix applied the load report is clean.
- **`The fast path is not available`** at load is expected on ROCm: `flash-linear-attention`
  and `causal-conv1d` are CUDA-only, so Gated DeltaNet runs the torch fallback. Correct,
  just slow. Do not try to pip-install those on ROCm.
- **Checkpoint layout is unusual** — 81 files as `layers-N.safetensors` plus
  `outside.safetensors` and `mtp.safetensors` (an MTP speculative-decoding head), ~29 GB.
- **Shared HF cache permission warnings.** If `/mnt/data_1.5t/hf_cache` was populated by
  another user you may see `Ignoring corrupted tree cache file ... Permission denied`.
  Cache-metadata only; results unaffected.

## Verdict

**WORKS — FP8 loaded natively, no BF16 fallback needed — but only with two fixes, one of
which prevents silent corruption.**

`Qwen/Qwen3.8-27B-FP8` runs correctly on MI355X / gfx950 / ROCm 7.2.4 through stock
Transformers + PyTorch, as `Qwen3_5ForConditionalGeneration` with `trust_remote_code=True`
exactly as expected. Weights stay in `torch.float8_e4m3fn` — gfx950's OCP
E4M3FN matches the checkpoint format, so the documented FP8 caveat did not bite
on the storage side. It ran on one GPU (28.4 GiB) and sharded across two
(12.2 + 16.3 GiB), producing **byte-identical, correct output** in both.

The two fixes are not optional:

1. `kernels>=0.12,<0.13` + `--no_deepgemm` — otherwise generation **crashes**
   (`OSError: libcudart.so`), because gfx950 reports capability 9.5 and slips past the
   Hopper gate into a CUDA-only code path.
2. `--fix_fp8_gate_proj` — otherwise generation **silently produces fluent garbage**,
   because `mlp.gate` prefix-matches `mlp.gate_proj` and strips its FP8 scales.

The second is the one that matters. A crash is honest; a model that loads cleanly, reports
no error, and emits confident nonsense is exactly the failure a correctness baseline exists
to catch. Anyone benchmarking this checkpoint on any vendor should check that
`gate_proj.weight` is `float8_e4m3fn` and not `bfloat16` before trusting a single number.

Throughput (1–10 tok/s) is poor and deliberately not the point — the Triton FP8 fallback
and the pure-torch Gated DeltaNet path both cost heavily on ROCm. Use vLLM for
performance and this folder for ground truth.

Recommended reference command:
`--model Qwen/Qwen3.8-27B-FP8 --device_map cuda:0 --max_new_tokens 96`.
