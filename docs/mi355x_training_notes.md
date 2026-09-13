# MI355X (AMD / ROCm) training notes

Cross-cutting lessons and multi-GPU traps for the training folders on AMD Instinct MI355X
(gfx950, 288 GB HBM), ROCm 7.2.4, Python 3.12. The
per-folder support table lives in the top-level [README](../README.md). Folder names refer
to the current layout (`training/<modality>/<framework>`).

## Lessons that apply to most folders

- **torch**: `2.11.0+rocm7.2` from `https://download.pytorch.org/whl/rocm7.2` satisfies a
  plain `torch==2.11.0` pin and installs clean. Always install it **before** the framework,
  and re-check afterwards — axolotl, rapidfire, composer and llmfoundry all replace it with
  a CUDA wheel. Recovery: `pip install --force-reinstall --no-deps torch==<v> --index-url …/rocm7.2`.
- **flash-attn does not exist on ROCm.** Every hardcoded `attn_implementation="flash_attention_2"`
  must fall back to `sdpa`. No folder needed a real FA2 kernel to train.
- **`tf32=True` raises on ROCm** (`--tf32 requires Ampere`) — guard it with `torch.version.cuda is not None`.
- **CUDA triton silently shadows ROCm triton.** If a dep pulls `triton`, evict it and
  restore `triton-rocm` (the exact version torch wants) or `import torch` breaks.
- **`CUDA_VISIBLE_DEVICES=""` hides all GPUs on ROCm**; scope with `HIP_VISIBLE_DEVICES`.
- Unrelated to AMD but it affects six folders: transformers 5.x `apply_chat_template(tokenize=True)`
  returns a `BatchEncoding`, so prompt-masking diffs silently drop **every** row — pass
  `return_dict=False`.

## Scaling to 8 GPUs

Single-GPU smoke runs prove a stack imports and steps; they do not prove *sharding*,
collectives, or that the batch geometry survives. Take a whole-node lock before an 8-GPU
run and sample `rocm-smi` *inside* the job, attributing by PID.

**All 23 trainable folders work at world size 8**, with one qualification:
`llm/openrlhf` counts here on its **SFT/DPO** tiers, which do scale to 8 — its
PPO/GRPO tiers stay dependency-blocked in the venv and are only validated in the
`rocm/verl` container at **4 GPUs, not 8**. So the strict reading is *22 folders fully at
world size 8, plus openrlhf partially*. (`llm/scalarlm` is excluded by design — it is an
HTTP client with no local training loop.) Commands live in each folder's own README.

| Folder | At 8 GPUs | Parallelism exercised | Notes |
|---|---|---|---|
| `llm/deepspeed` | **works, unmodified** | ZeRO-2, world 8 | SFT + DPO + GRPO all passed at 8 ranks |
| `llm/deepspeed_standalone` | **works** | ZeRO-3 full FT, world 8 | the "stage 3" banner never prints (logger reset) — introspect the live engine instead of trusting the log |
| `llm/fsdp` | **works, unmodified** | FSDP2 full shard | — |
| `llm/megatron` | **works, unmodified** | TP + PP + DP, 3 layouts | TP2/PP2/DP2 exercises every process-group type |
| `llm/torchtitan` | **works, unmodified** | FSDP2 + TP, 2 layouts | — |
| `llm/primus` | **works (DP8 only)** | DP 8 | ⚠️ TP/PP **blocked** — the image's checkpoint converter only ever emits TP1/PP1, so re-sharding is rejected at resume. LoRA still broken |
| `llm/verl` | **works, unmodified** | FSDP2 world 8 **colocated** with 4 vLLM engines × TP2 | the default topology already targets 8 GPUs; trainer and rollout share the same cards rather than splitting them |
| `classification/deepspeed` | **works, unmodified** | ZeRO-2 | use ZeRO-2, not ZeRO-3 — see the silent no-op save below |
| `llm/ray` | **works, unmodified** | 8 Ray workers on 8 distinct GPUs | `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` **still required** at 8 workers |
| `llm/composer` | **works, unmodified** | DDP **and** FSDP `FULL_SHARD` | only `-n 8` and batch geometry changed |
| `llm/llmfoundry` | **works** | FSDP `FULL_SHARD` | config only — geometry divisible by 8, `init_device: mixed` put back |
| `llm/nemo` | **works** | FSDP2, `dp_size 8` | ⚠️ the folder's CLI cannot change torchrun's rendezvous port (inherits the 29500 default) — collides on a shared box; use upstream's in-process launcher. TP/PP/CP meshes still untested |
| `llm/unsloth` | **works** | DDP 8, no sharding (Unsloth OSS is DDP-only) | ⚠️ the package still ships a `device_count() > 1` refusal; it fails to land only because trl 0.24.0 makes Unsloth's patch path bail out — **version luck, not a supported guarantee**. QLoRA and GRPO were not retested at 8 GPUs |
| `llm/axolotl` | **works, with changes** | plain DDP — no DeepSpeed, no FSDP | 8-GPU YAML + replicated dataset + pin override; no new packages, so the upstream "DeepSpeed is broken on ROCm" caveat never applies to LoRA |
| `llm/llamafactory` | **works, with changes** | DDP 8 (the CLI re-execs itself under `torchrun`) | environment and data plumbing only, no framework fix |
| `llm/lightning` | **works, with changes** | FSDP `FULL_SHARD`, and DDP | `--grad_clip 0` still required under FSDP; DDP needs `ddp_find_unused_parameters_true` |
| `llm/torchtune` | **works, with changes** | FSDP2 | needs a *separate* distributed config (recipes come in pairs, each owning its config surface) + a bigger dataset. Nothing AMD-specific |
| `llm/redhat` | **works, with changes** | world 8 | pin override + a dataset large enough to feed 8 ranks |
| `llm/peft` | **works, with one change** | DDP 8 — LoRA, QLoRA, DoRA | `--ddp_find_unused_parameters` (see multimodal note below). ⚠️ QLoRA skews VRAM heavily onto GPU 0 |
| `llm/rapidfire` | **works** | *config-level*: 8 configs, one per GPU | ⚠️ must **unset** `HIP_/CUDA_VISIBLE_DEVICES` — the opposite of every other folder (see below) |
| `embedding/sentence_transformers` | **works** | DDP + cross-rank negative gather | `gather_across_devices` grows the in-batch negative pool, which is the real reason to scale this one |
| `reranker/sentence_transformers` | **works** | DDP 8 | — |
| `llm/openrlhf` | **SFT/DPO work**; RL tiers unchanged | DeepSpeed, world 8 | batch geometry was the only change. PPO/GRPO remain blocked in the venv — a *dependency* limit (vLLM), not a scale one; they run in the `rocm/verl` container |

### What breaks when you go from 1-2 GPUs to 8

These are the failure modes that only appear at scale, and most of them **exit 0**:

- **Batch geometry fails silently.** `global = per_device
  × 8 × grad_accum`. Point an 8-rank run at a 10-100 row sample and it can perform *one*
  optimizer step (`reranker/sentence_transformers`) or *zero*
  (`embedding/sentence_transformers` at its registry default of 96 → a 768-row global batch
  against 90 training rows) while still reporting a plausible loss and exiting 0. The
  better-behaved folders assert instead — OpenRLHF wants
  `global // micro // world == 0`, veRL wants `train_batch × rollout.n % world == 0` — and
  those abort *before* any GPU work. **Always compute steps/epoch and check it is
  non-trivial before believing an 8-GPU pass.**
- **Leftover GPU pins in venvs.** A single-GPU session that appends
  `export HIP_VISIBLE_DEVICES=<n>` to a venv's `bin/activate` silently caps a later
  nominal "8-GPU" run to one card. Override both variables **after** sourcing and assert
  `torch.cuda.device_count() == 8`. If a venv may carry such a pin, rebuild it from the
  folder's requirements file.
- **`HIP_VISIBLE_DEVICES` outranks `CUDA_VISIBLE_DEVICES` on ROCm.** Measured directly:
  `HIP=0,1` plus `CUDA=3` yields 2 visible devices — the CUDA variable is ignored. Any
  framework that pins *its own* workers via `CUDA_VISIBLE_DEVICES` is therefore collapsed
  onto GPU 0 by the usual "export both to 0-7" advice, and the run still exits 0.
  RapidFire AI does exactly this (one spawned worker per GPU), so it needs both variables
  **unset** — the same class of fix as Ray's `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`.
- **Multimodal checkpoints break DDP but are fine on one GPU.** `gemma-4-E4B-it` adapts 379
  LM linears plus 114 vision and 135 audio; a text-only batch never runs the towers, so 249
  adapters receive no gradient and DDP raises *"Expected to have finished reduction in the
  prior iteration"*. Invisible at 1 GPU because there is no DDP wrapper at all. Fix:
  `find_unused_parameters`.
- **The shipped 10-row samples cannot feed 8 ranks.** Replicating them is a fine *pipeline*
  proof but it is not a learning result — duplicates inside one global batch turn
  contrastive "negatives" into copies of the positive, and embedding quality *fell* on a
  ×32-replicated set. Keep generated data out of the repo.
- **Disabling checkpoints is hard.** `save_strategy: "no"` is not enough:
  HF Trainer still writes at the end of `do_train`, `load_best_model_at_end=True` forces a
  save, Composer/Foundry write a fit-end checkpoint regardless, and RapidFire persists
  per-run adapters because that *is* its config-swap mechanism. At 8 ranks the footprint
  multiplies, and FSDP2's consolidated save is fp32 (~32 GiB for an 8B model).
- **Measuring utilization is its own trap.** An idle MI355X reports ~1% util and 0.28 GB
  VRAM, so a "busy = `>0%`" threshold will certify 8 working GPUs while your job is
  crashing. And a short run can exit before an externally-sampled `rocm-smi` even lands,
  capturing *another* job on a shared box. Sample from inside the job and attribute by PID.

### 4-GPU sharding probes — the collective *save*

Four paths were probed at 4 GPUs specifically to exercise the collective *save*, which is
where multi-GPU tends to break quietly:

| Path | Result |
|---|---|
| `llm/deepspeed` ZeRO-3, full FT (no LoRA), 4 GPUs | **holds** — the unguarded all-ranks `save_model` is the correct pattern (rank-0-guarding it is what deadlocked `deepspeed_standalone`) |
| `llm/fsdp` FSDP2, collective save enabled, 4 ranks | **holds** — but the consolidated save is **fp32**, so budget disk for it. During the save ranks 1-3 sit at 100% busy-waiting in the barrier — easy to misread as a hang |
| `llm/lightning` FSDP `FULL_SHARD`, 4 devices | **holds** — rank-0 gather produced a valid unsharded checkpoint |
| `classification/deepspeed` ZeRO-3, 4 GPUs | ⚠️ **silent no-op save**. ZeRO-2 on the same GPUs wrote weights correctly |

The classification ZeRO-3 bug is a *config* bug reachable only once parameters are
sharded — its `build_deepspeed_config()` never sets
`stage3_gather_16bit_weights_on_model_save`, so the HF save no-ops while still exiting 0.
No single-GPU run could have caught it. `llm/lightning` logs no loss
to stdout and its progress bar is swallowed when stdout is piped, so a healthy run looks
identical to a hang — read `train_loss` from the tfevents file to confirm liveness.

## Which route each folder needs on ROCm

Per-folder install detail lives in each folder's own README; the short version:

- **Plain ROCm torch + pip works** for deepspeed (+standalone), fsdp, peft (incl. QLoRA via
  the ROCm bitsandbytes wheels), lightning, axolotl, llamafactory, torchtune (the **stable**
  wheel, not nightly), megatron (no TE/Apex needed), nemo, ray (one env var), redhat, unsloth
  (`unsloth[amd]`), and the classification/embedding/reranker folders.
- **Needs the torch pin overridden** after install: composer, llmfoundry, rapidfire —
  restore the ROCm wheel with
  `pip install --force-reinstall --no-deps torch==<v> --index-url …/rocm7.2`.
- **Needs ROCm nightly torch**: torchtitan.
- **Container route**: verl and openrlhf's RL tiers (`rocm/verl`); primus (SFT works, LoRA is
  broken in the image).
