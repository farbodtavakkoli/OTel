# MI355X (AMD / ROCm) training notes

Lessons and multi-GPU traps for the training recipes on AMD Instinct MI355X (gfx950,
288 GB HBM), ROCm 7.2.4, Python 3.12. The per-folder support table is in the top-level
[README](../README.md).

## Lessons that apply to most folders

- **Install ROCm torch first, then re-check it.** `2.11.0+rocm7.2` from
  `https://download.pytorch.org/whl/rocm7.2` satisfies a plain `torch==2.11.0` pin and
  installs clean. Install it **before** the framework — axolotl, rapidfire, composer, and
  llmfoundry all replace it with a CUDA wheel. To recover:

  ```bash
  pip install --force-reinstall --no-deps torch==<v> --index-url https://download.pytorch.org/whl/rocm7.2
  ```
- **flash-attn does not exist on ROCm.** Every hard-coded
  `attn_implementation="flash_attention_2"` has to fall back to `sdpa`. No folder needed a
  real FA2 kernel to train.
- **`tf32=True` raises on ROCm** (`--tf32 requires Ampere`). Guard it with
  `torch.version.cuda is not None`.
- **CUDA triton silently shadows ROCm triton.** If a dependency pulls `triton`, remove it and
  restore `triton-rocm` at the exact version torch wants, or `import torch` breaks.
- **`CUDA_VISIBLE_DEVICES=""` hides all GPUs on ROCm.** Scope with `HIP_VISIBLE_DEVICES`.
- **Not AMD-specific, but it affects six folders:** transformers 5.x
  `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, so prompt-masking diffs
  silently drop every row. Pass `return_dict=False`.

## Scaling to 8 GPUs

A single-GPU smoke run proves a stack imports and steps. It does not prove sharding,
collectives, or that the batch geometry survives. Take a whole-node lock before an 8-GPU run,
and sample `rocm-smi` from *inside* the job, attributing by PID.

**All 23 trainable folders work at world size 8**, with one qualification: `llm/openrlhf`
counts on its SFT/DPO tiers only. Its PPO/GRPO tiers are dependency-blocked in the venv and
run in the `rocm/verl` container at 4 GPUs. So the strict reading is 22 folders fully at
world size 8, plus openrlhf partially. (`llm/scalarlm` is excluded by design — it is an HTTP
client with no local training loop.) Launch commands live in each folder's README.

| Folder | At 8 GPUs | Parallelism exercised | Notes |
|---|---|---|---|
| `llm/deepspeed` | **works, unmodified** | ZeRO-2, world 8 | SFT + DPO + GRPO all passed at 8 ranks |
| `llm/deepspeed_standalone` | **works** | ZeRO-3 full FT, world 8 | the "stage 3" banner never prints (logger reset) — introspect the live engine instead of trusting the log |
| `llm/fsdp` | **works, unmodified** | FSDP2 full shard | — |
| `llm/megatron` | **works, unmodified** | TP + PP + DP, 3 layouts | TP2/PP2/DP2 exercises every process-group type |
| `llm/torchtitan` | **works, unmodified** | FSDP2 + TP, 2 layouts | — |
| `llm/primus` | **works (DP8 only)** | DP 8 | TP/PP **blocked** — the image's checkpoint converter only ever emits TP1/PP1, so re-sharding is rejected at resume. LoRA still broken |
| `llm/verl` | **works, unmodified** | FSDP2 world 8 **colocated** with 4 vLLM engines × TP2 | the default topology already targets 8 GPUs; trainer and rollout share the same cards rather than splitting them |
| `classification/deepspeed` | **works, unmodified** | ZeRO-2 | use ZeRO-2, not ZeRO-3 — see the silent no-op save below |
| `llm/ray` | **works, unmodified** | 8 Ray workers on 8 distinct GPUs | `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` **still required** at 8 workers |
| `llm/composer` | **works, unmodified** | DDP **and** FSDP `FULL_SHARD` | only `-n 8` and batch geometry changed |
| `llm/llmfoundry` | **works** | FSDP `FULL_SHARD` | config only — geometry divisible by 8, `init_device: mixed` put back |
| `llm/nemo` | **works** | FSDP2, `dp_size 8` | the folder's CLI cannot change torchrun's rendezvous port (inherits the 29500 default) — collides on a shared box; use upstream's in-process launcher. TP/PP/CP meshes still untested |
| `llm/unsloth` | **works** | DDP 8, no sharding (Unsloth OSS is DDP-only) | the package still ships a `device_count() > 1` refusal; it fails to land only because trl 0.24.0 makes Unsloth's patch path bail out — **version luck, not a supported guarantee**. QLoRA and GRPO were not retested at 8 GPUs |
| `llm/axolotl` | **works, with changes** | plain DDP — no DeepSpeed, no FSDP | 8-GPU YAML + replicated dataset + pin override; no new packages, so the upstream "DeepSpeed is broken on ROCm" caveat never applies to LoRA |
| `llm/llamafactory` | **works, with changes** | DDP 8 (the CLI re-execs itself under `torchrun`) | environment and data plumbing only, no framework fix |
| `llm/lightning` | **works, with changes** | FSDP `FULL_SHARD`, and DDP | `--grad_clip 0` still required under FSDP; DDP needs `ddp_find_unused_parameters_true` |
| `llm/torchtune` | **works, with changes** | FSDP2 | needs a *separate* distributed config (recipes come in pairs, each owning its config surface) + a bigger dataset. Nothing AMD-specific |
| `llm/redhat` | **works, with changes** | world 8 | pin override + a dataset large enough to feed 8 ranks |
| `llm/peft` | **works, with one change** | DDP 8 — LoRA, QLoRA, DoRA | `--ddp_find_unused_parameters` (see multimodal note below). QLoRA skews VRAM heavily onto GPU 0 |
| `llm/rapidfire` | **works** | *config-level*: 8 configs, one per GPU | must **unset** `HIP_/CUDA_VISIBLE_DEVICES` — the opposite of every other folder (see below) |
| `embedding/sentence_transformers` | **works** | DDP + cross-rank negative gather | `gather_across_devices` grows the in-batch negative pool, which is the real reason to scale this one |
| `reranker/sentence_transformers` | **works** | DDP 8 | — |
| `llm/openrlhf` | **SFT/DPO work**; RL tiers unchanged | DeepSpeed, world 8 | batch geometry was the only change. PPO/GRPO remain blocked in the venv — a *dependency* limit (vLLM), not a scale one; they run in the `rocm/verl` container |

### What breaks when you go from 1-2 GPUs to 8

These failure modes only appear at scale, and most of them **exit 0**.

- **Batch geometry fails silently.** `global = per_device × 8 × grad_accum`. Point an 8-rank
  run at a 10-100 row sample and it can take one optimizer step, or zero, while still
  reporting a plausible loss and exiting 0. `embedding/sentence_transformers` at its default
  batch of 96 gives a 768-row global batch against 90 training rows — zero steps. Better
  behaved folders assert instead and abort before any GPU work. **Compute steps/epoch and
  confirm it is non-trivial before believing an 8-GPU pass.**
- **Leftover GPU pins in venvs.** A single-GPU session that appended
  `export HIP_VISIBLE_DEVICES=<n>` to a venv's `bin/activate` silently caps a later "8-GPU"
  run to one card. Export the variables **after** sourcing, and assert
  `torch.cuda.device_count() == 8`. If a venv may carry such a pin, rebuild it.
- **`HIP_VISIBLE_DEVICES` outranks `CUDA_VISIBLE_DEVICES` on ROCm.** Measured directly:
  `HIP=0,1` plus `CUDA=3` yields two visible devices — the CUDA variable is ignored. So any
  framework that pins its own workers with `CUDA_VISIBLE_DEVICES` gets collapsed onto GPU 0
  by the usual "export both to 0-7" advice, and still exits 0. RapidFire AI does exactly
  this, so it needs both variables **unset**.
- **Multimodal checkpoints break DDP but are fine on one GPU.** `gemma-4-E4B-it` adapts 379
  LM linears plus 114 vision and 135 audio. A text-only batch never runs the towers, so 249
  adapters get no gradient and DDP raises *"Expected to have finished reduction in the prior
  iteration"*. Invisible at 1 GPU because there is no DDP wrapper. Fix with
  `find_unused_parameters`.
- **The shipped 10-row samples cannot feed 8 ranks.** Replicating them proves the pipeline
  but is not a learning result — duplicates inside one global batch turn contrastive
  negatives into copies of the positive, and embedding quality *fell* on a 32x-replicated
  set. Keep generated data out of the repo.
- **Disabling checkpoints is hard.** `save_strategy: "no"` is not enough: HF Trainer still
  writes at the end of `do_train`, `load_best_model_at_end=True` forces a save,
  Composer and LLM Foundry write a fit-end checkpoint regardless, and RapidFire persists
  per-run adapters because that is its config-swap mechanism. At 8 ranks the footprint
  multiplies, and FSDP2's consolidated save is fp32 (~32 GiB for an 8B model).
- **Measuring utilization is its own trap.** An idle MI355X reports ~1% utilization and
  0.28 GB VRAM, so a "busy means >0%" threshold will certify 8 working GPUs while the job is
  crashing. A short run can also exit before an external `rocm-smi` sample lands, capturing
  another job on a shared box. Sample from inside the job and attribute by PID.

### 4-GPU probes of the collective save

The collective *save* is where multi-GPU tends to break quietly, so four paths were probed at
4 GPUs specifically to exercise it.

| Path | Result |
|---|---|
| `llm/deepspeed` ZeRO-3, full FT, 4 GPUs | **Holds.** The unguarded all-ranks `save_model` is the correct pattern — rank-0-guarding it is what deadlocked `deepspeed_standalone` |
| `llm/fsdp` FSDP2, collective save, 4 ranks | **Holds**, but the consolidated save is fp32, so budget disk. Ranks 1-3 sit at 100% busy-waiting in the barrier during the save — easy to misread as a hang |
| `llm/lightning` FSDP `FULL_SHARD`, 4 devices | **Holds.** Rank-0 gather produced a valid unsharded checkpoint |
| `classification/deepspeed` ZeRO-3, 4 GPUs | **Silent no-op save.** ZeRO-2 on the same GPUs wrote weights correctly |

The classification ZeRO-3 bug is a config bug reachable only once parameters are sharded:
`build_deepspeed_config()` never sets `stage3_gather_16bit_weights_on_model_save`, so the
save no-ops while the run exits 0. No single-GPU run could have caught it.

Separately, `llm/lightning` logs no loss to stdout and its progress bar is swallowed when
stdout is piped, so a healthy run looks identical to a hang. Read `train_loss` from the
tfevents file to confirm it is alive.

## Which install route each folder needs

Per-folder detail is in each folder's README. The short version:

- **Plain ROCm torch + pip works** for deepspeed and deepspeed_standalone, fsdp, peft
  (including QLoRA via the ROCm bitsandbytes wheels), lightning, axolotl, llamafactory,
  torchtune (the **stable** wheel, not nightly), megatron (no TE or Apex needed), nemo, ray
  (one env var), redhat, unsloth (`unsloth[amd]`), and the classification, embedding, and
  reranker folders.
- **Override the torch pin afterwards** for composer, llmfoundry, and rapidfire — restore the
  ROCm wheel with the `--force-reinstall --no-deps` command above.
- **Needs ROCm nightly torch:** torchtitan.
- **Container route:** verl and openrlhf's RL tiers (`rocm/verl`), and primus (SFT works;
  LoRA is broken in the image).
