# MI355X training campaign — notes and evidence (August 2026)

Field notes from executing every training folder on 8× AMD Instinct MI355X (gfx950,
288 GB HBM), ROCm 7.2.4, Ubuntu, Python 3.12.3. The per-folder verdict table lives in the
top-level [README](../README.md); this file holds the cross-cutting lessons, the 8-GPU
scaling evidence, and the upstream-claims audit. Folder names refer to the current layout
(`training/<modality>/<framework>`).

## What the MI355X runs taught us (applies to most folders)

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
- Unrelated to AMD but it bit six folders: transformers 5.x `apply_chat_template(tokenize=True)`
  returns a `BatchEncoding`, so prompt-masking diffs silently drop **every** row — pass
  `return_dict=False`.

## Scaling to all 8 GPUs

Single-GPU smoke runs prove a stack imports and steps; they do not prove *sharding*,
collectives, or that the batch geometry survives. So every trainable folder was re-run on
**all 8 MI355X**, each holding a machine-wide `flock` so it owned the whole node, with
`rocm-smi` sampled *inside* the locked job and attributed by PID.

**All 23 trainable folders work at world size 8**, with one qualification:
`llm/openrlhf` counts here on its **SFT/DPO** tiers, which do scale to 8 — its
PPO/GRPO tiers stay dependency-blocked in the venv and were only ever validated in the
`rocm/verl` container at **4 GPUs, not 8**. So the strict reading is *22 folders fully at
world size 8, plus openrlhf partially*. (`llm/scalarlm` is excluded by design — it is an
HTTP client with no local training loop.) Commands and full evidence live in each folder's
own "8-GPU run" section.

| Folder | At 8 GPUs | Parallelism exercised | Notes |
|---|---|---|---|
| `llm/deepspeed` | **works, unmodified** | ZeRO-2, world 8 | SFT + DPO + GRPO all passed at 8 ranks |
| `llm/deepspeed_standalone` | **works** | ZeRO-3 full FT, world 8 | per-GPU VRAM 167 → 47-53 GiB going 2 → 8 ranks. The "stage 3" banner never prints (logger reset) — introspect the live engine instead of trusting the log |
| `llm/fsdp` | **works, unmodified** | FSDP2 full shard | per-GPU VRAM ~41% → ~18.5% from 2 → 8 ranks: real sharding |
| `llm/megatron` | **works, unmodified** | TP + PP + DP, 3 layouts | DP8 fastest (92-112 TFLOP/s/GPU); TP2/PP2/DP2 exercises every process-group type. Zero NaN, zero skipped iters |
| `llm/torchtitan` | **works, unmodified** | FSDP2 + TP, 2 layouts | pure `dp_shard=8` is **2.2× faster** than `dp_shard=4 × tp=2` |
| `llm/primus` | **works (DP8 only)** | DP 8 | ~3.5× vs 2 GPUs (88% of ideal). ⚠️ TP/PP **blocked** — the image's checkpoint converter only ever emits TP1/PP1, so re-sharding is rejected at resume. LoRA still broken |
| `llm/verl` | **works, unmodified** | FSDP2 world 8 **colocated** with 4 vLLM engines × TP2 | the default topology already targets 8 GPUs; trainer and rollout share the same cards rather than splitting them |
| `classification/deepspeed` | **works, unmodified** | ZeRO-2 | 5.5× throughput |
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
| `llm/peft` | **works, with one change** | DDP 8 — LoRA, QLoRA, DoRA | `--ddp_find_unused_parameters` (see multimodal note below). ⚠️ QLoRA skews VRAM badly: GPU 0 at 76% vs 17-19% on ranks 1-7 |
| `llm/rapidfire` | **works** | *config-level*: 8 configs, one per GPU | ⚠️ must **unset** `HIP_/CUDA_VISIBLE_DEVICES` — the opposite of every other folder (see below). 4.8× wall clock, 8.1× on the training window |
| `embedding/sentence_transformers` | **works** | DDP + cross-rank negative gather | MTEB **0.5613** at 8 GPUs vs 0.5605 at 1, matched global batch. `gather_across_devices` grows the in-batch negative pool 48 → 384, which is the real reason to scale this one |
| `reranker/sentence_transformers` | **works** | DDP 8 | **6.4×** throughput (497 vs 77 samples/s) at matched global batch, ~80% efficiency |
| `llm/openrlhf` | **SFT/DPO work**; RL tiers unchanged | DeepSpeed, world 8 | batch geometry was the only change. PPO/GRPO remain blocked in the venv — a *dependency* limit (vLLM), not a scale one; they run in the `rocm/verl` container |

### What breaks when you go from 1-2 GPUs to 8

These are the failure modes that only appear at scale — and most of them **exit 0**, which
is exactly what makes them dangerous:

- **Batch geometry is the number one killer, and it fails silently.** `global = per_device
  × 8 × grad_accum`. Point an 8-rank run at a 10-100 row sample and it can perform *one*
  optimizer step (`reranker/sentence_transformers`, measured: `train_runtime 7.8s`, one
  step, exit 0) or *zero* (`embedding/sentence_transformers` at its registry default of 96
  → a 768-row global batch against 90 training rows) while still reporting a plausible
  loss. The better-behaved folders assert instead — OpenRLHF wants
  `global // micro // world == 0`, veRL wants `train_batch × rollout.n % world == 0` — and
  those abort *before* any GPU work. **Always compute steps/epoch and check it is
  non-trivial before believing an 8-GPU pass.**
- **Leftover GPU pins in venvs.** A single-GPU session that appends
  `export HIP_VISIBLE_DEVICES=<n>` to a venv's `bin/activate` silently caps a later
  nominal "8-GPU" run to one card. Override both variables **after** sourcing and assert
  `torch.cuda.device_count() == 8`. (The campaign venvs that carried these pins were
  removed in the repo reorg; rebuild from each folder's requirements file.)
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
  contrastive "negatives" into copies of the positive, and embedding's MTEB actually *fell*
  (0.5613 → 0.5208) on a ×32-replicated set. Keep generated data out of the repo.
- **Disabling checkpoints is harder than it looks.** `save_strategy: "no"` is not enough:
  HF Trainer still writes at the end of `do_train`, `load_best_model_at_end=True` forces a
  save, Composer/Foundry write a fit-end checkpoint regardless, and RapidFire persists
  per-run adapters because that *is* its config-swap mechanism. At 8 ranks the footprint
  multiplies, and FSDP2's consolidated save is fp32 (~32 GiB for an 8B model).
- **Measuring utilization is its own trap.** An idle MI355X reports ~1% util and 0.28 GB
  VRAM, so a "busy = `>0%`" threshold will happily certify 8 working GPUs while your job is
  crashing. And a short run can exit before an externally-sampled `rocm-smi` even lands,
  capturing *another* job on a shared box. Sample from inside the job and attribute by PID.

### Earlier 4-GPU sharding probes

Before the full 8-GPU sweep, four paths were probed at 4 GPUs specifically to exercise the
collective *save*, which is where multi-GPU tends to break quietly:

| Path | Result |
|---|---|
| `llm/deepspeed` ZeRO-3, full FT (no LoRA), 4 GPUs | **holds** — 8B sharded across 4 ranks, all-gather save wrote a single 14.9 GiB safetensors, no deadlock. The unguarded all-ranks `save_model` is the correct pattern (rank-0-guarding it is what deadlocked `deepspeed_standalone`) |
| `llm/fsdp` FSDP2, collective save enabled, 4 ranks | **holds** — but the consolidated save is **fp32 (~32 GiB for 8B)**, ~2× what ZeRO-3 writes, and one saving run costs ~99 GB with the optimizer checkpoint. During the save ranks 1-3 sit at 100% busy-waiting in the barrier — easy to misread as a hang |
| `llm/lightning` FSDP `FULL_SHARD`, 4 devices | **holds** — `Trainable params: 2.0 B` per rank (= 8B/4) confirms real sharding; rank-0 gather produced a valid unsharded checkpoint |
| `classification/deepspeed` ZeRO-3, 4 GPUs | ⚠️ **silent no-op save**. ZeRO-2 control on the same GPUs wrote weights correctly |

The classification ZeRO-3 bug is a *config* bug reachable only once parameters are
sharded — its `build_deepspeed_config()` never sets
`stage3_gather_16bit_weights_on_model_save`, so the HF save no-ops while still exiting 0.
No single-GPU run could have caught it. Also worth knowing: `llm/lightning` logs no loss
to stdout and its progress bar is swallowed when stdout is piped, so a healthy run looks
identical to a hang — read `train_loss` from the tfevents file to confirm liveness.

## Upstream support claims (audited)

Collected from upstream repos/docs; details and source links live in each folder's
"Hardware support & evidence" section. Where the MI355X runs contradict a claim, the run
wins — see the ⚠️ rows.

| Folder | AMD (as claimed upstream) | Audit result on MI355X |
|---|---|---|
| `llm/deepspeed` (+standalone) | official — hipcc supported, MI100/MI200 CI badge | confirmed |
| `llm/unsloth` | official — upstream lists NVIDIA/AMD/Intel + AMD training docs | confirmed (`unsloth[amd]`) |
| `llm/redhat` | reference env was MI355X, ROCm 7 | reproduced |
| `classification/deepspeed`, `embedding/*`, `reranker/*` | torch-level — ROCm wheels | confirmed |
| `llm/scalarlm` | official — TensorWave MI300X production | client verified on MI355X |
| `llm/fsdp` | device-agnostic FSDP2; Intel XPU too | confirmed |
| `llm/peft` | bitsandbytes official ROCm wheels (gfx90a/942/950) | confirmed incl. QLoRA |
| `llm/lightning` | via torch ROCm builds; not separately certified | works |
| `llm/axolotl` | official — "NVIDIA or AMD GPU" + AMD HPC guide | works (pip route) |
| `llm/llamafactory` | official — ROCm docker path + AMD-written guide | works (pip route) |
| `llm/rapidfire` | claimed **none** (NVIDIA CC 7.x/8.x) | ⚠️ **disproven** — works after restoring the ROCm wheel |
| `llm/verl` | first-party — `rocm/verl` images, MI300X/MI355X | confirmed (container) |
| `llm/openrlhf` | claimed none | ⚠️ RL tiers run on ROCm via the `rocm/verl` container |
| `llm/primus` | AMD-only — ROCm ≥ 7.0, gfx942/gfx950 | works (SFT; LoRA broken in image) |
| `llm/megatron` | official AMD builds — `rocm/megatron-lm` images | works from plain ROCm torch, no TE/Apex |
| `llm/torchtitan` | official — ROCm index swap + `AMD-AGI/torchtitan-amd` fork | works on upstream + ROCm **nightly** torch |
| `llm/torchtune` | claimed nightly-only | ⚠️ **stable** ROCm wheel works |
| `llm/nemo` | claimed **none** — NVIDIA-only | ⚠️ pip route works; NGC container genuinely unusable |
| `llm/ray` | AMD "experimental, community-supported" | works with one env var |
| `llm/composer` | none — "AMD + RoCM coming soon" | works after torch-pin override |
| `llm/llmfoundry` | beta — upstream MI250-tested path | works after torch-pin override |
