# H100 (NVIDIA / CUDA) training notes

Lessons from running the training recipes on **NVIDIA H100 80GB HBM3** (Hopper cc 9.0),
CUDA 13.0, Python 3.12, including what reverses when moving from ROCm to CUDA. The
per-folder support table is in the top-level [README](../README.md).

The companion AMD notes in [mi355x_training_notes.md](mi355x_training_notes.md) are about
half hardware-neutral and apply here too.

## The platform

- **VRAM is 80 GB, against 288 GB on MI355X.** A run the AMD notes call comfortable can OOM
  here — reduce batch size, sequence length, or model size, or add sharding and offload.
  Budget for the **save**, not just the step: `fsdp` full fine-tuning of an ~8B model trains
  steadily well under the card, but its consolidated fp32 save spikes to the full 80 GB even
  with CPU offload.
- **Build venvs on tmpfs (`/dev/shm`), not on a network mount.** NFS and CIFS mounts are slow
  for venv creation and imports, and can reject the CUDA-library symlink operations pip
  performs (`OSError: Operation not permitted` on `libcusparseLt.so.0`). Use:

  ```bash
  python3 -m venv /dev/shm/venv_<leaf> && ln -sf /dev/shm/venv_<leaf> .env_<leaf>
  ```

## torch on CUDA 13

- **`pip install torch==2.11.0` gives a native CUDA-13 build** that sees all 8 H100s. No
  `--index-url` is needed. Use
  `--index-url https://download.pytorch.org/whl/cu130` only if you want the explicit
  `+cu130` local version tag. Install torch **before** the framework so a later resolver does
  not replace it.
- **Re-check `torch.version.cuda` after every framework install.** Unlike on ROCm, a
  replacement here is usually harmless — a recent driver is backward-compatible across
  cu126, cu128, and cu130 builds. Observed downgrades, all of which run fine:
  composer to 2.7.0+cu126, redhat's `training-hub` to 2.11.0+cu130, tensorrt-llm to
  2.9.1+cu128. Confirm `torch.cuda.is_available()` plus one real GPU op and move on.

## ROCm workarounds that reverse on H100

- **flash-attn exists and builds.** `redhat`'s `training-hub[cuda] --no-build-isolation`
  compiles flash-attn 2.8.3.post1 via nvcc (set `MAX_JOBS=32`). The build is slow, and where
  a folder hard-codes `attn_implementation="sdpa"`, sdpa remains an acceptable fallback.
- **The `tf32` guard auto-enables.** What raised on ROCm simply turns tf32 on here. No action
  needed.
- **Drop `HIP_VISIBLE_DEVICES` and the NOSET escape hatches.** Plain `CUDA_VISIBLE_DEVICES`
  is authoritative. `RAY_EXPERIMENTAL_NOSET_*` is not needed, because Ray Core and Ray Train
  both scope the same variable on CUDA — the ROCm mismatch that caused `invalid device
  ordinal` cannot happen.
- **bitsandbytes needs no special handling.** The stock PyPI wheel loads
  `libbitsandbytes_cuda130.so`, so `peft` runs LoRA and QLoRA unmodified. QLoRA is easier
  here than on ROCm.

## Traps that apply on any vendor

- **transformers 5.x `apply_chat_template(tokenize=True)` returns a `BatchEncoding`**, so
  prompt-masking diffs silently drop every row. Pass `return_dict=False`.
- **`classification/deepspeed --zero_stage 3` saves no weights.** Its config never sets
  `stage3_gather_16bit_weights_on_model_save`, so the save no-ops while the run exits 0. Use
  ZeRO-2, or recover from the emitted `zero_to_fp32.py`. Only reachable once parameters are
  sharded, so a single-GPU run cannot catch it.
- **Batch geometry fails silently.** `global = per_device × world × grad_accum`. An 8-rank
  run against a 10-row sample can take one or zero optimizer steps and still print a
  plausible loss and exit 0. Compute steps/epoch and confirm it is non-trivial; on a tiny
  sample, raise the epoch count.
- **Disabling checkpoints is hard.** `save_strategy: "no"` does not stop HF Trainer's
  fit-end write, `load_best_model_at_end=True` forces a save, and Composer and LLM Foundry
  write a fit-end checkpoint regardless. Send outputs to `/dev/shm/<leaf>/` and delete large
  checkpoints once you have the evidence.
- **Prove GPU residency by PID, sampled from inside the job.** An H100 idles at ~0 MiB and
  0%, and a fast smoke run can finish before an external `nvidia-smi` sample lands — so an
  external sample can catch the card idle, or capture a co-tenant's process on a shared host.

## CUDA-13 dependency traps

- **`kernels 0.16.0` crashes all transformers 5.5.0 imports** (`LayerRepository …
  revision/version` ValueError). It is pulled by some `[cuda]` extras, including
  training-hub. Pin `kernels>=0.12,<0.13`.
- **`datasets` `.arrow` cache writes can fail on a network mount** (`OSError Errno 1
  Operation not permitted`). Set `HF_DATASETS_CACHE=/dev/shm/dscache_<leaf>`.
- **torch-2.13 + FSDP2 rejects tied embeddings.** `gemma-4`'s `tie_word_embeddings: True`
  collides with `fully_shard` (`Parameter embed_tokens.weight is shared`); torch 2.11
  tolerates it. Untie at load with `from_pretrained(..., tie_word_embeddings=False)`.
- **torchtune needs `torchao==0.10.0` pinned** with `--no-deps`. 0.18.x moved
  `torchao.dtypes.nf4tensor`, which torchtune 0.6.1 imports at load.

## Model cache

Check what is actually in `$HF_HOME` before assuming a model can start offline.
`LiquidAI/LFM2.5-350M` is a safe reference: tokenizer, chat template, and weights all cache
together. The frequently-defaulted `google/gemma-4-E4B-it` often caches as weights and config
only, with **no tokenizer files**, so it cannot start offline without borrowing a tokenizer
from a sibling gemma-4. `Qwen/Qwen3-0.6B`, `Qwen2.5-0.5B-Instruct`, and `Llama-3.2-3B-Instruct`
may not be cached at all. Per-folder swaps are documented in each recipe README, and any of
these can simply be downloaded.
