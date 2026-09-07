# H100 training notes and evidence

Notes from executing the training folders on **8× NVIDIA H100 80GB HBM3** (Hopper,
cc 9.0), driver 580.173.02, CUDA 13.0, Ubuntu, Python 3.12.3, covering the NVIDIA column
alongside the MI355X notes. The per-folder support table lives in the top-level
[README](../README.md); this file holds the cross-cutting lessons and the ROCm→CUDA
reversals. The companion AMD notes are in
[mi355x_training_notes.md](mi355x_training_notes.md); roughly half of that file is
hardware-neutral and applies here too. Folder names use the current layout
(`training/<modality>/<framework>`).

> **Method:** single-GPU smokes first, then real 2-GPU sharding passes. 8-GPU
> numbers are **extrapolated for reference and labelled as projections — not measured**;
> the rule is that every asserted result be backed by a real log.

## The platform, and the one constant that matters most

- **VRAM is 80 GB, vs 288 GB on MI355X (3.6× less).** This is the single biggest difference.
  A run the AMD notes call "comfortable" can OOM here. OOM is **not** "doesn't work" — reduce
  batch/seq/model or add sharding/offload, then document what it took; that adjustment *is*
  the finding. Concrete: `fsdp` full-FT of an ~8B model trains steady at ~46 GB but its
  **consolidated fp32 save spikes to the full 80 GB card even with CPU offload** (34.7 GB
  fp32 model; a fit-end checkpoint with optimizer state is ~126 GB on disk). Budget for the
  save, not just the step.
- **A proxy, not an air-gap.** Where a host exports `HTTP_PROXY`/`HTTPS_PROXY` to a corporate
  proxy, that proxy may return **403** for non-allowlisted hosts (huggingface.co,
  pypi.nvidia.com, download.pytorch.org, nvcr.io) while pypi.org **is** allowlisted (so
  `pip install torch` works with the proxy on). **Unsetting the proxy** (`unset HTTP_PROXY
  HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`) restores full egress. Treat such a
  host as online-with-a-gate, not offline. Pointing `HF_HOME` at a large pre-populated model
  cache means most runs never need the network at all.
- **venvs must live on tmpfs (`/dev/shm`), not on a CIFS/NFS mount.** Such mounts are
  pathologically slow for venv creation/imports *and* can reject CUDA-lib symlink ops during
  pip install (`OSError: Operation not permitted` on `libcusparseLt.so.0`). Standard pattern:
  `python3 -m venv /dev/shm/venv_<leaf> && ln -sf /dev/shm/venv_<leaf> .env_<leaf>`.

## torch on CUDA 13 — what actually resolves

- **Default stable `pip install torch` gives `2.13.0+cu130`** — a native CUDA-13 build that
  sees all 8 H100s. No `--index-url` needed. The repos pin `torch==2.11.0` (matching the ROCm
  wheel); **there is no cu130 wheel for 2.11.0**, so on H100 you install the default and
  document the deviation. This was the same story across peft, fsdp, torchtune, ray,
  transformers.
- **Re-check `torch.version.cuda` after every framework install** — the reverse of the ROCm
  trap. Observed clobbers, all benign because the replacement runs on this driver:
  - `mosaicml`/composer pins `torch<2.7.1` → downgrades to **2.7.0+cu126**; runs fine on a
    cu130/driver-580 host, so **no `--no-deps` override is needed** (unlike ROCm, where the pin
    is unsatisfiable). Cleaner on NVIDIA.
  - `training-hub` (redhat) downgraded 2.13→**2.11.0+cu130** (still CUDA 13) — fine.
  - `tensorrt-llm` pulled **2.9.1+cu128** — fine.
  Driver 580 is backward-compatible across cu126/cu128/cu130 builds, so "wrong minor CUDA" is
  usually a non-event; verify `torch.cuda.is_available()` + one real GPU op and move on.

## The ROCm workarounds that get REVERSED on H100

- **flash-attn exists here — and builds.** `redhat`'s `training-hub[cuda] --no-build-isolation`
  compiled **flash-attn 2.8.3.post1** via nvcc (~24 min, MAX_JOBS=32) and mini-trainer
  auto-selected `flash_attention_2` with no `TESTING=true` shim. This directly confirms the
  thesis that flash-attn is the whole point on NVIDIA. Caveat: the build is slow; where a
  folder hard-codes `attn_implementation="sdpa"` (e.g. `ray`) or the smoke budget is tight,
  sdpa selects an efficient Hopper kernel with no correctness cost and is an acceptable
  time-boxed fallback — document which one engaged.
- **`tf32` guard auto-enables on CUDA.** The `torch.version.cuda is not None` guard that
  *raised* on ROCm simply turns tf32 on here; no action needed, but note that several
  inference/exact paths keep `matmul.allow_tf32=False` by default in torch 2.13, so tf32 isn't
  always on the hot path.
- **Drop `HIP_VISIBLE_DEVICES` and the NOSET escape hatches.** Plain `CUDA_VISIBLE_DEVICES`
  is authoritative. `ray` confirmed the brief's hypothesis with evidence: on CUDA
  `RAY_EXPERIMENTAL_NOSET_*` is **not needed** — Ray's device manager and Ray Train both scope
  the *same* `CUDA_VISIBLE_DEVICES`, so the ROCm two-variable mismatch (`invalid device
  ordinal`) can't occur. Just `export CUDA_VISIBLE_DEVICES=<n>`, drop HIP + NOSET.
- **bitsandbytes "just works."** The stock PyPI wheel loads `libbitsandbytes_cuda130.so` (no
  ROCm `.so`); `peft` ran LoRA and QLoRA (nf4, paged_adamw_8bit) with no special handling —
  QLoRA is genuinely easier on NVIDIA.

## Hardware-neutral traps (these bite on any vendor)

- **transformers 5.x `apply_chat_template(tokenize=True)` returns a `BatchEncoding`**, so
  prompt-masking diffs silently drop every row — pass `return_dict=False`. (Same as MI355X.)
- **`classification/deepspeed --zero_stage 3` saves no weights** — its config never sets
  `stage3_gather_16bit_weights_on_model_save`, so the HF save no-ops while exiting 0. Use
  ZeRO-2, or recover from the emitted `zero_to_fp32.py`. Reachable only once params are
  sharded — a single-GPU run cannot catch it.
- **Batch geometry fails silently.** `global = per_device × world × grad_accum`. An 8-rank
  run against a 9–10 row sample can do one or zero optimizer steps and still print a plausible
  loss and exit 0. Always compute steps/epoch and confirm it's non-trivial. (Single-GPU
  smokes here bumped epochs to keep the step count meaningful.)
- **Disabling checkpoints is hard.** `save_strategy:"no"` doesn't stop HF Trainer's fit-end
  write, `load_best_model_at_end=True` forces a save, and Composer/Foundry write a fit-end
  checkpoint regardless. Write outputs to a scratch location such as
  `/dev/shm/<leaf>/` and delete large checkpoints once the evidence is captured.
- **Demand GPU-residency proof by PID, sampled from inside the job.** An H100 idles at ~0 MiB
  / 0 %, and a fast smoke (e.g. deepspeed's 87 s SFT) can finish before an externally-sampled
  `nvidia-smi` lands — an external residency sample can therefore catch the card *idle*, and
  must be re-run with an in-loop sampler. On a shared host an external sample can also capture
  a co-tenant's PID instead of your own.

## CUDA-13 dependency traps found in the field

- **`kernels 0.16.0`** (pulled by some `[cuda]` extras, e.g. training-hub) **crashes all
  transformers 5.5.0 imports** (`LayerRepository … revision/version` ValueError). Pin
  `kernels>=0.12,<0.13`.
- **`datasets` `.arrow` cache writes can fail on a CIFS/NFS mount** (`OSError Errno 1
  Operation not permitted`). Set `HF_DATASETS_CACHE=/dev/shm/dscache_<leaf>`.
- **torch-2.13 + FSDP2 rejects tied embeddings.** `gemma-4`'s `tie_word_embeddings:True`
  collides with `fully_shard` (`Parameter embed_tokens.weight is shared`); torch 2.11
  tolerated it. Untie at load (`from_pretrained(..., tie_word_embeddings=False)`). Seen in
  `fsdp`.
- **torchtune needs `torchao==0.10.0` pinned** (with `--no-deps`) — 0.18.x moved
  `torchao.dtypes.nf4tensor`, which torchtune 0.6.1 imports at load. Hardware-neutral.

## Model availability (offline-first, download-capable)

Check what is actually present in `$HF_HOME` before assuming a model can start offline. A
fully-cached, offline-usable causal-LM chat model such as **`LiquidAI/LFM2.5-350M`**
(tokenizer + chat template + weights) is a safe reference. The frequently-defaulted
`google/gemma-4-E4B-it` is often cached as weights+config only, with **no tokenizer files**, so
it cannot start offline without borrowing a tokenizer from a sibling gemma-4; `Qwen/Qwen3-0.6B`,
`Qwen2.5-0.5B-Instruct` and `Llama-3.2-3B-Instruct` may not be cached at all. Swaps are
documented per folder; with network egress available any of these can also just be downloaded.
**Model size doesn't matter for this exercise — the goal is proving the code runs.**

## Multi-GPU

Each folder that passes single-GPU also gets a **real 2-GPU sharding pass**; a 2-GPU pass fits
inside a small free GPU pool, so it is not gated on a co-tenant job holding the rest of the
node. **8-GPU figures in the README table are projections extrapolated from the 1→2-GPU
scaling and are labelled as such — not measured.** Per-folder 2-GPU evidence lives in each
folder's own H100 section.
