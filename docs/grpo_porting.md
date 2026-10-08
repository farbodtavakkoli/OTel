# Porting: running the same stacks on ROCm and on CUDA

The launchers, the reward and the data contract are identical on both platforms. Only two things
differ: how a GPU is selected, and how the environment is built.

## The platform switch

One environment variable, `GRPO_PLATFORM=rocm|cuda`, defaulting to `rocm`. It switches GPU selection
and nothing else: no training logic, no hyperparameters, no reward behaviour depends on it.

| | ROCm | CUDA |
|---|---|---|
| Process mask | `ROCR_VISIBLE_DEVICES` (`HIP_VISIBLE_DEVICES` nests inside an already-narrowed set) | `CUDA_VISIBLE_DEVICES` |
| Container grant | `--device /dev/kfd --device /dev/dri/renderD<n> --group-add video` | `--gpus "device=<n>"` |

Three consequences worth knowing before the first run:

- `/dev/kfd` and the `/dev/dri` render nodes do not exist on CUDA; `--gpus` is meaningless to the
  ROCm runtime. Anything passing either through is platform-specific.
- Render-node arguments become plain GPU indices on CUDA. ScalarLM keeps one `--render-nodes` flag
  whose entries are device paths on ROCm and indices on CUDA; the meaning is unchanged.
- TorchTitan RL's provisioner overwrites `CUDA_VISIBLE_DEVICES` with an absolute range starting at 0,
  so no parent-level mask keeps its workers off GPU 0. Keep the judge outside that range.

## vLLM version, the most common blocker

Almost no porting problem is the GPU. It is which vLLM the stack's framework can import, and both
failure modes below happen at import time, before any training starts.

| Stack | vLLM | Why |
|---|---|---|
| verl, Red Hat training-hub | **0.28.0** | 0.29 renamed `WeightsMapper.get_unstacked_mapper`, breaking LoRA hot-load |
| TRL + DeepSpeed, Axolotl, RapidFire AI, Unsloth | **≤ 0.27.1** | TRL 1.12 imports `NCCLTrainerSendWeightsArgs`, removed in 0.28 |
| OpenRLHF, TorchTitan RL | 0.29.0 | no constraint of their own |

Two rules follow. **Install vLLM first, then the framework `--no-deps`**, because vLLM pins torch and
the accelerator runtime consistently, and installing torch first makes vLLM replace it and drag in a
second runtime. **Check the architectures before anything else**: `check_archs` in
`../training/llm/grpo_common/setup_common.sh` fails the setup immediately if the installed vLLM cannot serve the policy
and the judge.

Note that the verl and TRL pins sit below the shared default that registers the large multimodal
policy's architecture. On released wheels those two constraints cannot both hold, so for a
multimodal policy expect to need a source build or a newer framework commit that drops the renamed
symbol. `check_archs` is what tells you, before a run rather than during one.

## Why ROCm uses an image and CUDA a venv

No released ROCm vLLM wheel registers the policy architecture, so on ROCm vLLM is compiled from
source into one shared base image (`../training/llm/grpo_common/build_base_image.sh`) and each stack is a thin
`--no-deps` layer on top. On CUDA the same versions install from PyPI, so each stack builds a
virtualenv instead. Both routes live in the same stack folder and neither is a fallback for the
other.

## Attention backends

- **TorchTitan RL on CUDA SM 9.0 needs `TITAN_ATTN_BACKEND=flex`.** The default path requires
  FlashAttention-3, which has no installable release and must be built from source.
- **OpenRLHF imports `flash_attn` at module scope**, so it is required even for an `sdpa` run. On
  CUDA no wheel matches the torch build and it compiles from source, which takes hours. On ROCm it is
  already in the base image.
- The ROCm patches for TorchTitan RL are ROCm-only, but one of them emits a peak-memory metric that
  is not platform-specific, so that metric is simply missing on CUDA.

## Venv-route notes

- `$GRPO_ENV_DIR` must be on a local, non-`noexec` disk: torch and vLLM cannot load compiled
  extensions from a `noexec` mount.
- `ninja` is a run-time dependency, not just a build one, because vLLM's FlashInfer backend JIT-compiles
  kernels on first use.
- Ray sizes its object store from `/dev/shm` and gets OOM-killed on a small one; set
  `GRPO_RAY_OBJECT_STORE_GB`. Console scripts must resolve next to the venv interpreter, so pass
  `PATH` explicitly to child processes and Ray actors.

## What a port has to prove

The gate is a short smoke run: two steps at a small batch and token budget with `--verify`. It passes
when the stack exits 0, the judge is called the expected number of times with zero failures, and the
exported adapter reloads with no unmatched tensors and generates coherent text. For a full-parameter
stack, the equivalent is a checkpoint round trip that reloads with the weights changed.

That establishes the stack runs on the platform. It establishes nothing about throughput: smoke
timers use a fraction of the tokens and steps of a real measurement, so they are never comparable
with a benchmark number.

## A note for anyone tidying the platform helpers

Four stacks (`axolotl`, `rapidfire`, `trl_deepspeed`, `scalarlm`) each carry their own copy of the
`GRPO_PLATFORM` / `_is_cuda` / `_visible_devices_var` block in their `utils.py` instead of importing a
shared one. That is deliberate. Every import of `grpo_common/` inside a stack's `utils.py` is deferred
into a function body, because those modules must also import standalone: verl's is loaded by hydra as
`data.custom_cls.path`, and Axolotl's is loaded as a plugin, in contexts where `grpo_common/` is not on
`sys.path`. `PLATFORM` is read at module scope, so hoisting it would need a module-level import and
would break exactly those paths. Keep the copies in step by hand.
