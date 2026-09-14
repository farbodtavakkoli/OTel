# ScalarLM server image — AMD Instinct MI355X (ROCm)

Build / run / customize / publish runbook for the **MI355X (gfx950, CDNA 4)** server image.

---

## 1. Pre-built images

| Tag | Hardware | Notes |
|---|---|---|
| `farbodatdocker/scalarlm:mi355-v1.7` | AMD MI355X, gfx950 / ROCm 7.2.4 | current |
| `farbodatdocker/scalarlm:mi355-v1.6` | AMD MI355X, gfx950 / ROCm 7.2.4 | previous current tag |
| `farbodatdocker/scalarlm:mi355-v1.5` | AMD MI355X, gfx950 / ROCm 7.2.4 | first image built from the unified AMD + NVIDIA tree |
| `farbodatdocker/scalarlm:mi355-v1.0` | AMD MI355X, gfx950 / ROCm 7.2.4 | first published tag (pre-fix history, see version records) |
| `farbodatdocker/scalarlm:h100-v1.5` | NVIDIA H100, sm_90 | see the NVIDIA doc |

The image is **58.9 GB** on disk (~15 GB compressed over the wire).

### Which source revision is inside a given image

There is deliberately **one MI355X tag**, so the tag name alone does not tell you what is in it.
The source revision is recorded inside the image instead, as an OCI label:

```bash
docker inspect farbodatdocker/scalarlm:mi355-v1.7 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

Cite **that hash**, not the tag, in a bug report — the tag moves when the image is rebuilt, the
label does not. `org.opencontainers.image.created` gives the build timestamp and
`org.opencontainers.image.authors` the maintainer. The revision refers to a commit in **this**
repository; `scalarlm_mi355.patch` (§7) reconstructs the same tree from upstream.

If you rebuild and republish under the same tag, pass the new revision to `--label` (see §2) so
this check stays accurate; the loss fingerprints in §4 are the independent cross-check.

### Run it

The image is **self-starting** — a bare `docker run` serves:

```bash
docker run -d --name scalarlm --init \
  --shm-size=64g --ulimit memlock=-1 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE --cap-add IPC_LOCK \
  -p 8000:8000 -p 8001:8001 \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B \
  -e SCALARLM_MAX_GPUS_PER_NODE=8 \
  -v "$PWD/models:/root/.cache/huggingface" \
  farbodatdocker/scalarlm:mi355-v1.7
```

Health check:
```bash
curl -s localhost:8000/v1/health
# {"api":"up","vllm":"up","megatron":"up","all":"up"}
```

Passing a script still overrides the default, so every existing invocation keeps working:
```bash
docker run ... farbodatdocker/scalarlm:mi355-v1.7 /app/cray/scripts/start_one_server.sh
```

### AMD flags that are not optional

| Flag / env | Why |
|---|---|
| `--device=/dev/kfd --device=/dev/dri` | ROCm device access (the AMD analogue of `--gpus`) |
| `--group-add video` | render-node permissions |
| `--shm-size=64g` | RCCL shared-memory transport; the 64 MB default deadlocks collectives |
| `--security-opt seccomp=unconfined` | HIP needs syscalls the default profile blocks |
| **`SCALARLM_MAX_GPUS_PER_NODE=8`** | **Default is 1.** Without it every job is silently capped to a single GPU no matter what `gpus` says. |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | optional; without network the boot can otherwise stall in tokenizer loading |

Multi-GPU training: set `gpus: 8, nodes: 1` in `train_args`. For multi-node, see §5.

### Deploying via the Helm chart

The container is the unit of deployment; in a cluster it is selected through the ScalarLM chart's
image block:

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: mi355-v1.7          # AMD MI355X / gfx950; use the H100 tag for NVIDIA
  pullPolicy: Always
```

The chart's stock `values.yaml` still points at `gdiamos/scalarlm-nvidia-12.0` — an **NVIDIA** image
that will not run on this hardware. Override the image block; do not rely on the default.

The AMD device flags in the table above have no `docker run` equivalent in a pod spec: request GPUs
through the AMD device plugin (`amd.com/gpu`), mount `/dev/kfd` and `/dev/dri`, and size
`/dev/shm` — a pod's default 64 MB `/dev/shm` deadlocks RCCL exactly as the Docker default does.

### Multi-GPU inference

- **Base model** — tensor-parallel across GPUs with `SCALARLM_TENSOR_PARALLEL_SIZE=<n>`.
- **Fine-tuned adapters** — serve at **tensor-parallel size 1**. vLLM's TP path rejects the
  hot-reloaded un-sharded adapter state dict.

Any `default_config.py` key can be overridden this way: `get_config()` synthesises
`SCALARLM_{KEY.upper()}` for every field at load time, which is why `SCALARLM_MAX_GPUS_PER_NODE`
and `SCALARLM_TENSOR_PARALLEL_SIZE` work despite appearing nowhere in the source as literals. Also
useful: `SCALARLM_ENABLE_LORA` (default `false`) and `SCALARLM_GPU_MEMORY_UTILIZATION` (default
`0.40` — the headroom that lets training and vLLM share GPU 0).

### If the HF cache is on a network mount

Point the datasets cache at local disk:

```bash
-e HF_DATASETS_CACHE=/tmp/hf_datasets
```

The `datasets` file lock returns `EACCES` on SMB for the second rank, so a multi-GPU job fails on
rank 1 while rank 0 looks healthy.

---

## 2. Build from scratch

The full ScalarLM server source ships in the `scalarlm` folder as **`repo/`** — a verbatim tree with the AMD
integration applied. Build from its Dockerfile, not by overlaying a published base: the
integration touches `ml/`, `infra/cray_infra/`, `scripts/` and `test/`, so an `ml/`-only overlay
cannot deliver it.

### The one-command build

```bash
cd training/llm/scalarlm
./build_image.sh                          # or: IMAGE_TAG=mi355-v1.7 ./build_image.sh
```

`build_image.sh` does three things you must not skip if you build by hand:

1. **Copies `ml/` into `repo/`** before building. `repo/ml/` is *gitignored and not source* —
   there is exactly one `ml/` tree, the one you edit, and staging it at build time is what
   guarantees the image bakes the current recipe.
2. **Creates `repo/vllm/`** if absent. The Dockerfile bind-mounts `./vllm`, and a missing directory
   fails the build with `failed to compute cache key: "/vllm": not found`. It is intentionally
   empty — `VLLM_SOURCE` defaults to `remote` and vLLM is cloned at the commit pinned in the
   Dockerfile.
3. **Labels the image** with the outer repository's HEAD, marking it `-dirty` if the tree has
   uncommitted changes.

### Pre-flight

```bash
git status --short                 # should be empty, or the image is labelled -dirty
git rev-parse HEAD                 # the revision that will be recorded in the image
rocminfo | grep -m1 gfx            # gfx950
```

### Building by hand

Equivalent to what the script does:

```bash
cd training/llm/scalarlm
rsync -a --exclude='__pycache__' --exclude='*.pyc' ml/ repo/ml/
mkdir -p repo/vllm
docker build --network=host --platform linux/amd64 \
  --build-arg BASE_NAME=amd \
  --build-arg TORCH_CUDA_ARCH_LIST=gfx942 \
  --build-arg VLLM_TARGET_DEVICE=rocm \
  --label org.opencontainers.image.revision=$(git rev-parse HEAD) \
  -t cray:mi355-build repo/
```

Two AMD-specific notes, both load-bearing:

- **`--network=host` is required.** Without it the in-build vLLM `git clone` fails TLS.
- **`TORCH_CUDA_ARCH_LIST=gfx942` is correct despite the name** (the build arg is inherited from
  the CUDA path). Inside the Dockerfile `PYTORCH_ROCM_ARCH` governs and includes gfx950, so
  gfx950 kernels *are* produced:
  ```
  torch.cuda.get_arch_list() -> [... 'gfx942', 'gfx950' ...]
  ```
  Do not "fix" this to `gfx950` without re-running the arch check below.

### Arch verification — the `roc-obj-ls` probe is not valid here

The intuitive check `roc-obj-ls <lib> | grep gfx950` **does not work on this image** and must not be
used as a gate. It reports **0 bundles for every library**, including `libtorch_hip.so`, which
demonstrably works. The bundle format is not one this `roc-obj-ls` parses.

Use functional evidence instead:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  farbodatdocker/scalarlm:mi355-v1.7 python -c "
import torch
print(torch.cuda.get_device_name(0))          # AMD Instinct MI355X
print('gfx950' in torch.cuda.get_arch_list()) # True
a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
print((a@a).shape)                            # a real HIP kernel executes
"
```

### Make it self-starting + sign it

Metadata-only build — no layers rebuilt:

```bash
BUILDDIR=$(mktemp -d)
cat > "$BUILDDIR/Dockerfile" <<EOF
FROM cray:amd4
CMD ["/app/cray/scripts/start_one_server.sh"]
LABEL org.opencontainers.image.authors="Farbod Tavakkoli"
LABEL org.opencontainers.image.source="https://github.com/farbodtavakkoli/OTel"
LABEL org.opencontainers.image.revision="$(git rev-parse HEAD)"
EOF
docker build -t farbodatdocker/scalarlm:mi355-v1.7 "$BUILDDIR"
rm -rf "$BUILDDIR"
```

`$(git rev-parse HEAD)` is expanded by the shell **before** the heredoc is written, so the label
records the revision you actually built from. Confirm it landed:

```bash
docker inspect farbodatdocker/scalarlm:mi355-v1.7 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

---

## 3. Customizing the training recipe (`ml/`)

**In most cases you do not need to rebuild or overlay anything.** The SDK ships whichever copy of
`ml/` is next to you:

- **Client side** — `find_ml_dir()` returns `$CWD/ml` if it exists, else the `ml/` installed beside
  the SDK package, else nothing. Whatever it finds is tarred and uploaded with the job.
- **Server side** — the upload is extracted into the job directory. The image's own
  `/app/cray/ml` is copied in **only if the upload contained no `ml/`**.

So:

| You run the client from | Which `ml/` actually trains |
|---|---|
| `training/llm/scalarlm/` | **yours** — edit `ml/`, rerun, done. No rebuild. |
| anywhere without an `ml/` dir | the image's baked default |

That is the supported way to modify the recipe: **edit `ml/` in the `scalarlm` folder and submit a job.** Your
changes take effect on the next run.

> **Note.** Because the client's copy wins, a
> fix you bake into the *server* image does nothing if your client folder still has an old `ml/` —
> the job runs the old code and still reports COMPLETED. Symptom: a loss fingerprint identical to
> the pre-fix baseline. If a change seems to have no effect, check which `ml/` the client sent
> before you debug anything else.

### When you do need to rebuild

An overlay only reaches `ml/`:

```dockerfile
FROM farbodatdocker/scalarlm:mi355-v1.7
COPY ml/ /app/cray/ml/          # changes the DEFAULT for clients that have no ml/
```

Anything outside `ml/` — the collectives layer (`infra/cray_infra/training/distributed.py`), the
launcher, the Slurm wiring, the ROCm/PyTorch versions — is **not** in the `scalarlm` folder and cannot be
changed by an overlay. Those need the full source: see §7.

---

## 4. Acceptance gates

Run these against the built image with **zero source mounts** unless noted:

| Gate | Check | Expected |
|---|---|---|
| 0a | corrections present in image | all AMD fixes + all H100 ports found |
| 0b | gfx950 kernels | `roc-obj-ls` probe is invalid (returns 0 for every lib) — use the functional proof instead: MI355X matmul + `gfx950` in the arch list |
| 1 | unit suite in-image — run it with `./run_unit_tests.sh <tag>` | 811 passed, 2 deselected |
| 2 | collective correctness, 8 ranks | `RESULT: ALL COLLECTIVES CORRECT` |
| 3 | end-to-end, image as **both** server and client | losses match the fingerprints below |
| 4 | serving regression | 5 passed *(this script mounts the repo tree; with a clean tree at the built commit the mounts are byte-identical to what is baked)* |
| 5 | legacy collective jobs, 8 ranks | 5/5 `RESULT: … passed` |

Use `./run_unit_tests.sh` for gate 1 rather than calling `pytest` directly. A bare
`python3 -m pytest test/unit` reports 18 failures that are harness artefacts, not defects:

- **16 fsdp / pytorch_fsdp tests.** The FSDP wrap path calls `get_rank()`, which initialises
  `torch.distributed` through the `env://` rendezvous and needs `RANK`, `LOCAL_RANK`,
  `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`. Those are normally set by `torchrun`, so a bare
  `pytest` fails with `ValueError: environment variable RANK expected, but not set`. `ddp` is
  unaffected because it does not initialise at wrap time.
- **2 `test_live_test_command.py` tests.** They read `/app/cray/cmd/test_command.sh`. The
  Dockerfile copies `infra`, `sdk`, `test`, `ml` and `scripts` but not `cmd/`, which is a
  host-side developer CLI that is deliberately not shipped. The script deselects those two so the
  gate needs no source mount; run `WITH_CMD=1 ./run_unit_tests.sh <tag>` to bind-mount `repo/cmd`
  and get 813 passed instead.

The script sets the rendezvous variables and picks the GPU flags for the tag's vendor.

### Release records

Identify an image you pulled by its digest or its `org.opencontainers.image.revision` label, not
by the tag:

| Tag | Digest | `org.opencontainers.image.revision` |
|---|---|---|
| `mi355-v1.7` | `sha256:0f9ebeee1f8871b280f421ec915ea5f66cbeeaa74b6db38b8a9ab1793846961e` | `b9f27453b38839f4287a8e8e01cc323402085763` |
| `mi355-v1.6` | `sha256:ee40edab891cd3208dd1a122d0d6472d698e955a9bd7e064bff624164e86bfa4` | `6bc7d82a3a8a0185b88a3741e51272a92fd5d40e` |
| `mi355-v1.5` | `sha256:37c536eb4b2fe9be6866b9139904caa29a99fa7c859ebcc49f6482a3aef51220` | `218d2f9d31161e371205e61793ade4a140f87c7d` |
| `mi355-v1.2` | `sha256:2614b47eb78d8d2c79fa21e3fb3fd061f2b4a23d7f555556883d70b6e5327c59` | `b693ffa412c7650f23ec418384ec08c872eb1481` |
| `mi355-v1.1` | `sha256:53e3184c6d41ee2bbafc5c8107560ecd8090e0c6b27f7b4b4e57ec7e9029946e` | `a9da8f7cd3c89ad2ebd1b744654298c38a4af592` |

`mi355-v1.7` bakes the same training recipe as `mi355-v1.6` with all Python docstrings removed,
and corrects `org.opencontainers.image.source`, which earlier images pointed at an unrelated
repository. Nothing in the training or serving path changed. Verified two ways: the unit suite
gives the same result on both tags, and an identical 8-GPU `fsdp` job run against each image
produced per-step losses equal to the last digit.

### Reproducible training fingerprints

Qwen3-0.6B · 128 records · 30 steps · `adapter_type: none` · `gpus: 8, nodes: 1`, with the
`scalarlm` folder's `ml/`:

| Strategy | Final loss (30 steps) |
|---|---|
| `fsdp` | **0.0038679696153849363** |
| `ddp` | **0.005165183916687965** |
| `pytorch_fsdp` | **0.0036232012789696455** |

These reproduce bit-exactly. A mismatch means the client's `ml/` and the image's baked copy are
crossed (§3) — check which one the client uploaded before debugging anything else. The baked
`ml/` in `mi355-v1.0` predates the `pack()` tail-padding fix and yields an older set instead
(`fsdp` 0.067634217441082, `ddp` 0.07000554352998734).

> **`0.0734812841` from an fsdp run means the FSDP gradient-mean fix did not load** — a stale image
> on one side (most likely the client, see §3). That exact value is the non-execution fingerprint.

---

## 5. Known limitations and caveats

- **Multi-node works, with one caveat.** A 2-node × 8-GPU job runs in containers
  (`ALLREDUCE OK world=16`) and training completes across both hosts and writes a checkpoint. The
  blocker was asymmetric RDMA device enumeration, which hangs RCCL silently; the launcher now
  derives `NCCL_IB_HCA` as the intersection of devices active on every node. Launching multi-node
  **via `sbatch`** is not covered — use torchrun directly unless your Slurm registers every node
  and its hostnames resolve for mpirun. See "Multi-node" in `readme_scalarlm.md`.
- **`gradient_accumulation_steps` > 1 is not covered** by the fingerprints above; they all use 1,
  while the config default is 4.
- **At large model sizes use a sharded strategy.** Qwen3-32B trains under `pytorch_fsdp` and
  `fsdp`; `ddp` OOMs outright.
- **Embedding mode runs**, given `sentence_transformers`. Two caveats: the dependency is **not in
  the image**, and CoSENT returns exactly 0.0 loss at `batch_size: 1` while still reporting
  COMPLETED — use `batch_size >= 2`.
- **`upload_to_hf` works, but is dead in the image.** The image sets `HF_HUB_OFFLINE=1`, and
  because the upload path swallows failures a blocked upload still reports COMPLETED. The upload
  clears the offline flags for its own duration, but the image default is still worth changing.
- **FA2 is downgraded, not supported.** `attn_implementation=flash_attention_2` is silently mapped
  to `sdpa`. It is not a supported training backend here: FA2 loads cleanly and then aborts
  mid-training with a device-side `HSA_STATUS_ERROR_EXCEPTION`, which is not a Python exception and
  cannot be caught by the `from_pretrained` fallback.

---

## 6. Publishing

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker login -u farbodatdocker          # paste a Personal Access Token, not the password
docker push farbodatdocker/scalarlm:mi355-v1.7
```

One tag; the source revision travels inside the image as a label (§1). The push is resumable —
re-run it on a drop.

If Docker runs as root on your host, prefix `login` and `push` with `sudo` consistently — otherwise
they use different credential stores and the push fails on auth after uploading.

### Before publishing publicly

`repo/infra/slurm_configs/munge.key` and `slurm.key` are **static secrets committed in the upstream
ScalarLM repository**. They are therefore present in three places: upstream's public repo, this
vendored `repo/` tree, and — via `COPY ./infra` in the Dockerfile — every image built from it,
including the pre-existing public ones. This is upstream behaviour, not something introduced here,
and they are kept verbatim because there is no generate-if-missing path: removing them breaks the
build.

A MUNGE key is the shared secret Slurm uses to authenticate. Anyone who obtains one can forge Slurm
credentials against any cluster that trusts it.

For a single-container deployment the practical exposure is low, because slurmctld and slurmd run
inside the container and are not published. For any multi-node deployment, treat these keys as
compromised the moment the image is public, and regenerate them per cluster rather than relying on
the baked-in pair.

Also present: `/app/cray/vllm/.git` (245 MB) — not a secret, but pure image bloat.

---

## 7. Where the full source lives, and cutting the next release

Everything needed to rebuild is in the `scalarlm` folder. There is no separate repository to track.

### What is here

Paths are relative to `training/llm/scalarlm/`:

| Path | What it is |
|---|---|
| `ml/` | **the training recipe — the one canonical copy.** Edit this. |
| `repo/` | the full ScalarLM server source with the AMD integration applied |
| `docs/build_image.sh` | stages `ml/` into `repo/` and builds the image |
| `docs/run_unit_tests.sh` | runs acceptance gate 1 against a built image |
| `docs/scalarlm_mi355.patch` | the integration as one patch against public upstream |
| `docs/DOCKER_IMAGE_MI355.md`, `docs/DOCKER_IMAGE_H100.md` | the two image runbooks |
| `train.py`, `inference.py`, `data/` | the client |

`repo/` is a verbatim upstream tree at commit `4566a84` with the AMD integration applied. It
carries `infra/cray_infra/training/distributed.py` (the NCCL/RCCL collectives layer),
`scripts/train_job_entrypoint.sh`, the `Dockerfile` and the test suite — none of which an `ml/`
overlay can reach.

**`repo/` has no `.git`**, and **`repo/ml/` is deliberately absent from version control** — it is
created by `build_image.sh` from `ml/` (§2).

### Reconstructing from upstream

`scalarlm_mi355.patch` is the same integration as a single patch. Use it when you want the change
on top of a fresh upstream clone rather than this vendored copy:

```bash
git clone https://github.com/supermassive-intelligence/scalarlm.git
cd scalarlm && git checkout 4566a84
git apply /path/to/scalarlm_mi355.patch
```

### Release checklist

1. **Clean tree** — `git status --short` empty, or the image is labelled `-dirty`.
2. **Build** — `./build_image.sh`. It stages `ml/` and labels the revision for you.
3. **Verify the label** — `docker inspect` reports the revision you just built.
4. **Acceptance gates** — §4. At minimum the unit suite, the 8-rank collective check, and one FSDP
   plus one DDP run whose final loss matches the fingerprints there. A fingerprint mismatch means
   the tree you shipped is not the tree you think you shipped.
5. **Smoke test** — bare `docker run` with no command reaches `{"all":"up"}`.
6. **Push** — §6, one tag.
7. **Tag the commit** — so the published image maps to a named source revision.

### When upstream ScalarLM moves

To rebase, clone upstream at the newer commit and apply `scalarlm_mi355.patch` to it, resolving any
rejected hunks by hand. The hunks most likely to conflict are the ones this integration rewrote:
`infra/cray_infra/training/distributed.py` (replaced wholesale),
`ml/cray_megatron/megatron/distribution/{ddp,fsdp}.py` (the gradient-mean fixes), and
`scripts/train_job_entrypoint.sh` (the torchrun launcher). Re-run gate 5 afterwards — the
gradient-semantics tests catch a silently dropped fix. Then copy the rebased tree back over
`repo/` and regenerate the patch.
