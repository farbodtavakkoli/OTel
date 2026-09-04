# ScalarLM server image — AMD Instinct MI355X (ROCm)

Build / run / customize / publish runbook for the **MI355X (gfx950, CDNA 4)** server image.

---

## 1. Pre-built images

| Tag | Hardware | Notes |
|---|---|---|
| `farbodatdocker/scalarlm:mi355-v1.6` | AMD MI355X, gfx950 / ROCm 7.2.4 | current — built from `amd-nvidia-merge` @ `6bc7d82`, the same Dockerfile revision as `h100-v1.5` |
| `farbodatdocker/scalarlm:mi355-v1.5` | AMD MI355X, gfx950 / ROCm 7.2.4 | first image off the `amd-nvidia-merge` branch (`218d2f9`) |
| `farbodatdocker/scalarlm:mi355-v1.0` | AMD MI355X, gfx950 / ROCm 7.2.4 | first published tag (pre-fix history, see version records) |
| `farbodatdocker/scalarlm:h100-v1.5` | NVIDIA H100, sm_90 | see the NVIDIA doc |

`docker images` reports **58.9 GB** (vs ~45 GB for the NVIDIA image — the ROCm base and the in-tree
vLLM build account for the difference). That is the **uncompressed on-disk** size; the registry
stores and transfers compressed layers, so the actual upload is several times smaller. For
measured: this image is 58.9 GB locally and **15.3 GB on Docker Hub** (the H100 image is ~45 GB
locally, 13.6 GB published). Plan disk around the local figure and transfer around a quarter of it.
Most of the base layers are shared with `rocm/primus` and are `Mounted from` rather than uploaded,
so the first push moves far less than 15.3 GB. The push is resumable — re-run it
on a dropped connection.

### Which source revision is inside a given image

There is deliberately **one MI355X tag**, so the tag name alone does not tell you what is in it.
The source revision is recorded inside the image instead, as an OCI label:

```bash
docker inspect farbodatdocker/scalarlm:mi355-v1.6 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

Cite **that hash**, not the tag, in any benchmark table or bug report — the tag moves when the image
is rebuilt, the label does not. `org.opencontainers.image.created` gives the build timestamp, and
`org.opencontainers.image.authors` the maintainer.

The revision refers to a commit in **this** repository — the one that carries `repo/` and `ml/`. To
see the underlying ScalarLM integration history instead, clone the bundle (§7); its
`amd-integration` tip is the tree vendored here.

If you rebuild and republish under the same tag, pass the new revision to `--label` (see §2) so this
check keeps telling the truth. An image whose label does not match the tree you think you built is
the single most expensive failure mode in this project — see the fingerprints in §4 for the
independent cross-check.

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
  farbodatdocker/scalarlm:mi355-v1.6
```

Health check:
```bash
curl -s localhost:8000/v1/health
# {"api":"up","vllm":"up","megatron":"up","all":"up"}   (~110 s from cold on this box)
```

Passing a script still overrides the default, so every existing invocation keeps working:
```bash
docker run ... farbodatdocker/scalarlm:mi355-v1.6 /app/cray/scripts/start_one_server.sh
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

Multi-GPU training: set `gpus: 8, nodes: 1` in `train_args`. Multi-node is **implemented but never
executed** — see §5.

### Deploying via the Helm chart

The container is the unit of deployment; in a cluster it is selected through the ScalarLM chart's
image block:

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: mi355-v1.6          # AMD MI355X / gfx950; use h100-v1.5 for NVIDIA H100
  pullPolicy: Always
```

The chart's stock `values.yaml` still points at `gdiamos/scalarlm-nvidia-12.0` — an **NVIDIA** image
that will not run on this hardware. Override the image block; do not rely on the default.

The AMD device flags in the table above have no `docker run` equivalent in a pod spec: request GPUs
through the AMD device plugin (`amd.com/gpu`), mount `/dev/kfd` and `/dev/dri`, and size
`/dev/shm` — a pod's default 64 MB `/dev/shm` deadlocks RCCL exactly as the Docker default does.

### Multi-GPU inference

Training and inference scale differently, and the knobs are separate:

- **Base model** — tensor-parallel across GPUs with `SCALARLM_TENSOR_PARALLEL_SIZE=<n>`.
- **Fine-tuned adapters** — serve at **tensor-parallel size 1**. vLLM's TP path rejects the
  hot-reloaded un-sharded adapter state dict. This is a vLLM constraint, not an AMD one, and it
  applies here for the same reason it applies on NVIDIA.

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
rank 1 while rank 0 looks healthy. This is a HuggingFace-library behaviour and is hardware-agnostic.

---

## 2. Build from scratch (the route actually used)

The full ScalarLM server source ships in this folder as **`repo/`** — a verbatim tree with the AMD
integration applied. The image is built from its Dockerfile, not by overlaying a published base.

**Why not the `FROM published-base + COPY ml/` shortcut:** the integration touches four trees —
`ml/`, `infra/cray_infra/`, `scripts/`, and `test/` — so an `ml/`-only overlay cannot deliver it. A
partial overlay produces a plausible-looking hybrid, which is exactly the silent-failure mode this
work exists to remove.

### The one-command build

```bash
cd training/llm/scalarlm
./build_image.sh                          # or: IMAGE_TAG=mi355-v1.6 ./build_image.sh
```

`build_image.sh` does three things you must not skip if you build by hand:

1. **Copies `ml/` into `repo/`** before building. `repo/ml/` is *gitignored and not source* —
   there is exactly one `ml/` tree in this repository, the one you edit, and staging it at build
   time is what guarantees the image bakes the current recipe. This is the single most important
   property of the layout; two committed copies of `ml/` would drift silently.
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
- **`TORCH_CUDA_ARCH_LIST=gfx942` is correct despite the name.** It is an unfortunate build-arg
  name inherited from the CUDA path. Inside the Dockerfile `PYTORCH_ROCM_ARCH` governs and includes
  gfx950, so gfx950 kernels *are* produced. Verified in the built image:
  ```
  torch.cuda.get_arch_list() -> [... 'gfx942', 'gfx950' ...]
  ```
  Do not "fix" this to `gfx950` without re-running the arch check below.

### Arch verification — and a warning about the obvious probe

The intuitive check `roc-obj-ls <lib> | grep gfx950` **does not work on this image** and must not be
used as a gate. It reports **0 bundles for every library**, including `libtorch_hip.so`, which
demonstrably works. The bundle format is not one this `roc-obj-ls` parses.

Use functional evidence instead:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  farbodatdocker/scalarlm:mi355-v1.6 python -c "
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
LABEL org.opencontainers.image.source="https://github.com/farbodtavakkoli/training_junk"
LABEL org.opencontainers.image.revision="$(git rev-parse HEAD)"
EOF
docker build -t farbodatdocker/scalarlm:mi355-v1.6 "$BUILDDIR"
rm -rf "$BUILDDIR"
```

`$(git rev-parse HEAD)` is expanded by the shell **before** the heredoc is written, so the label
records the revision you actually built from. Confirm it landed:

```bash
docker inspect farbodatdocker/scalarlm:mi355-v1.6 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

---

## 3. Customizing the training recipe (`ml/`)

**In most cases you do not need to rebuild or overlay anything.** `ml/` is the training recipe, and
the SDK ships whichever copy is next to you:

- **Client side** — `find_ml_dir()` returns `$CWD/ml` if it exists, else the `ml/` installed beside
  the SDK package, else nothing. Whatever it finds is tarred and uploaded with the job.
- **Server side** — the upload is extracted into the job directory. The image's own
  `/app/cray/ml` is copied in **only if the upload contained no `ml/`**.

So:

| You run the client from | Which `ml/` actually trains |
|---|---|
| `training/llm/scalarlm/` (this folder) | **yours** — edit `ml/`, rerun, done. No rebuild. |
| anywhere without an `ml/` dir | the image's baked default |

That is the supported way to modify the recipe: **edit `ml/` in this folder and submit a job.** Your
changes take effect on the next run.

> **The corollary is the most expensive trap in this project.** Because the client's copy wins, a
> fix you bake into the *server* image does nothing if your client folder still has an old `ml/` —
> the job runs the old code and still reports COMPLETED. Symptom: a loss fingerprint identical to
> the pre-fix baseline. If a change seems to have no effect, check which `ml/` the client sent
> before you debug anything else.

### When you do need to rebuild

An overlay only reaches `ml/`:

```dockerfile
FROM farbodatdocker/scalarlm:mi355-v1.6
COPY ml/ /app/cray/ml/          # changes the DEFAULT for clients that have no ml/
```

Anything outside `ml/` — the collectives layer (`infra/cray_infra/training/distributed.py`), the
launcher, the Slurm wiring, the ROCm/PyTorch versions — is **not** in this folder and cannot be
changed by an overlay. Those need the full source: see §7.

---

## 4. Acceptance gates — results actually observed

All run against the built image with **zero source mounts** unless noted.

| Gate | Check | Result |
|---|---|---|
| 0a | corrections present in image | all AMD fixes + all H100 ports found |
| 0b | gfx950 kernels | `roc-obj-ls` probe invalid (0 for every lib); functional proof passed — MI355X matmul + `gfx950` in arch list |
| 1 | unit suite in-image | **14 passed** |
| 2 | collective correctness, 8 ranks | `RESULT: ALL COLLECTIVES CORRECT` |
| 3 | end-to-end, image as **both** server and client | see fingerprints below |
| 4 | serving regression | **5 passed** *(this script mounts the repo tree; the tree was clean at the built commit, so the mounts are byte-identical to what is baked)* |
| 5 | legacy collective jobs, 8 ranks | 5/5 `RESULT: … passed` |

### Release verification — run against the published image itself

Re-run immediately before publishing, against `farbodatdocker/scalarlm:mi355-v1.0` with no source
mounts (the client used the image's own baked `ml/`):

| Check | Result |
|---|---|
| `ml/` in image == `ml/` in this folder | identical (git tree `a735e194…`) |
| FSDP, 30 steps, 8 GPUs | COMPLETED · 0.254 s/step · loss **0.067634217441082** |
| DDP, 30 steps, 8 GPUs | COMPLETED · 0.174 s/step · loss **0.07000554352998734** |
| GPUs engaged | 8/8 both runs (+12 GB FSDP, +13 GB DDP per device) |

Both losses matched the fingerprints for that tree exactly, which is what certifies the published
artifact runs the code this folder documented at release. Records:
`scalarlm_mi355_release_2026-08-31/`.

#### `mi355-v1.1` — published 2026-08-31

Rebuilt to carry the `pack()` tail-padding fix into the image's baked `ml/`. That copy is what a
client with no `ml/` of its own uploads (§3), so before v1.1 such a client still got the ragged-block
bug — which crashes at `batch_size > 1`, the client default.

| Check | Result |
|---|---|
| digest | `sha256:53e3184c6d41ee2bbafc5c8107560ecd8090e0c6b27f7b4b4e57ec7e9029946e` |
| `org.opencontainers.image.revision` | `a9da8f7cd3c89ad2ebd1b744654298c38a4af592`, no `-dirty` |
| fix present in baked `ml/` | yes — 3 occurrences of the padding path in `load_language_model_dataset.py` |
| FSDP, 30 steps, 8 GPUs | COMPLETED · 0.253 s/step · loss **0.0038679696153849363** |
| DDP, 30 steps, 8 GPUs | COMPLETED · 0.166 s/step · loss **0.005165183916687965** |

Both gates were run with the client executing **from the v1.1 image with no `ml/` mount**, so
`find_ml_dir()` resolved to the image's own `/app/cray/ml`. That is deliberately the path v1.1 exists
to fix, and both losses match the post-`pack()` fingerprints exactly.

`mi355-v1.0` remains published and its record above stands — it documents the pre-fix tree, which is
why v1.1 is a new tag rather than an overwrite.

#### `mi355-v1.2` — published 2026-09-01

Adds the `pytorch_fsdp` strategy (PyTorch FSDP2) and the training-loop change it required: no
collective and no host sync anywhere between forward and `optimizer_step()`. Rebuilt so the baked
`ml/` carries it — a client shipping no `ml/` of its own could not select `pytorch_fsdp` on v1.1, and
would still hit the double-checkpointing crash described in the commit.

| Check | Result |
|---|---|
| digest | `sha256:2614b47eb78d8d2c79fa21e3fb3fd061f2b4a23d7f555556883d70b6e5327c59` |
| `org.opencontainers.image.revision` | `b693ffa412c7650f23ec418384ec08c872eb1481`, no `-dirty` |
| FSDP2 present in baked `ml/` | yes — `distribution/pytorch_fsdp.py` |
| `pytorch_fsdp`, 30 steps, 8 GPUs | COMPLETED · 0.175 s/step · loss **0.0036232012789696455** |
| `fsdp`, 30 steps, 8 GPUs | COMPLETED · 0.241 s/step · loss **0.0038679696153849363** |
| `ddp`, 30 steps, 8 GPUs | COMPLETED · 0.158 s/step · loss **0.005165183916687965** |

All three gates ran with the client executing **from the v1.2 image with no `ml/` mount**, so
`find_ml_dir()` resolved to the image's own `/app/cray/ml`. `fsdp` and `ddp` returned their v1.1
fingerprints **bit-identically**, which is the check that the loop restructure was
behaviour-preserving for the existing strategies.

#### `mi355-v1.5` — published 2026-09-03

First AMD image built from the unified `amd-nvidia-merge` branch. The baked `ml/` gains the
classification `pad_token_id` fix (transformers raises at `batch_size > 1` without it) and the baked
requirements pin `sentence_transformers>=5.0,<6`; the `gpu_aware_mpi` shim directory and the
`local_training_config.yaml` sidecar are gone from the tree. v1.3/v1.4 remain published; they were
built from the pre-merge `amd` branch.

| Check | Result |
|---|---|
| digest | `sha256:37c536eb4b2fe9be6866b9139904caa29a99fa7c859ebcc49f6482a3aef51220` |
| `org.opencontainers.image.revision` | `218d2f9d31161e371205e61793ade4a140f87c7d`, no `-dirty` |
| imports (build-time guard + runtime) | `transformers 5.12.1`, `tokenizers 0.22.2`, `sentence_transformers 5.7.0` |
| `ddp`, 30 steps, 8 GPUs | COMPLETED · 0.161 s/step · loss **0.005165183916687965** |
| `pytorch_fsdp`, 30 steps, 8 GPUs | COMPLETED · 0.174 s/step · loss **0.0036232012789696455** |
| `fsdp`, 30 steps, 8 GPUs | COMPLETED · 0.243 s/step · loss **0.0038679696153849363** |
| classification, `batch_size: 2`, 30 steps | COMPLETED · loss 1.21 → 5.5e-07 (fails on v1.4's baked `ml/`) |

Server **and** client ran from the v1.5 image with no `ml/` mount, so both the server infra and the
baked `/app/cray/ml` are what was gated. All three strategy fingerprints are bit-identical to the
recorded values.

#### `mi355-v1.6` — published 2026-09-04

Built from `amd-nvidia-merge` @ `6bc7d82`, the same revision `h100-v1.5` was built from, so the two
hardware images now share one `repo/Dockerfile`. The only source change since v1.5 is that
Dockerfile: an explicit `scikit-learn` install plus an import gate for the embedding path, and an
NVIDIA-only venv `torchrun` shim guarded on `VLLM_TARGET_DEVICE=cuda`. Both are inert on ROCm — the
`rocm/primus` base already ships scikit-learn, and the AMD build passes `rocm` — so this image is
v1.5 with a new label. `ml/` is byte-identical to `53807f8` (git tree `5967c061…`).

| Check | Result |
|---|---|
| digest | `sha256:ee40edab891cd3208dd1a122d0d6472d698e955a9bd7e064bff624164e86bfa4` |
| `org.opencontainers.image.revision` | `6bc7d82a3a8a0185b88a3741e51272a92fd5d40e`, no `-dirty` |
| build gate | `transformers 5.12.1 + tokenizers 0.22.2 import OK`, `sklearn import OK 1.9.0` |
| `pip freeze` vs `mi355-v1.5` | identical — 432 packages, zero diff |
| `ddp`, 30 steps, 8 GPUs | COMPLETED · loss **0.005165183916687965** — bit-identical to v1.5 |

Server **and** client ran from the v1.6 image with no `ml/` mount. `fsdp` / `pytorch_fsdp` /
classification were not re-run: with an identical package set and identical `ml/`, the v1.5 records
above apply unchanged.

### Reproducible training fingerprints

Qwen3-0.6B · 128 records · 30 steps · `adapter_type: none` · `gpus: 8, nodes: 1`.

**There are two valid sets, and which one you get depends on whose `ml/` runs** — the client's copy
overrides the image's baked one (§3). The `pack()` tail-padding fix landed after `mi355-v1.0` was
published, and it changes the data: 390 tokens per epoch that were previously discarded are now
retained, taking the dataset from 52 blocks to 53.

As of `mi355-v1.2` the image's baked `ml/` matches this folder's, so a client that ships no `ml/` of
its own now produces the same numbers as one that does — verified by the v1.2 gates above. Only
`mi355-v1.0` still yields the old set.

Current source tree (this folder's `ml/`, post-fix):

| Strategy | Median step | Final loss (30 steps) |
|---|---|---|
| `fsdp` | 0.241–0.251 s | **0.0038679696153849363** |
| `ddp` | 0.160–0.182 s | **0.005165183916687965** |
| `pytorch_fsdp` | 0.177–0.181 s | **0.0036232012789696455** |

`pytorch_fsdp` (PyTorch FSDP2) is 1.36× faster than `fsdp` and reproduces bit-exactly across three
independent runs. Its loss differs from `fsdp` in the 3rd significant figure because the two shard at
different granularity, so the bf16 reductions accumulate in a different order — expected, not a
regression. The `fsdp` and `ddp` fingerprints above are **unchanged** by the FSDP2 work, which is the
check that the loss-reduction restructure it required was behaviour-preserving.

With `gradient_checkpointing: true` and `gradient_accumulation_steps: 4` (train.py's defaults) rather
than the benchmark's settings:

| Strategy | Median step | Final loss (30 steps) |
|---|---|---|
| `fsdp` | 1.355 s | 0.0054761627689003944 |
| `pytorch_fsdp` | 0.850 s | 0.008069735020399094 |

Published image `mi355-v1.0` with its own baked `ml/` (pre-fix):

| Strategy | Median step | Final loss (30 steps) |
|---|---|---|
| `fsdp` | 0.245–0.254 s | **0.067634217441082** |
| `ddp` | 0.164–0.174 s | **0.07000554352998734** |

Both sets reproduce bit-exactly within their own tree. A mismatch means the two `ml/` copies are
crossed — check which one the client uploaded before debugging anything else. See
`AWS_ScalarLM_port.md` §8 for why the values moved.

> **`0.0734812841` from an fsdp run means the FSDP gradient-mean fix did not load** — a stale image
> on one side (most likely the client, see §3). That exact value is the non-execution fingerprint.

All 8 GPUs confirmed engaged, VRAM sampled per device during a live FSDP run:
```
GPU0 258024 MB (also hosts vLLM)   GPU1-3 12910 MB   GPU4 12846 MB   GPU5-7 12782 MB
```

Also verified on this image: LoRA/FSDP completes (2,293,760 / 598,343,680 trainable = 0.38 %), and a
job explicitly requesting `flash_attention_2` now completes with `attn_implementation=sdpa` in the
log and zero HSA aborts (see §5).

---

## 5. Open items — carried verbatim, do not quietly drop

- **Multi-node now works, with one caveat.** A 2-node x 8-GPU job runs in containers
  (`ALLREDUCE OK world=16`) and real training completes 30 steps across both hosts and writes a
  checkpoint. The blocker was asymmetric RDMA device enumeration, which hangs RCCL silently; the
  launcher now derives `NCCL_IB_HCA` as the intersection of devices active on every node. Still
  untested: launching multi-node **via `sbatch`** — the runs above used torchrun directly, because
  this cluster's slurm registers a single node and its hostnames do not resolve for mpirun.
  See "Multi-node" in `readme_scalarlm.md`.
- **`gradient_accumulation_steps` > 1 never executed post-fix.** Every measured run used 1; the
  config default is 4.
- **Sharding is now proven at scale.** Qwen3-32B trains on both sharded strategies
  (`pytorch_fsdp` 0.523 s/step, `fsdp` 1.100 s) while **`ddp` OOMs outright** (254 GiB of 288 GiB) —
  so sharding is genuinely required, not just overhead. The older 0.6B VRAM figures still measure
  FSDP overhead rather than its benefit; do not use *those* for capacity planning.
- **No post-training inference/eval of a trained checkpoint.** "COMPLETED with a plausible loss" is
  the only end-to-end quality signal for the artifact itself.
- **DDP issues ~310 unbucketed, non-overlapped allreduces per step** (~9 % of a step at 0.6B). A
  known scaling cliff at production size.
- **Embedding mode now runs**, after adding `sentence_transformers` and fixing two undefined names
  in `load_embedding_dataset.py`. Two caveats: the dependency is **not yet in the image**, and
  CoSENT returns exactly 0.0 loss at `batch_size: 1` while still reporting COMPLETED — use
  `batch_size >= 2`.
- **`upload_to_hf` works, but is dead in the image.** A 3.58 GB checkpoint uploaded successfully
  once offline mode was suspended; the image sets `HF_HUB_OFFLINE=1`, and because the upload path
  swallows failures a blocked upload still reports COMPLETED. The upload now clears the offline
  flags for its duration, but the image default is still worth changing.
- **FA2 is downgraded, not supported.** `attn_implementation=flash_attention_2` is silently mapped
  to `sdpa`. It is not a supported training backend here: FA2 loads cleanly and then aborts
  mid-training with a device-side `HSA_STATUS_ERROR_EXCEPTION`, which is not a Python exception and
  cannot be caught by the `from_pretrained` fallback.

---

## 6. Publishing

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker login -u farbodatdocker          # paste a Personal Access Token, not the password
docker push farbodatdocker/scalarlm:mi355-v1.6
```

One tag. 58.9 GB on disk, materially less over the wire (§1). Resumable — re-run on a drop. Layers
shared with a base show `Mounted from …`; only new layers are `Pushed`. The source revision travels
inside the image as a label, so the single tag loses nothing (§1).

If Docker runs as root on your host, prefix `login` and `push` with `sudo` consistently — otherwise
they use different credential stores and the push fails on auth after uploading.

### Before publishing publicly — read this

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

Also present: `/app/cray/vllm/.git` (245 MB). Not a secret — the vLLM fork's public remote, no
embedded credentials — but it is pure image bloat and a candidate for removal in a future squashed
build.

---

## 7. Where the full source lives, and cutting the next release

Everything needed to rebuild is in this folder. There is no separate repository to track.

### What is here

| Path | What it is |
|---|---|
| `ml/` | **the training recipe — the one canonical copy.** Edit this. |
| `repo/` | the full ScalarLM server source with the AMD integration applied |
| `build_image.sh` | stages `ml/` into `repo/` and builds the image |
| `scalarlm_mi355.patch` | the integration as one patch against public upstream |

| `train.py`, `inference.py`, `data/` | the client |

`repo/` is a verbatim upstream tree at commit `4566a84` with 11 commits of AMD integration applied —
61 files changed, of which only 15 are under `ml/`. The other 46 are why this folder needs the
server source at all: `infra/cray_infra/training/distributed.py` (the whole NCCL/RCCL collectives
layer), `scripts/train_job_entrypoint.sh`, the `Dockerfile`, and the test suite.

**`repo/` has no `.git`.** A nested repository would be committed as a gitlink — a submodule pointer
to commits on no remote, which clones as an empty directory.

**`repo/ml/` is deliberately absent from version control.** It is created by `build_image.sh` from
`ml/`. One tree, one source of truth; see §2.

### Recovering the development history

The branch-level history (`baseline-pre-amd`, `base-ddpfix`, `amd-pr-5`, `amd-integration`) was kept
as `scalarlm_amd_history.bundle`. It is **no longer tracked in this repo**: `git bundle verify`
reports it "records a complete history", i.e. it is a full clone of a repository whose base is the
**public** upstream, so it carried ~9.3 MB to preserve a branch topology that the patch below and the
vendored trees already reproduce. If you have a copy, it is still a complete clone source:

```bash
git clone scalarlm_amd_history.bundle scalarlm-history
cd scalarlm-history
git log --oneline 4566a84..amd-integration      # the 11 integration commits
git diff baseline-pre-amd..amd-integration      # everything the integration changed
```

### Reconstructing from upstream instead

`scalarlm_mi355.patch` is the same integration as a single patch, verified to reproduce the tree
**exactly** (git tree-hash equality). Use it when you want the change on top of a fresh upstream
clone rather than this vendored copy:

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

To rebase, clone the bundle, add upstream as a remote, and rebase `amd-integration` onto the newer
commit. The hunks most likely to conflict are the ones this integration rewrote:
`infra/cray_infra/training/distributed.py` (replaced wholesale),
`ml/cray_megatron/megatron/distribution/{ddp,fsdp}.py` (the gradient-mean fixes), and
`scripts/train_job_entrypoint.sh` (the torchrun launcher). Re-run gate 5 afterwards — the
gradient-semantics tests are what catch a silently dropped fix. Then copy the rebased tree back over
`repo/` and regenerate the patch and bundle.
