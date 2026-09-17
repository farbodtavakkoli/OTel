# ScalarLM server image — AMD Instinct MI355X (ROCm)

Run, build, customize, and publish the **MI355X (gfx950, CDNA 4)** server image. One
container serves both the vLLM inference endpoint and the Megatron-via-Slurm training
endpoint, with this folder's `ml/` recipe baked in. Client usage is in
[`../readme_scalarlm.md`](../readme_scalarlm.md); the NVIDIA image is in
[`DOCKER_IMAGE_H100.md`](DOCKER_IMAGE_H100.md).

## Image

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:mi355-v1.7` | AMD MI355X, gfx950 / ROCm 7.2.4 |

58.9 GB on disk, ~15 GB compressed over the wire.

```bash
docker pull farbodatdocker/scalarlm:mi355-v1.7
```

There is deliberately one MI355X tag, so the tag does not tell you what is inside. The
source revision is recorded as an OCI label — cite that hash, not the tag, in a bug report:

```bash
docker inspect farbodatdocker/scalarlm:mi355-v1.7 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
# mi355-v1.7 -> b9f27453b38839f4287a8e8e01cc323402085763
# digest     -> sha256:0f9ebeee1f8871b280f421ec915ea5f66cbeeaa74b6db38b8a9ab1793846961e
```

`org.opencontainers.image.created` gives the build timestamp. If you rebuild and republish
under the same tag, pass the new revision to `--label` or this check goes stale.

## Run

The image is self-starting — a bare `docker run` serves:

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

curl -s localhost:8000/v1/health
# {"api":"up","vllm":"up","megatron":"up","all":"up"}
```

Passing a script still overrides the default, so existing invocations keep working:
`docker run ... farbodatdocker/scalarlm:mi355-v1.7 /app/cray/scripts/start_one_server.sh`.

### AMD flags that are not optional

| Flag / env | Why |
|---|---|
| `--device=/dev/kfd --device=/dev/dri` | ROCm device access (the AMD analogue of `--gpus`) |
| `--group-add video` | render-node permissions |
| `--shm-size=64g` | RCCL shared-memory transport; the 64 MB default deadlocks collectives |
| `--security-opt seccomp=unconfined` | HIP needs syscalls the default profile blocks |
| `SCALARLM_MAX_GPUS_PER_NODE=8` | **Default is 1.** Without it every job is silently capped to a single GPU no matter what `gpus` says. |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | optional; without network the boot can otherwise stall in tokenizer loading |

For multi-GPU training submit `gpus: 8, nodes: 1` in `train_args`.

### Serving and config overrides

- Base model: tensor-parallel across GPUs with `SCALARLM_TENSOR_PARALLEL_SIZE=<n>`.
- Fine-tuned adapters: serve at **tensor-parallel size 1** — vLLM's TP path rejects the
  hot-reloaded un-sharded adapter state dict.
- Any `default_config.py` key is overridable as `SCALARLM_{KEY.upper()}`, which is why
  `SCALARLM_MAX_GPUS_PER_NODE` and `SCALARLM_TENSOR_PARALLEL_SIZE` work without appearing
  as literals in the source. Also useful: `SCALARLM_ENABLE_LORA` (default `false`) and
  `SCALARLM_GPU_MEMORY_UTILIZATION` (default `0.40`, the headroom that lets training and
  vLLM share GPU 0).
- If the HF cache is on a network mount, add `-e HF_DATASETS_CACHE=/tmp/hf_datasets`. The
  `datasets` file lock returns `EACCES` on SMB for the second rank, so a multi-GPU job
  fails on rank 1 while rank 0 looks healthy.

### Helm chart

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: mi355-v1.7          # AMD MI355X / gfx950; use the H100 tag for NVIDIA
  pullPolicy: Always
```

The chart's stock `values.yaml` points at `gdiamos/scalarlm-nvidia-12.0`, an NVIDIA image
that will not run on this hardware — override the image block. The AMD device flags above
have no `docker run` equivalent in a pod spec: request GPUs through the AMD device plugin
(`amd.com/gpu`), mount `/dev/kfd` and `/dev/dri`, and size `/dev/shm` — a pod's default
64 MB deadlocks RCCL exactly as the Docker default does.

## Customizing the training recipe (`ml/`)

You rarely need to rebuild or overlay anything. The SDK ships whichever copy of `ml/` sits
next to the client and the server extracts it into the job directory; the image's
`/app/cray/ml` is used **only if the upload contained no `ml/`**.

| You run the client from | Which `ml/` trains |
|---|---|
| `training/llm/scalarlm/` | yours — edit `ml/`, rerun, done. No rebuild. |
| anywhere without an `ml/` dir | the image's baked default |

Because the client's copy wins, a fix baked into the server image does nothing if your
client folder still has an old `ml/`: the job runs the old code and still reports
COMPLETED. If a change seems to have no effect, check which `ml/` the client sent before
debugging anything else.

An overlay can only reach `ml/`:

```dockerfile
FROM farbodatdocker/scalarlm:mi355-v1.7
COPY ml/ /app/cray/ml/          # changes the DEFAULT for clients that have no ml/
```

Anything outside `ml/` — the collectives layer
(`infra/cray_infra/training/distributed.py`), the launcher, the Slurm wiring, the
ROCm/PyTorch versions — needs a full rebuild from `repo/`.

## Build from scratch

Build from `repo/`'s Dockerfile, not by overlaying a published base: the AMD integration
touches `ml/`, `infra/cray_infra/`, `scripts/` and `test/`.

```bash
rocminfo | grep -m1 gfx            # gfx950
git status --short                 # empty, or the image is labelled -dirty

cd training/llm/scalarlm/docs
./build_image.sh                          # or: IMAGE_TAG=mi355-v1.7 ./build_image.sh
```

Building by hand, do the three things the script does — stage `ml/` into `repo/ml/`
(gitignored, not source; staging is what makes the image bake the current recipe), create
`repo/vllm/` (the Dockerfile bind-mounts `./vllm`; a missing directory fails the build with
`failed to compute cache key: "/vllm": not found`, and it is intentionally empty because
`VLLM_SOURCE` defaults to `remote`), and label the revision:

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

- `--network=host` is required; without it the in-build vLLM `git clone` fails TLS.
- `TORCH_CUDA_ARCH_LIST=gfx942` is correct despite the name (the build arg is inherited
  from the CUDA path). Inside the Dockerfile `PYTORCH_ROCM_ARCH` governs and includes
  gfx950, so gfx950 kernels are produced. Do not "fix" it to `gfx950` without re-running
  the arch check below.

Verify the arch functionally. `roc-obj-ls <lib> | grep gfx950` is **not** a valid gate on
this image — it reports 0 bundles for every library, including the working
`libtorch_hip.so`:

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

Add the self-start CMD and the labels with a metadata-only build (no layers rebuilt):

```bash
BUILDDIR=$(mktemp -d)
cat > "$BUILDDIR/Dockerfile" <<EOF
FROM cray:mi355-build
CMD ["/app/cray/scripts/start_one_server.sh"]
LABEL org.opencontainers.image.authors="Farbod Tavakkoli"
LABEL org.opencontainers.image.source="https://github.com/farbodtavakkoli/OTel"
LABEL org.opencontainers.image.revision="$(git rev-parse HEAD)"
EOF
docker build -t farbodatdocker/scalarlm:mi355-v1.7 "$BUILDDIR"
rm -rf "$BUILDDIR"
```

`$(git rev-parse HEAD)` is expanded by the shell before the heredoc is written, so the
label records the revision you actually built from.

## Acceptance checks

Run these against the built image with no source mounts unless noted.

| Gate | Check | Expected |
|---|---|---|
| 0a | corrections present in image | all AMD fixes + all H100 ports found |
| 0b | gfx950 kernels | the functional probe above: MI355X matmul + `gfx950` in the arch list |
| 1 | unit suite in-image — `./run_unit_tests.sh <tag>` | 811 passed, 2 deselected (`WITH_CMD=1`: 813 passed) |
| 2 | collective correctness — `python3 -m pytest test/infra/distribution_strategy -q` | passes |
| 3 | end-to-end, image as both server and client | losses match the fingerprints below |
| 4 | serving regression — `python3 -m pytest test/integration/api/test_slurm_api.py -q` | 5 passed |
| 5 | legacy collective jobs, 8 ranks | 5/5 `RESULT: ... passed` |

Use `./run_unit_tests.sh` for gate 1 rather than calling `pytest` directly: a bare
`python3 -m pytest test/unit` reports 18 failures that are harness artefacts — 16 FSDP
tests need the `torchrun` rendezvous variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`MASTER_ADDR`, `MASTER_PORT`), and 2 read `/app/cray/cmd/test_command.sh`, which the
Dockerfile deliberately does not ship. The script sets the rendezvous variables, deselects
those two, and picks the GPU flags for the tag's vendor; `WITH_CMD=1` bind-mounts
`repo/cmd` and runs them.

### Reproducible training fingerprints

Qwen3-0.6B, 128 records, 30 steps, `adapter_type: none`, `gpus: 8, nodes: 1`, with this
folder's `ml/`:

| Strategy | Final loss (30 steps) |
|---|---|
| `fsdp` | `0.0038679696153849363` |
| `ddp` | `0.005165183916687965` |
| `pytorch_fsdp` | `0.0036232012789696455` |

These reproduce bit-exactly. A mismatch means the client's `ml/` and the image's baked copy
are crossed — check which one the client uploaded before debugging anything else.

## Caveats

- **At large model sizes use a sharded strategy.** Qwen3-32B trains under `pytorch_fsdp`
  and `fsdp`; `ddp` OOMs outright.
- **Embedding mode needs `batch_size >= 2`** and `sentence_transformers`, which is not in
  the image. CoSENT returns exactly 0.0 loss at `batch_size: 1` while still reporting
  COMPLETED.
- **`upload_to_hf` is effectively dead in the image.** It sets `HF_HUB_OFFLINE=1`, and the
  upload path swallows failures, so a blocked upload still reports COMPLETED.
- **`flash_attention_2` is downgraded, not supported.** It is mapped to `sdpa`. Forced, it
  loads cleanly and then aborts mid-training with a device-side `HSA_STATUS_ERROR_EXCEPTION`
  that is not a Python exception and cannot be caught.
- The fingerprints above all use `gradient_accumulation_steps: 1`, while the config default
  is 4.
- Multi-node runs (2 nodes x 8 GPUs) complete in containers; launching multi-node via
  `sbatch` is not covered — use torchrun directly unless your Slurm registers every node
  and its hostnames resolve for mpirun. See the multi-node notes in
  [`../readme_scalarlm.md`](../readme_scalarlm.md).

## Publishing

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker login -u farbodatdocker          # paste a Personal Access Token, not the password
docker push farbodatdocker/scalarlm:mi355-v1.7
```

The push is resumable — re-run it on a drop. If Docker runs as root on your host, prefix
`login` and `push` with `sudo` consistently, or they use different credential stores and
the push fails on auth after uploading.

**Before publishing publicly:** `repo/infra/slurm_configs/munge.key` and `slurm.key` are
static secrets committed upstream, so they are in this vendored `repo/` tree and — via
`COPY ./infra` — in every image built from it. A MUNGE key is the shared secret Slurm uses
to authenticate, so anyone holding one can forge Slurm credentials against a cluster that
trusts it. Exposure is low for a single container (slurmctld and slurmd are not published);
for any multi-node deployment treat the baked-in pair as compromised and regenerate per
cluster.

## Full source

Everything needed to rebuild is in this folder; there is no separate repository. Paths
relative to `training/llm/scalarlm/`:

| Path | What it is |
|---|---|
| `ml/` | the training recipe — the one canonical copy. Edit this. |
| `repo/` | the full ScalarLM server source with the AMD integration applied |
| `docs/build_image.sh` | stages `ml/` into `repo/` and builds the image |
| `docs/run_unit_tests.sh` | runs acceptance gate 1 against a built image |
| `docs/scalarlm_mi355.patch` | the integration as one patch against public upstream |
| `train.py`, `inference.py`, `data/` | the client |

`repo/` is a verbatim upstream tree at commit `4566a84` with the integration applied. It
has no `.git`, and `repo/ml/` is created by `build_image.sh`. Reconstruct it from upstream
with:

```bash
git clone https://github.com/supermassive-intelligence/scalarlm.git
cd scalarlm && git checkout 4566a84
git apply /path/to/scalarlm_mi355.patch
```

### Release checklist

1. Clean tree (`git status --short` empty), or the image is labelled `-dirty`.
2. `./build_image.sh` — it stages `ml/` and labels the revision.
3. `docker inspect` reports the revision you just built.
4. Acceptance checks above: at minimum the unit suite, the 8-rank collective check, and one
   `fsdp` plus one `ddp` run whose final loss matches the fingerprints.
5. Bare `docker run` with no command reaches `{"all":"up"}`.
6. Push, one tag.
7. Tag the commit, so the published image maps to a named source revision.

To rebase onto a newer upstream, apply `scalarlm_mi355.patch` to a fresh clone at that
commit and resolve rejected hunks by hand — the likely conflicts are
`infra/cray_infra/training/distributed.py` (replaced wholesale),
`ml/cray_megatron/megatron/distribution/{ddp,fsdp}.py` (the gradient-mean fixes), and
`scripts/train_job_entrypoint.sh` (the torchrun launcher). Re-run gate 5 afterwards, then
copy the rebased tree over `repo/` and regenerate the patch.
