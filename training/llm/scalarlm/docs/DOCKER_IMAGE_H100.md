# ScalarLM server image: NVIDIA H100 (CUDA)

Run, build, extend and push the **H100 (`sm_90`)** server image. One container serves vLLM
inference and Megatron training through Slurm, with this folder's `ml/` baked in. Client usage
is in [`../readme_scalarlm.md`](../readme_scalarlm.md). The AMD image and the shared gates are in
[`DOCKER_IMAGE_MI355.md`](DOCKER_IMAGE_MI355.md).

## Image

See the [canonical hardware image table](../readme_scalarlm.md#server-image).
`farbodatdocker/scalarlm:h100-v2.5` is local only and has not been pushed.
Do not pull it or configure a remote registry deployment until the owner publishes it.

| Provenance | Value |
|---|---|
| Code revision | `f5fb130d262a3ea58e905fcc2e3284f11218a884` |
| Local image ID | `sha256:eba4a92a5824344d0f52c30dc08e6332767e6c01022f0a20316ec5512e33b9d5` |
| Recipe | `../repo/Dockerfile.h100-v2.5`, overlay using an already cached base |
| Maintainer and author | `Farbod Tavakkoli <farbodtavakoli@gmail.com>` |

```bash
docker image inspect farbodatdocker/scalarlm:h100-v2.5 \
  --format '{{.Id}} {{json .Config.Labels}}'
```

Use the image ID for reproducible validation. Later test fixture corrections are working
tree changes, not part of this image.

## Component versions

This table preserves the original CUDA stack record independently of image tags.
It is historical dependency guidance, not a fresh inventory of the v2.5 image.
Inspect the actual image before relying on these versions.

| Component | Version |
|---|---|
| Base | `nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, no CUDA compat layer |
| PyTorch | `2.13.0+cu129`, triton 3.7.1 |
| vLLM | upstream v0.32.0 plus `scripts/vllm_patches/vllm-v0.32.0-scalarlm.patch` |
| transformers | 5.19.0 (pin `>=5.17,<5.20`), tokenizers 0.23.3 |
| opencv | 5.0.0.93 built against system OpenSSL 3, no bundled OpenSSL |
| Extras | `mlflow-skinny` |
| EFA | EFA installer 1.51.0 userspace: libfabric `2.7.0amzn2.0`, `aws-ofi-nccl` 1.22.0, `rdma-core` 65 |

The original stack required a driver supporting CUDA 12.9 or newer.

## Run

The image starts the server on its own:

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v2.5

curl http://localhost:8000/v1/health
# {"api":"up","vllm":"up","all":"up"}
```

Baked runtime config: `ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]`,
`CMD ["/app/cray/scripts/start_one_server.sh"]`, `WORKDIR /app/cray`.

Notes:

1. `--shm-size=64g` is required for jobs with several ranks. The 64 MB Docker default kills a
   rank with SIGBUS, and Slurm then relaunches it in a loop. `docker-compose.yaml` sets
   `shm_size: 64gb`, but a raw `docker run` must pass the flag.
2. For training on several GPUs, pass them with `--gpus` and set `SCALARLM_MAX_GPUS_PER_NODE`.
   It defaults to 1. Ranks map to GPUs by local rank. If the HF cache is on an SMB or NFS mount,
   add `-e HF_DATASETS_CACHE=/tmp/hf_datasets`, or the `datasets` file lock returns EACCES for
   the second rank.
3. Serve fine tuned adapters at tensor parallel size 1. The base model can use
   `SCALARLM_TENSOR_PARALLEL_SIZE`.
4. Write job output to fast local disk. Checkpoint save time must stay below the NCCL timeout
   configured by `process_group_timeout_seconds` (default 3600 seconds). The current ordinary
   AdamW smoke checkpoints were 170,973,462,868 bytes each.
5. LoRA runs under `pytorch_fsdp` save only trainable weights. Current image LoRA checkpoint
   size remains unmeasured. Full parameter runs save the whole model.
6. `flash_attention_2` is mapped to `sdpa` by the loader.
7. On a host without the NVIDIA container toolkit, pass `/dev/nvidia*` and the driver libraries
   with `--device` and `-v` instead of `--gpus`, and run `ldconfig` in the container. The
   entrypoint then prints "driver was not detected", yet `torch.cuda.is_available()` is true.
8. The recipe configures `FI_PROVIDER=efa` and `FI_EFA_USE_DEVICE_RDMA=1`.
   Historically NCCL loaded the plugin automatically: EFA hosts logged `NET/OFI` with
   provider `efa`; hosts without EFA logged `aws-ofi-nccl initialization failed` once
   per rank and fell back to InfiniBand or sockets. The original InfiniBand test recorded
   bus bandwidth falling from 10 to 3 GB/s when `FI_PROVIDER=efa` was removed, because
   the plugin took over through libfabric. This is historical context, not a v2.5 result.
   Preserve the provider setting and verify transport with `NCCL_DEBUG=INFO` on the target
   hardware. EFA and multiple node behavior have not been validated on this image.
9. When the server shuts down it kills only its own process tree (the uvicorn supervisor, the
   worker, vLLM and their children), so other processes in the same container survive.

## Current acceptance evidence

Recorded smoke evidence for `meta-models/Muse-Glimmer-30B`:

| Check | Result and limit |
|---|---|
| SFT | 20 optimizer updates, 512 complete sample conversations, 1024 token context, GPUs 0, 1, 2, 3, 5, 6 and 7; all text parameters trained, vision frozen |
| Precision | Ordinary BF16 AdamW, FP32 reductions and chunked loss; no convergence equivalence claim |
| Checkpoints | Indices 10 and 19 validated; each 170,973,462,868 bytes, all 1,436 model keys checked |
| Full Hugging Face export | All 1,436 keys exactly match the checkpoint |
| Serving | TP 1, PP 7, HTTP 200, 64 generated tokens; role marker and token limit truncation, mechanics only |
| Unit suite | 1,039 passed, 70 skipped with patched test mounts, a test only `/app/ml` alias, checkout `cmd/test_command.sh` and single rank `torchrun` |
| Native image unit suite | Untested without those mounts; fixture corrections are not baked into the image |
| GRPO | 80 saved rollouts, zero optimizer updates; original attempt blocked by HTTP 401 |
| Endurance | Prepared but not run; no long duration acceptance evidence |
| Multiple nodes and EFA | Not validated on this image |

Local artifacts outside the repository: `../../../../../smoke/sft20_7gpu_cleaned/run01/`,
including `checkpoint_verification.json`, `export_verification.json` and `serving_query.json`.
The local unit log is `../../../../../smoke/logs/h100_v25_full_unit_gpu_inspector.log`.
These paths are relative to this document and are not shipped repository artifacts.
Refreshed credentials passed a separate API check; GRPO and endurance have not resumed.
The sample does not validate the full training dataset or answer quality.

## Unit test runner and historical evidence

Run gate 1 through `docs/run_unit_tests.sh`, from this directory, with an explicit tag:

```bash
./run_unit_tests.sh h100-v2.5
WITH_CMD=1 ./run_unit_tests.sh h100-v2.5
```

These are unverified v2.5 guidance, not commands executed for the current evidence above.
The recorded v2.5 suite needed additional test mounts and single rank `torchrun`.
The runner supplies rendezvous variables and either deselects the two command tests
or mounts `repo/cmd` when `WITH_CMD=1`. It uses `--gpus all`; without the container
toolkit use the manual device passthrough described in run note 7.

Historically bare `python3 -m pytest test/unit` reported 18 harness failures on both
vendor images: FSDP tests needed `torchrun` rendezvous variables and two tests read
`cmd/test_command.sh`, which the Dockerfile did not ship. The original runner handled
those issues. This count is historical, not a current v2.5 failure count.

Original acceptance record at revision `320f00f4c951619d0c4d2000d7d9cc71b0c6d698`,
using `meta-models/Muse-Glimmer-30B`, not v2.5:

| Check | Historical result |
|---|---|
| Unit suite | 855 passed, 2 deselected; original `WITH_CMD=1` expectation was 857 passed |
| `hf_xet` | 1.8.0, import and Xet availability checks passed |
| Full parameter SFT, 8 GPUs | 5 steps, final loss 0.0047, completed with exit 0 |
| Final SFT checkpoint | `checkpoint_4.pt`, about 171 GB, saved successfully |
| Persistent LoRA GRPO, 8 GPUs | 5 cycles, completed with exit 0 |

These historical tiny sample smoke tests are not model quality evaluations.
MLflow uploads were unavailable and metrics stayed in logs. An earlier five record
SFT attempt left three of eight ranks empty and timed out; use at least one record
per rank. The final GRPO server kill was expected teardown.

Earlier historical record at revision `e93b0fb1384ef11cb1afa5de90a314c9b9cb2c4d`,
8 H100 GPUs, driver 580, `meta-models/Muse-Glimmer-30B`, not v2.5:

| Check | Historical result |
|---|---|
| Build time torch stack checks | Passed after base, vLLM and infra layers |
| `/etc/ld.so.conf.d/` | No `cuda-compat` entry |
| Compressed size | 13.35 GB; original AI Runtime limit 20 GB |
| `import cv2` with `OPENSSL_FORCE_FIPS_MODE=1` | OK |
| Unit suite, 1 GPU visible | 837 passed, 2 deselected, 0 skipped |
| vLLM generation, TP 1 | Correct answers, about 75 tok/s single stream |
| Full parameter SFT, `pytorch_fsdp`, 8 GPUs, 4096 tokens | Step 0 16.5 s, then about 3.1 s/step, 55.5 GiB peak |
| NCCL all reduce, 8 GPUs, NVLink | 242 GB/s bus bandwidth |
| NCCL over InfiniBand, no EFA | `NET/IB`, 10.2 GB/s |
| Server shutdown after vLLM engine failure | Exit 137; unrelated Python process survived |
| LoRA GRPO, 4 steps of 4 prompts x 4 completions | About 80 s/step, 0 rollout errors, old log probs 16/16, ratio 0.996 to 1.0003, reward 0.12 to 0.62 |

That host had InfiniBand but no EFA; the original planned two node AI Runtime check
was not evidence of EFA validation. Gates 2 to 5 in the AMD runbook were not rerun
for that historical record and are not claimed for v2.5. Gates 3 to 5 require a running
server; gate 3 must run with no client `ml/` directory to test the baked recipe.
Gate 2 counts vary by vendor because tests marked `arch="rocm"` are AMD only.
An older integration record had `test/integration/api/test_vllm_api.py::test_lora_adapter_endpoints`
returning HTTP 500 instead of 4xx for a nonexistent adapter path. That historical
failure was not rechecked for the later records and is not a current v2.5 claim.

## Exact overlay build

Use a clean checkout of revision `f5fb130d262a3ea58e905fcc2e3284f11218a884`
with the recipe's base already cached. Run from that checkout's `training/llm/scalarlm`:

```bash
set -e
REVISION=$(git rev-parse HEAD)
test "$REVISION" = f5fb130d262a3ea58e905fcc2e3284f11218a884
test -z "$(git status --porcelain --untracked-files=no)"
diff -qr ml repo/ml
docker build --pull=false -f repo/Dockerfile.h100-v2.5 \
  --build-arg REVISION="$REVISION" \
  -t farbodatdocker/scalarlm:h100-v2.5 repo
docker image inspect farbodatdocker/scalarlm:h100-v2.5 \
  --format '{{.Id}} {{json .Config.Labels}}'
```

The recipe copies `ml/`, infrastructure, scripts and unit tests from `repo/`;
it removes obsolete optimizer files and stamps the real code revision plus author labels.
Do not use the generic NVIDIA target and call it this overlay image.
Rebuilding from later corrected tests produces a different image and requires its own
committed code hash and validation; do not label changed code with the original revision.
No build or GPU validation was run during this documentation update.

## Build from scratch

The generic `build_image.sh` stages canonical `ml/` into `repo/`, builds the NVIDIA
configuration of `repo/Dockerfile` and stamps provenance labels. It is a different
recipe from the verified v2.5 overlay and needs its own tag and acceptance evidence.
From the OTel repository root:

```bash
cd training/llm/scalarlm/docs
TARGET=nvidia IMAGE_TAG=h100-custom ./build_image.sh
```

The original CUDA 13 variant remains untested:

```bash
EXTRA_BUILD_ARGS="--build-arg CUDA_VERSION=13.0.3 --build-arg TORCH_CUDA_INDEX=cu130" \
  TARGET=nvidia IMAGE_TAG=h100-custom-cu130 ./build_image.sh
```

Build from a clean committed checkout. The helper appends a dirty suffix to the revision
when source changes are present or `REVISION` is supplied; do not publish that as verified
provenance. The Dockerfile uses BuildKit `RUN --mount` cache syntax, so the host needs
`docker buildx` and `DOCKER_BUILDKIT=1`. Behind a proxy, unset the proxy variables first:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export DOCKER_BUILDKIT=1
```

Use `sudo docker` when Docker runs as root. The original vLLM compile took about
50 minutes; this is historical timing, not a current build estimate.
Architectures other than `sm_90` require recompiling vLLM's `_C` extension with the
appropriate `TORCH_CUDA_ARCH_LIST`.

The shared collective layer is `cray_infra.training.distributed` in
`repo/infra/cray_infra/training/distributed.py`, using `torch.distributed` and NCCL.
It requires `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, set by
`repo/scripts/train_job_entrypoint.sh`.

## Build on top of a published image

After the owner publishes an image, verify its revision and substitute its immutable
reference below. The local v2.5 image has not been published. This custom overlay example
changes only `ml/` and the default model; it does not reproduce the v2.5 recipe.

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY ml/ /app/cray/ml/
ENV SCALARLM_MODEL=Qwen/Qwen3-0.6B
```

```bash
docker build --build-arg BASE_IMAGE='<published-image>@sha256:<digest>' \
  -t <your-namespace>/scalarlm:my-tag ./my-scalarlm
```

Changes outside `ml/`, including collectives, launcher, Slurm, CUDA or PyTorch, need a
full recipe build rather than this minimal overlay. All build examples are guidance only;
none was executed during this documentation update.

## Push to a registry

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
read -rsp 'Docker Hub token: ' DOCKER_TOKEN; printf '\n'
printf '%s' "$DOCKER_TOKEN" | docker login -u farbodatdocker --password-stdin
unset DOCKER_TOKEN
docker push farbodatdocker/scalarlm:h100-v2.5
```

These commands are for the owner to run; this image has not been pushed.
Docker Hub needs a Personal Access Token as the password. Compressed upload size has not
been measured for this image. With `sudo docker`, use `sudo` for both `login` and `push`.
