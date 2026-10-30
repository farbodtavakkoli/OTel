# ScalarLM server image: NVIDIA H100 (CUDA)

Run, build, extend and push the **H100 (`sm_90`)** server image. One container serves vLLM
inference and Megatron training through Slurm, with this folder's `ml/` baked in. Client usage
is in [`../readme_scalarlm.md`](../readme_scalarlm.md). The AMD image and the shared gates are in
[`DOCKER_IMAGE_MI355.md`](DOCKER_IMAGE_MI355.md).

## Image

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:h100-v1.9` | NVIDIA H100 (`sm_90`), GRPO, native Muse Glimmer, EFA |
| `farbodatdocker/scalarlm:mi355-v1.8` | AMD MI355X (ROCm), see the other runbook |

```bash
docker pull farbodatdocker/scalarlm:h100-v1.9
```

`h100-v1.9` supports `ddp`, `fsdp` and `pytorch_fsdp`, LoRA, full parameter training and the
GRPO mode, and serves and trains `meta-models/Muse-Glimmer-30B`. It replaces `h100-v1.8` and adds
EFA for multi node NCCL and a server shutdown that stops only its own processes. `h100-v1.8`
replaced `h100-v1.7`.

| Component | Version |
|---|---|
| Base | `nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, no CUDA compat layer |
| PyTorch | `2.13.0+cu129`, triton 3.7.1 |
| vLLM | upstream v0.32.0 plus `scripts/vllm_patches/vllm-v0.32.0-scalarlm.patch` |
| transformers | 5.19.0 (pin `>=5.17,<5.20`), tokenizers 0.23.3 |
| opencv | 5.0.0.93 built against system OpenSSL 3, no bundled OpenSSL |
| Extras | `mlflow-skinny` |
| EFA | EFA installer 1.51.0 userspace: libfabric `2.7.0amzn2.0`, `aws-ofi-nccl` 1.22.0, `rdma-core` 65 |

The image runs natively on drivers with CUDA 12.9 or newer. Identify a pulled image by its
revision label, not the tag:

```bash
docker inspect farbodatdocker/scalarlm:h100-v1.9 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
# h100-v1.9 -> e93b0fb1384ef11cb1afa5de90a314c9b9cb2c4d
# h100-v1.8 -> c44453419e41d831b7caaebce7ebcedc3f608468
# v1.8 digest -> sha256:8094eb166244ae6c9087944f863a10272860a4e747dbd65ef195812899aa7709
```

The `h100-v1.8` label `c444534` is a local build commit. On GitHub the same image code is commit
`36c54e359456bb1be4cdb964d36c9197198749cd`.

Helm chart:

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: h100-v1.9
  pullPolicy: Always
```

## Run

The image starts the server on its own:

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.9

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
4. Write job output to fast local disk. The final full parameter checkpoint (about 171 GB for
   Glimmer 30B) must finish inside the 10 minute NCCL barrier timeout.
5. LoRA runs under `pytorch_fsdp` save only the trainable weights (about 1.3 GB per step for
   Glimmer 30B). Full parameter runs save the whole model.
6. `flash_attention_2` is mapped to `sdpa` by the loader.
7. On a host without the NVIDIA container toolkit, pass `/dev/nvidia*` and the driver libraries
   with `--device` and `-v` instead of `--gpus`, and run `ldconfig` in the container. The
   entrypoint then prints "driver was not detected", yet `torch.cuda.is_available()` is true.
8. Multi node over EFA needs no job setting. The image sets `FI_PROVIDER=efa` and
   `FI_EFA_USE_DEVICE_RDMA=1`, and NCCL loads the plugin on its own. With `NCCL_DEBUG=INFO`, an
   EFA host logs `NET/OFI` with provider `efa`. A host without EFA logs
   `aws-ofi-nccl initialization failed` once per rank, and NCCL falls back to InfiniBand or sockets. Keep
   `FI_PROVIDER=efa` on InfiniBand hosts: without it the plugin takes over through libfabric and
   cut bus bandwidth from 10 to 3 GB/s in our test.
9. When the server shuts down it kills only its own process tree (the uvicorn supervisor, the
   worker, vLLM and their children), so other processes in the same container survive.

## Acceptance checks

Run gate 1 with the runner script, not `pytest` directly:

```bash
./run_unit_tests.sh h100-v1.9              # expect: 837 passed, 2 deselected
WITH_CMD=1 ./run_unit_tests.sh h100-v1.9   # also runs the two cmd/ tests: 839 passed
```

A bare `python3 -m pytest test/unit` reports 18 failures on either vendor's image. They come
from the harness: the FSDP tests need the `torchrun` rendezvous variables, and two tests read
`cmd/test_command.sh`, which the Dockerfile does not ship. The script handles both.

The script uses `--gpus all`. On a host without the container toolkit, run the same command it
runs inside the container with the manual passthrough from note 7.

Results on 8xH100 (driver 580) for `h100-v1.9`, model `meta-models/Muse-Glimmer-30B`:

| Check | Result |
|---|---|
| Build time torch stack checks | pass after base, vLLM and infra layers |
| `/etc/ld.so.conf.d/` | no `cuda-compat` entry |
| Compressed size | 13.35 GB (AI Runtime limit 20 GB) |
| `import cv2` with `OPENSSL_FORCE_FIPS_MODE=1` | OK |
| Unit suite, 1 GPU visible | 837 passed, 2 deselected, 0 skipped |
| vLLM generation, TP 1 | correct answers, about 75 tok/s single stream |
| Full parameter SFT, `pytorch_fsdp`, 8 GPUs, 4096 tokens | step 0 16.5 s, then about 3.1 s/step, 55.5 GiB peak |
| NCCL all reduce, 8 GPUs, NVLink | 242 GB/s bus bandwidth (`h100-v1.8`: 231 GB/s) |
| NCCL over InfiniBand, no EFA | stays on `NET/IB`, 10.2 GB/s (`h100-v1.8`: 10.0 GB/s) |
| Server shutdown after the vLLM engine dies | exits 137, an unrelated Python process in the container survives (`h100-v1.8` killed it) |
| LoRA GRPO, 4 steps of 4 prompts x 4 completions | about 80 s/step, 0 rollout errors, old log probs 16/16, ratio 0.996 to 1.0003, reward 0.12 to 0.62 |

EFA itself was not tested here: this host has InfiniBand and no EFA devices. The 2 node run on
AI Runtime is the check for it.

Gates 2 to 5 from `DOCKER_IMAGE_MI355.md` were not rerun on `h100-v1.9`. When you run them:

1. Gates 3 to 5 need a running server.
2. Run gate 3 with no `ml/` in the client's working directory, so the client uses the image's
   baked copy.
3. Gate 2's count depends on the vendor: several tests under `test/infra/distribution_strategy`
   are marked `arch="rocm"` and are collected only on the AMD image.
4. Known failure on `h100-v1.7` and `h100-v1.6`, not rechecked on `h100-v1.8`:
   `test/integration/api/test_vllm_api.py::test_lora_adapter_endpoints`. vLLM returns 500
   rather than 4xx for an adapter path that does not exist.

## Build from scratch

`build_image.sh` stages `ml/` into `repo/`, builds the `nvidia` target of `repo/Dockerfile`
and stamps the provenance labels:

```bash
cd training/llm/scalarlm/docs
TARGET=nvidia IMAGE_TAG=h100-v1.9 ./build_image.sh
```

For a CUDA 13 host (untested):

```bash
EXTRA_BUILD_ARGS="--build-arg CUDA_VERSION=13.0.3 --build-arg TORCH_CUDA_INDEX=cu130" \
  TARGET=nvidia IMAGE_TAG=h100-v1.9 ./build_image.sh
```

Build from a clean checkout, or the revision label ends in `-dirty`. The Dockerfile uses BuildKit
`RUN --mount` cache syntax, so the host needs `docker buildx` and `DOCKER_BUILDKIT=1`. Behind a proxy, unset it first:
`unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`. Use `sudo docker` if
Docker runs as root. The vLLM compile takes about 50 minutes.

Another GPU architecture than `sm_90` means recompiling vLLM's `_C` extension for it with
`TORCH_CUDA_ARCH_LIST`.

The collective layer is `cray_infra.training.distributed`
(`repo/infra/cray_infra/training/distributed.py`), pure Python on `torch.distributed` and NCCL,
shared with the AMD build. It needs `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, which
`repo/scripts/train_job_entrypoint.sh` sets.

## Build on top of a published image

To change only `ml/` or pin a model:

```dockerfile
FROM farbodatdocker/scalarlm:h100-v1.9
COPY ml/ /app/cray/ml/
ENV SCALARLM_MODEL=Qwen/Qwen3-0.6B
```

```bash
docker build -t <your-namespace>/scalarlm:my-tag ./my-scalarlm
```

Anything outside `ml/` (collectives, launcher, Slurm, CUDA or PyTorch) needs the build from
scratch.

## Push to a registry

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
docker login -u <dockerhub-username>
docker push <namespace>/scalarlm:<tag>
```

Docker Hub needs a Personal Access Token as the password (hub.docker.com, Account Settings,
Personal access tokens). The upload is about 13 GB and
resumes if you rerun `push`. With `sudo docker`, use `sudo` for both `login` and `push`.
