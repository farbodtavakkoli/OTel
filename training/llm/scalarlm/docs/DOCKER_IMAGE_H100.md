# ScalarLM server image: NVIDIA H100 (CUDA)

Run, build, extend and push the **H100 (`sm_90`)** server image. One container serves vLLM
inference and Megatron training through Slurm, with this folder's `ml/` baked in. Client usage
is in [`../readme_scalarlm.md`](../readme_scalarlm.md). The AMD image and the shared gates are in
[`DOCKER_IMAGE_MI355.md`](DOCKER_IMAGE_MI355.md).

## Image

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:h100-v1.8` | NVIDIA H100 (`sm_90`), GRPO, native Muse Glimmer |
| `farbodatdocker/scalarlm:mi355-v1.8` | AMD MI355X (ROCm), see the other runbook |

```bash
docker pull farbodatdocker/scalarlm:h100-v1.8
```

`h100-v1.8` supports `ddp`, `fsdp` and `pytorch_fsdp`, LoRA, full parameter training and the
GRPO mode, and serves and trains `meta-models/Muse-Glimmer-30B`. It replaces `h100-v1.7`.

| Component | Version |
|---|---|
| Base | `nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, no CUDA compat layer |
| PyTorch | `2.13.0+cu129`, triton 3.7.1 |
| vLLM | upstream v0.32.0 plus `scripts/vllm_patches/vllm-v0.32.0-scalarlm.patch` |
| transformers | 5.19.0 (pin `>=5.17,<5.20`), tokenizers 0.23.3 |
| opencv | 5.0.0.93 built against system OpenSSL 3, no bundled OpenSSL |
| Extras | `mlflow-skinny` |

The image runs natively on drivers with CUDA 12.9 or newer. Identify a pulled image by its
revision label, not the tag:

```bash
docker inspect farbodatdocker/scalarlm:h100-v1.8 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
# h100-v1.8 -> c44453419e41d831b7caaebce7ebcedc3f608468
```

Helm chart:

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: h100-v1.8
  pullPolicy: Always
```

## Run

The image starts the server on its own:

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.8

curl http://localhost:8000/v1/health
# {"api":"up","vllm":"up","all":"up"}
```

Baked runtime config: `ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]`,
`CMD ["/app/cray/scripts/start_one_server.sh"]`, `WORKDIR /app/cray`.

Notes:

1. `--shm-size=64g` is required for jobs with several ranks. The 64 MB Docker default kills a
   rank with SIGBUS, and Slurm then relaunches it in a loop.
2. For training on several GPUs, pass them with `--gpus` and set `SCALARLM_MAX_GPUS_PER_NODE`.
   It defaults to 1.
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

## Acceptance checks

Run gate 1 with the runner script, not `pytest` directly:

```bash
./run_unit_tests.sh h100-v1.8              # expect: 835 passed, 2 deselected
```

The script uses `--gpus all`. On a host without the container toolkit, run the same command it
runs inside the container with the manual passthrough from note 7.

Results on 8xH100 (driver 580) for `h100-v1.8`, model `meta-models/Muse-Glimmer-30B`:

| Check | Result |
|---|---|
| Build time torch stack checks | pass after base, vLLM and infra layers |
| `/etc/ld.so.conf.d/` | no `cuda-compat` entry |
| Compressed size | 13.3 GB (AI Runtime limit 20 GB) |
| `import cv2` with `OPENSSL_FORCE_FIPS_MODE=1` | OK |
| Unit suite, 1 GPU visible | 835 passed, 2 deselected, 0 skipped |
| vLLM generation, TP 1 | correct answers, about 75 tok/s single stream |
| Full parameter SFT, `pytorch_fsdp`, 8 GPUs, 4096 tokens | step 0 18 s, then about 3.0 s/step, 55.5 GiB peak, final checkpoint saved |
| LoRA GRPO, 4 prompts x 4 completions | about 77 s/step, 0 rollout errors, old log probs 16/16, ratio 0.997 to 1.0006 |

Gates 2 to 5 from `DOCKER_IMAGE_MI355.md` were not rerun on `h100-v1.8`.

## Build from scratch

`build_image.sh` stages `ml/` into `repo/`, builds the `nvidia` target of `repo/Dockerfile`
and stamps the provenance labels:

```bash
cd training/llm/scalarlm/docs
TARGET=nvidia IMAGE_TAG=h100-v1.8 ./build_image.sh
```

For a CUDA 13 host (untested):

```bash
EXTRA_BUILD_ARGS="--build-arg CUDA_VERSION=13.0.3 --build-arg TORCH_CUDA_INDEX=cu130" \
  TARGET=nvidia IMAGE_TAG=h100-v1.8 ./build_image.sh
```

Build from a clean checkout, or the revision label ends in `-dirty`. The host needs Docker with
`buildx`. Behind a proxy, unset it first:
`unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`. Use `sudo docker` if
Docker runs as root. The vLLM compile takes about 50 minutes.

## Build on top of a published image

To change only `ml/` or pin a model:

```dockerfile
FROM farbodatdocker/scalarlm:h100-v1.8
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

Docker Hub needs a Personal Access Token as the password. The upload is about 13 GB and
resumes if you rerun `push`. With `sudo docker`, use `sudo` for both `login` and `push`.
