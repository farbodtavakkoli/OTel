# ScalarLM server image — NVIDIA H100 (CUDA)

Run, build, extend, and push the **H100 (sm_90)** server image. One container serves both
the vLLM inference endpoint and the Megatron-via-Slurm training endpoint, with this
folder's `ml/` recipe baked in. Client usage is in
[`../readme_scalarlm.md`](../readme_scalarlm.md); the AMD image and the shared acceptance
gates are in [`DOCKER_IMAGE_MI355.md`](DOCKER_IMAGE_MI355.md).

## Image

| Tag | Hardware |
|---|---|
| `farbodatdocker/scalarlm:h100-v1.7` | NVIDIA H100 / Hopper (`sm_90`), includes GRPO |
| `farbodatdocker/scalarlm:mi355-v1.8` | AMD MI355X (ROCm) — see the other runbook |

```bash
docker pull farbodatdocker/scalarlm:h100-v1.7
```

`h100-v1.7` supports `ddp`, `fsdp` and `pytorch_fsdp` training, classification at
`batch_size > 1`, LoRA, and the GRPO training mode. It shares one `ml/` training recipe with
`mi355-v1.8`, and replaces both `h100-v1.6` and `h100-grpo-v1.0`. Use `h100-v1.7`.

Identify a pulled image by its digest or revision label, not by the tag:

```bash
docker inspect farbodatdocker/scalarlm:h100-v1.7 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
# h100-v1.7 -> 0b99841b6dba38085738442f0276033c6b3bdb5e
# digest    -> sha256:91b77deb6694c6f0fa930e98a1408d3113be8cce3cbfec5801eaa21edf00a27a
```

Helm chart:

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: h100-v1.7          # NVIDIA H100 / Hopper; use mi355-v1.8 for AMD MI355X
  pullPolicy: Always
```

## `h100-v1.8` (Muse Glimmer)

Built locally, not yet pushed. Revision `4dcb06de8797388dd01087b9ff10ca1daff1165a`.

| Component | Version |
|---|---|
| Base | `nvcr.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, no CUDA compat layer |
| PyTorch | `2.13.0+cu129`, triton 3.7.1 |
| vLLM | upstream v0.32.0 plus `scripts/vllm_patches/vllm-v0.32.0-scalarlm.patch` |
| transformers | 5.19.0 (pin `>=5.17,<5.20`), tokenizers 0.23.3 |
| opencv | 5.0.0.93 built against system OpenSSL 3, no bundled OpenSSL |
| Extras | `mlflow-skinny` |

Runs natively on drivers with CUDA 12.9 or newer. For a CUDA 13 host, build with
`EXTRA_BUILD_ARGS="--build-arg CUDA_VERSION=13.0.3 --build-arg TORCH_CUDA_INDEX=cu130" TARGET=nvidia ./build_image.sh`.
This variant is untested.

Results on 8xH100 (driver 580), `meta-models/Muse-Glimmer-30B`:

| Check | Result |
|---|---|
| Build time torch stack checks | pass after base, vLLM and infra layers |
| `/etc/ld.so.conf.d/` | no `cuda-compat` entry |
| Compressed size | 13.3 GB (limit 20 GB) |
| `import cv2` with `OPENSSL_FORCE_FIPS_MODE=1` | OK |
| Unit suite with current `ml/` and `test/`, 1 GPU visible | 835 passed, 2 deselected, 0 skipped |
| vLLM generation, TP 1 | correct answers, about 75 tok/s single stream |
| Full parameter SFT, `pytorch_fsdp`, 8 GPUs, 8 steps of 4096 tokens | step 0 18.6 s, then about 3.0 s/step, 55.5 GiB peak, final 171 GB checkpoint saved |
| LoRA GRPO, 8 steps, 4 prompts x 4 completions | about 77 s/step, 0 rollout errors, old log probs 16/16, ratio 0.997 to 1.0006 |

Notes:

1. The image bakes `ml/` from `4dcb06d`, which saves the full model on every LoRA step. The LoRA
   only FSDP checkpoint (`85b8a3b`, `c444534`) is active only when the submitting client has the
   current `ml/` in its working directory or beside the SDK, so the SDK ships it in the job tar.
   Jobs that fall back to the image's `/app/cray/ml` lack it. That includes the infra kit's
   `SCALARLM_HOME=/app/cray` and `TRAINER_REF` paths. Rebuild from `c444534` or later, or push that
   commit for `TRAINER_REF`.
2. Write job output to fast local disk. The final full parameter checkpoint must finish inside
   the 10 minute NCCL barrier timeout.
3. On a host without the NVIDIA container toolkit, pass `/dev/nvidia*` and the driver
   libraries with `--device` and `-v` instead of `--gpus`, and run `ldconfig` in the container.
   `run_unit_tests.sh` uses `--gpus all` and does not work there. The v1.8 suite above ran with
   that manual passthrough, `CUDA_VISIBLE_DEVICES=0`, the rendezvous variables and the same
   deselects as the script. The NVIDIA entrypoint then prints "driver was not detected", yet
   `torch.cuda.is_available()` is true.

## Run

The image is self-starting — no entrypoint override needed:

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.7

curl http://localhost:8000/v1/health
# {"api":"up","vllm":"up","all":"up"}
```

`--shm-size=64g` is required for multi-rank jobs. Docker defaults `/dev/shm` to 64 MB;
multi-rank MPI maps its shared-memory segments there and an 8-rank job overflows even
16 GB, killing a rank with SIGBUS, which Slurm then relaunches in a loop.
`docker-compose.yaml` sets `shm_size: 64gb`, but a raw `docker run` must pass the flag.

Baked-in runtime config: `ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]`,
`CMD ["/app/cray/scripts/start_one_server.sh"]`, `WORKDIR /app/cray`.

### Multi-GPU, single node

- **Training:** add `--gpus '"device=0,1"'` and `-e SCALARLM_MAX_GPUS_PER_NODE=2`, then
  submit with `gpus=2` and `distribution_strategy` `fsdp` or `ddp`. Ranks map to distinct
  GPUs by per-node local rank. If the HF cache is on a network (SMB/NFS) mount, add
  `-e HF_DATASETS_CACHE=/tmp/hf_datasets` — the `datasets` file lock otherwise returns
  EACCES on SMB for the second rank.
- **Inference:** tensor-parallel across GPUs for the **base** model
  (`-e SCALARLM_TENSOR_PARALLEL_SIZE=2`). Serve fine-tuned **adapters at tensor-parallel
  size 1**; vLLM's TP path rejects the hot-reloaded un-sharded adapter state dict.

`flash_attention_2` crashes in the varlen kernel on this image, so the loader maps it to
`sdpa` unconditionally — the same as on MI355X.

## Acceptance checks

The gate list is shared with the AMD image (see `DOCKER_IMAGE_MI355.md`). Run gate 1 with
the runner script, not `pytest` directly:

```bash
./run_unit_tests.sh h100-v1.7              # expect: 827 passed, 2 deselected
WITH_CMD=1 ./run_unit_tests.sh h100-v1.7   # also runs the two cmd/ tests: 829 passed
```

A bare `python3 -m pytest test/unit` reports 18 failures on either vendor's image; they are
harness artefacts (the FSDP tests need the `torchrun` rendezvous variables, and two read
`cmd/test_command.sh`, which the Dockerfile does not ship). The script handles both.

Results on 8xH100 for `h100-v1.7`:

| Gate | Check | Result |
|---|---|---|
| 0a | corrections present in image | baked `/app/cray/ml` byte-identical to the tree at the revision label, 44 `.py` files |
| 0b | `sm_90` kernels | `sm_90` in `torch.cuda.get_arch_list()`, capability `(9,0)`, matmul runs |
| 1 | unit suite in-image | 827 passed, 2 deselected, 16 of them GRPO (`WITH_CMD=1`: 829 passed) |
| 2 | collective correctness — `python3 -m pytest test/infra/distribution_strategy -q` | 14 passed |
| 3 | end-to-end, image as **both** server and client | `ddp`, `fsdp`, `pytorch_fsdp`: 30 per-step losses bit-identical to `h100-v1.6` |
| 4 | serving regression — `python3 -m pytest test/integration/api/test_slurm_api.py -q` | 5 passed |
| 5 | legacy collective jobs, 8 ranks | 5/5 `RESULT: ... passed` |
| GRPO | `train_llm_grpo_scalarlm.py --verify`, Qwen3 1.7B, 2 cycles | exit 0, judge failures 0, old log probs 32/32 per cycle, continuation, PEFT and served adapter proofs pass |

Gates 3-5 need a running server. Run gate 3 with no `ml/` in the client's working
directory, so the client uses the image's own baked copy. Gate 2's count is
vendor-specific: several tests under `test/infra/distribution_strategy` are marked
`arch="rocm"` and are collected only on the AMD image.

Known failure on `h100-v1.7`, unchanged from `h100-v1.6`:
`test/integration/api/test_vllm_api.py::test_lora_adapter_endpoints` — vLLM returns 500
rather than 4xx for a nonexistent adapter path.

## Build from scratch

One command: it stages `ml/` into `repo/`, builds the `nvidia` target of `repo/Dockerfile`
(vLLM compiled for `sm_90`), and stamps the provenance labels.

```bash
cd training/llm/scalarlm/docs
TARGET=nvidia ./build_image.sh                 # or: TARGET=nvidia IMAGE_TAG=h100-v1.7 ./build_image.sh
```

The Dockerfile uses BuildKit `RUN --mount` cache syntax, so the host needs `docker buildx`
and `DOCKER_BUILDKIT=1`. Prerequisites: an H100 host with Docker + nvidia-container-toolkit.
Behind a proxy, unset it before any registry pull or push:
`unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`. Use
`sudo docker` if Docker runs as root on your host.

Targeting a GPU architecture other than `sm_90` means recompiling vLLM's `_C` extension
for that arch (`TORCH_CUDA_ARCH_LIST`) — the `sm_90` kernels are already in this image.

The collective layer is `cray_infra.training.distributed`
(`repo/infra/cray_infra/training/distributed.py`), a pure-Python `torch.distributed`/NCCL
module shared with the AMD build; it needs the torchrun environment
(`RANK`/`LOCAL_RANK`/`WORLD_SIZE`), which `repo/scripts/train_job_entrypoint.sh` sets up.

## Build your own image on top of a published one

The fast path for iterating on `ml/`, pinning a model, or re-tagging under your own
namespace:

```dockerfile
# my-scalarlm/Dockerfile
FROM farbodatdocker/scalarlm:h100-v1.7

COPY ml/ /app/cray/ml/                  # overlay your own ml/ tree
ENV SCALARLM_MODEL=Qwen/Qwen3-0.6B      # bake a default model

# entrypoint/CMD/workdir are inherited — the server still starts on its own
```

```bash
docker build -t <your-namespace>/scalarlm:my-tag ./my-scalarlm
docker run -d --gpus '"device=0"' --ipc host --shm-size=64g -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  <your-namespace>/scalarlm:my-tag
```

An overlay reaches `ml/` only. Anything outside it (collectives, launcher, Slurm wiring,
CUDA/PyTorch versions) needs the from-scratch build above.

## Push to a registry

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # if behind a proxy
docker login -u <dockerhub-username>          # use a Personal Access Token as the password
docker push <namespace>/scalarlm:<tag>
```

Docker Hub rejects account passwords from the CLI — create a Personal Access Token
(hub.docker.com -> Account Settings -> Personal access tokens). The image is ~45 GB and the
upload is resumable, so re-run the same `push` if it drops. If Docker runs as root, prefix
`login` and `push` with `sudo` consistently so they share one credential store.
