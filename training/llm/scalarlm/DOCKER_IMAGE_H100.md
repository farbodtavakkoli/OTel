# ScalarLM server image — run, build, customize, push

The ScalarLM server runs as a Docker image (one container serves both the vLLM inference
endpoint and the Megatron-LM-via-Slurm training endpoint, with this folder's `ml/` tree baked
in). This document covers running the pre-built images, building one from scratch, building your
**own** image on top of a published one, and pushing.

## Pre-built images

Drop into the ScalarLM Kubernetes Helm chart via `image.repository` / `.tag` / `.pullPolicy`:

| Image | Hardware |
|---|---|
| `farbodatdocker/scalarlm:h100-v1.5` | NVIDIA H100 / Hopper (`sm_90`) — current: from-scratch `repo/Dockerfile` build of `amd-nvidia-merge` @ `6bc7d82` |
| `farbodatdocker/scalarlm:mi355-v1.6` | AMD MI355X (ROCm) — same source revision; see `DOCKER_IMAGE_MI355.md` |

```yaml
image:
  repository: farbodatdocker/scalarlm
  tag: h100-v1.5          # NVIDIA H100 / Hopper; use mi355-v1.6 for AMD MI355X
  pullPolicy: Always
```

> `h100-v1.5` is the first H100 image built by `TARGET=nvidia ./build_image.sh` from the unified
> tree (digest `sha256:be7aac2525bfa06c45adf10ac44a15cfd5b5cebd7495556d2215dae520189611`, revision
> label `6bc7d82`). vLLM is compiled for `sm_90` in the Dockerfile's `vllm` stage; the merged
> inference server runs natively. Verified on 8×H100: `ddp` and `fsdp` losses bit-identical to the
> pre-merge `nvidia` branch, `pytorch_fsdp` within ~1.7e-5 (different reduction granularity),
> classification at `batch_size > 1` and LoRA complete.
>
> Earlier tags were a chain of hand-built overlays: `h100-v1.1` (base sm_90 + training fixes),
> `h100-v1.2` (torch.distributed/NCCL collective backend), `h100-v1.3` (adapter/mode/distribution
> fixes + `sentence_transformers` for the embedding path). They predate the amd/nvidia merge and
> the `mpirun -> torchrun` launcher; use `h100-v1.5` for anything load-bearing.

### Run it directly

The image restores the real entrypoint, so it starts the server on its own (no override):

```bash
docker run -d --name scalarlm --gpus '"device=0"' --ipc host --shm-size=64g \
  -e SCALARLM_MODEL=Qwen/Qwen3-0.6B -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --cap-add SYS_PTRACE -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  farbodatdocker/scalarlm:h100-v1.5
# curl http://localhost:8000/v1/health  ->  {"api":"up","vllm":"up","all":"up"}
```

> **`--shm-size=64g` is required for multi-rank jobs.** Docker defaults `/dev/shm` to 64 MB;
> multi-rank MPI maps its shared-memory segments there and an 8-rank job overflows even 16 GB,
> killing a rank with SIGBUS (which Slurm then relaunches in a loop). `docker-compose.yaml` sets
> `shm_size: 64gb` on the `cray` anchor, but a raw `docker run` must pass the flag (FIX 5).

Baked-in runtime config: `ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]`,
`CMD ["/app/cray/scripts/start_one_server.sh"]`, `WORKDIR /app/cray`.

### Multi-GPU (single node)

- **Training:** add `--gpus '"device=0,1"'` and `-e SCALARLM_MAX_GPUS_PER_NODE=2`, submit with
  `gpus=2` and `distribution_strategy` fsdp or ddp. Ranks map to distinct GPUs by per-node local
  rank (rank0→GPU0, rank1→GPU1). Point the HF datasets cache at local disk with
  `-e HF_DATASETS_CACHE=/tmp/hf_datasets` if the HF cache is on a network (SMB/NFS) mount — the
  `datasets` file lock returns EACCES on SMB for the second rank otherwise.
- **Inference:** tensor-parallel across GPUs for the **base** model (`-e SCALARLM_TENSOR_PARALLEL_SIZE=2`).
  **Serve fine-tuned adapters at tensor-parallel size 1** — vLLM's TP path rejects the
  hot-reloaded un-sharded adapter state dict. Base-model TP works; adapter serving works at TP=1.

---

## Build the H100 image from scratch

The supported route is the one-command build — it stages `ml/`, builds the `nvidia` target of
`repo/Dockerfile` (vLLM compiled for `sm_90`), and stamps the provenance labels:

```bash
cd training/llm/scalarlm
TARGET=nvidia ./build_image.sh                 # or: TARGET=nvidia IMAGE_TAG=h100-v1.5 ./build_image.sh
```

The Dockerfile uses BuildKit `RUN --mount` cache syntax, so the host needs `docker buildx` and
`DOCKER_BUILDKIT=1`. Two NVIDIA-specific fixes live in the Dockerfile and need no action: a
`torchrun` shim installed into the venv (on the NVIDIA base torch lives in system `dist-packages`,
so a bare `torchrun` otherwise ran under the system interpreter and workers died at
`import cryptography`), and an explicit `scikit-learn` install for the embedding path
(`sentence_transformers` imports it eagerly but the megatron requirements install `--no-deps`).

The rest of this section is the **legacy hand-build route** used for `h100-v1.1`–`v1.3`, kept for
reference.

The stock prebuilt `gdiamos/scalarlm-nvidia-8.0` targets **A100 / `sm_80`**; on an H100 (`sm_90`)
its vLLM CUDA kernels fail at startup with `CUDA error: no kernel image is available for
execution on the device`. No published gdiamos tag targets Hopper, so vLLM's custom extension
must be recompiled for `sm_90`. The base torch already includes `sm_90`; only vLLM's `_C` extension
is rebuilt.

> Prerequisites: an H100 host with Docker + nvidia-container-toolkit. If behind a proxy, unset it
> before any registry pull/push: `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy`.
> Use `sudo docker` if Docker runs as root on your host.

### 1 — Rebuild vLLM for `sm_90`

```bash
docker pull gdiamos/scalarlm-nvidia-8.0:latest
docker tag  gdiamos/scalarlm-nvidia-8.0:latest cray:latest

docker run -d --name vllm_build --gpus '"device=0"' --entrypoint sleep cray:latest infinity
docker exec vllm_build bash -c '
  cd /app/cray/vllm
  rm -rf build *.egg-info; find .deps -name CMakeCache.txt -delete   # clear stale sm_80 cache
  export TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release MAX_JOBS=48
  python use_existing_torch.py --prefix
  pip install --no-build-isolation -e .'          # ~18 min
docker commit vllm_build cray:h100-sm90
docker rm -f vllm_build
```

- **Use `bash -c`, not `bash -lc`** — a login shell re-sources the image's
  `ENV TORCH_CUDA_ARCH_LIST=8.0` and silently rebuilds for the wrong arch.
- Verify sm_90 kernels: `docker run --rm --entrypoint bash cray:h100-sm90 -c 'cuobjdump --list-elf /app/cray/vllm/vllm/_C.abi3.so | grep -c sm_90'`

`cray:h100-sm90` is now a full deployable server; its only remaining issues are the clobbered
`[sleep]` entrypoint and that it still carries the stock `ml/`.

### 1b — Collective backend: NCCL

The collective layer is `cray_infra.training.distributed`
(`repo/infra/cray_infra/training/distributed.py`), a pure-Python `torch.distributed`/NCCL module
shared with the AMD build. Because it is Python, it is carried by baking `ml/`/`infra/` — no wheel
rebuild. NCCL reduces bf16/fp16 natively, so the older C++ `mpi_allreduce` bit-pattern issue does
not arise. It requires the torchrun environment (`RANK`/`LOCAL_RANK`/`WORLD_SIZE`), which
`repo/scripts/train_job_entrypoint.sh` sets up (`mpirun` -> one `torchrun` per node).

> **Since the amd/nvidia merge:** `h100-v1.3` was built against the earlier `gpu_aware_mpi` shim
> and the plain `mpirun -> python` launcher. Those were replaced by the module and launcher above
> (the `gpu_aware_mpi/` directory and its `setup.py` are gone). `h100-v1.5` is the first image built
> from the unified tree and was verified on 8×H100 — `ddp`/`fsdp` bit-identical to the pre-merge
> `nvidia` branch. Note that `flash_attention_2` crashes in the varlen kernel on this image, which is
> why the loader maps it to `sdpa` unconditionally.

### 2 — Bake in this folder's `ml/`

```bash
SRC=training/llm/scalarlm/ml ; DST=/tmp/ml_clean
cp -a "$SRC" "$DST"
find "$DST" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$DST" -type f \( -name '*.pyc' -o -name local_training_config.yaml -o -name 'checkpoint_*.pt' \) -delete

docker run -d --name mlbake --entrypoint sleep -v "$DST":/tmp/ml_clean:ro cray:h100-sm90 infinity
docker exec mlbake bash -c 'rm -rf /app/cray/ml && mkdir -p /app/cray/ml && cp -a /tmp/ml_clean/. /app/cray/ml/'
```

### 3 — Restore entrypoint/CMD/WORKDIR and tag

```bash
docker commit \
  --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
  --change 'CMD ["/app/cray/scripts/start_one_server.sh"]' \
  --change 'WORKDIR /app/cray' \
  mlbake farbodatdocker/scalarlm:h100-v1.1
docker rm -f mlbake
```

Verify: `docker inspect farbodatdocker/scalarlm:h100-v1.1` shows the entrypoint, cmd, and workdir
above, so the image starts the server with no `--entrypoint` override.

### 4 — (optional) Sign with author/maintainer/links

`docker commit` needs a container; to add metadata to an *image*, use a metadata-only build
(reuses all layers, adds only label layers):

```bash
BUILDDIR=$(mktemp -d)
cat > "$BUILDDIR/Dockerfile" <<'EOF'
FROM farbodatdocker/scalarlm:h100-v1.1
LABEL org.opencontainers.image.authors="Your Name"
LABEL maintainer="Your Name"
LABEL org.opencontainers.image.source="https://github.com/<you>"
EOF
docker build -t farbodatdocker/scalarlm:h100-v1.1 "$BUILDDIR"
rm -rf "$BUILDDIR"
```

---

## Build your OWN image on top of a published one

You don't need the full rebuild to customize — start `FROM` a published image and layer your
changes. This is the fast path for iterating on `ml/`, pinning a model, or re-tagging under your
own namespace.

```dockerfile
# my-scalarlm/Dockerfile
FROM farbodatdocker/scalarlm:h100-v1.5

# example: overlay your own modified ml/ tree
COPY ml/ /app/cray/ml/

# example: bake a default model / env
ENV SCALARLM_MODEL=Qwen/Qwen3-0.6B

# entrypoint/CMD/workdir are inherited from the base — the server still starts on its own
```

```bash
docker build -t <your-namespace>/scalarlm:my-tag ./my-scalarlm
docker run -d --gpus '"device=0"' --ipc host -p 8000:8000 -p 8001:8001 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  <your-namespace>/scalarlm:my-tag
```

Notes:
- The `sm_90` vLLM kernels are already in the base — you only rebuild vLLM (§ "from scratch") if
  you change the vLLM version or target a different GPU architecture.
- If you only tweak `ml/` (the training backend), a `COPY ml/ /app/cray/ml/` layer is enough.
- To publish under your own account, use your namespace in the tag and log in as that account.

---

## Push to a registry

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy   # if behind a proxy
docker login -u <dockerhub-username>          # use a Personal Access Token as the password
docker push <namespace>/scalarlm:<tag>
```

- Docker Hub rejects account passwords from the CLI — create a Personal Access Token
  (hub.docker.com → Account Settings → Personal access tokens) and paste it at the password prompt.
- The image is ~45 GB; the upload is slow but resumable — re-run the same `push` if it drops.
  Layers shared with the base show `Mounted from …`; only new layers are `Pushed`.
- If Docker runs as root on your host, prefix `login` and `push` with `sudo` consistently so they
  share the same credential store.
