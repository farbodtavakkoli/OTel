# gpu_aware_mpi

`gpu_aware_mpi` is the C++/PyTorch extension that does all cross-rank communication during training. It sits between PyTorch and the MPI runtime, exposing a small Python API (`allreduce`, `allgather`, `reduce_scatter`, `alltoall`, `send`, `recv`, `isend`, `irecv`, `barrier`, `get_rank`, `get_size`, `finalize_mpi`) and dispatching each call to one of two transports:

- **Standard MPI** — uses `MPI_Isend` / `MPI_Irecv` on the tensor's data pointer directly. Works across nodes, relies on the underlying MPI implementation (OpenMPI, HPC-X, etc.) for GPU-aware transport where available.
- **SHM fast path** — when two ranks share `/dev/shm`, data is staged through a mmap'd file on the shared tmpfs and only a 1-byte control signal crosses MPI. Eliminates serialization overhead between intra-node ranks.

The SHM path is the reason this extension exists. Standard GPU-aware MPI works, but on AMD MI300 systems and on mixed-vendor clusters the latency/throughput profile of intra-node transfers via the MPI implementation is uneven. The SHM channel gives ScalarLM a predictable fast path under its own control.

It lives at `infra/cray_infra/training/gpu_aware_mpi/` in the ScalarLM tree and is built as a PyTorch C++ extension during image build. The training loop in `ml/` imports it as `gpu_aware_mpi`.

---

## 1. Why Not Just NCCL / RCCL?

Three reasons the training stack goes through its own extension rather than `torch.distributed` + NCCL:

1. **GPU-vendor agnostic.** The extension builds against CUDA, ROCm, or a CPU-only backend (`setup.py:detect_gpu_platform`). The same Python code drives NVIDIA H100, AMD MI300, and CPU fallback. NCCL is NVIDIA-only; RCCL works on AMD but isn't API-identical; neither runs on CPU.
2. **Single communication primitive for the whole stack.** Training, collectives, and `@main_rank_only` (file I/O serialization) all use the same `barrier` / `get_rank` / `allreduce` calls. No split between a PyTorch-native `torch.distributed` path and an MPI path for things like `mpirun` rank coordination.
3. **Explicit control over intra-node transport.** With NCCL you get whatever NCCL's topology detection picks. With this extension, `/dev/shm` is the deterministic intra-node fast path and everything else goes through MPI — visible to the operator, tunable, profileable.

This design puts the complexity in one place (the extension) and keeps the training loop free of device-specific code.

---

## 2. Build and Packaging

`infra/cray_infra/training/gpu_aware_mpi/setup.py` is a `torch.utils.cpp_extension.CppExtension` that detects platform at build time:

```python
def detect_gpu_platform():
    if os.path.exists('/opt/rocm'):       return 'rocm'
    if os.path.exists('/usr/local/cuda'): return 'cuda'
    return 'cpu'
```

Per-platform adjustments:

| Platform | Compiler | Includes | Libs | `-D` flags |
|---|---|---|---|---|
| `rocm` | `/opt/ompi-rocm/bin/mpicxx` | `/opt/rocm/include`, `/opt/ompi-rocm/include` | `mpi`, `rt` | `USE_ROCM=1` |
| `cuda` | default (uses `mpicxx` from `$PATH`) | `/usr/local/cuda/include`, `/opt/hpcx/ompi/include` | `cudart`, `rt` | `USE_CUDA=1` |
| `cpu` | `/usr/bin/mpicxx` | `/usr/lib/aarch64-linux-gnu/openmpi/include` | `mpi`, `rt` | — |

Four compilation units, one shared library named `gpu_aware_mpi.so`:

```
gpu_aware_mpi/
├── common.h             # MPI lifecycle, dtype mapping, MpiRequest, wait helpers
├── gpu_aware_mpi.cpp    # Public API + pybind11 module + dispatch logic
├── shm_channel.h/cpp    # /dev/shm discovery, channel creation, duplex layout
├── shm_transport.h/cpp  # SHM send/recv + isend/irecv (with control signaling)
└── mpi_transport.h/cpp  # Standard MPI send/recv + isend/irecv
```

Build is triggered from the Dockerfile. Training images always include it; the CPU-backend build exists primarily for CI and development against the extension itself.

---

## 3. Public API

The pybind11 module (`gpu_aware_mpi.cpp:258`) exports ten functions:

| Python symbol | C++ function | Description |
|---|---|---|
| `gpu_aware_mpi.get_rank()` | `get_rank` | `MPI_Comm_rank(MPI_COMM_WORLD)`. Lazy-initializes MPI. |
| `gpu_aware_mpi.get_size()` | `get_size` | `MPI_Comm_size(MPI_COMM_WORLD)`. |
| `gpu_aware_mpi.barrier()` | `barrier` | `MPI_Barrier(MPI_COMM_WORLD)`. |
| `gpu_aware_mpi.finalize_mpi()` | `finalize_mpi` | Tears down SHM channels, then `MPI_Finalize`. Idempotent. |
| `gpu_aware_mpi.send(tensor, dest)` | `mpi_send` | Blocking send. SHM if peer shares `/dev/shm`, else MPI. |
| `gpu_aware_mpi.recv(tensor, source)` | `mpi_recv` | Blocking recv. |
| `gpu_aware_mpi.isend(tensor, dest)` | `mpi_isend` | Non-blocking send; returns `MpiRequest`. |
| `gpu_aware_mpi.irecv(tensor, source)` | `mpi_irecv` | Non-blocking recv. |
| `gpu_aware_mpi.allreduce(tensor)` | `mpi_allreduce` | In-place `SUM` allreduce via `MPI_Allreduce`. |
| `gpu_aware_mpi.allgather(sendbuf, recvbuf)` | `mpi_allgather` | Hand-rolled from `isend`/`irecv`. |
| `gpu_aware_mpi.reduce_scatter(sendbuf, recvbuf)` | `mpi_reduce_scatter` | Hand-rolled from `isend`/`irecv`. |
| `gpu_aware_mpi.alltoall(sendbuf, recvbuf)` | `mpi_alltoall` | Direct `MPI_Alltoall`. |

All tensor-taking APIs:

- Mutate the tensor in place on the receiving side (or in-place-sum for `allreduce`).
- Call `tensor.contiguous()` first if not already contiguous.
- Call `torch::cuda::synchronize()` before and after the transfer when the tensor is on CUDA, so producer GPU writes are flushed and consumer reads see fresh data.

### 3.1 MPI lifecycle

`common.h:20` — `ensure_mpi_initialized()`:

```c++
inline bool& mpi_initialized_flag() { static bool flag = false; return flag; }
inline void ensure_mpi_initialized() {
    if (!mpi_initialized_flag()) {
        MPI_Init(nullptr, nullptr);
        mpi_initialized_flag() = true;
    }
}
```

Every public function calls this first, so Python code never needs an explicit init. `finalize_mpi()` does `shm_cleanup()` then `MPI_Finalize()` and resets the flag — called explicitly from `ml/cray_megatron/main.py:48` at the end of training.

Because MPI is started from inside Python rather than from an `MPI_Init`-at-startup C program, `mpirun` is responsible for setting the environment (`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, etc.). The sbatch entrypoint is `mpirun --allow-run-as-root python ml/cray_megatron/main.py` — MPI already thinks the world exists; the Python process just joins it lazily on first call.

### 3.2 dtype → MPI_Datatype mapping

`common.h:38` — `get_typesize`:

| Torch dtype | MPI type | Byte size |
|---|---|---|
| `kFloat32` | `MPI_FLOAT` | 4 |
| `kFloat64` | `MPI_DOUBLE` | 8 |
| `kFloat16` | `MPI_SHORT` | 2 |
| `kBFloat16` | `MPI_SHORT` | 2 |
| `kInt32` | `MPI_INT` | 4 |
| `kInt64` | `MPI_LONG_LONG` | 8 |
| `kUInt8` | `MPI_UNSIGNED_CHAR` | 1 |
| `kInt8` | `MPI_CHAR` | 1 |

Both float16 and bfloat16 are transported as `MPI_SHORT` — MPI has no native bf16/fp16 type, but `MPI_SUM` over `MPI_SHORT` is nonsense math. Safe to use for point-to-point transfers; **not safe for `allreduce`**, because MPI will sum the 16-bit integer bit patterns rather than do a floating-point sum. Training-side gradient sync callers always upcast to fp32 before allreduce or use the DDP path's `backward_sync` which handles each param separately (the training loop uses fp32 optimizer state).

---

## 4. SHM Discovery and Dispatch

### 4.1 Peer discovery — `shm_channel.cpp:58`

The discovery protocol is a four-step probe:

1. Each rank creates `/dev/shm/scalarlm_probe_{rank}` (1 byte, `O_CREAT | O_RDWR`).
2. `MPI_Barrier` so all probes are visible.
3. Each rank attempts `shm_open` on every other rank's probe (read-only). Successes go into `g_shm_peers`.
4. Another barrier, then unlink own probe.

If step 1 fails (no `/dev/shm` — rare, but possible in constrained containers), the peer set stays empty and everything falls through to standard MPI. If `/dev/shm` is per-pod (Kubernetes default), same-pod ranks see each other and cross-pod ranks don't — exactly the topology we want.

`shm_is_peer(peer_rank)` is the constant-time check used by `mpi_send` / `mpi_recv` / `mpi_isend` / `mpi_irecv` to pick a transport (`gpu_aware_mpi.cpp:58-74, 84-97`).

### 4.2 Channel naming

Each pair of ranks gets one shm file shared between them (`shm_channel.cpp:44`):

```c++
static std::string channel_name(int a, int b) {
    int lo = std::min(a, b);
    int hi = std::max(a, b);
    return "scalarlm_ch_" + std::to_string(lo) + "_" + std::to_string(hi);
}
```

Pair `(0, 3)` and pair `(3, 0)` both produce `scalarlm_ch_0_3`. The lower-ranked peer owns creation and teardown; the higher-ranked peer opens read-write once the file exists.

### 4.3 Dispatch, in four lines

```c++
void mpi_send(torch::Tensor& tensor, int dest) {
    ensure_mpi_initialized();
    ensure_shm_discovered();
    if (shm_is_peer(dest))  shm_send(tensor, dest);
    else                    mpi_send_standard(tensor, dest);
}
```

Same pattern for `recv`/`isend`/`irecv`. Discovery happens lazily on first transfer, not at import time — so a training job that only talks across nodes never creates probe files.

---

## 5. Duplex SHM Channel Layout

A channel file contains two independent regions — one for each direction:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │ Half 0 (lo → hi direction)                                       │
    │   ┌──────────────┐ ┌──────────────────────────────────────────┐  │
    │   │  ShmHeader   │ │ data buffer (capacity bytes)              │  │
    │   │  { nbytes,   │ │                                           │  │
    │   │   capacity}  │ │                                           │  │
    │   └──────────────┘ └──────────────────────────────────────────┘  │
    ├──────────────────────────────────────────────────────────────────┤
    │ Half 1 (hi → lo direction)                                       │
    │   ┌──────────────┐ ┌──────────────────────────────────────────┐  │
    │   │  ShmHeader   │ │ data buffer (capacity bytes)              │  │
    │   └──────────────┘ └──────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────┘
```

`ShmHeader` (`shm_channel.h:23`) is 16 bytes:

```c++
struct ShmHeader {
    uint64_t nbytes;    // bytes written by sender this transfer
    uint64_t capacity;  // total data buffer capacity (fixed per-channel)
};
```

The `ShmChannel` struct (L31) exposes helper methods that pick the right half based on whether the local rank is lower or higher than the peer:

```c++
ShmHeader* send_header()   // local's outgoing header
void*      send_data()
ShmHeader* recv_header()   // local's incoming header
void*      recv_data()
```

This way both peers use the same API and the two halves never collide. A single channel supports simultaneous bidirectional traffic without any extra synchronization.

### 5.1 Page-locked host memory for GPU DMA

When the extension is built with `USE_CUDA` or `USE_ROCM`, `create_channel` calls `cudaHostRegister(ptr, total, cudaHostRegisterDefault)` on the mmap'd region (`shm_channel.cpp:196`). With ROCm the same symbols are aliased to `hipHostRegister` / `hipSuccess`:

```c++
#ifdef USE_ROCM
#define cudaHostRegister       hipHostRegister
#define cudaHostUnregister     hipHostUnregister
...
```

Page-locking makes GPU → shm copies faster because the GPU DMA engine can write directly to pinned host memory without going through a staging buffer. If `cudaHostRegister` fails (rare — IOMMU restrictions, exhausted pinned memory budget) the channel still works, just slower. The init log prints `cudaHostRegister=yes|no` so operators can see this.

### 5.2 Default capacity and grow

`DEFAULT_CAPACITY = 256 MB` per direction (`shm_channel.cpp:34`). On first use of a channel, the region is sized to `max(requested, 256 MB)`.

If a later transfer asks for more capacity than the existing channel has (`shm_get_channel(peer, min_capacity)` at L210), the channel is torn down and re-created:

```c++
if (it != g_channels.end() && it->second.capacity() >= min_capacity)
    return it->second;

// Need to grow. Tear down old first:
cudaHostUnregister(old.mapped);
munmap(old.mapped, old.mapped_size);
if (rank < peer_rank) shm_unlink_safe(channel_name(rank, peer_rank));
g_channels.erase(it);

// Recreate at max(requested, DEFAULT_CAPACITY)
size_t cap = std::max(min_capacity, DEFAULT_CAPACITY);
g_channels[peer_rank] = create_channel(peer_rank, cap);
```

Grow is rare because most training tensors fit in 256 MB. When it happens — for example, optimizer state allgather on a large shard — both peers coordinate via `peer_sync` before reading the new region.

### 5.3 The barrier-correctness bug this code avoids

`peer_sync` (`shm_channel.cpp:139`) is a pairwise `MPI_Sendrecv` rather than `MPI_Barrier(MPI_COMM_WORLD)`:

```c++
static constexpr int PEER_SYNC_TAG = 43;

static void peer_sync(int peer_rank) {
    char s = 0, r = 0;
    MPI_Sendrecv(&s, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 &r, 1, MPI_CHAR, peer_rank, PEER_SYNC_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

The comment at L131 explains why: channel creation is lazy and triggered by the first transfer between a pair. Different rank pairs hit that trigger at different points in the program. If `create_channel` called `MPI_Barrier(MPI_COMM_WORLD)`, rank 0↔1 would block waiting for ranks 2 and 3 to join a barrier they have no reason to enter, and ranks 2 and 3 would later trigger their own barrier with a different set of participants, causing deadlock or mismatched barriers that corrupt state. Pairwise `MPI_Sendrecv` only blocks the two peers that actually share this channel — always safe, regardless of what other pairs are doing.

Tag `43` is chosen to be distinct from `SHM_CTRL_TAG = 42` (the control-byte tag used for transfers) so a mid-flight signal doesn't get matched against a peer-sync.

---

## 6. The SHM Transfer Protocol

Protocol is deliberately simple: data via shared memory, completion via a 1-byte MPI message. No spin loops, no cache-line flag polling, no memory barriers beyond what `MPI_Send`/`MPI_Recv` already guarantee.

### 6.1 Blocking send — `shm_transport.cpp:13`

```c++
void shm_send(torch::Tensor& tensor, int dest) {
    // 1. Get or grow channel to fit
    ShmChannel& ch = shm_get_channel(dest, nbytes);
    sync_cuda_if_needed(tensor);

    // 2. Copy tensor into shm (from GPU or CPU)
    torch::Tensor shm_tensor = torch::from_blob(ch.send_data(), {numel}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);

    // 3. Write header
    ch.send_header()->nbytes = nbytes;

    // 4. Wake peer with 1-byte MPI_Send
    char sig = 1;
    MPI_Send(&sig, 1, MPI_CHAR, dest, SHM_CTRL_TAG, MPI_COMM_WORLD);
}
```

The `torch::from_blob` call wraps the shm region as a CPU tensor without copying — `copy_` then uses PyTorch's optimized cross-device copy path (including GPU → pinned-host DMA when the tensor is on CUDA).

### 6.2 Blocking recv — `shm_transport.cpp:36`

```c++
void shm_recv(torch::Tensor& tensor, int source) {
    ShmChannel& ch = shm_get_channel(source, nbytes);

    // 1. Wait for 1-byte signal
    char sig;
    MPI_Recv(&sig, 1, MPI_CHAR, source, SHM_CTRL_TAG, MPI_COMM_WORLD, &status);

    // 2. Sanity-check byte count matches
    uint64_t sent_bytes = ch.recv_header()->nbytes;
    if (sent_bytes != nbytes) throw;

    // 3. Copy out of shm
    torch::Tensor shm_tensor = torch::from_blob(ch.recv_data(), {numel}, cpu_opts);
    tensor.copy_(shm_tensor);
    sync_cuda_if_needed(tensor);
}
```

The header check catches any shape/dtype mismatch between sender and receiver — a mis-matched `allgather` slice would raise instead of silently truncating or over-reading.

### 6.3 Why this is fast

The MPI message is 1 byte: `MPI_Send` of 1 byte goes via the "eager" protocol on every MPI implementation, so there's no rendezvous. The actual payload sits in shm and is visible to the peer as soon as the memcpy completes. On CUDA/ROCm, the page-locked mapping lets the memcpy DMA directly — one copy from GPU to shm, one from shm to peer GPU. That's two copies total.

Compared to what standard MPI does on CUDA-aware OpenMPI (often: GPU → pinned staging → MPI eager/rendezvous over shared memory → pinned staging → GPU, i.e. three copies), this is a constant-factor win. The win is larger on ROCm where not every MPI implementation has good GPU-aware transport.

---

## 7. Truly Non-Blocking Variants

The async path is where the design gets interesting. Both sender and receiver need to return from `isend`/`irecv` quickly so caller code can do other work, but the payload hasn't actually moved yet. The trick is splitting the work between eager steps and deferred callbacks.

### 7.1 `MpiRequest` — the handle

`common.h:79`:

```c++
struct MpiRequest {
    MPI_Request           mpi_req    = MPI_REQUEST_NULL;
    bool                  completed  = false;
    char                  shm_sig    = 0;    // stable storage for 1-byte signal
    std::function<void()> on_complete;       // fired after MPI_Wait succeeds
};
```

Two subtleties:

- **`shm_sig` lives inside the request.** `MPI_Isend`/`MPI_Irecv` require the buffer to stay alive until MPI_Wait completes. Putting the char inside the request struct guarantees that — as long as the Python caller holds the `MpiRequest`, the 1-byte buffer is valid.
- **`on_complete` is an `std::function`.** It's empty for the MPI path and for `shm_isend`. It holds a capturing lambda for `shm_irecv` — the deferred shm → tensor copy.

### 7.2 `shm_isend` — eager copy, lazy signal

`shm_transport.cpp:67`:

```c++
MpiRequest shm_isend(torch::Tensor& tensor, int dest) {
    ShmChannel& ch = shm_get_channel(dest, nbytes);
    sync_cuda_if_needed(tensor);

    // Copy data into shm eagerly — this is cheap (local memcpy / DMA)
    torch::Tensor shm_tensor = torch::from_blob(ch.send_data(), {numel}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);
    ch.send_header()->nbytes = nbytes;

    // Post the 1-byte signal non-blocking
    MpiRequest req;
    req.shm_sig = 1;
    MPI_Isend(&req.shm_sig, 1, MPI_CHAR, dest, SHM_CTRL_TAG, MPI_COMM_WORLD, &req.mpi_req);
    return req;
}
```

The data copy happens synchronously during `isend`. Why not defer that too? Because there's no MPI completion event to hang it off — the sender has no way to know when it's safe to discard the source tensor if the copy hasn't happened yet. Copying eagerly into shm gives the caller the property it actually wants: as soon as `shm_isend` returns, the source tensor can be freely modified.

At `mpi_wait` time there's literally nothing left for the send side to do.

### 7.3 `shm_irecv` — eager post, deferred copy-out

`shm_transport.cpp:103`:

```c++
MpiRequest shm_irecv(torch::Tensor& tensor, int source) {
    ShmChannel& ch = shm_get_channel(source, nbytes);

    MpiRequest req;
    req.on_complete = [tensor, &ch, nbytes]() mutable {
        uint64_t sent_bytes = ch.recv_header()->nbytes;
        if (sent_bytes != nbytes) throw ...;

        torch::Tensor shm_tensor = torch::from_blob(ch.recv_data(), {numel}, cpu_opts);
        tensor.copy_(shm_tensor);
        sync_cuda_if_needed(tensor);
    };
    MPI_Irecv(&req.shm_sig, 1, MPI_CHAR, source, SHM_CTRL_TAG, MPI_COMM_WORLD, &req.mpi_req);
    return req;
}
```

The receiver side is the mirror image of send. The MPI_Irecv for the 1-byte signal is posted immediately — MPI starts watching for the signal in the background. The actual copy-out from shm into the destination tensor can't happen until the signal arrives, so it's captured in the `on_complete` lambda.

Capture details:

- `tensor` is captured by value — `torch::Tensor` is a reference-counted handle, copying it cheaply keeps the underlying storage alive until the callback runs.
- `&ch` is captured by reference — channels live until `shm_cleanup()`, which is after all waits.
- `nbytes` is plain scalar.

### 7.4 Wait / Waitall

`common.h:90`:

```c++
inline void mpi_wait(MpiRequest& req) {
    if (req.completed) return;
    MPI_Wait(&req.mpi_req, &st);
    if (req.on_complete) req.on_complete();
    req.completed = true;
}
```

`mpi_waitall` (L100) is the batched version. It carefully separates "real MPI requests" from "requests with callbacks":

```c++
std::vector<MPI_Request> pending;
std::vector<size_t>      indices;
for (size_t i = 0; i < reqs.size(); ++i) {
    if (!reqs[i].completed) {
        pending.push_back(reqs[i].mpi_req);
        indices.push_back(i);
    }
}
MPI_Waitall(pending.size(), pending.data(), statuses.data());

// All MPI ops have landed; run callbacks in posting order
for (size_t idx : indices) {
    if (reqs[idx].on_complete) reqs[idx].on_complete();
    reqs[idx].completed = true;
}
```

The index mapping preserves posting order so that `on_complete` callbacks run deterministically, even when MPI returns completions out of order. For `mpi_allgather`'s N-1 receives, this means the per-slice shm copies run in rank order after all the 1-byte signals have landed.

### 7.5 What the MPI path does asynchronously

For comparison, `mpi_isend_standard` (`mpi_transport.cpp:51`) is a one-liner: `MPI_Isend(tensor.data_ptr(), ...)`. The `MpiRequest` returned has an empty `on_complete` — MPI takes care of everything. The same `mpi_wait` / `mpi_waitall` code handles both paths uniformly.

---

## 8. Hand-Rolled Collectives

`allreduce` and `alltoall` delegate directly to MPI (`MPI_Allreduce` and `MPI_Alltoall`). `allgather` and `reduce_scatter` are hand-rolled in terms of `isend` / `irecv` so they benefit from the SHM fast path automatically.

### 8.1 `mpi_allgather` — `gpu_aware_mpi.cpp:123`

Algorithm:

```
1. Copy own slice: recvbuf[rank * count : (rank+1) * count] ← sendbuf
2. For each peer i ≠ rank:
     - issue isend(sendbuf, i)               ─┐
     - issue irecv(recvbuf[i * count:], i)   ─┤ all concurrent
3. waitall(recv_reqs)                        ─┤  (shm callbacks fire here)
4. waitall(send_reqs)                        ─┘
5. sync_cuda(recvbuf)
```

Two subtleties:

- **Sends and receives are all posted before any waits.** This is the key to actually getting concurrency: issuing `isend`s first lets the interconnect start moving data immediately, while issues `irecv`s register receive buffers so the sender's data has somewhere to land without MPI staging.
- **`recv_slices` keeps per-peer slice tensors alive.** `recvbuf.slice()` returns a view; the vector pins the slice so the `shm_irecv` callback can copy into it safely after `waitall` returns.

### 8.2 `mpi_reduce_scatter` — `gpu_aware_mpi.cpp:171`

```
1. recvbuf ← sendbuf[rank * count : (rank+1) * count]    // seed with own contribution
2. For each peer i ≠ rank:
     - issue isend(sendbuf[i * count:], i)  (from send_slices, which are .contiguous())
     - issue irecv(tmp_bufs[i], i)
3. waitall(recv_reqs)
4. For each tmp: recvbuf.add_(tmp)                        // in-place reduction
5. waitall(send_reqs)                                     // after the reduction
6. sync_cuda(recvbuf)
```

The ordering of step 5 matters: sends are waited on **after** the reduction. This is because the shm send path (`shm_isend`) has already copied into shm synchronously, and the peer's `on_complete` callback reads from shm during *its* waitall. If we finished our send-waits before the peer finished its recv-waits, in theory it's fine (the peer's copy has already completed by then), but ScalarLM's code is defensive: don't exit the function while any peer might still be referencing your shm buffer. This is also why `shm_cleanup` is called from `finalize_mpi` rather than from channel destructors — it's unsafe to tear down shm while any outstanding transfer might touch it.

### 8.3 Why not use `MPI_Allgather` / `MPI_Reduce_scatter` directly?

Because MPI's collective implementations pick their own topology — typically ring or recursive-doubling — and their own transport. By decomposing into pairwise `isend`/`irecv`, each pair-wise transfer independently chooses the SHM fast path or standard MPI. Same-pod ranks go through shm; cross-pod pairs go through MPI. An MPI-level `MPI_Allgather` would use whatever the MPI implementation's allgather transport is for the *whole* communicator, which is usually a ring that can't selectively use shm per-link.

The cost is algorithmic: these are O(N) communication rounds, not O(log N). For N=4, this is fine (3 concurrent transfers). For N≫8 across many nodes, the MPI implementation's recursive-doubling allgather would be better. ScalarLM's sweet spot is small training worlds (1 node × 4-8 GPUs), so the simpler algorithm wins.

---

## 9. Python-Side Usage

### 9.1 Importing

```python
from gpu_aware_mpi import (
    allreduce, allgather, reduce_scatter, alltoall,
    send, recv,          # blocking point-to-point
    isend, irecv,        # non-blocking (not yet exported at Python level in all builds)
    barrier,
    get_rank, get_size,
    finalize_mpi,
)
```

### 9.2 `@main_rank_only`

`ml/cray_megatron/collectives/main_rank_only.py` wraps rank-0 I/O with barriers so the whole world waits for the rank-0 write to complete:

```python
_in_main_rank_only = False

def main_rank_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _in_main_rank_only
        if _in_main_rank_only:                     # re-entrant call
            return func(*args, **kwargs)
        _in_main_rank_only = True
        try:
            barrier()                              # everyone rendezvous
            result = func(*args, **kwargs) if is_main_rank() else None
            barrier()                              # wait for rank 0 to finish
            return result
        finally:
            _in_main_rank_only = False
    return wrapper
```

The `_in_main_rank_only` guard prevents deadlocks when a `@main_rank_only` function calls another `@main_rank_only` function — the inner call skips the barriers because the outer one already has everyone synchronized.

Used for `save_status` (training_harness.py), `save_checkpoint` / `update_history` / `print_training_step_info` (training_loop.py). See `docs/training-lifecycle.md` §4.5.

### 9.3 Distribution strategies

**DDP** (`ml/cray_megatron/megatron/distribution/ddp.py`) uses `allreduce` for gradient sync:

```python
def backward_sync(self):
    for param_name, param in self.model.named_parameters(recurse=False):
        if param.requires_grad and param.grad is not None:
            allreduce(param.grad)
```

**FSDP** (`ml/cray_megatron/megatron/distribution/fsdp.py`) uses `allgather` to reassemble sharded parameters at forward time and `reduce_scatter` to redistribute gradients at backward time — the two costliest collectives in sharded training, both of which get the SHM fast path for same-pod peers.

### 9.4 Loss sync in the training loop

`ml/cray_megatron/megatron/training_loop.py:450`:

```python
class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_tmp = input.clone()
        allreduce(input_tmp)
        return input_tmp

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_tmp = grad_output.clone()
        allreduce(grad_output_tmp)
        return grad_output_tmp

allreduce_op = _AllReduce.apply
```

An autograd-aware allreduce used by `sync_loss` (L209) for logging the cross-rank average of each step's loss. The `clone()` is defensive — allreduce is in-place, and reusing the original tensor would modify the value that `ctx.save_for_backward` held.

---

## 10. Tests

`test/collectives/` contains both Python wrappers that exercise the extension via the ScalarLM training harness and a standalone test of the shm protocol:

| File | What it tests |
|---|---|
| `test-hello.py` | MPI init + `get_rank`/`get_size`. |
| `test-send-recv.py` | Point-to-point blocking round-trip. |
| `test-all-reduce.py` | `allreduce` across N ranks. |
| `test-all-gather.py` | `allgather` correctness. |
| `test-reduce-scatter.py` | `reduce_scatter` correctness. |
| `test-all-to-all.py` | `alltoall` correctness. |
| `benchmark-send-recv.py` | Bandwidth/latency sweep across message sizes for both shm and MPI paths. |
| `test_shm_channel.py` | **Standalone** Python reimplementation of the shm protocol using `multiprocessing.shared_memory`. No MPI or C++ required. Validates channel layout, duplex operation, data integrity across dtypes, reuse, and multi-channel concurrency. |
| `test_shm_channel_cpp.cpp` | C++ unit test that links directly against the extension for lower-level debugging. |

The first group submits itself as a SLURM job via `llm.submit_slurm_job` — they run inside the same `mpirun` context as a training job. The standalone `test_shm_channel.py` runs in a single Python process with `multiprocessing` subprocesses and is what you reach for to debug protocol bugs without spinning up a full container.

---

## 11. Design Notes

**Why shm over a UNIX socket?** Sockets would add a kernel copy per transfer; shm gives zero-copy after the initial memcpy. The 1-byte MPI control signal is the synchronization primitive, so we're not paying for a socket's flow control.

**Why use MPI for the signal at all, not a cache-line flag with `std::atomic`?** Spinning on a cache line wastes CPU, burns power, and doesn't give MPI a chance to make progress on other transfers. Using MPI for the signal lets MPI's progress engine handle waiting efficiently (including `MPI_Waitall` over many pending operations). It also guarantees correct memory ordering without fences — after `MPI_Recv` returns, every write the sender did before calling `MPI_Send` is visible, by MPI's own memory model.

**Why one shm file per pair, not one shared multi-writer file?** Multi-writer shm needs per-writer locking or lock-free queues. One-pair-per-file is a single-producer/single-consumer channel per direction — no locking needed, trivial correctness reasoning, no contention.

**Why duplex (two halves per file) rather than two files?** One file means one `shm_open`, one `mmap`, one `cudaHostRegister`. Both directions share the same pinned-memory registration. Half the file descriptors and half the registration overhead.

**Why `shm_sig` inside `MpiRequest`?** MPI's `Isend`/`Irecv` require the buffer to outlive the operation. Putting the byte inside the handle struct couples the buffer's lifetime to the handle's lifetime — the Python caller holding the handle keeps the byte alive. A static or module-global byte wouldn't work because you'd need one per in-flight operation.

**Why `sync_cuda_if_needed` before *and* after transfers?** Before: the sending side's previous kernels might have outstanding writes to the source tensor. After: the receiving side needs to ensure the copy into the destination is visible before the next kernel reads it. Both are no-ops on CPU tensors, so there's no overhead to always calling them.

**Why call `tensor.contiguous()` instead of requiring it?** The caller shouldn't have to worry. MPI and memcpy both require contiguous buffers; if the tensor isn't contiguous we make a contiguous copy. Usually the training code passes contiguous tensors and the check is free.

**Why 256 MB default capacity?** Typical gradient shards, parameter allgathers, and activation exchanges fit. Larger than 256 MB triggers a grow (file recreation + re-register), which is slow — so we pick a size big enough that growth is rare but small enough not to burn a GB of `/dev/shm` per rank pair.

**Why `MPI_SHORT` for bf16/fp16?** MPI has no standard half-float type. For point-to-point transfers we just need the right byte count, and `MPI_SHORT` gives us that. For `allreduce` with `MPI_SUM` this would be wrong — but ScalarLM's gradient sync doesn't allreduce half-precision tensors. Don't do that.

**Why no NCCL / RCCL at all?** Simplicity and vendor-agnosticism. NCCL is excellent at its job but brings a separate topology-detection layer, a separate init protocol, and a separate debuggability story. MPI + shm is one system to reason about, builds on both NVIDIA and AMD without changes, and hits a large fraction of NCCL's performance for the small-world training workloads ScalarLM targets.

---

## 12. Limitations

- **No half-precision allreduce.** `MPI_SUM` on `MPI_SHORT` sums integer bit patterns. Callers must upcast before `allreduce`.
- **`/dev/shm` capacity.** Each channel consumes `2 × (16 + capacity)` bytes in `/dev/shm`. With 8 ranks per pod, that's 28 pairs × 2 × 256 MB = 14 GB. Ensure the pod's `/dev/shm` PVC/emptyDir is sized accordingly — Kubernetes defaults to 64 MB, which will break silently.
- **MPI_COMM_WORLD only.** All collectives operate on the full world. No sub-communicators, no tensor-parallel groups. FSDP sharding is done in application code using per-rank slicing, not separate communicators.
- **Lazy init is global.** `ensure_mpi_initialized` is process-wide. If a consumer imports `gpu_aware_mpi` but never intends to use it, the first call to anything (including `get_rank`) still fires `MPI_Init`. For non-MPI processes, don't import it.
- **Grow is expensive.** A transfer that exceeds the current channel capacity tears the channel down and recreates it, re-paying `cudaHostRegister` (slow — tens to hundreds of ms for large buffers). Size a transfer pattern so that 256 MB covers the common case.
- **No GPU direct transport.** The shm path goes through host memory. For GPU-resident tensors, this is GPU → pinned host → peer pinned host → peer GPU. For truly large transfers, a GPU-direct path (NVLink, xGMI) via NCCL/RCCL would be faster. But cross-vendor, cross-topology GPU-direct is hard to get right — shm is the robust fallback.
- **Discovery is one-shot.** `shm_discover_peers` runs once per process. If the world changes (rank joins/leaves — not currently supported anyway), discovery state is stale. Re-run `finalize_mpi()` to reset everything.
