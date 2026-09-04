"""NCCL/RCCL collectives via torch.distributed.

Ported from tensorwavecloud/ScalarLM PR #5 ("move from mpi/ucx collectives to
nccl"). This is a drop-in replacement for the hand-written ``gpu_aware_mpi`` C++
extension: same function names, same call signatures, so every call site only
changes its import line.

Why AMD recommend it on MI300/MI355:

* RCCL is AMD's own collective library and is topology-aware — it drives xGMI /
  Infinity Fabric between GPUs on a node, where the OpenMPI+UCX path went through
  generic transports.
* torch.distributed handles bf16/fp16 natively. The MPI extension had no such
  datatype and mapped both onto ``MPI_SHORT``; that is fine for byte-moving
  collectives but silently corrupts ``allreduce``, the one collective that does
  arithmetic (MPI_SUM adds raw IEEE bit patterns as int16). Moving to NCCL
  removes that entire class of bug rather than patching it.
* No custom C++ extension to compile and keep working against each new ROCm.

Requires the torchrun-style environment (RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT); ``scripts/train_job_entrypoint.sh`` sets it up.
NCCL collectives require CUDA tensors, so every buffer is moved on-device and
made contiguous here — callers that used to hand in CPU tensors keep working.
"""

import os
import sys
import time

import torch
import torch.distributed as dist

from cray_infra.training.train_debug import is_train_debug_enabled


def _trace(msg: str) -> None:
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[rank={rank}] dist [{time.monotonic():.3f}]: {msg}\n")
    sys.stderr.flush()


def _local_rank() -> int:
    """This process's GPU index on its own host.

    torchrun sets LOCAL_RANK; OpenMPI and Slurm are accepted as fallbacks so the
    module still works if the job is launched by mpirun directly.
    """
    for var in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        raw = os.environ.get(var)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def _cuda_device() -> torch.device:
    return torch.device("cuda", _local_rank())


def cuda_device() -> torch.device:
    return _cuda_device()


def _to_cuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_cuda else tensor.to(_cuda_device())


def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _prepare(tensor: torch.Tensor) -> torch.Tensor:
    return _ensure_contiguous(_to_cuda(tensor))


def init():
    _trace(
        f"init() pid={os.getpid()} RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}"
    )
    if dist.is_initialized():
        _trace("init() skipped, already initialized")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for distributed training")

    # set_device BEFORE init_process_group: NCCL binds its communicator to the
    # current device, and every rank defaulting to device 0 is the classic cause
    # of a hang on the first collective.
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    _trace(
        f"init_process_group complete rank={dist.get_rank()} "
        f"world_size={dist.get_world_size()} backend={dist.get_backend()} "
        f"device=cuda:{local_rank}"
    )


def _ensure_initialized():
    if not dist.is_initialized():
        init()


def get_rank():
    _ensure_initialized()
    return dist.get_rank()


def get_size():
    _ensure_initialized()
    return dist.get_world_size()


def barrier():
    if not dist.is_initialized():
        return
    dist.barrier()


def allgather(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _prepare(sendbuf)
    device_recv = _prepare(recvbuf)
    dist.all_gather_into_tensor(device_recv, sendbuf)
    # _prepare() COPIES a CPU tensor to the GPU, so NCCL fills the copy and the
    # caller's buffer would silently stay zero. The MPI extension wrote through
    # to CPU buffers, so any call site that still passes one must keep working:
    # copy the result back. (Call sites on the hot path allocate on-device and
    # skip this entirely.)
    if device_recv is not recvbuf:
        recvbuf.copy_(device_recv.view_as(recvbuf))
        return recvbuf
    return device_recv


def reduce_scatter(sendbuf, recvbuf):
    _ensure_initialized()
    sendbuf = _prepare(sendbuf)
    device_recv = _prepare(recvbuf)
    dist.reduce_scatter_tensor(device_recv, sendbuf, op=dist.ReduceOp.SUM)
    if device_recv is not recvbuf:
        recvbuf.copy_(device_recv.view_as(recvbuf))
        return recvbuf
    return device_recv


def allreduce(tensor, op=dist.ReduceOp.SUM):
    _ensure_initialized()
    device_tensor = _prepare(tensor)
    dist.all_reduce(device_tensor, op=op)
    # allreduce is in-place in the MPI extension; preserve that for CPU callers.
    if device_tensor is not tensor:
        tensor.copy_(device_tensor.view_as(tensor))
        return tensor
    return device_tensor


def alltoall(sendbuf, recvbuf=None):
    _ensure_initialized()
    sendbuf = _prepare(sendbuf)
    world_size = get_size()
    if sendbuf.numel() % world_size != 0:
        raise ValueError("alltoall send buffer numel must be divisible by world size")

    input_list = [_ensure_contiguous(c) for c in sendbuf.chunk(world_size, dim=0)]

    if recvbuf is None:
        recvbuf = torch.empty_like(sendbuf)
    device_recv = _prepare(recvbuf)
    output_list = list(device_recv.chunk(world_size, dim=0))

    dist.all_to_all(output_list, input_list)
    # Same CPU write-back contract as allgather/reduce_scatter/allreduce/recv:
    # _prepare() copies a CPU tensor to the GPU, so without this the caller's
    # buffer would silently keep its old contents.
    if device_recv is not recvbuf:
        recvbuf.copy_(device_recv.view_as(recvbuf))
        return recvbuf
    return device_recv


def send(tensor, dest: int):
    _ensure_initialized()
    tensor = _prepare(tensor)
    dist.send(tensor, dst=dest)
    return tensor


def recv(tensor, source: int):
    _ensure_initialized()
    device_tensor = _prepare(tensor)
    dist.recv(device_tensor, src=source)
    # recv fills its buffer in place. If _prepare() had to stage a CPU tensor on
    # the GPU, the caller's tensor is a different object and would silently keep
    # its old contents — write the received data back. (test_sendrecv_cpu asserts
    # on the original buffer and fails without this.)
    if device_tensor is not tensor:
        tensor.copy_(device_tensor.view_as(tensor))
        return tensor
    return device_tensor


def finalize():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# The MPI extension spelled teardown `finalize_mpi`; keep the name so the
# existing call sites (main.py, training_job_context.py) stay unchanged.
finalize_mpi = finalize
