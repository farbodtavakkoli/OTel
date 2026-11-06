import logging
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from cray_infra.training.train_debug import is_train_debug_enabled
from cray_infra.util.get_job_config import get_job_config


def _trace(msg: str) -> None:
    if not is_train_debug_enabled():
        return
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "?"))
    sys.stderr.write(f"[rank={rank}] dist [{time.monotonic():.3f}]: {msg}\n")
    sys.stderr.flush()


def _local_rank() -> int:
    for var in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        raw = os.environ.get(var)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available() and hasattr(torch, "neuron"):
        return torch.device("neuron")
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
    if not torch.cuda.is_available() and hasattr(torch, "neuron"):
        dist.init_process_group(backend="neuron")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is required for distributed training")

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)

    if os.environ.get("CRAY_TRAINING_JOB_CONFIG_PATH"):
        timeout_seconds = get_job_config()["process_group_timeout_seconds"]
    else:
        from cray_infra.util.default_job_config import JobConfig
        timeout_seconds = JobConfig.model_fields["process_group_timeout_seconds"].default
    logging.getLogger(__name__).info("Initializing NCCL process group with timeout_seconds=%s", timeout_seconds)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
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
    if device_tensor is not tensor:
        tensor.copy_(device_tensor.view_as(tensor))
        return tensor
    return device_tensor


def finalize():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


finalize_mpi = finalize
