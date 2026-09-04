from cray_megatron.megatron.distribution.fsdp import SimpleFSDP
from cray_megatron.megatron.distribution.ddp import DDP
from cray_megatron.megatron.distribution.no_distribution import NoDistribution
from cray_infra.util.get_job_config import get_job_config

from cray_infra.training.distributed import get_size, get_rank

import torch

import socket
import os
import json
import time
import subprocess

import logging

logger = logging.getLogger(__name__)


def load_distribution_strategy():
    device = get_device()

    strategy = {
        "device": device,
    }

    if get_size() > 1:
        distribution_strategy = get_job_config()["distribution_strategy"]

        if distribution_strategy == "ddp":
            logger.info("Using DDP distribution strategy.")
            strategy["strategy"] = DDP
        elif distribution_strategy == "pytorch_fsdp":
            # Imported lazily: fully_shard needs torch >= 2.4, and this module must load on older torch.
            from cray_megatron.megatron.distribution.pytorch_fsdp import PyTorchFSDP

            logger.info("Using PyTorch FSDP2 distribution strategy.")
            strategy["strategy"] = PyTorchFSDP
        elif distribution_strategy == "fsdp":
            # "fsdp" is JobConfig's default, so it must not be reported as unknown below.
            logger.info("Using SimpleFSDP distribution strategy.")
            strategy["strategy"] = SimpleFSDP
        else:
            logger.warning(
                f"Unknown distribution strategy '{distribution_strategy}' "
                "specified. Defaulting to SimpleFSDP."
            )
            strategy["strategy"] = SimpleFSDP
    else:
        logger.info("Using NoDistribution distribution strategy.")
        strategy["strategy"] = NoDistribution

    return strategy


def get_device():
    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        # Honour torchrun's LOCAL_RANK: it must match the device distributed.init() selected, or NCCL hangs.
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            try:
                return torch.device(f"cuda:{int(local_rank_env) % gpu_count}")
            except ValueError:
                pass

        selected_gpu = select_gpu()

        if gpu_count > 1:
            return torch.device(f"cuda:{selected_gpu}")

        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def apply_distribution_strategy(model_info):
    distribution_strategy = load_distribution_strategy()
    model_info["distribution_strategy"] = distribution_strategy
    return model_info


def get_local_rank_and_size():
    """This process's rank within its host, and how many ranks share the host."""
    for rank_var, size_var in (
        # torchrun first: it is authoritative, and OMPI_* may describe launcher processes, not workers.
        ("LOCAL_RANK", "LOCAL_WORLD_SIZE"),
        ("OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE"),
        ("SLURM_LOCALID", "SLURM_NTASKS_PER_NODE"),
    ):
        raw_rank = os.environ.get(rank_var)
        raw_size = os.environ.get(size_var)
        if raw_rank is None or raw_size is None:
            continue
        try:
            # SLURM_NTASKS_PER_NODE can be "2(x3)" on heterogeneous allocations.
            return int(raw_rank), int(str(raw_size).split("(")[0])
        except ValueError:
            continue

    return 0, 1


def select_gpu():
    rank = get_rank()
    gpu_count = torch.cuda.device_count()

    # Single-node multi-GPU: ranks share a hostname, so the hostname-position scheme cannot separate them.
    local_rank, local_size = get_local_rank_and_size()
    if local_size > 1 and gpu_count > 0:
        gpu_index = local_rank % gpu_count
        logger.info(
            f"Rank {rank} is local rank {local_rank} of {local_size} on this host; "
            f"assigned GPU {gpu_index} out of {gpu_count} available GPUs."
        )
        return gpu_index

    machine_id = get_machine_id()
    my_hostname = socket.gethostname()

    attempts = 10

    for attempt in range(attempts):
        try:
            hosts_on_this_machine = get_hosts_on_machine(machine_id)
            break
        except Exception as e:
            logger.error(f"Error getting hosts on machine (attempt {attempt + 1}/{attempts}): {e}")
            hosts_on_this_machine = []
            time.sleep(1)

    gpu_index = 0

    for i, host in enumerate(hosts_on_this_machine):
        if host == my_hostname:
            gpu_index = i % gpu_count
            break

    logger.info(
        f"Rank {rank} on host {my_hostname} out of {len(hosts_on_this_machine)} with "
        f"machine ID {machine_id} assigned GPU {gpu_index} out of {gpu_count} available GPUs."
    )
    return gpu_index


def get_machine_id():
    machine_id = None
    try:
        machine_id = get_board_serial()
    except Exception as e:
        logger.error(f"Error reading machine ID: {e}")
    return machine_id


def get_board_serial() -> str | None:
    result = subprocess.run(
        ["dmidecode", "-s", "baseboard-serial-number"],
        capture_output=True, text=True
    )
    serial = result.stdout.strip()
    return serial if serial else None


def get_hosts_on_machine(machine_id):
    node_path = "/app/cray/nfs/nodes"

    hostnames = []

    for filename in os.listdir(node_path):
        if filename.endswith(".json"):
            with open(os.path.join(node_path, filename), "r") as f:
                node_info = json.load(f)
                if node_info.get("machine_id") == machine_id:
                    hostnames.append(node_info["hostname"])

    return list(sorted(hostnames))
