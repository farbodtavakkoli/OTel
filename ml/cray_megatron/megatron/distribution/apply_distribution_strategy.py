import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from cray_infra.training.distribution_strategy.fsdp.fsdp import SimpleFSDP
from gpu_aware_mpi import get_size, get_rank
from ml.get_local_job_config import load_local_training_config
import logging
import os
import socket

logger = logging.getLogger(__name__)


from gpu_aware_mpi import allreduce, get_size

import torch.nn as nn


class SimpleDDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def backward_sync(self):
        for param_name, param in self.model.named_parameters(recurse=False):
            if param.requires_grad and param.grad is not None:
                allreduce(param.grad)
 
def load_distribution_strategy():
    device = get_device()

    strategy = {
        "device": device,
    }

    if get_size() > 1:
        try:
            job_config = load_local_training_config()
            dist_strategy = job_config.get("distribution_strategy", "fsdp")
        except Exception as e:
            logger.warning(f"Could not load local training config: {e}. Defaulting to FSDP.")
            dist_strategy = "fsdp"
        
        if dist_strategy == "ddp":
            if get_rank() == 0:
                logger.info("Using DDP (Distributed Data Parallel) - full model on each GPU")
            strategy["strategy"] = SimpleDDP
        else:
            if get_rank() == 0:
                logger.info("Using FSDP (Fully Sharded Data Parallel) - model sharded across GPUs")
            strategy["strategy"] = SimpleFSDP

    return strategy


def get_device():
    rank = get_rank()
    world_size = get_size()
    
    # Get hostname to verify node distribution
    hostname = socket.gethostname()
    
    # Get environment info
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    node_rank = int(os.environ.get('NODE_RANK', '0'))
    
    logger.info(f"=" * 80)
    logger.info(f"DEVICE SELECTION DIAGNOSTICS - Rank {rank}/{world_size}")
    logger.info(f"Hostname: {hostname}")
    logger.info(f"LOCAL_RANK (from env): {local_rank}")
    logger.info(f"NODE_RANK (from env): {node_rank}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU count on this node: {gpu_count}")
        logger.info(f"Total GPUs for training job: {world_size}")
        
        # List all available GPUs
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.2f} GB")
        
        # Calculate which GPU this rank should use
        selected_gpu = rank % gpu_count
        logger.info(f"Calculated selected_gpu: {selected_gpu} (rank {rank} % gpu_count {gpu_count})")
        
        # Create device
        device = torch.device(f"cuda:{selected_gpu}")
        
        # Set as current device for this process
        torch.cuda.set_device(device)
        current = torch.cuda.current_device()
        logger.info(f"Set current device to: cuda:{current}")
        logger.info(f"Returning device: {device}")
        logger.info(f"=" * 80)
        
        return device
    else:
        logger.info("CUDA not available, using CPU")
        logger.info(f"=" * 80)
        return torch.device("cpu")


def apply_distribution_strategy(model_info):
    distribution_strategy = load_distribution_strategy()
    model_info["distribution_strategy"] = distribution_strategy
    return model_info
