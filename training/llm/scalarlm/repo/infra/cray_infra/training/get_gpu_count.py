from cray_infra.training.slurm_capacity import count_slurm_gpus


def get_gpu_count():
    """
    Total GPUs scalarlm has for training.

    Sums `Gres=gpu:N` across nodes registered with slurmctld, which is
    exactly one entry per running megatron pod and reflects the
    `max_gpus_per_node` cap applied in `write_node_config`. Falls back
    to this host's CUDA device count when slurmctld isn't reachable
    (local dev, early startup).
    """
    return count_slurm_gpus()
