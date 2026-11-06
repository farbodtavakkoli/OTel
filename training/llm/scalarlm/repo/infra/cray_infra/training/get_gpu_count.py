from cray_infra.training.slurm_capacity import count_slurm_gpus


def get_gpu_count():
    return count_slurm_gpus()
