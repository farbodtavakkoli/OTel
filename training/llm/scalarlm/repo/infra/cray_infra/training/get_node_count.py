from cray_infra.training.slurm_capacity import count_slurm_nodes


def get_node_count():
    """
    Number of MPI nodes scalarlm has for training.

    One entry per slurmd-registered node, which is one per running
    scalarlm-megatron pod. Falls back to 1 (the current host) for
    local dev / early startup, where slurmctld isn't up yet.
    """
    return count_slurm_nodes()
