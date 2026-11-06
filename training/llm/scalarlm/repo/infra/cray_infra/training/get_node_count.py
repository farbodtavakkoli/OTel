from cray_infra.training.slurm_capacity import count_slurm_nodes


def get_node_count():
    return count_slurm_nodes()
