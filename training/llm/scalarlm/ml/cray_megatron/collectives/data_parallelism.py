from cray_infra.training.distributed import get_rank, get_size

def get_data_parallel_rank():
    return get_rank()


def get_data_parallel_world_size():
    return get_size()
