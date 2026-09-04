from cray_megatron.megatron.dataset.load_dataset import load_dataset

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import torch


class DataLoader:
    def __init__(self, model, tokenizer, starting_epoch=0):

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = get_batch_size()
        # On resume the trainer passes the saved epoch so the shuffle seed reproduces the ordering.
        self.epoch = starting_epoch

        self.dataset = load_dataset(
            model=self.model,
            tokenizer=self.tokenizer,
            epoch=self.epoch,
        )

        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size
        )

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.dataset = load_dataset(
                model=self.model,
                tokenizer=self.tokenizer,
                epoch=self.epoch,
            )
            self.loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size
            )
            self.iterator = iter(self.loader)

            try:
                return next(self.iterator)
            except StopIteration:
                # An empty reloaded epoch means this rank's shard is empty, not that the epoch ended.
                world_size = get_data_parallel_world_size()
                rank = get_data_parallel_rank()
                raise RuntimeError(
                    f"Data-parallel rank {rank} of {world_size} received an EMPTY "
                    f"dataset shard. split_dataset_by_node assigns record i to "
                    f"rank i % {world_size}, so the dataset needs at least "
                    f"{world_size} records for every rank to get one. Use a "
                    f"larger dataset or request fewer GPUs (gpus <= number of "
                    f"training records)."
                ) from None


def get_batch_size():
    job_config = get_job_config()
    return job_config["batch_size"]
