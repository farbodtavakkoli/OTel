"""Dataset loader for DiffusionGemma (`DiffusionGemmaForBlockDiffusion`)."""

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from cray_megatron.megatron.dataset.diffusion_canvas import (
    anchor_token_id,
    tokenize_canvas_batch,
)

import datasets
import jsonlines

import logging

logger = logging.getLogger(__name__)

# Fallback when a job omits the diffusion block; matches config.canvas_length.
_DEFAULT_CANVAS_LENGTH = 256


def load_diffusion_dataset(model, tokenizer, epoch):
    canvas_length = _get_canvas_length()
    anchor_id = _resolve_anchor_id(tokenizer)
    supervise_termination = _supervise_termination()
    pad_loss_weight = _pad_loss_weight()

    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "input": datasets.Value(dtype="string"),
                "output": datasets.Value(dtype="string"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_canvas_tokenize_function(
            tokenizer,
            canvas_length,
            anchor_id,
            supervise_termination,
            pad_loss_weight,
        ),
        batched=True,
        remove_columns=["input", "output"],
    )

    torch_dataset = tokenized_dataset.with_format("torch")

    return torch_dataset


def make_dataset_generator():
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                yield obj

    return read_dataset


def get_dataset_path():
    job_config = get_job_config()
    return job_config["training_data_path"]


def split_dataset_by_node(dataset):
    data_parallel_rank = get_data_parallel_rank()
    data_parallel_world_size = get_data_parallel_world_size()

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset


def _get_canvas_length():
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    # The nested block may be a dict or a pydantic model depending on the caller.
    if hasattr(diffusion, "canvas_length"):
        return diffusion.canvas_length
    return diffusion.get("canvas_length", _DEFAULT_CANVAS_LENGTH)


def _anchor_enabled():
    """Whether the canvas anchor token is requested via the diffusion job config."""
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    if hasattr(diffusion, "anchor_token"):
        return bool(diffusion.anchor_token)
    return bool(diffusion.get("anchor_token", False))


def _supervise_termination():
    """Whether to supervise the full canvas instead of masking the tail with -100."""
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    if hasattr(diffusion, "supervise_termination"):
        return bool(diffusion.supervise_termination)
    return bool(diffusion.get("supervise_termination", False))


def _pad_loss_weight():
    """Relative CE weight on the supervised pad tail; 1.0 = uniform."""
    job_config = get_job_config()
    diffusion = job_config.get("diffusion") or {}
    if hasattr(diffusion, "pad_loss_weight"):
        return float(diffusion.pad_loss_weight)
    return float(diffusion.get("pad_loss_weight", 1.0))


def _resolve_anchor_id(tokenizer):
    """Resolve the anchor token id when enabled, else None."""
    if not _anchor_enabled():
        return None
    anchor_id = anchor_token_id(tokenizer)
    if anchor_id is None:
        logger.warning(
            "diffusion.anchor_token is set but the tokenizer has no bos_token_id; "
            "training without a canvas anchor."
        )
    return anchor_id


def get_canvas_tokenize_function(
    tokenizer,
    canvas_length,
    anchor_id=None,
    supervise_termination=False,
    pad_loss_weight=1.0,
):
    def tokenize(dataset):
        return tokenize_canvas_batch(
            tokenizer,
            canvas_length,
            dataset["input"],
            dataset["output"],
            anchor_id,
            supervise_termination,
            pad_loss_weight,
        )

    return tokenize
