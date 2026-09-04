"""Sequence-classification dataset loading."""

from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.dataset.load_language_model_dataset import (
    split_dataset_by_node,
    get_dataset_path,
)

import datasets
import jsonlines

import logging

logger = logging.getLogger(__name__)


def load_classification_dataset(model, tokenizer, epoch):
    """Load dataset for sequence-classification training."""
    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "text": datasets.Value(dtype="string"),
                "label": datasets.Value(dtype="int64"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_tokenize_function_classification(model, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    return tokenized_dataset.with_format("torch")


def make_dataset_generator():
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                yield obj

    return read_dataset


def get_tokenize_function_classification(model, tokenizer):
    """Tokenize for sequence classification; padding MUST be on the left."""
    job_config = get_job_config()
    max_length = job_config["max_token_block_size"]

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(dataset):
        tokens = tokenizer(
            dataset["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        tokens["labels"] = dataset["label"]

        return tokens

    return tokenize
