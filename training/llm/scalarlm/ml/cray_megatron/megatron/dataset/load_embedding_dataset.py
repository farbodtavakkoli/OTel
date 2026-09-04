
# The causal loader's generator is schema-agnostic, so it is reused here.
from cray_megatron.megatron.dataset.load_language_model_dataset import (
    split_dataset_by_node,
    make_dataset_generator,
)

from cray_infra.util.get_job_config import get_job_config

import datasets

import logging

logger = logging.getLogger(__name__)

def _reject_degenerate_batch_size():
    """Reject batch_size < 2: CoSENT is pairwise within a batch, so one pair yields no gradient."""
    job_config = get_job_config()
    batch_size = job_config.get("batch_size", 1)

    if batch_size is not None and int(batch_size) < 2:
        raise ValueError(
            f"training_mode 'embedding' requires batch_size >= 2, got {batch_size}. "
            "CoSENT is a pairwise ranking loss computed within a batch: with a "
            "single example there is nothing to rank, so the loss is exactly 0.0 "
            "at every step and the job would report COMPLETED without learning "
            "anything. Set batch_size to 2 or more (8 works well)."
        )


def load_embedding_dataset(model, tokenizer, epoch):
    """Load dataset for embedding model training."""
    _reject_degenerate_batch_size()

    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(),
        features=datasets.Features(
            {
                "sentence1": datasets.Value(dtype="string"),
                "sentence2": datasets.Value(dtype="string"),
                "score": datasets.Value(dtype="float"),
            }
        ),
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    tokenized_dataset = split_dataset.map(
        get_tokenize_function_embedding(model, tokenizer),
        batched=True,
        remove_columns=[
            "sentence1",
            "sentence2",
            "score",
        ],
    )

    torch_dataset = tokenized_dataset.with_format("torch")

    return torch_dataset


def get_tokenize_function_embedding(model, tokenizer):
    """Tokenize function for embedding model training."""

    def tokenize(dataset):

        sentence1_tokens = tokenizer(dataset["sentence1"])
        sentence2_tokens = tokenizer(dataset["sentence2"])

        job_config = get_job_config()
        max_length = job_config["max_token_block_size"]

        tokens = {
            "sentence1_input_ids": [],
            "sentence1_attention_mask": [],
            "sentence2_input_ids": [],
            "sentence2_attention_mask": []
        }

        for s1_ids, s1_mask, s2_ids, s2_mask in zip(
            sentence1_tokens["input_ids"],
            sentence1_tokens["attention_mask"],
            sentence2_tokens["input_ids"],
            sentence2_tokens["attention_mask"]
        ):
            s1_ids = s1_ids[:max_length]
            s1_mask = s1_mask[:max_length]
            s2_ids = s2_ids[:max_length]
            s2_mask = s2_mask[:max_length]

            padding_length_s1 = max_length - len(s1_ids)
            padding_length_s2 = max_length - len(s2_ids)

            s1_ids = s1_ids + ([tokenizer.pad_token_id] * padding_length_s1)
            s1_mask = s1_mask + ([0] * padding_length_s1)
            s2_ids = s2_ids + ([tokenizer.pad_token_id] * padding_length_s2)
            s2_mask = s2_mask + ([0] * padding_length_s2)

            tokens["sentence1_input_ids"].append(s1_ids)
            tokens["sentence1_attention_mask"].append(s1_mask)
            tokens["sentence2_input_ids"].append(s2_ids)
            tokens["sentence2_attention_mask"].append(s2_mask)

        tokens["labels"] = dataset["score"]

        return tokens

    return tokenize

