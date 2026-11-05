from cray_infra.util.get_job_config import get_job_config

from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

import datasets
import jsonlines
import random

import logging

logger = logging.getLogger(__name__)


def load_language_model_dataset(model, tokenizer, epoch):
    pretokenized = get_job_config().get("pretokenized_dataset", False)
    if pretokenized:
        features = datasets.Features(
            {
                key: datasets.Sequence(datasets.Value(dtype="int64"))
                for key in ("input_ids", "labels", "attention_mask")
            }
        )
        validate_row = get_pretokenized_validate_function(model)
    else:
        features = datasets.Features(
            {
                "input": datasets.Value(dtype="string"),
                "output": datasets.Value(dtype="string"),
            }
        )
        validate_row = None
    hf_dataset = datasets.IterableDataset.from_generator(
        make_dataset_generator(validate_row),
        features=features,
    )
    shuffled_dataset = hf_dataset.shuffle(seed=42 + epoch, buffer_size=256)
    split_dataset = split_dataset_by_node(shuffled_dataset)

    if pretokenized:
        tokenized_dataset = split_dataset
    else:
        tokenized_dataset = split_dataset.map(
            get_tokenize_function(model, tokenizer),
            batched=True,
            remove_columns=[
                "input",
                "output",
            ],
        )

    packed_dataset = tokenized_dataset.map(
        get_pack_function(model, tokenizer),
        batched=True,
    )

    torch_dataset = packed_dataset.with_format("torch")

    return torch_dataset


def make_dataset_generator(validate_row=None):
    def read_dataset():
        dataset_path = get_dataset_path()
        with open(dataset_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            for obj in reader:
                yield validate_row(obj) if validate_row is not None else obj

    return read_dataset


def get_pretokenized_validate_function(model):
    config = getattr(model.config, "text_config", model.config)
    vocab_size = config.vocab_size
    budget = min(
        get_max_position_embeddings(model.config),
        get_job_config()["max_token_block_size"],
    )

    def validate_row(row):
        keys = ("input_ids", "labels", "attention_mask")
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), list) for key in keys
        ):
            raise ValueError("Pretokenized rows require input_ids, labels and attention_mask lists")
        ids, labels, attention = (row[key] for key in keys)
        if len(ids) < 2 or len(ids) > budget:
            raise ValueError(f"Pretokenized row length must be between 2 and packing budget {budget}")
        if len(labels) != len(ids) or len(attention) != len(ids):
            raise ValueError("Pretokenized row has misaligned labels or attention_mask")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError("Pretokenized input_ids must be integer token ids within the model vocabulary")
        if any(
            type(label) is not int or (label != -100 and label != token)
            for token, label in zip(ids, labels)
        ):
            raise ValueError("Pretokenized labels must be -100 or the corresponding input token id")
        if any(type(mask) is not int or mask not in (0, 1) for mask in attention):
            raise ValueError("Pretokenized attention_mask must contain integer 0 or 1 values")
        if any(label != -100 and mask == 0 for label, mask in zip(labels, attention)):
            raise ValueError("Pretokenized labels must mask unattended tokens with -100")
        if not any(
            labels[i] != -100 and attention[i - 1] == 1 and attention[i] == 1
            for i in range(1, len(ids))
        ):
            raise ValueError("Pretokenized row must contain a supervised shifted target")
        return {key: row[key] for key in keys}

    return validate_row


def get_dataset_path():
    job_config = get_job_config()

    # custom_data_path overrides the tarball's training_data_path when set.
    custom_path = job_config.get("custom_data_path")
    if custom_path:
        logger.info(f"Using custom_data_path override: {custom_path}")
        return custom_path

    return job_config["training_data_path"]


def split_dataset_by_node(dataset):
    data_parallel_rank = get_data_parallel_rank()
    data_parallel_world_size = get_data_parallel_world_size()

    num_shards = data_parallel_world_size
    index = data_parallel_rank

    filtered_dataset = dataset.filter(
        lambda example, idx: idx % data_parallel_world_size == data_parallel_rank,
        with_indices=True,
    )

    return filtered_dataset


def get_tokenize_function(model, tokenizer):
    def tokenize(dataset):
        text = [
            input_text + output_text
            for input_text, output_text in zip(dataset["input"], dataset["output"])
        ]

        tokens = tokenizer(text)

        tokens = add_eos_token(tokens, model, tokenizer)

        input_text_lengths = [
            len(tokenizer(input_text)["input_ids"]) for input_text in dataset["input"]
        ]

        # Labels are -100 over the input text and the token ids over the output.
        tokens["labels"] = [
            [-100] * input_text_length + input_ids[input_text_length:]
            for input_text_length, input_ids in zip(
                input_text_lengths, tokens["input_ids"]
            )
        ]

        return tokens

    return tokenize


def add_eos_token(tokens, model, tokenizer):
    # add stop token to the end of the sequence
    if model.generation_config is None:
        eos_token = tokenizer.eos_token_id
    elif model.generation_config.eos_token_id is None:
        eos_token = tokenizer.eos_token_id
    else:
        if isinstance(model.generation_config.eos_token_id, list):
            eos_token = model.generation_config.eos_token_id[-1]
        else:
            eos_token = model.generation_config.eos_token_id

    if eos_token is None:
        return tokens
    for i, input_ids in enumerate(tokens["input_ids"]):
        if not input_ids or input_ids[-1] != eos_token:
            tokens["input_ids"][i] = input_ids + [eos_token]
            tokens["attention_mask"][i] = tokens["attention_mask"][i] + [1]
            if "labels" in tokens:
                tokens["labels"][i] = tokens["labels"][i] + [eos_token]

    return tokens


def get_pack_function(model, tokenizer=None):
    job_config = get_job_config()

    # Pads the tail block; never learned from, but must be a valid id, not a sentinel.
    pad_token_id = None
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    # Cap block size at max_position_embeddings so position_ids stay in-bounds for RoPE.
    budget = min(
        get_max_position_embeddings(model.config), job_config["max_token_block_size"]
    )

    if budget <= 0:
        raise ValueError("max_token_block_size must be positive")

    if job_config.get("pretokenized_dataset", False):
        validate_row = get_pretokenized_validate_function(model)

        def pack_whole_rows(dataset):
            keys = list(dict.fromkeys([*dataset, "position_ids", "document_ids"]))
            result = {key: [] for key in keys}
            block = {key: [] for key in keys}
            doc_idx = 0

            def flush():
                pad_count = budget - len(block["input_ids"])
                pad_values = {
                    "input_ids": pad_token_id,
                    "labels": -100,
                    "position_ids": 0,
                    "document_ids": doc_idx,
                }
                for key in keys:
                    result[key].append(block[key] + [pad_values.get(key, 0)] * pad_count)
                    block[key] = []

            for i in range(len(dataset["input_ids"])):
                row = {key: dataset[key][i] for key in dataset}
                validate_row(row)
                doc_len = len(row["input_ids"])
                for key, values in row.items():
                    if len(values) != doc_len:
                        raise ValueError(f"Tokenized row {i} has misaligned {key}")
                if len(block["input_ids"]) + doc_len > budget:
                    flush()
                    doc_idx = 0
                row["labels"] = [-100] + row["labels"][1:]
                row["position_ids"] = list(range(doc_len))
                row["document_ids"] = [doc_idx] * doc_len
                for key in keys:
                    block[key].extend(row[key])
                doc_idx += 1
            if block["input_ids"]:
                flush()
            return result

        return pack_whole_rows

    def pack(dataset):
        # position_ids and document_ids are what stop packed docs bleeding into each other.

        compressed_input_ids = []
        compressed_labels = []
        compressed_position_ids = []
        compressed_document_ids = []

        doc_idx = 0
        for i in range(len(dataset["input_ids"])):
            ids = dataset["input_ids"][i]
            labels = dataset["labels"][i]
            doc_len = len(ids)

            if doc_len > budget:
                raise ValueError(
                    f"Tokenized row {i} has {doc_len} tokens, exceeding packing budget {budget}; "
                    "no tokens were truncated. Increase max_token_block_size or prepare the data explicitly."
                )
            if len(labels) != doc_len:
                raise ValueError(f"Tokenized row {i} has misaligned labels")
            compressed_input_ids.extend(ids)
            compressed_labels.extend([-100] + labels[1:] if doc_len else [])
            compressed_position_ids.extend(range(doc_len))
            compressed_document_ids.extend([doc_idx] * doc_len)
            doc_idx += 1

        concatenated_dataset = {
            "input_ids": compressed_input_ids,
            "labels": compressed_labels,
            "position_ids": compressed_position_ids,
            "document_ids": compressed_document_ids,
        }
        extra_keys = [k for k in dataset.keys() if k not in concatenated_dataset]
        for k in extra_keys:
            concatenated_dataset[k] = []
            for i, val in enumerate(dataset[k]):
                if len(val) != len(dataset["input_ids"][i]):
                    raise ValueError(f"Tokenized row {i} has misaligned {k}")
                concatenated_dataset[k].extend(val)

        # Pad the tail to a whole block rather than truncating: short blocks fail torch.stack.
        total_length = len(concatenated_dataset.get("input_ids", []))

        if total_length == 0:
            return {k: [] for k in concatenated_dataset}

        padded_length = ((total_length + budget - 1) // budget) * budget
        pad_count = padded_length - total_length

        if pad_count > 0:
            # A document id past the last real document, so real documents cannot attend into padding.
            pad_values = {
                "input_ids": pad_token_id,
                "labels": -100,
                "position_ids": 0,
                "document_ids": doc_idx,
            }
            for k in concatenated_dataset:
                # 0 marks a position as not-real for attention_mask and other auxiliary streams.
                concatenated_dataset[k].extend([pad_values.get(k, 0)] * pad_count)

        result = {
            k: [t[i : i + budget] for i in range(0, padded_length, budget)]
            for k, t in concatenated_dataset.items()
        }

        return result


    return pack

def get_max_position_embeddings(config):
    if hasattr(config, 'max_position_embeddings'):
        return config.max_position_embeddings
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings'):
        return config.text_config.max_position_embeddings
    if hasattr(config, 'n_positions'):
        return config.n_positions
    raise AttributeError(f"Cannot find max_position_embeddings in {type(config)}")

