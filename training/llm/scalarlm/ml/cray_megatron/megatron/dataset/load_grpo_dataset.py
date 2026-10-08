from __future__ import annotations

import logging
import math
import random

import jsonlines
import torch

from cray_infra.util.get_job_config import get_job_config
from cray_megatron.collectives.data_parallelism import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from cray_megatron.megatron.dataset.load_language_model_dataset import (
    get_dataset_path,
    get_max_position_embeddings,
)

logger = logging.getLogger(__name__)


def load_grpo_dataset(model, tokenizer, epoch):
    job_config = get_job_config()
    budget = min(
        get_max_position_embeddings(model.config), job_config["max_token_block_size"]
    )
    rows = _read_rows(get_dataset_path())
    return GrpoDataset(
        rows,
        tokenizer,
        epoch=epoch,
        rank=get_data_parallel_rank(),
        world_size=get_data_parallel_world_size(),
        max_len=budget,
    )


def _read_rows(path):
    with open(path) as f:
        rows = list(jsonlines.Reader(f))
    bad = [i for i, r in enumerate(rows) if "advantage" not in r or "prompt" not in r]
    if bad:
        raise ValueError(
            f"grpo dataset {path}: {len(bad)} row(s) lack 'prompt'/'advantage', e.g. rows {bad[:5]}"
        )
    return rows


def _pad_id(tokenizer, model):
    for cand in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
    ):
        if cand is not None:
            return int(cand)
    return 0


class GrpoDataset(torch.utils.data.IterableDataset):
    def __init__(self, rows, tokenizer, epoch, rank, world_size, max_len, pad_token_id=None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.rank = rank
        self.world_size = world_size
        self.max_len = max_len
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else _pad_id(tokenizer, None)
        )
        self.n_truncated = 0
        self.n_on_policy = 0

    # one row -> tensors
    def encode(self, row):
        p_ids = row.get("prompt_token_ids")
        c_ids = row.get("completion_token_ids")
        if p_ids is None or c_ids is None:
            # Same tokenisation as the SFT loader: prefix length of the prompt masks the prompt.
            full = self.tokenizer(row["prompt"] + row["completion"])["input_ids"]
            n_prompt = len(self.tokenizer(row["prompt"])["input_ids"])
            p_ids, c_ids = full[:n_prompt], full[n_prompt:]
        p_ids, c_ids = list(p_ids), list(c_ids)

        old = row.get("old_logprobs")
        if old is not None and len(old) != len(c_ids):
            raise ValueError(
                f"grpo row: old_logprobs has {len(old)} entries but the completion has "
                f"{len(c_ids)} tokens; they must align 1:1 (send vLLM token_ids with logprobs)"
            )
        if old is None:
            self.n_on_policy += 1
            old = [float("nan")] * len(c_ids)

        if len(p_ids) + len(c_ids) > self.max_len:
            # Keep the prompt, drop the completion tail; old_logprobs are cut with it.
            keep = max(0, self.max_len - len(p_ids))
            c_ids, old = c_ids[:keep], old[:keep]
            self.n_truncated += 1
        if not c_ids:
            return None  # nothing to learn from

        ids = p_ids + c_ids
        n_p = len(p_ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "completion_mask": torch.tensor([0.0] * n_p + [1.0] * len(c_ids)),
            "old_logprobs": torch.tensor([0.0] * n_p + [float(x) for x in old], dtype=torch.float32),
            "advantage": torch.tensor(float(row["advantage"]), dtype=torch.float32),
        }

    def __iter__(self):
        order = list(range(len(self.rows)))
        random.Random(42 + self.epoch).shuffle(order)
        mine = [i for k, i in enumerate(order) if k % self.world_size == self.rank]
        for i in mine:
            item = self.encode(self.rows[i])
            if item is not None:
                yield item

    def __len__(self):
        return math.ceil(len(self.rows) / self.world_size)

    # right-pad a micro-batch
    def collate_fn(self, items):
        L = max(x["input_ids"].shape[0] for x in items)
        B = len(items)
        out = {
            "input_ids": torch.full((B, L), self.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, L), dtype=torch.long),
            "completion_mask": torch.zeros((B, L), dtype=torch.float32),
            "old_logprobs": torch.zeros((B, L), dtype=torch.float32),
            "advantage": torch.stack([x["advantage"] for x in items]),
        }
        for b, x in enumerate(items):
            n = x["input_ids"].shape[0]
            for k in ("input_ids", "attention_mask", "completion_mask", "old_logprobs"):
                out[k][b, :n] = x[k]
        return out
