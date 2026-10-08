"""CPU tests for the grpo dataset (load_grpo_dataset.py) and init_adapter_from (adapters/).

No job config / tokenizer download: the dataset class is exercised directly with a stub
tokenizer, and the strict adapter-state loader with a tiny nn.Module wrapped like PEFT.
"""
import math

import pytest
import torch
from torch import nn

from adapters.init_adapter_from import load_adapter_state_into, resolve_init_adapter_path
from cray_megatron.megatron.dataset.load_grpo_dataset import GrpoDataset


class StubTokenizer:
    """Whitespace tokenizer: token id = 100 + word length; the SFT-style prefix rule applies."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text):
        return {"input_ids": [100 + len(w) for w in text.split()]}


def _ds(rows, world=1, rank=0, max_len=64):
    return GrpoDataset(rows, StubTokenizer(), epoch=0, rank=rank, world_size=world, max_len=max_len)


def test_token_ids_used_verbatim_and_aligned_with_old_logprobs():
    row = {"prompt": "p", "completion": "c", "advantage": 0.5,
           "prompt_token_ids": [2, 7, 8], "completion_token_ids": [30, 31, 106],
           "old_logprobs": [-0.1, -0.2, -0.3]}
    item = _ds([row]).encode(row)
    assert item["input_ids"].tolist() == [2, 7, 8, 30, 31, 106]
    assert item["completion_mask"].tolist() == [0, 0, 0, 1, 1, 1]
    assert item["old_logprobs"].tolist()[3:] == pytest.approx([-0.1, -0.2, -0.3])
    assert item["advantage"].item() == 0.5


def test_text_fallback_masks_prompt_prefix():
    row = {"prompt": "aa bbb", "completion": " cccc d", "advantage": -1.0, "old_logprobs": None}
    item = _ds([row]).encode(row)
    # prompt tokens: 102, 103 ; completion tokens: 104, 101
    assert item["input_ids"].tolist() == [102, 103, 104, 101]
    assert item["completion_mask"].tolist() == [0, 0, 1, 1]
    assert all(math.isnan(x) for x in item["old_logprobs"].tolist()[2:])  # on-policy marker


def test_misaligned_old_logprobs_raise():
    row = {"prompt": "p", "completion": "c", "advantage": 0.0,
           "prompt_token_ids": [2], "completion_token_ids": [3, 4], "old_logprobs": [-1.0]}
    with pytest.raises(ValueError):
        _ds([row]).encode(row)


def test_truncation_keeps_prompt_and_cuts_completion_tail():
    row = {"prompt": "p", "completion": "c", "advantage": 1.0,
           "prompt_token_ids": [2, 3], "completion_token_ids": [10, 11, 12, 13],
           "old_logprobs": [-1, -2, -3, -4]}
    ds = _ds([row], max_len=4)
    item = ds.encode(row)
    assert item["input_ids"].tolist() == [2, 3, 10, 11]
    assert item["old_logprobs"].tolist()[2:] == [-1.0, -2.0]
    assert ds.n_truncated == 1


def test_collate_right_pads_and_sharding():
    rows = [{"prompt": "p", "completion": "c", "advantage": float(i),
             "prompt_token_ids": [2], "completion_token_ids": [5] * (i + 1),
             "old_logprobs": [-0.5] * (i + 1)} for i in range(4)]
    ds = _ds(rows)
    batch = ds.collate_fn([ds.encode(r) for r in rows])
    assert batch["input_ids"].shape == (4, 5)
    assert batch["attention_mask"].sum(1).tolist() == [2, 3, 4, 5]
    assert batch["completion_mask"].sum(1).tolist() == [1, 2, 3, 4]
    assert batch["input_ids"][0].tolist() == [2, 5, 0, 0, 0]  # pad id 0
    # Two ranks split the shuffled rows disjointly and completely.
    seen = []
    for rank in range(2):
        seen += [int(it["advantage"]) for it in _ds(rows, world=2, rank=rank)]
    assert sorted(seen) == [0, 1, 2, 3]


# --- init_adapter_from ------------------------------------------------------------------

class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)


class Wrapper(nn.Module):  # PEFT-like: wrapper.model is the module the checkpoint keys refer to
    def __init__(self):
        super().__init__()
        self.model = Inner()


def test_strict_load_into_descendant_and_count():
    src, dst = Wrapper(), Wrapper()
    sd = {"lora_A.weight": src.model.lora_A.weight.detach().clone(),
          "lora_B.weight": src.model.lora_B.weight.detach().clone()}
    n, target = load_adapter_state_into(dst, sd)
    assert n == 2 and target == "Inner"
    assert torch.equal(dst.model.lora_B.weight, src.model.lora_B.weight)
    assert not torch.equal(dst.model.lin.weight, src.model.lin.weight)  # untouched frozen base


def test_unknown_checkpoint_key_fails_loudly():
    with pytest.raises(RuntimeError):
        load_adapter_state_into(Wrapper(), {"lora_C.weight": torch.zeros(2, 2)})


def test_resolve_job_hash_picks_highest_step(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "abc").mkdir(parents=True)
    for s in (9, 100, 25):
        (jobs / "abc" / f"checkpoint_{s}.pt").write_bytes(b"x")
    me = jobs / "def"
    me.mkdir()
    assert resolve_init_adapter_path("abc", str(me)).endswith("checkpoint_100.pt")
    assert resolve_init_adapter_path(str(jobs / "abc"), str(me)).endswith("checkpoint_100.pt")
    assert resolve_init_adapter_path(str(jobs / "abc" / "checkpoint_9.pt"), str(me)).endswith("checkpoint_9.pt")
    with pytest.raises(FileNotFoundError):
        resolve_init_adapter_path("nope", str(me))
