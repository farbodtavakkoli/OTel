from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import LATEST_CHECKPOINT_SRC

_namespace: dict = {}
exec(LATEST_CHECKPOINT_SRC, _namespace)
latest_checkpoint = _namespace["_scalarlm_latest_checkpoint"]


def _files(*names):
    return [Path("/jobs/hash") / name for name in names]


def test_picks_highest_step_not_first_alphabetically():
    chosen = latest_checkpoint(_files("checkpoint_1000.pt", "checkpoint_1999.pt"))

    assert chosen.name == "checkpoint_1999.pt"


def test_lexical_order_puts_a_smaller_step_last():
    chosen = latest_checkpoint(
        _files("checkpoint_1000.pt", "checkpoint_1999.pt", "checkpoint_500.pt")
    )

    assert chosen.name == "checkpoint_1999.pt"


def test_single_checkpoint_is_returned():
    chosen = latest_checkpoint(_files("checkpoint_1999.pt"))

    assert chosen.name == "checkpoint_1999.pt"


def test_non_numeric_names_lose_to_numbered_ones():
    chosen = latest_checkpoint(_files("adapter.pt", "checkpoint_10.pt"))

    assert chosen.name == "checkpoint_10.pt"


def test_all_non_numeric_falls_back_deterministically():
    chosen = latest_checkpoint(_files("b.pt", "a.pt", "c.pt"))

    assert chosen.name == "a.pt"


def test_selection_is_order_independent():
    names = ("checkpoint_500.pt", "checkpoint_1999.pt", "checkpoint_1000.pt")
    forward = latest_checkpoint(_files(*names))
    backward = latest_checkpoint(_files(*reversed(names)))

    assert forward == backward == Path("/jobs/hash/checkpoint_1999.pt")
