"""Unit tests for adapter checkpoint selection.

Both vLLM-side call sites picked `sorted(glob("*.pt"))[0]`, which orders
lexically. A job that saved several checkpoints therefore served an
arbitrary earlier one: it loads cleanly, reports every tensor matched,
and answers with garbage because the model is undertrained.

Observed on a 2000-step run whose final checkpoint produced the correct
answer under HuggingFace while vLLM served an earlier one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import LATEST_CHECKPOINT_SRC  # noqa: E402

_namespace: dict = {}
exec(LATEST_CHECKPOINT_SRC, _namespace)
latest_checkpoint = _namespace["_scalarlm_latest_checkpoint"]


def _files(*names):
    return [Path("/jobs/hash") / name for name in names]


def test_picks_highest_step_not_first_alphabetically():
    """The exact ordering that caused the bug: 1000 sorts before 1999."""
    chosen = latest_checkpoint(_files("checkpoint_1000.pt", "checkpoint_1999.pt"))

    assert chosen.name == "checkpoint_1999.pt"


def test_lexical_order_puts_a_smaller_step_last():
    """checkpoint_500.pt sorts AFTER both, so neither `[0]` nor `[-1]`
    is correct -- only numeric comparison is."""
    chosen = latest_checkpoint(
        _files("checkpoint_1000.pt", "checkpoint_1999.pt", "checkpoint_500.pt")
    )

    assert chosen.name == "checkpoint_1999.pt"


def test_single_checkpoint_is_returned():
    chosen = latest_checkpoint(_files("checkpoint_1999.pt"))

    assert chosen.name == "checkpoint_1999.pt"


def test_non_numeric_names_lose_to_numbered_ones():
    """A stray file shouldn't outrank a real checkpoint."""
    chosen = latest_checkpoint(_files("adapter.pt", "checkpoint_10.pt"))

    assert chosen.name == "checkpoint_10.pt"


def test_all_non_numeric_falls_back_deterministically():
    """With nothing to compare numerically both call sites must still
    agree, so the choice has to be a function of the sorted order."""
    chosen = latest_checkpoint(_files("b.pt", "a.pt", "c.pt"))

    assert chosen.name == "a.pt"


def test_selection_is_order_independent():
    """glob() order is filesystem-dependent; the result must not be."""
    names = ("checkpoint_500.pt", "checkpoint_1999.pt", "checkpoint_1000.pt")
    forward = latest_checkpoint(_files(*names))
    backward = latest_checkpoint(_files(*reversed(names)))

    assert forward == backward == Path("/jobs/hash/checkpoint_1999.pt")
