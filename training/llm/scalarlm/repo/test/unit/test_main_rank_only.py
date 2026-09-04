"""
Unit tests for the @main_rank_only decorator.

Contract under test: docs/architecture.md §13 — the decorator gates
rank-0-only I/O behind two surrounding barriers, with a re-entrance guard
that prevents nested @main_rank_only calls from double-barriering.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_reentry_flag():
    """
    The module-level `_in_main_rank_only` flag is intentionally global
    (shared across all @main_rank_only-wrapped functions in a process). The
    wrapper restores it to False in a finally block, but a previous test
    that aborts mid-call could leave it True. Reset it defensively.
    """
    import cray_megatron.collectives.main_rank_only as m
    m._in_main_rank_only = False
    yield
    m._in_main_rank_only = False


def _patch_rank(monkeypatch, rank: int):
    """
    Replace `get_rank` and `barrier` in the decorator's module namespace.
    The symbols are imported by-name at module load, so patching at the
    gpu_aware_mpi origin would not affect the already-bound local names.
    """
    barriers = []
    monkeypatch.setattr(
        "cray_megatron.collectives.main_rank_only.get_rank",
        lambda: rank,
    )
    monkeypatch.setattr(
        "cray_megatron.collectives.main_rank_only.barrier",
        lambda: barriers.append("b"),
    )
    return barriers


def test_runs_on_rank_zero_and_returns_value(monkeypatch):
    barriers = _patch_rank(monkeypatch, rank=0)
    from cray_megatron.collectives.main_rank_only import main_rank_only

    invocations = []

    @main_rank_only
    def do_thing(x):
        invocations.append(x)
        return x * 2

    result = do_thing(21)

    assert result == 42
    assert invocations == [21]
    # One barrier before, one after.
    assert len(barriers) == 2


def test_skips_on_non_zero_rank_and_returns_none(monkeypatch):
    barriers = _patch_rank(monkeypatch, rank=1)
    from cray_megatron.collectives.main_rank_only import main_rank_only

    invocations = []

    @main_rank_only
    def do_thing():
        invocations.append("ran")
        return "value"

    result = do_thing()

    # Non-rank-0: function body does not execute, wrapper returns None.
    assert result is None
    assert invocations == []
    # Barriers still execute on every rank — they are collective.
    assert len(barriers) == 2


def test_is_main_rank_helper_tracks_rank(monkeypatch):
    monkeypatch.setattr(
        "cray_megatron.collectives.main_rank_only.get_rank", lambda: 0
    )
    from cray_megatron.collectives.main_rank_only import is_main_rank

    assert is_main_rank() is True

    monkeypatch.setattr(
        "cray_megatron.collectives.main_rank_only.get_rank", lambda: 2
    )

    assert is_main_rank() is False


def test_reentrant_call_skips_inner_barriers(monkeypatch):
    barriers = _patch_rank(monkeypatch, rank=0)
    from cray_megatron.collectives.main_rank_only import main_rank_only

    invocations = []

    @main_rank_only
    def inner():
        invocations.append("inner")

    @main_rank_only
    def outer():
        invocations.append("outer")
        inner()

    outer()

    assert invocations == ["outer", "inner"]
    # Outer contributes 2 barriers; inner detects it is re-entrant and runs
    # the body directly with no additional barriers.
    assert len(barriers) == 2


def test_exception_inside_function_resets_reentry_flag(monkeypatch):
    _patch_rank(monkeypatch, rank=0)
    import cray_megatron.collectives.main_rank_only as m
    from cray_megatron.collectives.main_rank_only import main_rank_only

    @main_rank_only
    def bad():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        bad()

    # A raised exception must not leave _in_main_rank_only sticky.
    assert m._in_main_rank_only is False
