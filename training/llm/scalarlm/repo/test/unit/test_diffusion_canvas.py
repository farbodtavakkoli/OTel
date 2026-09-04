"""
Unit tests for cray_megatron.megatron.dataset.diffusion_canvas — the pure
canvas-tokenization used by the DiffusionGemma dataset loader.

Verifies canvas pad/truncate at the boundary, clean-labels construction (-100 on
padding), encoder/canvas field shapes, and the pad-token fallback. A fake
tokenizer keeps this free of transformers/datasets/gpu_aware_mpi imports.
"""

import logging

from cray_megatron.megatron.dataset.diffusion_canvas import (
    anchor_token_id,
    pad_token_id,
    tokenize_canvas_batch,
)


class _FakeTokenizer:
    """Maps each character to a distinct in-vocab token id (>= 2). add_special_tokens
    is accepted and ignored (the fake adds none either way)."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, add_special_tokens=True):
        ids = [[(ord(c) % 40) + 2 for c in t] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def test_canvas_pads_short_output_with_clean_labels():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(tok, canvas_length=8, inputs=["hi"], outputs=["ab"])

    # Encoder side is the prompt tokenized as-is.
    assert out["encoder_input_ids"] == [[(ord("h") % 40) + 2, (ord("i") % 40) + 2]]
    assert out["encoder_attention_mask"] == [[1, 1]]

    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    assert len(ci) == 8 and len(cl) == 8

    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    # Clean canvas is pad-filled; labels are the clean tokens with -100 on padding.
    assert ci == [t0, t1, 0, 0, 0, 0, 0, 0]
    assert cl == [t0, t1, -100, -100, -100, -100, -100, -100]


def test_canvas_exact_length_has_no_padding():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(tok, canvas_length=3, inputs=["x"], outputs=["abc"])
    cl = out["canvas_labels"][0]
    assert len(cl) == 3
    assert all(v != -100 for v in cl)  # fully supervised, no padding


def test_canvas_truncates_oversized_output(caplog):
    tok = _FakeTokenizer()
    with caplog.at_level(logging.WARNING):
        out = tokenize_canvas_batch(tok, canvas_length=3, inputs=["x"], outputs=["abcde"])

    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    assert len(ci) == 3 and len(cl) == 3
    # Truncated to the first 3 tokens; all supervised (no -100 padding).
    assert all(v != -100 for v in cl)
    t0, t1, t2 = ((ord(c) % 40) + 2 for c in "abc")
    assert ci == [t0, t1, t2]
    assert "truncating" in caplog.text.lower()


def test_multiple_rows_independent_padding():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(
        tok, canvas_length=4, inputs=["a", "bb"], outputs=["a", "bbb"]
    )
    assert len(out["canvas_labels"]) == 2
    assert out["canvas_labels"][0].count(-100) == 3  # 1 token + 3 pad
    assert out["canvas_labels"][1].count(-100) == 1  # 3 tokens + 1 pad


class _FakeTokenizerBOS(_FakeTokenizer):
    """Adds a BOS id distinct from every content token (content ids are >= 2 and
    < 42; 99 can't collide)."""

    bos_token_id = 99


def test_anchor_prepends_supervised_bos():
    tok = _FakeTokenizerBOS()
    anchor = anchor_token_id(tok)
    assert anchor == 99

    out = tokenize_canvas_batch(
        tok, canvas_length=8, inputs=["hi"], outputs=["ab"], anchor_id=anchor
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]

    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    # Anchor is a clean, supervised prefix at position 0; the output shifts right by
    # one and padding fills the rest.
    assert ci == [99, t0, t1, 0, 0, 0, 0, 0]
    assert cl == [99, t0, t1, -100, -100, -100, -100, -100]
    assert len(ci) == 8 and len(cl) == 8


def test_anchor_shrinks_output_budget_by_one():
    tok = _FakeTokenizerBOS()
    # 3-token output into a length-3 canvas: with the anchor, budget is 2 -> the
    # third output token is truncated to make room for the anchor.
    out = tokenize_canvas_batch(
        tok, canvas_length=3, inputs=["x"], outputs=["abc"], anchor_id=99
    )
    ci = out["canvas_input_ids"][0]
    t0, t1 = ((ord(c) % 40) + 2 for c in "ab")
    assert ci == [99, t0, t1]  # anchor + first 2 output tokens, no room for 'c'


def test_anchor_none_is_byte_identical_to_pre_anchor():
    tok = _FakeTokenizerBOS()
    with_default = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"]
    )
    with_explicit_none = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"], anchor_id=None
    )
    assert with_default == with_explicit_none
    assert with_default["canvas_input_ids"][0][0] != 99  # no anchor leaked in


def test_anchor_token_id_none_without_bos():
    # The plain fake has no bos_token_id attribute -> resolver returns None so the
    # loader can warn and train without an anchor rather than inventing an id.
    assert anchor_token_id(_FakeTokenizer()) is None


def test_supervise_termination_appends_eos_and_supervises_tail():
    tok = _FakeTokenizer()  # pad_token_id=0, eos_token_id=1
    out = tokenize_canvas_batch(
        tok, canvas_length=8, inputs=["hi"], outputs=["ab"],
        supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]

    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    # Answer, then a terminating EOS (1), then the pad tail (0) — and every slot is
    # supervised (labels mirror inputs; no -100 anywhere).
    assert ci == [t0, t1, 1, 0, 0, 0, 0, 0]
    assert cl == [t0, t1, 1, 0, 0, 0, 0, 0]
    assert -100 not in cl
    # No loss-weight field at the default uniform weight.
    assert "canvas_loss_weight" not in out


def test_supervise_termination_over_budget_reserves_eos_slot():
    tok = _FakeTokenizer()
    # 5-token output into a length-3 canvas: budget = 3 - 0 (no anchor) - 1 (EOS) = 2.
    out = tokenize_canvas_batch(
        tok, canvas_length=3, inputs=["x"], outputs=["abcde"],
        supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    t0, t1 = ((ord(c) % 40) + 2 for c in "ab")
    # First 2 answer tokens + EOS fill the canvas exactly; EOS is never truncated.
    assert ci == [t0, t1, 1]
    assert cl == [t0, t1, 1]
    assert len(ci) == 3


def test_supervise_termination_off_is_byte_identical():
    tok = _FakeTokenizer()
    default = tokenize_canvas_batch(tok, canvas_length=6, inputs=["hi"], outputs=["ab"])
    explicit_off = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"],
        supervise_termination=False,
    )
    assert default == explicit_off
    assert -100 in default["canvas_labels"][0]  # tail still masked when off


def test_supervise_termination_with_anchor():
    tok = _FakeTokenizerBOS()  # bos=99
    out = tokenize_canvas_batch(
        tok, canvas_length=8, inputs=["hi"], outputs=["ab"],
        anchor_id=99, supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    # anchor(99) + answer + EOS(1) + pad(0) tail, all supervised.
    assert ci == [99, t0, t1, 1, 0, 0, 0, 0]
    assert cl == [99, t0, t1, 1, 0, 0, 0, 0]
    assert len(ci) == 8


def test_supervise_termination_emits_loss_weight_when_downweighted():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"],
        supervise_termination=True, pad_loss_weight=0.25,
    )
    w = out["canvas_loss_weight"][0]
    # answer(2) + EOS(1) weight 1.0; the 3 pad-tail slots weight 0.25.
    assert w == [1.0, 1.0, 1.0, 0.25, 0.25, 0.25]
    assert len(w) == 6
    # Uniform weight (1.0) emits no field even with supervise_termination on.
    out_uniform = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"],
        supervise_termination=True, pad_loss_weight=1.0,
    )
    assert "canvas_loss_weight" not in out_uniform


def test_supervise_termination_no_eos_token():
    class NoEos(_FakeTokenizer):
        eos_token_id = None
        pad_token_id = 3

    out = tokenize_canvas_batch(
        NoEos(), canvas_length=5, inputs=["x"], outputs=["ab"],
        supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    # No EOS to append -> answer + supervised pad(3) tail, no reserved EOS slot.
    assert ci == [t0, t1, 3, 3, 3]
    assert cl == [t0, t1, 3, 3, 3]
    assert -100 not in cl


def test_pad_token_id_fallbacks():
    class NoPad:
        pad_token_id = None
        eos_token_id = 7

    assert pad_token_id(NoPad()) == 7

    class NoPadNoEos:
        pad_token_id = None
        eos_token_id = None

    assert pad_token_id(NoPadNoEos()) == 0

    class HasPad:
        pad_token_id = 5
        eos_token_id = 9

    assert pad_token_id(HasPad()) == 5
