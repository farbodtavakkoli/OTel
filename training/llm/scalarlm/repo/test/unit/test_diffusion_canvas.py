import logging

from cray_megatron.megatron.dataset.diffusion_canvas import (
    anchor_token_id,
    pad_token_id,
    tokenize_canvas_batch,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, texts, add_special_tokens=True):
        ids = [[(ord(c) % 40) + 2 for c in t] for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def test_canvas_pads_short_output_with_clean_labels():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(tok, canvas_length=8, inputs=["hi"], outputs=["ab"])

    assert out["encoder_input_ids"] == [[(ord("h") % 40) + 2, (ord("i") % 40) + 2]]
    assert out["encoder_attention_mask"] == [[1, 1]]

    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    assert len(ci) == 8 and len(cl) == 8

    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    assert ci == [t0, t1, 0, 0, 0, 0, 0, 0]
    assert cl == [t0, t1, -100, -100, -100, -100, -100, -100]


def test_canvas_exact_length_has_no_padding():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(tok, canvas_length=3, inputs=["x"], outputs=["abc"])
    cl = out["canvas_labels"][0]
    assert len(cl) == 3
    assert all(v != -100 for v in cl)


def test_canvas_truncates_oversized_output(caplog):
    tok = _FakeTokenizer()
    with caplog.at_level(logging.WARNING):
        out = tokenize_canvas_batch(tok, canvas_length=3, inputs=["x"], outputs=["abcde"])

    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    assert len(ci) == 3 and len(cl) == 3
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
    assert out["canvas_labels"][0].count(-100) == 3
    assert out["canvas_labels"][1].count(-100) == 1


class _FakeTokenizerBOS(_FakeTokenizer):
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
    assert ci == [99, t0, t1, 0, 0, 0, 0, 0]
    assert cl == [99, t0, t1, -100, -100, -100, -100, -100]
    assert len(ci) == 8 and len(cl) == 8


def test_anchor_shrinks_output_budget_by_one():
    tok = _FakeTokenizerBOS()
    out = tokenize_canvas_batch(
        tok, canvas_length=3, inputs=["x"], outputs=["abc"], anchor_id=99
    )
    ci = out["canvas_input_ids"][0]
    t0, t1 = ((ord(c) % 40) + 2 for c in "ab")
    assert ci == [99, t0, t1]


def test_anchor_none_is_byte_identical_to_pre_anchor():
    tok = _FakeTokenizerBOS()
    with_default = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"]
    )
    with_explicit_none = tokenize_canvas_batch(
        tok, canvas_length=6, inputs=["hi"], outputs=["ab"], anchor_id=None
    )
    assert with_default == with_explicit_none
    assert with_default["canvas_input_ids"][0][0] != 99


def test_anchor_token_id_none_without_bos():
    assert anchor_token_id(_FakeTokenizer()) is None


def test_supervise_termination_appends_eos_and_supervises_tail():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(
        tok, canvas_length=8, inputs=["hi"], outputs=["ab"],
        supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]

    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
    assert ci == [t0, t1, 1, 0, 0, 0, 0, 0]
    assert cl == [t0, t1, 1, 0, 0, 0, 0, 0]
    assert -100 not in cl
    assert "canvas_loss_weight" not in out


def test_supervise_termination_over_budget_reserves_eos_slot():
    tok = _FakeTokenizer()
    out = tokenize_canvas_batch(
        tok, canvas_length=3, inputs=["x"], outputs=["abcde"],
        supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    t0, t1 = ((ord(c) % 40) + 2 for c in "ab")
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
    assert -100 in default["canvas_labels"][0]


def test_supervise_termination_with_anchor():
    tok = _FakeTokenizerBOS()
    out = tokenize_canvas_batch(
        tok, canvas_length=8, inputs=["hi"], outputs=["ab"],
        anchor_id=99, supervise_termination=True,
    )
    ci = out["canvas_input_ids"][0]
    cl = out["canvas_labels"][0]
    t0, t1 = (ord("a") % 40) + 2, (ord("b") % 40) + 2
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
    assert w == [1.0, 1.0, 1.0, 0.25, 0.25, 0.25]
    assert len(w) == 6
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
