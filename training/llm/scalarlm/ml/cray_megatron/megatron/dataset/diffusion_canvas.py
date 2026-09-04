"""Pure canvas-tokenization helpers for DiffusionGemma."""

import logging

logger = logging.getLogger(__name__)


def pad_token_id(tokenizer):
    """A valid embedding index to fill canvas padding: pad, else eos, else 0."""
    for tok in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if tok is not None:
            return tok
    return 0


def anchor_token_id(tokenizer):
    """Resolve the canvas anchor token (BOS), or ``None`` if the tokenizer has none."""
    return getattr(tokenizer, "bos_token_id", None)


def tokenize_canvas_batch(
    tokenizer,
    canvas_length,
    inputs,
    outputs,
    anchor_id=None,
    supervise_termination=False,
    pad_loss_weight=1.0,
):
    """Tokenize ``{input, output}`` rows into DiffusionGemma canvas fields."""
    pad_id = pad_token_id(tokenizer)
    # Only append EOS when the tail is supervised; otherwise it is unsupervised noise.
    eos_id = tokenizer.eos_token_id if supervise_termination else None
    append_eos = supervise_termination and eos_id is not None

    prefix_len = 1 if anchor_id is not None else 0
    # The anchor and the EOS each consume a canvas slot.
    output_budget = canvas_length - prefix_len - (1 if append_eos else 0)

    encoder = tokenizer(inputs)
    output_tokens = tokenizer(outputs, add_special_tokens=False)["input_ids"]

    emit_weight = supervise_termination and pad_loss_weight != 1.0

    canvas_input_ids = []
    canvas_labels = []
    canvas_loss_weight = [] if emit_weight else None
    for toks in output_tokens:
        if len(toks) > output_budget:
            logger.warning(
                "DiffusionGemma output has %d tokens > canvas budget %d "
                "(canvas_length %d%s%s); truncating to the first %d.",
                len(toks),
                output_budget,
                canvas_length,
                " minus 1 anchor slot" if anchor_id is not None else "",
                " minus 1 EOS slot" if append_eos else "",
                output_budget,
            )
            toks = toks[:output_budget]

        body = list(toks) + ([eos_id] if append_eos else [])
        # The anchor slot must stay clean: pair this with corrupt_canvas(protect_prefix=1).
        prefix_input = [anchor_id] if anchor_id is not None else []
        prefix_label = [anchor_id] if anchor_id is not None else []

        pad = canvas_length - len(prefix_input) - len(body)
        canvas_input_ids.append(prefix_input + body + [pad_id] * pad)
        if supervise_termination:
            # Give the pad tail a real target so it is corruptible and teaches termination.
            canvas_labels.append(prefix_label + body + [pad_id] * pad)
            if emit_weight:
                canvas_loss_weight.append(
                    [1.0] * (len(prefix_label) + len(body)) + [pad_loss_weight] * pad
                )
        else:
            canvas_labels.append(prefix_label + body + [-100] * pad)

    result = {
        "encoder_input_ids": encoder["input_ids"],
        "encoder_attention_mask": encoder["attention_mask"],
        "canvas_input_ids": canvas_input_ids,
        "canvas_labels": canvas_labels,
    }
    if emit_weight:
        result["canvas_loss_weight"] = canvas_loss_weight
    return result
