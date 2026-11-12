"""Verify full export, structure, and baseline/trained metrics in a fresh process."""

import argparse
import gc
import json
import sys
from pathlib import Path
from decider_recipe_utils import load_rows, StrictTokenizer, require, write_json


def main(args):
    import torch
    from decider_checkpoints import verify_checkpoint_export
    from decider_model_io import load_full
    from prepare_decider_model import architecture

    sys.path.insert(0, str(args.upstream))
    from training.data.collate import CollatorConfig, SystemOneCollator
    from training.data.format import Example

    torch.set_num_threads(6)
    rows = load_rows(args.data)
    train_rows = load_rows(args.training_data)
    overlap = {" ".join(r["state"].split()) for r in rows} & {
        " ".join(r["state"].split()) for r in train_rows
    }
    require(
        args.split == "training" or not overlap,
        "Evaluation overlaps training states; provide a disjoint split",
    )
    model = load_full(args.final).eval()
    require(
        not model.config.use_lora
        and not any("lora_" in n for n, _ in model.named_parameters()),
        "Unmerged adapter",
    )
    require(
        all(torch.isfinite(p).all() for p in model.parameters()),
        "Nonfinite exported weights",
    )
    actual = architecture(model)
    expected = json.loads(args.architecture.read_text())
    # HF writes this descriptive class label during save; actual module classes
    # above are authoritative and remain strictly compared.
    expected["torso_config"].pop("architectures", None)
    require(
        actual["parameters"] == expected["parameters"], "Parameter names/shapes changed"
    )
    require(actual["modules"] == expected["modules"], "Module classes changed")
    for field in (
        "torso_config",
        "decider_config",
        "tokenizer_vocab_sha256",
        "special_tokens",
        "dropout",
    ):
        require(
            actual[field] == expected[field],
            "Architecture/tokenizer mismatch: " + field,
        )
    if args.checkpoint:
        verify_checkpoint_export(args.checkpoint, model)
    examples = [Example.from_dict(r) for r in rows]

    def metrics(candidate):
        candidate.config.temperature = 1.0
        candidate.config.temperature_by_kind = {}
        collator = SystemOneCollator(
            StrictTokenizer(candidate.tokenizer, args.pad_to),
            CollatorConfig(max_length=args.pad_to),
            train=False,
        )
        losses = []
        correct = 0
        with torch.no_grad():
            for row in examples:
                out = candidate(**collator([row]))
                losses.append(float(out["loss"]))
                correct += int(out["logits"].argmax()) == row.label
        return dict(loss=sum(losses) / len(losses), accuracy=correct / len(losses))

    trained = metrics(model)
    del model
    gc.collect()
    baseline = metrics(load_full(args.merged).eval())
    write_json(
        args.output,
        dict(
            status="passed",
            architecture_equal=True,
            all_weights_finite=True,
            export_parameter_parity="passed" if args.checkpoint else "not_checked",
            rows=len(rows),
            split=args.split,
            baseline=baseline,
            trained=trained,
            temperature=1.0,
            independent_quality_evaluation=args.split != "training",
            selection="last_step",
        ),
    )
    print("Verification passed:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "upstream",
        "data",
        "training-data",
        "final",
        "merged",
        "architecture",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--split", choices=["training", "validation", "test"], required=True
    )
    parser.add_argument("--pad-to", type=int, default=768)
    main(parser.parse_args())
