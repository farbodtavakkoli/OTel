"""Dataset validation and deterministic split preparation."""

import hashlib
import json
import math
from pathlib import Path
import random

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def row_id(row, keys):
    canonical = json.dumps(
        {key: row[key] for key in keys},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def prompt(q, d):
    return f"question:{q} \n \n passage:{d}"


def load_and_split(args, out):
    rows = []
    for line_number, line in enumerate(Path(args.data).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object on line {line_number}")
        rows.append(row)
    keys = ["anchor", "positive"] + [f"negative_{i}" for i in range(1, args.n_neg + 1)]
    for i, row in enumerate(rows):
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in keys):
            raise ValueError(f"Invalid/missing text in row {i}")
        if row["positive"].strip() in [row[k].strip() for k in keys[2:]]:
            raise ValueError(f"Positive labeled as negative in row {i}")
    random.Random(args.seed).shuffle(rows)
    k = int(len(rows) * (1 - args.eval_frac))
    train, valid = rows[:k], rows[k:]
    if not train or not valid:
        raise ValueError("Empty training or validation split")
    # Reject leakage instead of silently changing the requested split.
    if {r["anchor"].strip() for r in train} & {r["anchor"].strip() for r in valid}:
        raise ValueError("Query overlap between training and validation")
    train_docs = {r[key].strip() for r in train for key in keys[1:]}
    valid_docs = {r[key].strip() for r in valid for key in keys[1:]}
    overlap = train_docs & valid_docs
    if overlap:
        raise ValueError(f"{len(overlap)} passages overlap training and validation")
    pairs = [(prompt(r["anchor"], r[key]), float(key == "positive"))
             for r in train for key in keys[1:]]
    if len(pairs) < args.batch:
        raise ValueError("Training split has fewer pairs than --batch; lower --batch or add data")
    if args.smoke and len(pairs) // args.batch * args.batch < 2 * args.microbatch:
        raise ValueError("Smoke requires at least two complete microbatches after dropping pairs")
    planned_steps = 2 if args.smoke else math.ceil(
        (len(pairs) // args.batch) / args.gradient_accumulation) * args.epochs
    if not args.smoke and planned_steps <= 1:
        raise ValueError("Only one optimizer step planned: warmup would use zero learning rate. "
                         "Lower --batch/--gradient_accumulation, increase --epochs, or add data")
    audit = {"rows": len(rows), "training_queries": len(train),
             "validation_queries": len(valid), "training_pairs": len(pairs),
             "unique_queries": len({r["anchor"] for r in rows}),
             "document_overlap": len(overlap),
             "dropped_pairs_per_epoch": len(pairs) % args.batch,
             "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
             "validation_query_sha256": [hashlib.sha256(r["anchor"].encode()).hexdigest() for r in valid]}
    write_json(out / "audit.json", audit)
    # Record reproducible split membership without copying dataset text or extra fields.
    write_json(out / "split.json", {
        "format": "sha256-required-fields-v1",
        "train": [row_id(row, keys) for row in train],
        "validation": [row_id(row, keys) for row in valid],
    })
    return pairs, valid
