"""Compare uninterrupted/resumed full checkpoints, including the pointer head."""

import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--resumed", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
if {
    path.relative_to(args.reference) for path in args.reference.rglob("*.safetensors")
} != {path.relative_to(args.resumed) for path in args.resumed.rglob("*.safetensors")}:
    raise ValueError("Checkpoint file sets differ")
differences = {}
compared = 0
for source in sorted(args.reference.rglob("*.safetensors")):
    relative = source.relative_to(args.reference)
    target = args.resumed / relative
    with (
        safe_open(source, framework="pt", device="cpu") as left,
        safe_open(target, framework="pt", device="cpu") as right,
    ):
        if not (set(left.keys()) == set(right.keys())):
            raise ValueError("Validation failed: set(left.keys()) == set(right.keys())")
        for key in left.keys():
            reference_tensor, resumed_tensor = left.get_tensor(key), right.get_tensor(
                key
            )
            compared += 1
            if not torch.equal(reference_tensor, resumed_tensor):
                differences[f"{relative}:{key}"] = float(
                    (reference_tensor - resumed_tensor).abs().max()
                )
if not (compared > 0):
    raise ValueError("No checkpoint tensors found")
result = dict(
    tensors_compared=compared,
    bitwise_equal=not differences,
    differing_tensors=len(differences),
    max_absolute_difference=max(differences.values(), default=0.0),
    differences=differences,
)
with args.output.open("x") as out:
    out.write(json.dumps(result, indent=2))
print(json.dumps(result))
if differences:
    raise RuntimeError("Resumed model differs from uninterrupted model; inspect report")
