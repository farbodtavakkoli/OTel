#!/usr/bin/env python3
"""CLI for native TorchNeuron fine-tuning of NVIDIA's bidirectional reranker."""

import argparse
import math
from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
from reranker_data import load_and_split, write_json
from reranker_service import MODEL, REVISION, run_finetuning


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PACKAGE_DIR / "OTel_reranker_sample_100.jsonl"))
    parser.add_argument("--out", default="output", help="New directory for checkpoints and metrics")
    parser.add_argument("--weights", default="./weights")
    parser.add_argument("--cache", default="./compiled")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--n_neg", type=int, default=5)
    parser.add_argument("--eval_frac", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("batch", "microbatch", "gradient_accumulation", "max_len", "epochs", "n_neg"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not 0 < args.eval_frac < 1:
        parser.error("--eval_frac must be between 0 and 1")
    if args.batch % args.microbatch:
        parser.error("--microbatch must divide --batch")
    return args


def public_config(args):
    config = {key: value for key, value in vars(args).items()
              if key not in {"data", "out", "weights", "cache"}}
    config.update({"data_file": Path(args.data).name, "model": MODEL, "revision": REVISION})
    return config


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", public_config(args))
    try:
        pairs, valid = load_and_split(args, out)
        if not args.prepare_only:
            run_finetuning(args, out, pairs, valid)
    except BaseException as exc:
        write_json(out / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
