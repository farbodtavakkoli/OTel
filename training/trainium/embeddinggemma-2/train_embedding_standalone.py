#!/usr/bin/env python3
"""CLI for standalone native TorchNeuron EmbeddingGemma-2 fine-tuning."""

import argparse
import json
import logging
import os
from pathlib import Path
import time

from prepare_embedding_data import prepare_data
from embedding_model import self_test
from embedding_training_config import DEFAULT_MODEL, DEFAULT_REVISION
from embedding_run_artifacts import (
    prepare_run,
    runtime_preflight,
    validate_arguments,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "OTel_embedding_sample_100.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/embeddinggemma2"),
    )
    parser.add_argument(
        "--model_cache_dir",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub",
    )
    parser.add_argument("--compile_cache_dir", type=Path, default=None)
    parser.add_argument("--device", default="neuron", choices=["neuron"])
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--retrieval_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--eval_max_seq_length", type=int, default=1024)
    parser.add_argument(
        "--matryoshka_dims",
        type=int,
        nargs="+",
        default=[768, 512, 256, 128],
    )
    parser.add_argument(
        "--num_negatives", type=int, choices=range(1, 6), default=1
    )
    parser.add_argument(
        "--eval_corpus_size",
        type=int,
        default=20000,
        help="Additional hash-ranked distractors.",
    )
    parser.add_argument(
        "--query_prompt", default="task: search result | query: "
    )
    parser.add_argument(
        "--document_prompt", default="title: none | text: "
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Prepare data, load model/processor, and test a small compiled "
            "Neuron update; no fine-tuning."
        ),
    )
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def configure_process_environment():
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def run(args):
    validate_arguments(args, DEFAULT_MODEL)
    configure_process_environment()
    if args.self_test:
        self_test()
        return

    recipe_directory = Path(__file__).resolve().parent
    output = prepare_run(args, recipe_directory)
    started = time.monotonic()
    try:
        rows, spec, audit = prepare_data(args, output / "data")
        if len(rows["train"]) < args.batch_size or not rows["validation"]:
            raise ValueError(
                "Insufficient train/validation rows after grouping; "
                "use more data or another seed/smaller batch."
            )
        if args.prepare_only:
            write_json(
                output / "result.json",
                {"status": "prepared", "audit": audit},
            )
            print(json.dumps(audit))
            return

        write_json(
            output / "preflight-environment.json", runtime_preflight()
        )
        from train_embedding_on_trainium import train

        result = train(args, output, rows)
        result.update(
            elapsed_seconds=time.monotonic() - started,
            audit=audit,
            evaluation_id=spec["evaluation_id"],
        )
        write_json(output / "result.json", result)
        logging.info("Completed: %s", output / "result.json")
    except Exception as error:
        write_json(
            output / "result.json",
            {
                "status": "failed",
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
