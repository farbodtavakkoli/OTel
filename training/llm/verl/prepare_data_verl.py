"""Convert chat JSONL into the parquet column contract verl's RL trainers expect (schema details in readme_verl.md)."""

import argparse
import json
import logging

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_rows(path, data_source, ability, split):
    """Turn each chat record into one verl row — final assistant turn is the ground truth, the rest is the prompt."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages")

            if not messages:
                logger.warning("row %d: no 'messages' field, skipping", index)
                continue
            if messages[-1].get("role") != "assistant":
                logger.warning("row %d: last turn is not 'assistant', skipping", index)
                continue

            prompt = messages[:-1]
            ground_truth = messages[-1].get("content", "")

            if not prompt or not ground_truth:
                logger.warning("row %d: empty prompt or empty answer, skipping", index)
                continue

            rows.append(
                {
                    "data_source": data_source,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": {"style": "rule", "ground_truth": ground_truth},
                    "extra_info": {"split": split, "index": index},
                }
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="chat JSONL -> verl parquet")
    parser.add_argument("--input", default="data/OTel_LLM_sample_10.jsonl", help="source chat JSONL")
    parser.add_argument("--output", default="data/otel_train.parquet", help="destination .parquet path")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--data-source",
        default="otel_local",
        help="tag your reward fn switches on (see reward_verl.py)",
    )
    parser.add_argument("--ability", default="general", help="free-form task tag")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = build_rows(args.input, args.data_source, args.ability, args.split)
    if not rows:
        raise SystemExit(f"no usable rows found in {args.input}")

    pd.DataFrame(rows).to_parquet(args.output, index=False)
    logger.info("wrote %d rows -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
