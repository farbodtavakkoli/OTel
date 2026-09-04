"""Flatten chat JSONL into the {text} rows torchtune's text_completion_dataset wants -- see readme_torchtune.md."""

import argparse
import json
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="chat JSONL -> torchtune {text} JSONL")
    parser.add_argument(
        "--input",
        default="data/OTel_LLM_sample_10.jsonl",
        help="source chat JSONL (default: the shipped sample)",
    )
    parser.add_argument(
        "--output",
        default="data/otel_cpt.jsonl",
        help="destination {text} JSONL (default: data/otel_cpt.jsonl)",
    )
    return parser.parse_args()


def flatten_record(record):
    """Return `text` if present, else every `messages[].content` joined by blank lines."""
    if record.get("text"):
        return record["text"]
    messages = record.get("messages") or []
    parts = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else ""
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)


def main():
    args = parse_args()

    n = 0
    with open(args.input, "r", encoding="utf-8") as src, open(
        args.output, "w", encoding="utf-8"
    ) as dst:
        for index, line in enumerate(src):
            if not line.strip():
                continue
            text = flatten_record(json.loads(line))
            if not text:
                logger.warning("row %d: empty after flatten, skipping", index)
                continue
            dst.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    if n == 0:
        raise SystemExit(f"no usable rows in {args.input}")
    logger.info("wrote %d rows -> %s", n, args.output)


if __name__ == "__main__":
    main()
