"""Derive data/classification_sample.csv from the chat JSONL sample — see readme_classification.md."""
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Derive a labeled classification CSV from a chat JSONL")
    parser.add_argument("--source", type=str,
                        default="../../llm/deepspeed_standalone/OTel_LLM_sample_10.jsonl",
                        help="Chat JSONL where each row has messages plus a flow field")
    parser.add_argument("--out", type=str, default="data/classification_sample.csv", help="Output CSV path")
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate the user-turn text to this length")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.source) as f:
        for line in f:
            record = json.loads(line)
            # First user turn is the text; the flow field is the class label.
            user_text = next(m["content"] for m in record["messages"] if m["role"] == "user")
            rows.append({"text": user_text[: args.max_chars], "label": record["flow"]})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    labels = sorted({r["label"] for r in rows})
    print(f"Wrote {len(rows)} rows, {len(labels)} classes {labels} to {out_path}")


if __name__ == "__main__":
    main()
