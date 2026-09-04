"""Convert the repo's anchor/positive/negative_N embedding JSONL into Tevatron's retriever training JSONL."""

import argparse
import json
import os

DEFAULT_INPUT = "../sentence_transformers/OTel_embedding_sample_100.jsonl"
DEFAULT_OUTPUT = "OTel_tevatron_sample_100.jsonl"


def parse_args():
    """Parse converter flags."""
    parser = argparse.ArgumentParser(description="Convert triplet JSONL to Tevatron retriever JSONL")
    parser.add_argument("--input_file", default=DEFAULT_INPUT,
                        help="Source JSONL with anchor/positive/negative_1..N columns")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT,
                        help="Destination JSONL in Tevatron's query/positive_passages/negative_passages format")
    parser.add_argument("--anchor_field", default="anchor", help="Column holding the query text")
    parser.add_argument("--positive_field", default="positive", help="Column holding the relevant passage")
    parser.add_argument("--negative_prefix", default="negative_",
                        help="Prefix of the hard-negative columns (negative_1, negative_2, ...)")
    parser.add_argument("--max_negatives", type=int, default=5,
                        help="Keep at most this many hard negatives per row")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N rows")
    parser.add_argument("--docid_prefix", default="otel", help="Prefix for the generated docid values")
    return parser.parse_args()


def read_rows(path, limit):
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_negatives(row, prefix, max_negatives):
    """Return the row's negative_* values in numeric order, dropping empties."""
    keys = sorted((k for k in row if k.startswith(prefix) and row[k]),
                  key=lambda k: int(k[len(prefix):]) if k[len(prefix):].isdigit() else 0)
    return [row[k] for k in keys[:max_negatives]]


def build_example(row, index, args):
    """Build one Tevatron training record from one triplet row."""
    query = (row.get(args.anchor_field) or "").strip()
    positive = (row.get(args.positive_field) or "").strip()
    if not query or not positive:
        return None

    docid = f"{args.docid_prefix}-{index}"
    negatives = collect_negatives(row, args.negative_prefix, args.max_negatives)
    return {
        "query_id": f"q{index}",
        "query": query,
        "positive_passages": [{"docid": f"{docid}-pos", "text": positive}],
        "negative_passages": [{"docid": f"{docid}-neg{n}", "text": text.strip()}
                              for n, text in enumerate(negatives)],
    }


def main():
    """Read the triplet JSONL, emit the Tevatron JSONL, and report the counts."""
    args = parse_args()
    if not os.path.isfile(args.input_file):
        raise SystemExit(f"input file not found: {args.input_file}")

    written = skipped = 0
    with open(args.output_file, "w", encoding="utf-8") as out:
        for index, row in enumerate(read_rows(args.input_file, args.limit)):
            example = build_example(row, index, args)
            if example is None:
                skipped += 1
                continue
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows to {args.output_file} (skipped {skipped})")


if __name__ == "__main__":
    main()
