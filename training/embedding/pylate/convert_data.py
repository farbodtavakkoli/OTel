"""Convert the repo's anchor/positive/negative_N JSONL into PyLate's ColBERT triplet format."""
import argparse
import json
import os


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Convert triplet JSONL to PyLate ColBERT format")
    parser.add_argument("--src", type=str, default="../sentence_transformers/OTel_embedding_sample_100.jsonl",
                        help="Source JSONL with anchor/positive/negative_1..negative_N columns")
    parser.add_argument("--dst", type=str, default="OTel_embedding_pylate_100.jsonl",
                        help="Destination JSONL in PyLate format")
    parser.add_argument("--n_neg", type=int, default=1,
                        help="Hard negatives per query to emit as negative_1..negative_N columns")
    parser.add_argument("--explode", action="store_true",
                        help="Emit one (query, positive, negative) row per hard negative instead of one wide row")
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate each text to this many characters")
    return parser.parse_args()


def negatives(row, n_neg, max_chars):
    """Collect up to n_neg non-empty hard negatives from a source row."""
    out = []
    for i in range(1, n_neg + 1):
        text = row.get(f"negative_{i}")
        if text:
            out.append(text[:max_chars])
    return out


def convert_row(row, n_neg, explode, max_chars):
    """Map one anchor/positive/negative_N row to one or more PyLate training records."""
    query = row["anchor"][:max_chars]
    positive = row["positive"][:max_chars]
    negs = negatives(row, n_neg, max_chars)
    if explode:
        return [{"query": query, "positive": positive, "negative": neg} for neg in negs]
    record = {"query": query, "positive": positive}
    if len(negs) == 1:
        record["negative"] = negs[0]
    else:
        for i, neg in enumerate(negs, start=1):
            record[f"negative_{i}"] = neg
    return [record]


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    src = args.src if os.path.isabs(args.src) else os.path.join(here, args.src)
    dst = args.dst if os.path.isabs(args.dst) else os.path.join(here, args.dst)

    with open(src) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    written = 0
    with open(dst, "w") as f:
        for row in rows:
            for record in convert_row(row, args.n_neg, args.explode, args.max_chars):
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} rows from {len(rows)} source rows to {dst}")


if __name__ == "__main__":
    main()
