"""Convert the repo's anchor/positive/negative_N JSONL into FlagEmbedding's query/pos/neg reranker format."""
import argparse
import json
import os


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Convert triplet JSONL to FlagEmbedding reranker format")
    parser.add_argument("--src", type=str, default="../sentence_transformers/OTel_reranker_sample_100.jsonl",
                        help="Source JSONL with anchor/positive/negative_1..negative_N columns")
    parser.add_argument("--dst", type=str, default="OTel_reranker_flagembedding_100.jsonl",
                        help="Destination JSONL in FlagEmbedding format")
    parser.add_argument("--n_neg", type=int, default=5, help="Hard negatives per query to carry over")
    parser.add_argument("--prompt", type=str,
                        default="Predict whether passage B contains an answer to query A.",
                        help="Instruction stored in the 'prompt' field (used by the LLM-based rerankers)")
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate each text to this many characters")
    return parser.parse_args()


def convert_row(row, n_neg, prompt, max_chars):
    """Map one anchor/positive/negative_N row to a FlagEmbedding reranker record."""
    neg = [row[f"negative_{i}"][:max_chars] for i in range(1, n_neg + 1) if row.get(f"negative_{i}")]
    return {
        "query": row["anchor"][:max_chars],
        "pos": [row["positive"][:max_chars]],
        "neg": neg,
        "prompt": prompt,
    }


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    src = args.src if os.path.isabs(args.src) else os.path.join(here, args.src)
    dst = args.dst if os.path.isabs(args.dst) else os.path.join(here, args.dst)

    with open(src) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(dst, "w") as f:
        for row in rows:
            f.write(json.dumps(convert_row(row, args.n_neg, args.prompt, args.max_chars)) + "\n")

    print(f"Wrote {len(rows)} rows to {dst}")


if __name__ == "__main__":
    main()
