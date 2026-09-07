"""Embedding client for a vLLM OpenAI-compatible server — see README.md."""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the server needs it for the gated EmbeddingGemma repo.
load_dotenv("dev.env")

DEFAULT_QUERY = "Which inference engines support AMD ROCm?"
DEFAULT_DOCS = [
    "vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.",
    "PostgreSQL is a relational database management system.",
]


def parse_args():
    """Parse CLI arguments; every tunable of the request is exposed here."""
    parser = argparse.ArgumentParser(description="Embed texts against a vLLM /v1/embeddings endpoint")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--model", type=str, default="embeddinggemma",
                        help="Served model name (matches --served-model-name)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY,
                        help="Query text; cosine similarity is reported against each document")
    parser.add_argument("--document", type=str, action="append", default=None,
                        help="Document text; repeat the flag for several documents")
    parser.add_argument("--encoding_format", type=str, default="float", choices=["float", "base64"],
                        help="Embedding encoding requested from the server")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--show_dims", type=int, default=5, help="How many leading vector components to print")
    return parser.parse_args()


def post_json(url, payload, timeout):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    args = parse_args()
    documents = args.document if args.document else list(DEFAULT_DOCS)
    texts = [args.query] + documents

    url = f"http://{args.host}:{args.port}/v1/embeddings"
    payload = {"model": args.model, "input": texts, "encoding_format": args.encoding_format}

    try:
        result = post_json(url, payload, args.timeout)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    vectors = [item["embedding"] for item in sorted(result["data"], key=lambda d: d["index"])]

    print(f"endpoint    : {url}")
    print(f"model       : {result.get('model')}")
    print(f"vectors     : {len(vectors)}")
    print(f"dimensions  : {len(vectors[0])}")
    print(f"usage       : {result.get('usage')}")
    print()
    print(f"query       : {args.query}")
    print(f"  first {args.show_dims} dims: {[round(x, 5) for x in vectors[0][:args.show_dims]]}")
    print()
    for doc, vector in zip(documents, vectors[1:]):
        print(f"  cos={cosine(vectors[0], vector):+.4f}  {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
