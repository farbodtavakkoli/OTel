"""Reranker client for a vLLM OpenAI-compatible server — see README.md."""

import argparse
import json
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the server uses it when it pulls the reranker repo.
load_dotenv("dev.env")

DEFAULT_QUERY = "Which inference engines support AMD ROCm?"
DEFAULT_DOCS = [
    "vLLM supports AMD ROCm and runs on MI300/MI350 Instinct GPUs.",
    "PostgreSQL is a relational database management system.",
    "SGLang also provides a ROCm build for AMD GPUs.",
    "The Eiffel Tower is located in Paris, France.",
]


def parse_args():
    """Parse CLI arguments; every tunable of the rerank request is exposed here."""
    parser = argparse.ArgumentParser(description="Rerank documents against a vLLM /v1/rerank endpoint")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--model", type=str, default="qwen3-reranker",
                        help="Served model name (matches --served-model-name)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to rank the documents against")
    parser.add_argument("--document", type=str, action="append", default=None,
                        help="Candidate document; repeat the flag for several documents")
    parser.add_argument("--top_n", type=int, default=None, help="Return only the top N results (default: all)")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def post_json(url, payload, timeout):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    args = parse_args()
    documents = args.document if args.document else list(DEFAULT_DOCS)

    url = f"http://{args.host}:{args.port}/v1/rerank"
    payload = {"model": args.model, "query": args.query, "documents": documents}
    if args.top_n is not None:
        payload["top_n"] = args.top_n

    try:
        result = post_json(url, payload, args.timeout)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    print(f"endpoint    : {url}")
    print(f"model       : {result.get('model')}")
    print(f"query       : {args.query}")
    print(f"usage       : {result.get('usage')}")
    print()
    print("rank  index  score       document")
    for rank, item in enumerate(result["results"], start=1):
        text = item.get("document", {}).get("text", documents[item["index"]])
        print(f"{rank:>4}  {item['index']:>5}  {item['relevance_score']:.6f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
