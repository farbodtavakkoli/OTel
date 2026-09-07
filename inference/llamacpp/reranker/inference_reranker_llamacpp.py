"""Client for a llama.cpp llama-server reranking endpoint — see README.md."""
import argparse
import json
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; llama-server needs it only for gated GGUF pulls, not for this client.
load_dotenv("dev.env")

DEFAULT_QUERY = "What is an OpenTelemetry span?"
DEFAULT_DOCUMENTS = [
    "A span represents a single unit of work in a distributed trace and carries a start time, duration and attributes.",
    "To bake sourdough, feed the starter twelve hours before mixing the dough.",
    "OpenTelemetry spans nest inside a trace to describe the path of a request through a system.",
    "The 1998 football World Cup final was played in Saint-Denis.",
]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="llama.cpp reranker smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="llama-server host")
    parser.add_argument("--port", type=int, default=8202, help="llama-server port")
    parser.add_argument("--model", type=str, default="qwen3-reranker-0.6b", help="Model name echoed in the request body")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to rank documents against")
    parser.add_argument("--documents", type=str, nargs="*", default=None,
                        help="Candidate documents (defaults to 2 relevant + 2 irrelevant)")
    parser.add_argument("--top_n", type=int, default=None, help="Return only the top N documents")
    parser.add_argument("--endpoint", type=str, default="/v1/rerank",
                        choices=["/v1/rerank", "/rerank", "/v1/reranking"], help="Rerank route to call")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the raw JSON response")
    return parser.parse_args()


def wait_for_health(base_url: str, retries: int):
    """Block until /health reports the model is loaded, or raise."""
    for attempt in range(retries):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"llama-server at {base_url} never became healthy after {retries} attempts")


def rerank(base_url: str, endpoint: str, model: str, query: str, documents, top_n, timeout: int):
    """POST one rerank request and return the parsed response plus latency."""
    payload = {"model": model, "query": query, "documents": documents}
    if top_n:
        payload["top_n"] = top_n
    started = time.time()
    r = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def main():
    args = parse_args()
    documents = args.documents if args.documents else DEFAULT_DOCUMENTS
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    body, elapsed = rerank(base_url, args.endpoint, args.model, args.query, documents, args.top_n, args.timeout)

    results = sorted(body["results"], key=lambda row: row["relevance_score"], reverse=True)
    print(f"endpoint  : {base_url}{args.endpoint}")
    print(f"model     : {body.get('model')}")
    print(f"latency_s : {elapsed:.3f}")
    print(f"query     : {args.query}")
    print("--- ranked documents (best first) ---")
    for rank, row in enumerate(results, start=1):
        doc = documents[row["index"]]
        print(f"{rank}. score={row['relevance_score']:+.5f}  idx={row['index']}  {doc[:90]}")

    top_doc = documents[results[0]["index"]]
    print(f"top_hit_is_relevant : {'span' in top_doc.lower()}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
