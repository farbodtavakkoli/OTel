"""Client for an SGLang Qwen3 reranker server — see readme_sglang_reranker.md."""
import argparse
import json
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN for the server-side model pull; the reranker itself is ungated.
load_dotenv("dev.env")


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="SGLang reranker smoke client (/v1/rerank)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=8102, help="SGLang server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Reranker-0.6B",
                        help="Model id as served (must match --model-path)")
    parser.add_argument("--query", type=str, default="Which inference engines support AMD ROCm?",
                        help="Rerank query")
    parser.add_argument("--documents", type=str, nargs="+",
                        default=["vLLM and SGLang both support AMD ROCm GPUs.",
                                 "PostgreSQL is a relational database.",
                                 "ROCm is AMD's open compute platform for Instinct accelerators."],
                        help="Candidate documents to score against the query")
    parser.add_argument("--top_n", type=int, default=None, help="Return only the top N documents")
    parser.add_argument("--endpoint", type=str, default="/v1/rerank", help="Endpoint path to call")
    parser.add_argument("--wait", type=int, default=600, help="Seconds to wait for /health before giving up")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds")
    return parser.parse_args()


def post_json(url: str, payload: dict, timeout: int):
    """POST a JSON body and return the decoded JSON response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_health(base: str, wait_s: int) -> float:
    """Block until the server answers /health; return the seconds waited."""
    start = time.time()
    while time.time() - start < wait_s:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
                if resp.status == 200:
                    return time.time() - start
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise RuntimeError(f"server at {base} did not become healthy within {wait_s}s")


def extract_ranking(out, documents):
    """Normalize SGLang's rerank response into (score, document) pairs, best first."""
    rows = out["results"] if isinstance(out, dict) and "results" in out else out
    ranking = []
    for row in rows:
        score = row.get("score", row.get("relevance_score"))
        doc = row.get("document")
        if isinstance(doc, dict):
            doc = doc.get("text")
        if doc is None:
            doc = documents[row["index"]]
        ranking.append((score, doc))
    return sorted(ranking, key=lambda r: r[0], reverse=True)


def main():
    """Wait for the server, rerank the documents, print the real relevance scores in order."""
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    waited = wait_for_health(base, args.wait)
    print(f"[health] server ready after {waited:.1f}s")

    payload = {"model": args.model, "query": args.query, "documents": args.documents}
    if args.top_n is not None:
        payload["top_n"] = args.top_n

    start = time.time()
    out = post_json(f"{base}{args.endpoint}", payload, args.timeout)
    elapsed = time.time() - start

    ranking = extract_ranking(out, args.documents)
    print(f"[latency] {elapsed:.2f}s | pairs={len(args.documents)} | query={args.query!r}")
    for rank, (score, doc) in enumerate(ranking, start=1):
        print(f"[rank {rank}] score={score:.6f}  {doc}")


if __name__ == "__main__":
    main()
