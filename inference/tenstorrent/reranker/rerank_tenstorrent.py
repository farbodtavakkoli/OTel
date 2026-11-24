import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv("dev.env")

# Same query/document set as inference/transformers/reranker and inference/mlx/reranker.
DEFAULT_QUERY = "Which inference engines support AMD ROCm?"

DEFAULT_DOCUMENTS = [
    "vLLM supports AMD ROCm.",
    "SGLang ships a ROCm build for AMD Instinct accelerators.",
    "PostgreSQL is a relational database.",
    "The Eiffel Tower is located in Paris, France.",
]

# Index of the documents above that are genuinely relevant to DEFAULT_QUERY.
DEFAULT_RELEVANT = [0, 1]


def parse_args():
    parser = argparse.ArgumentParser(description="Reranker client for the Tenstorrent Nemotron rerank server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="nvidia/llama-nemotron-rerank-1b-v2", help="Served model name")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to rerank documents against")
    parser.add_argument("--documents_file", type=str, default=None,
                        help="JSON/JSONL/text file of document strings; omit for the built-in set")
    parser.add_argument("--top_n", type=int, default=None, help="Return only the best N documents (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Path for the reference-output JSON artifact")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def read_texts(path, fallback):
    if path is None:
        return list(fallback)
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l)["text"] if l.lstrip().startswith("{") else l for l in raw.splitlines() if l.strip()]


def make_request(url, payload):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
    return urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)


def rerank(url, args, documents):
    payload = {"model": args.model, "query": args.query, "documents": documents,
               "top_n": args.top_n or len(documents), "return_documents": False}
    started = time.time()
    with urllib.request.urlopen(make_request(url, payload), timeout=args.timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result, time.time() - started


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/rerank"
    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    try:
        rerank(url, args, documents[:1])  # warm-up: the first request after start pays one-time setup
        result, elapsed = rerank(url, args, documents)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    ranked = result["results"]
    scores = [None] * len(documents)
    logits = [None] * len(documents)
    for item in ranked:
        scores[item["index"]] = item["relevance_score"]
        logits[item["index"]] = item.get("logit")

    print(f"endpoint      : {url}")
    print(f"model         : {result.get('model')}")
    print(f"latency       : {elapsed * 1000:.1f} ms ({len(documents)} pairs)")
    print(f"usage         : {result.get('usage')}")
    print(f"\nQ: {args.query}")
    for rank, item in enumerate(ranked):
        j = item["index"]
        print(f"  #{rank + 1} score={item['relevance_score']:.6f} logit={item.get('logit', float('nan')):+.4f}  {documents[j]}")

    ok = True
    if args.documents_file is None and args.top_n is None:
        irrelevant = [i for i in range(len(documents)) if i not in DEFAULT_RELEVANT]
        best_rel = min(logits[i] for i in DEFAULT_RELEVANT)
        worst_irr = max(logits[i] for i in irrelevant)
        ok = best_rel > worst_irr
        top = sorted(item["index"] for item in ranked[:len(DEFAULT_RELEVANT)])
        print(f"\nsanity: min(relevant)={best_rel:+.4f} > max(irrelevant)={worst_irr:+.4f} -> {'PASS' if ok else 'FAIL'}")
        print(f"sanity: top-{len(DEFAULT_RELEVANT)} == relevant set -> {'PASS' if top == DEFAULT_RELEVANT else 'FAIL'}")
        ok = ok and top == DEFAULT_RELEVANT

    artifact = {
        "model": result.get("model"), "endpoint": url, "latency_ms": round(elapsed * 1000, 1),
        "query": args.query, "documents": documents,
        "scores": scores, "logits": logits,
        "ranking": [item["index"] for item in ranked],
        "sanity_check_passed": bool(ok),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nwrote reference artifact: {args.output}")

    print(f"\nSANITY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
