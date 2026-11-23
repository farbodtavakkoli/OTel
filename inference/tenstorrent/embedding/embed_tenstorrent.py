import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv("dev.env")

# Same query/document sets as inference/transformers/embedding and inference/mlx/embedding.
DEFAULT_QUERIES = [
    "What GPU runtimes support ROCm?",
    "Which database is embedded and serverless?",
]

DEFAULT_DOCUMENTS = [
    "vLLM supports NVIDIA CUDA and AMD ROCm.",
    "SGLang provides a ROCm build for AMD Instinct accelerators.",
    "SQLite is an embedded database.",
    "The Eiffel Tower is located in Paris, France.",
]

# EmbeddingGemma's retrieval prompts (config_sentence_transformers.json "query" / "document").
# The server embeds the text it receives, so the client adds them.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding client for the Tenstorrent EmbeddingGemma-2 server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8132, help="Server port")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-2", help="Served model name")
    parser.add_argument("--queries_file", type=str, default=None, help="JSON/JSONL/text queries; omit for the built-in set")
    parser.add_argument("--documents_file", type=str, default=None, help="JSON/JSONL/text documents; omit for the built-in set")
    parser.add_argument("--truncate_dim", type=int, default=None, help="Matryoshka output dim (768/512/256/128); omit for full")
    parser.add_argument("--no_prompts", action="store_true", help="Send the texts without the query/document prompts")
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


def embed(url, args, texts, prompt):
    if not args.no_prompts:
        texts = [prompt + t for t in texts]
    started = time.time()
    with urllib.request.urlopen(make_request(url, {"model": args.model, "input": texts}), timeout=args.timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    vectors = [item["embedding"] for item in sorted(result["data"], key=lambda item: item["index"])]
    if args.truncate_dim:
        vectors = [v[:args.truncate_dim] for v in vectors]
    return [normalize(v) for v in vectors], result, time.time() - started


def normalize(v):
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/embeddings"
    queries = read_texts(args.queries_file, DEFAULT_QUERIES)
    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    try:
        embed(url, args, ["warm-up"], QUERY_PROMPT)
        q_emb, result, q_s = embed(url, args, queries, QUERY_PROMPT)
        d_emb, _, d_s = embed(url, args, documents, DOCUMENT_PROMPT)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    sims = [[sum(a * b for a, b in zip(q, d)) for d in d_emb] for q in q_emb]
    print(f"endpoint      : {url}")
    print(f"model         : {result.get('model')}")
    print(f"latency       : queries {q_s * 1000:.1f} ms ({len(queries)} texts), documents {d_s * 1000:.1f} ms ({len(documents)} texts)")
    print(f"dim           : {len(q_emb[0])}")
    print(f"query[0] first8={[round(x, 6) for x in q_emb[0][:8]]}")

    ok = True
    for i, query in enumerate(queries):
        order = sorted(range(len(documents)), key=lambda j: -sims[i][j])
        print(f"\nQ{i}: {query}")
        for rank, j in enumerate(order):
            print(f"  #{rank + 1} cos={sims[i][j]:+.6f}  {documents[j]}")
        if args.queries_file is None and args.documents_file is None:
            relevant, irrelevant = ([0, 1], [2, 3]) if i == 0 else ([2], [0, 1, 3])
            best_rel = min(sims[i][j] for j in relevant)
            worst_irr = max(sims[i][j] for j in irrelevant)
            passed = best_rel > worst_irr
            ok &= passed
            print(f"  sanity: min(relevant)={best_rel:+.6f} > max(irrelevant)={worst_irr:+.6f} -> {'PASS' if passed else 'FAIL'}")

    artifact = {
        "model": result.get("model"), "endpoint": url, "truncate_dim": args.truncate_dim,
        "prompts": not args.no_prompts, "embedding_dim": len(q_emb[0]),
        "queries": queries, "documents": documents,
        "cosine_similarities": [[round(x, 6) for x in row] for row in sims],
        "query_embeddings_first16": [[round(x, 6) for x in q[:16]] for q in q_emb],
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
