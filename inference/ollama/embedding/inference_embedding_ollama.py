"""Client for an Ollama server serving EmbeddingGemma-300M GGUF — see README.md."""
import argparse
import json
import math
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; Ollama needs it only for gated `hf.co/...` pulls, never for this client.
load_dotenv("dev.env")

# EmbeddingGemma's own prompt templates. The GGUF carries NO template, so the client must add
# them or cosine geometry drifts badly from the Transformers baseline — see the README.
QUERY_TEMPLATE = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: none | text: {text}"

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


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Ollama EmbeddingGemma smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Ollama server host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama server port")
    parser.add_argument("--model", type=str, default="embeddinggemma", help="Ollama model name created from the GGUF")
    parser.add_argument("--api", type=str, default="native", choices=["native", "openai"],
                        help="native=/api/embed, openai=/v1/embeddings")
    parser.add_argument("--queries", type=str, nargs="+", default=DEFAULT_QUERIES, help="Query strings to embed")
    parser.add_argument("--documents", type=str, nargs="+", default=DEFAULT_DOCUMENTS, help="Document strings to embed")
    parser.add_argument("--no_prompt_template", action="store_true",
                        help="Send raw text without EmbeddingGemma's task prefixes (degrades agreement; see README)")
    parser.add_argument("--keep_alive", type=str, default="5m", help="How long Ollama keeps the model resident")
    parser.add_argument("--reference", type=str, default=None,
                        help="Optional Transformers reference JSON to cross-check cosine similarities against")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Max allowed absolute cosine delta vs the reference before the check fails")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the results JSON")
    return parser.parse_args()


def wait_for_health(base_url: str, retries: int):
    """Block until the Ollama server answers on /api/tags, or raise."""
    for _ in range(retries):
        try:
            if requests.get(f"{base_url}/api/tags", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"Ollama at {base_url} never became healthy after {retries} attempts")


def embed(base_url: str, args, texts):
    """Embed a list of strings and return (vectors, elapsed_seconds)."""
    if args.api == "openai":
        url = f"{base_url}/v1/embeddings"
        payload = {"model": args.model, "input": texts}
    else:
        url = f"{base_url}/api/embed"
        payload = {"model": args.model, "input": texts, "keep_alive": args.keep_alive}

    started = time.time()
    response = requests.post(url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    body = response.json()
    elapsed = time.time() - started

    if args.api == "openai":
        return [row["embedding"] for row in body["data"]], elapsed
    return body["embeddings"], elapsed


def cosine(a, b):
    """Cosine similarity between two vectors."""
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / denominator


def cross_check(matrix, reference_path, tolerance):
    """Compare a cosine matrix against a saved Transformers reference and print the deltas."""
    with open(reference_path) as f:
        reference = json.load(f)
    expected = reference["cosine_similarities"]

    print(f"--- cross-check vs {reference['model']} ({reference['dtype']}) ---")
    worst = 0.0
    for i, (got_row, want_row) in enumerate(zip(matrix, expected)):
        delta = max(abs(g - w) for g, w in zip(got_row, want_row))
        worst = max(worst, delta)
        print(f"q{i} ollama       : {[round(v, 4) for v in got_row]}")
        print(f"q{i} transformers : {[round(v, 4) for v in want_row]}")
        print(f"q{i} max_abs_delta: {delta:.4f}")
    print(f"worst_abs_delta : {worst:.4f}  (tolerance {tolerance})")
    print(f"agreement       : {'PASS' if worst <= tolerance else 'FAIL'}")
    return worst


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    queries, documents = args.queries, args.documents
    if not args.no_prompt_template:
        queries = [QUERY_TEMPLATE.format(text=t) for t in args.queries]
        documents = [DOCUMENT_TEMPLATE.format(text=t) for t in args.documents]

    wait_for_health(base_url, args.health_retries)
    query_vectors, query_seconds = embed(base_url, args, queries)
    document_vectors, document_seconds = embed(base_url, args, documents)

    endpoint = "/v1/embeddings" if args.api == "openai" else "/api/embed"
    print(f"endpoint        : {base_url}{endpoint}")
    print(f"model           : {args.model}")
    print(f"prompt_template : {'off (raw text)' if args.no_prompt_template else 'on (EmbeddingGemma task prefixes)'}")
    print(f"latency_s       : {query_seconds + document_seconds:.3f}")
    print(f"n_vectors       : {len(query_vectors) + len(document_vectors)}")
    print(f"dim             : {len(query_vectors[0])}")
    print(f"norm[q0]        : {math.sqrt(sum(x * x for x in query_vectors[0])):.4f}")
    print(f"head[q0]        : {[round(x, 5) for x in query_vectors[0][:8]]}")

    print("--- cosine similarity (semantic sanity check) ---")
    matrix = [[cosine(q, d) for d in document_vectors] for q in query_vectors]
    for i, row in enumerate(matrix):
        best = row.index(max(row))
        print(f"q{i} {args.queries[i]!r}")
        for j, value in enumerate(row):
            marker = "  <-- best" if j == best else ""
            print(f"    {value:.4f}  {args.documents[j]!r}{marker}")

    worst = cross_check(matrix, args.reference, args.tolerance) if args.reference else None

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "model": args.model,
                "endpoint": f"{base_url}{endpoint}",
                "prompt_template": not args.no_prompt_template,
                "embedding_dim": len(query_vectors[0]),
                "queries": args.queries,
                "documents": args.documents,
                "cosine_similarities": matrix,
                "query_embeddings_first16": [v[:16] for v in query_vectors],
                "worst_abs_delta_vs_reference": worst,
            }, f, indent=2)
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
