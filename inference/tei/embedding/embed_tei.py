"""Client for a Hugging Face TEI text-embeddings-router endpoint — see README.md."""
import argparse
import json
import math
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the router needs it to resolve the gated EmbeddingGemma
# repo, this client does not. Loaded anyway so the folder behaves like its siblings.
load_dotenv("dev.env")

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
    parser = argparse.ArgumentParser(description="TEI embedding smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="text-embeddings-router host")
    parser.add_argument("--port", type=int, default=8301, help="text-embeddings-router port")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m",
                        help="Model name echoed in the /v1/embeddings request body")
    parser.add_argument("--api", type=str, default="both", choices=["embed", "openai", "both"],
                        help="Which endpoint to exercise: native /embed, OpenAI /v1/embeddings, or both")
    parser.add_argument("--queries", type=str, nargs="*", default=None, help="Query texts")
    parser.add_argument("--documents", type=str, nargs="*", default=None, help="Document texts")
    parser.add_argument("--query_prompt_name", type=str, default="query",
                        help="sentence-transformers prompt name applied to queries (empty string disables)")
    parser.add_argument("--document_prompt_name", type=str, default="document",
                        help="sentence-transformers prompt name applied to documents (empty string disables)")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Ask the router to L2-normalize the returned vectors")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false",
                        help="Return raw unnormalized vectors")
    parser.add_argument("--truncate", action="store_true", help="Let the router truncate over-long inputs")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to a Transformers baseline reference JSON to compare against")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the run summary JSON")
    return parser.parse_args()


def wait_for_health(base_url, retries):
    """Block until /health returns 200, or raise."""
    for _ in range(retries):
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"TEI router at {base_url} never became healthy after {retries} attempts")


def server_info(base_url, timeout):
    """Fetch /info so the run records what the server actually loaded."""
    r = requests.get(f"{base_url}/info", timeout=timeout)
    r.raise_for_status()
    return r.json()


def embed_native(base_url, texts, prompt_name, normalize, truncate, timeout):
    """POST TEI's native /embed and return (vectors, seconds)."""
    payload = {"inputs": texts, "normalize": normalize, "truncate": truncate}
    if prompt_name:
        payload["prompt_name"] = prompt_name
    started = time.time()
    r = requests.post(f"{base_url}/embed", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def embed_openai(base_url, model, texts, timeout):
    """POST the OpenAI-compatible /v1/embeddings and return (vectors, seconds)."""
    started = time.time()
    r = requests.post(f"{base_url}/v1/embeddings", json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    return [row["embedding"] for row in body["data"]], time.time() - started


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def l2_norm(vec):
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def compare_to_reference(path, query_vectors, similarity_rows):
    """Compare this run against a Transformers baseline reference JSON."""
    with open(path) as handle:
        ref = json.load(handle)
    report = {"reference_path": path, "reference_dim": ref.get("embedding_dim")}
    ref_head = ref.get("query_embeddings_first16") or []
    head_cosines = []
    for idx, row in enumerate(ref_head):
        if idx < len(query_vectors):
            head_cosines.append(round(cosine(row, query_vectors[idx][:len(row)]), 6))
    report["first16_cosine_vs_reference"] = head_cosines
    ref_sims = ref.get("cosine_similarities") or []
    deltas = []
    for r_row, m_row in zip(ref_sims, similarity_rows):
        deltas.append([round(m - r, 6) for r, m in zip(r_row, m_row)])
    report["reference_similarities"] = ref_sims
    report["measured_similarities"] = [[round(v, 6) for v in row] for row in similarity_rows]
    report["similarity_delta"] = deltas
    flat = [abs(v) for row in deltas for v in row]
    report["max_abs_similarity_delta"] = round(max(flat), 6) if flat else None
    return report


def main():
    args = parse_args()
    queries = args.queries if args.queries else DEFAULT_QUERIES
    documents = args.documents if args.documents else DEFAULT_DOCUMENTS
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    info = server_info(base_url, args.timeout)
    print(f"endpoint    : {base_url}")
    print(f"model_id    : {info.get('model_id')}")
    print(f"dtype       : {info.get('model_dtype')}")
    print(f"pooling     : {info.get('model_type')}")
    print(f"tei_version : {info.get('version')} (sha {info.get('sha')})")

    summary = {"base_url": base_url, "info": info, "queries": queries, "documents": documents}

    if args.api in ("embed", "both"):
        q_vecs, q_secs = embed_native(base_url, queries, args.query_prompt_name,
                                      args.normalize, args.truncate, args.timeout)
        d_vecs, d_secs = embed_native(base_url, documents, args.document_prompt_name,
                                      args.normalize, args.truncate, args.timeout)
        print("--- POST /embed ---")
        print(f"n_query_vec : {len(q_vecs)}   n_doc_vec : {len(d_vecs)}")
        print(f"dim         : {len(q_vecs[0])}")
        print(f"l2_norm[q0] : {l2_norm(q_vecs[0]):.6f}")
        print(f"head[q0]    : {[round(x, 5) for x in q_vecs[0][:8]]}")
        print(f"latency_s   : queries {q_secs:.3f} / documents {d_secs:.3f}")

        sims = [[cosine(q, d) for d in d_vecs] for q in q_vecs]
        print("--- cosine similarity (query x document) ---")
        for qi, query in enumerate(queries):
            print(f"  Q{qi}: {query}")
            ranked = sorted(range(len(documents)), key=lambda di: sims[qi][di], reverse=True)
            for di in ranked:
                print(f"     {sims[qi][di]:+.4f}  {documents[di]}")
        best = [max(range(len(documents)), key=lambda di: sims[qi][di]) for qi in range(len(queries))]
        print(f"top1_doc_index_per_query : {best}")

        summary["dim"] = len(q_vecs[0])
        summary["embed_latency_seconds"] = {"queries": round(q_secs, 4), "documents": round(d_secs, 4)}
        summary["cosine_similarities"] = [[round(v, 6) for v in row] for row in sims]
        summary["query_embeddings_first16"] = [[round(x, 6) for x in v[:16]] for v in q_vecs]
        summary["top1_doc_index_per_query"] = best

        if args.reference:
            print("--- vs Transformers baseline ---")
            report = compare_to_reference(args.reference, q_vecs, sims)
            print(f"reference_dim            : {report['reference_dim']}")
            print(f"first16_cosine_vs_ref    : {report['first16_cosine_vs_reference']}")
            print(f"max_abs_similarity_delta : {report['max_abs_similarity_delta']}")
            summary["reference_comparison"] = report

    if args.api in ("openai", "both"):
        o_vecs, o_secs = embed_openai(base_url, args.model, queries, args.timeout)
        print("--- POST /v1/embeddings (OpenAI-compatible) ---")
        print(f"n_vectors   : {len(o_vecs)}   dim : {len(o_vecs[0])}")
        print(f"l2_norm[0]  : {l2_norm(o_vecs[0]):.6f}")
        print(f"latency_s   : {o_secs:.3f}")
        summary["openai_dim"] = len(o_vecs[0])
        summary["openai_latency_seconds"] = round(o_secs, 4)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"summary written to {args.out}")


if __name__ == "__main__":
    main()
