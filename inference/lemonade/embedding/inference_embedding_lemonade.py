"""OpenAI-compatible client for a Lemonade Server embedding endpoint — see README.md."""
import argparse
import json
import math
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; Lemonade needs it only for gated checkpoint pulls, not for this client.
load_dotenv("dev.env")

DEFAULT_TEXTS = [
    "A span is the basic unit of work recorded by a distributed trace.",
    "Preheat the oven to 200 degrees and butter a cake tin.",
    "OpenTelemetry collectors batch and export telemetry to a backend.",
]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Lemonade Server embedding smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Lemonade Server host")
    parser.add_argument("--port", type=int, default=8350, help="Lemonade Server port")
    parser.add_argument("--model", type=str, default="EmbeddingGemma-300M",
                        help="Model id as registered in Lemonade (no user. prefix)")
    parser.add_argument("--endpoint", type=str, default="/v1/embeddings",
                        choices=["/v1/embeddings", "/api/v1/embeddings"], help="Embeddings route to call")
    parser.add_argument("--texts", type=str, nargs="*", default=None,
                        help="Texts to embed (defaults to a 3-sentence semantic-similarity probe)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Optional JSON file of reference vectors to cross-check against; when --texts is "
                             "omitted the baseline's own queries are embedded so the comparison is meaningful")
    parser.add_argument("--reference_prompt", type=str, default="task: search result | query: ",
                        help="Prefix applied to baseline queries (EmbeddingGemma's encode_query template)")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--load_model", action="store_true", help="Ask Lemonade to load the model before embedding")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the raw JSON response")
    return parser.parse_args()


def wait_for_health(base_url: str, retries: int):
    """Block until Lemonade Server answers /api/v1/health, or raise."""
    for _ in range(retries):
        try:
            if requests.get(f"{base_url}/api/v1/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"Lemonade Server at {base_url} never became healthy after {retries} attempts")


def load_model(base_url: str, model: str, timeout: int):
    """POST /api/v1/load so the llama.cpp subprocess is warm before timing anything."""
    r = requests.post(f"{base_url}/api/v1/load", json={"model_name": model}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def embed(base_url: str, endpoint: str, model: str, texts, timeout: int):
    """POST one embeddings request and return the parsed response plus latency."""
    started = time.time()
    r = requests.post(f"{base_url}{endpoint}", json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_reference(path):
    """Read a baseline JSON and return (reference_vectors, source_texts_or_None)."""
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict):
        if "data" in blob:
            return [row["embedding"] for row in blob["data"]], None
        for key in ("embeddings", "vectors", "query_embeddings_first16"):
            if key in blob:
                return blob[key], blob.get("queries")
    return blob, None


def main():
    args = parse_args()
    reference, reference_texts = load_reference(args.reference) if args.reference else (None, None)

    # Comparing our vectors against the baseline only means anything if we embed the same
    # strings, with the same EmbeddingGemma task prompt the baseline's encode_query applied.
    if args.texts:
        texts = args.texts
    elif reference_texts:
        texts = [f"{args.reference_prompt}{t}" for t in reference_texts]
    else:
        texts = DEFAULT_TEXTS
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    if args.load_model:
        load_model(base_url, args.model, args.timeout)

    body, elapsed = embed(base_url, args.endpoint, args.model, texts, args.timeout)
    vectors = [row["embedding"] for row in body["data"]]

    print(f"endpoint   : {base_url}{args.endpoint}")
    print(f"model      : {body.get('model')}")
    print(f"latency_s  : {elapsed:.3f}")
    print(f"n_vectors  : {len(vectors)}")
    print(f"dim        : {len(vectors[0])}")
    print(f"norm[0]    : {math.sqrt(sum(x * x for x in vectors[0])):.4f}")
    print(f"head[0]    : {[round(x, 5) for x in vectors[0][:8]]}")

    if len(vectors) >= 3:
        print("--- cosine similarity (semantic sanity check) ---")
        print(f"related   (0 vs 2) : {cosine(vectors[0], vectors[2]):.4f}")
        print(f"unrelated (0 vs 1) : {cosine(vectors[0], vectors[1]):.4f}")

    if reference:
        print("--- agreement vs reference baseline ---")
        for i, ref in enumerate(reference[:len(vectors)]):
            width = min(len(ref), len(vectors[i]))
            print(f"cosine(mine[{i}][:{width}], ref[{i}][:{width}]) : {cosine(vectors[i][:width], ref[:width]):.6f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
