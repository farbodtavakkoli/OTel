"""OpenAI-compatible client for a llama.cpp llama-server embedding endpoint — see readme_embedding_llamacpp.md."""
import argparse
import json
import math
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; llama-server needs it only for gated GGUF pulls, not for this client.
load_dotenv("dev.env")

DEFAULT_TEXTS = [
    "A span is the basic unit of work recorded by a distributed trace.",
    "Preheat the oven to 200 degrees and butter a cake tin.",
    "OpenTelemetry collectors batch and export telemetry to a backend.",
]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="llama.cpp embedding smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="llama-server host")
    parser.add_argument("--port", type=int, default=8201, help="llama-server port")
    parser.add_argument("--model", type=str, default="embeddinggemma-300m", help="Model name echoed in the request body")
    parser.add_argument("--texts", type=str, nargs="*", default=None,
                        help="Texts to embed (defaults to a 3-sentence semantic-similarity probe)")
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


def embed(base_url: str, model: str, texts, timeout: int):
    """POST one embeddings request and return the parsed response plus latency."""
    started = time.time()
    r = requests.post(f"{base_url}/v1/embeddings", json={"model": model, "input": texts}, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    args = parse_args()
    texts = args.texts if args.texts else DEFAULT_TEXTS
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    body, elapsed = embed(base_url, args.model, texts, args.timeout)

    vectors = [row["embedding"] for row in body["data"]]
    print(f"endpoint   : {base_url}/v1/embeddings")
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

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
