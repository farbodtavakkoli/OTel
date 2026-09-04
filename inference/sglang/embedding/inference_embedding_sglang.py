"""OpenAI-compatible client for an SGLang embedding server — see readme_sglang_embedding.md."""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; embeddinggemma-300m is gated, so the server needs it at launch.
load_dotenv("dev.env")


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="SGLang embedding smoke client (/v1/embeddings)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=8101, help="SGLang server port")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m",
                        help="Model id as served (must match --model-path)")
    parser.add_argument("--texts", type=str, nargs="+",
                        default=["vLLM supports AMD ROCm.", "SQLite is an embedded database."],
                        help="Documents to embed")
    parser.add_argument("--query", type=str, default="Which inference engines support AMD ROCm?",
                        help="Query embedded and scored against every document")
    parser.add_argument("--endpoint", type=str, default="/v1/embeddings", help="Endpoint path to call")
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


def cosine(a, b) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    """Wait for the server, embed a query plus documents, print dims and cosine scores."""
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    waited = wait_for_health(base, args.wait)
    print(f"[health] server ready after {waited:.1f}s")

    inputs = [args.query] + list(args.texts)
    start = time.time()
    out = post_json(f"{base}{args.endpoint}", {"model": args.model, "input": inputs}, args.timeout)
    elapsed = time.time() - start

    vecs = [row["embedding"] for row in out["data"]]
    print(f"[latency] {elapsed:.2f}s | n_vectors={len(vecs)} dim={len(vecs[0])}")
    print(f"[vector0 head] {[round(v, 5) for v in vecs[0][:8]]}")
    for text, vec in zip(args.texts, vecs[1:]):
        print(f"[cosine] {cosine(vecs[0], vec):+.4f}  <- {text}")


if __name__ == "__main__":
    main()
