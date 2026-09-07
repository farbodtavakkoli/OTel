"""OpenAI-compatible client for an SGLang LLM server — see README.md."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the server needs it for gated models, the client only for parity.
load_dotenv("dev.env")


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="SGLang LLM smoke client (/v1/chat/completions)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=8100, help="SGLang server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.8-27B-FP8",
                        help="Model id as served (must match --model-path or --served-model-name)")
    parser.add_argument("--prompt", type=str, default="Name the two GPU vendors ROCm and CUDA belong to, in one line.",
                        help="User prompt sent to the chat endpoint")
    parser.add_argument("--system", type=str, default="You are a terse assistant.", help="System prompt")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions", help="Endpoint path to call")
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


def main():
    """Wait for the server, send one chat completion, print the real generated text."""
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    waited = wait_for_health(base, args.wait)
    print(f"[health] server ready after {waited:.1f}s")

    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": args.system},
                     {"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    start = time.time()
    out = post_json(f"{base}{args.endpoint}", payload, args.timeout)
    elapsed = time.time() - start

    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    print(f"[latency] {elapsed:.2f}s | prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}")
    print(f"[response] {text}")


if __name__ == "__main__":
    main()
