"""OpenAI-compatible client for a llama.cpp llama-server LLM endpoint — see readme_llm_llamacpp.md."""
import argparse
import json
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; llama-server needs it only for gated GGUF pulls, not for this client.
load_dotenv("dev.env")


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="llama.cpp LLM chat-completions smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="llama-server host")
    parser.add_argument("--port", type=int, default=8200, help="llama-server port")
    parser.add_argument("--model", type=str, default="qwen3.8-27b-gguf", help="Model name echoed in the request body")
    parser.add_argument("--prompt", type=str, default="In one sentence, what is an OpenTelemetry span?",
                        help="User prompt sent to /v1/chat/completions")
    parser.add_argument("--system", type=str, default="You are a concise technical assistant.",
                        help="System prompt")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability mass")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds")
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


def chat(base_url: str, args, timeout: int):
    """POST one chat completion and return the parsed response."""
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    started = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    body, elapsed = chat(base_url, args, args.timeout)

    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(f"endpoint      : {base_url}/v1/chat/completions")
    print(f"model         : {body.get('model')}")
    print(f"latency_s     : {elapsed:.2f}")
    print(f"prompt_tokens : {usage.get('prompt_tokens')}")
    print(f"output_tokens : {usage.get('completion_tokens')}")
    if elapsed > 0 and usage.get("completion_tokens"):
        print(f"tok_per_s     : {usage['completion_tokens'] / elapsed:.1f}")
    print("--- generated text ---")
    print(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
