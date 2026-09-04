"""LLM chat/classification client for a vLLM OpenAI-compatible server — see readme_llm_vllm.md."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; the server needs it to pull the Qwen FP8 repo.
load_dotenv("dev.env")

DEFAULT_PROMPT = "In two sentences, explain what vLLM is and which GPU vendors it supports."
CLASSIFY_LABELS = ["billing", "network_outage", "device_setup", "other"]


def parse_args():
    """Parse CLI arguments; every tunable of the completion request is exposed here."""
    parser = argparse.ArgumentParser(description="Chat or prompted-classification client for a vLLM server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="qwen38-27b-fp8",
                        help="Served model name (matches --served-model-name)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--classify", type=str, default=None,
                        help="Text to classify into one of the built-in labels via prompting")
    parser.add_argument("--labels", type=str, default=",".join(CLASSIFY_LABELS),
                        help="Comma-separated label set used by --classify")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Nucleus sampling top-p")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed for reproducible output")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def post_json(url, payload, timeout):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_messages(args):
    """Build the chat message list for either free generation or prompted classification."""
    if args.classify is not None:
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        system = ("You are a strict text classifier. Reply with exactly one label from this list "
                  f"and nothing else: {', '.join(labels)}.")
        return [{"role": "system", "content": system},
                {"role": "user", "content": args.classify}]

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    payload = {
        "model": args.model,
        "messages": build_messages(args),
        "max_tokens": args.max_tokens,
        "temperature": 0.0 if args.classify is not None else args.temperature,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    started = time.time()
    try:
        result = post_json(url, payload, args.timeout)
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0

    print(f"endpoint      : {url}")
    print(f"model         : {result.get('model')}")
    print(f"finish_reason : {choice.get('finish_reason')}")
    print(f"latency       : {elapsed:.2f}s")
    if completion_tokens:
        print(f"decode rate   : {completion_tokens / elapsed:.1f} tok/s ({completion_tokens} completion tokens)")
    print(f"usage         : {usage}")

    reasoning = message.get("reasoning_content")
    if reasoning:
        print()
        print("--- reasoning ---")
        print(reasoning.strip())

    print()
    print("--- response ---")
    print((message.get("content") or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
