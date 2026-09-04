"""Client for an Ollama server serving a Qwen3.8-27B GGUF — see readme_llm_ollama.md."""
import argparse
import json
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; Ollama needs it only for gated `hf.co/...` pulls, never for this client.
load_dotenv("dev.env")


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Ollama LLM generate/chat smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Ollama server host")
    parser.add_argument("--port", type=int, default=11434, help="Ollama server port")
    parser.add_argument("--model", type=str, default="qwen3.8-27b-q8", help="Ollama model name created from the GGUF")
    parser.add_argument("--api", type=str, default="chat", choices=["chat", "generate", "openai"],
                        help="chat=/api/chat, generate=/api/generate, openai=/v1/chat/completions")
    parser.add_argument("--prompt", type=str, default="In one sentence, what is an OpenTelemetry span?",
                        help="User prompt")
    parser.add_argument("--system", type=str, default="You are a concise technical assistant.",
                        help="System prompt (ignored by --api generate)")
    parser.add_argument("--think", action="store_true",
                        help="Keep the model's thinking block on; default off (see the thinking quirk in the README)")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability mass")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility")
    parser.add_argument("--num_ctx", type=int, default=None,
                        help="Override the context window; unset uses Ollama's VRAM-derived default")
    parser.add_argument("--keep_alive", type=str, default="5m", help="How long Ollama keeps the model resident")
    parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the raw JSON response")
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


def build_options(args):
    """Assemble the Ollama `options` block from the tunables."""
    options = {
        "num_predict": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    return options


def call_endpoint(base_url: str, args):
    """POST one completion on the selected API and return (body, elapsed_seconds)."""
    if args.api == "openai":
        url = f"{base_url}/v1/chat/completions"
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
    elif args.api == "generate":
        url = f"{base_url}/api/generate"
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
            "think": args.think,
            "keep_alive": args.keep_alive,
            "options": build_options(args),
        }
    else:
        url = f"{base_url}/api/chat"
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            "stream": False,
            "think": args.think,
            "keep_alive": args.keep_alive,
            "options": build_options(args),
        }

    started = time.time()
    response = requests.post(url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    return url, response.json(), time.time() - started


def report(url: str, body: dict, elapsed: float, api: str):
    """Print the generated text plus whatever timing the chosen API exposes."""
    print(f"endpoint      : {url}")
    print(f"model         : {body.get('model')}")
    print(f"latency_s     : {elapsed:.2f}")

    if api == "openai":
        usage = body.get("usage", {})
        message = body["choices"][0]["message"]
        text, thinking = message.get("content", ""), message.get("reasoning")
        print(f"prompt_tokens : {usage.get('prompt_tokens')}")
        print(f"output_tokens : {usage.get('completion_tokens')}")
        print(f"finish_reason : {body['choices'][0].get('finish_reason')}")
    else:
        text = body.get("response") if api == "generate" else body["message"]["content"]
        thinking = body.get("thinking") if api == "generate" else body["message"].get("thinking")
        eval_count, eval_ns = body.get("eval_count"), body.get("eval_duration")
        print(f"prompt_tokens : {body.get('prompt_eval_count')}")
        print(f"output_tokens : {eval_count}")
        print(f"load_s        : {body.get('load_duration', 0) / 1e9:.2f}")
        if eval_count and eval_ns:
            print(f"tok_per_s     : {eval_count / (eval_ns / 1e9):.1f}")

    if thinking:
        print("--- thinking (truncated) ---")
        print(thinking[:400])
    print("--- generated text ---")
    print(text)


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    url, body, elapsed = call_endpoint(base_url, args)
    report(url, body, elapsed, args.api)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
