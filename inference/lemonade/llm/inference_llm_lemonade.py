"""OpenAI-compatible chat client for a Lemonade Server LLM endpoint — see readme_llm_lemonade.md."""
import argparse
import json
import os
import time
import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; Lemonade needs it only for gated checkpoint pulls, not for this client.
load_dotenv("dev.env")

DEFAULT_PROMPT = "In exactly two sentences, explain what an OpenTelemetry span is."


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Lemonade Server LLM smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Lemonade Server host")
    parser.add_argument("--port", type=int, default=8350, help="Lemonade Server port")
    parser.add_argument("--model", type=str, default="Qwen38-Q8",
                        help="Model id as registered in Lemonade (no user. prefix)")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions",
                        choices=["/v1/chat/completions", "/api/v1/chat/completions", "/v1/completions"],
                        help="Chat route to call")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt to send")
    parser.add_argument("--system", type=str, default=None, help="Optional system message")
    parser.add_argument("--max_tokens", type=int, default=256, help="Generation cap")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=120, help="Health-check attempts before giving up")
    parser.add_argument("--load_model", action="store_true", help="Ask Lemonade to load the model before prompting")
    parser.add_argument("--show_reasoning", action="store_true", help="Print the model's reasoning_content too")
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


def build_messages(prompt: str, system):
    """Assemble the OpenAI chat message list."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def chat(base_url: str, endpoint: str, model: str, messages, max_tokens: int, temperature: float, timeout: int):
    """POST one chat-completions request and return the parsed response plus latency."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    started = time.time()
    r = requests.post(f"{base_url}{endpoint}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def print_timings(body):
    """Print llama.cpp's own timing block when Lemonade passes it through."""
    timings = body.get("timings") or {}
    if not timings:
        return
    print("--- llama.cpp timings ---")
    print(f"prompt_n            : {timings.get('prompt_n')}")
    print(f"predicted_n         : {timings.get('predicted_n')}")
    print(f"predicted_per_second: {timings.get('predicted_per_second', 0):.2f} tok/s")
    if timings.get("draft_n"):
        accepted, drafted = timings.get("draft_n_accepted", 0), timings["draft_n"]
        print(f"draft_n / accepted  : {drafted} / {accepted} ({100.0 * accepted / drafted:.1f}% accepted)")


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    wait_for_health(base_url, args.health_retries)
    if args.load_model:
        load_model(base_url, args.model, args.timeout)

    messages = build_messages(args.prompt, args.system)
    body, elapsed = chat(base_url, args.endpoint, args.model, messages, args.max_tokens, args.temperature, args.timeout)
    message = body["choices"][0]["message"]

    print(f"endpoint   : {base_url}{args.endpoint}")
    print(f"model      : {body.get('model')}")
    print(f"fingerprint: {body.get('system_fingerprint')}")
    print(f"latency_s  : {elapsed:.3f}")
    print(f"finish     : {body['choices'][0].get('finish_reason')}")
    print(f"prompt     : {args.prompt}")
    if args.show_reasoning and message.get("reasoning_content"):
        print("--- reasoning_content ---")
        print(message["reasoning_content"].strip())
    print("--- generated text ---")
    print(message.get("content", "").strip())
    print_timings(body)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(body, f, indent=2)
        print(f"raw response written to {args.out}")


if __name__ == "__main__":
    main()
