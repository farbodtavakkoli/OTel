import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv("dev.env")

DEFAULT_PROMPT = "In three sentences, explain the O-RAN F1 interface and which nodes it connects."
CLASSIFY_LABELS = ["billing", "network_outage", "device_setup", "other"]


def parse_args():
    parser = argparse.ArgumentParser(description="Chat or prompted-classification client for a tt-inference-server endpoint")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it", help="Served model name")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt")
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--classify", type=str, default=None,
                        help="Text to classify into one of the built-in labels via prompting")
    parser.add_argument("--labels", type=str, default=",".join(CLASSIFY_LABELS),
                        help="Comma-separated label set used by --classify")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate, reasoning included")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed for reproducible output")
    parser.add_argument("--no_thinking", action="store_true", help="Send enable_thinking false and skip reasoning")
    parser.add_argument("--stream", action="store_true", help="Stream tokens over SSE")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def make_request(url, payload):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
    return urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)


def build_messages(args):
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


def run_stream(req, timeout):
    reasoning, content = [], []
    section = None
    ttft = None
    usage = {}
    finish = None
    model = None
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            model = event.get("model", model)
            if event.get("usage"):
                usage = event["usage"]
            if not event.get("choices"):
                continue
            choice = event["choices"][0]
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            for name, text, sink in (("reasoning", delta.get("reasoning") or delta.get("reasoning_content"), reasoning),
                                     ("response", delta.get("content"), content)):
                if not text:
                    continue
                if ttft is None:
                    ttft = time.time() - started
                if section != name:
                    print(f"\n--- {name} ---", flush=True)
                    section = name
                sink.append(text)
                print(text, end="", flush=True)
    print()
    return model, finish, usage, ttft, time.time() - started


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    payload = {
        "model": args.model,
        "messages": build_messages(args),
        "max_tokens": args.max_tokens,
        "temperature": 0.0 if args.classify is not None else args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    try:
        if args.stream:
            model, finish, usage, ttft, elapsed = run_stream(make_request(url, payload), args.timeout)
        else:
            started = time.time()
            with urllib.request.urlopen(make_request(url, payload), timeout=args.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - started
            ttft = None
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    if not args.stream:
        choice = result["choices"][0]
        message = choice["message"]
        model, finish, usage = result.get("model"), choice.get("finish_reason"), result.get("usage", {})

    completion_tokens = usage.get("completion_tokens") or 0
    if args.stream:
        print()
    print(f"endpoint      : {url}")
    print(f"model         : {model}")
    print(f"thinking      : {'off' if args.no_thinking else 'on'}")
    print(f"finish_reason : {finish}")
    print(f"latency       : {elapsed:.2f}s")
    if ttft is not None:
        print(f"ttft          : {ttft:.2f}s")
    if completion_tokens:
        if ttft is not None and completion_tokens > 1 and elapsed > ttft:
            rate = (completion_tokens - 1) / (elapsed - ttft)
        else:
            rate = completion_tokens / elapsed
        print(f"decode rate   : {rate:.1f} tok/s ({completion_tokens} completion tokens)")
    print(f"usage         : {usage}")

    if args.stream:
        return 0

    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        print()
        print("--- reasoning ---")
        print(reasoning.strip())

    print()
    print("--- response ---")
    content = message.get("content")
    if content is None and finish == "length":
        print("(no answer: max_tokens was spent on reasoning; raise --max_tokens or pass --no_thinking)")
    else:
        print((content or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
