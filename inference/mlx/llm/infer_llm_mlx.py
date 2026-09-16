import argparse
import json, os, subprocess, time

import requests
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN; mlx_lm.server needs it only for gated repos, not for this client.
load_dotenv("dev.env")

# OTel-LLM-E4B-IT is a context-grounded model (OTel-LLM + OTel-Safety): it is meant to answer
# from a retrieved passage. Without one it sometimes abstains ("I do not have enough
# information based on the provided context ...") and sometimes answers from memory, so the
# smoke prompt carries a context. Pass --context '' to see the ungrounded behaviour.
DEFAULT_CONTEXT = (
    "In NR sidelink resource allocation mode 2 (TS 38.214 clause 8.1.4), the UE monitors the "
    "sensing window and decodes SCI format 1-A to identify resources reserved by other UEs. "
    "Candidate resources in the selection window whose RSRP exceeds a threshold are excluded. "
    "If fewer than 20% of the initial candidates remain, the threshold is raised by 3 dB and the "
    "procedure repeats."
)
DEFAULT_PROMPT = "What is the sensing window used for in Mode 2, and what happens if too few candidates remain?"


def parse_args():
    parser = argparse.ArgumentParser(description="mlx_lm.server chat-completions smoke client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="mlx_lm.server host")
    parser.add_argument("--port", type=int, default=8080, help="mlx_lm.server port")
    parser.add_argument("--model", type=str, default="./OTel-LLM-E4B-IT-4bit",
                        help="Model sent in the request body. Must be the same path/id serve.sh was started with: "
                             "mlx_lm.server loads whatever the request names (and GET /v1/models lists the whole "
                             "HF cache, not the loaded model)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User question")
    parser.add_argument("--context", type=str, default=DEFAULT_CONTEXT,
                        help="Grounding context prepended to the question; pass '' for an ungrounded question")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy, reproducible)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling probability mass")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds")
    parser.add_argument("--health_retries", type=int, default=60, help="Health-check attempts before giving up")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the raw JSON response")
    return parser.parse_args()


def wait_for_health(base_url: str, retries: int):
    for _ in range(retries):
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"mlx_lm.server at {base_url} never became healthy after {retries} attempts")


def chat(base_url: str, args):
    user = f"Context:\n{args.context}\n\nQuestion: {args.prompt}" if args.context else args.prompt
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": user}]
    payload = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "top_p": args.top_p}
    started = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=args.timeout)
    r.raise_for_status()
    return r.json(), time.time() - started


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    wait_for_health(base_url, args.health_retries)

    chat(base_url, args)  # warm-up: prompt-cache fill and Metal kernel compilation
    resp, elapsed = chat(base_url, args)
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    comp = usage.get("completion_tokens", 0)
    print(f"endpoint      : {base_url}/v1/chat/completions")
    print(f"model         : {args.model}")
    print(f"prompt_tokens : {usage.get('prompt_tokens')}")
    print(f"output_tokens : {comp}")
    print(f"latency       : {elapsed:.2f}s (~{comp / elapsed:.1f} tok/s end-to-end, warm prompt cache)")
    print("--- generated text ---")
    print(text.strip())

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"request": vars(args), "response": resp, "latency_seconds": round(elapsed, 3)}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
