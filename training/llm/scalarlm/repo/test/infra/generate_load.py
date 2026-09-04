#!/usr/bin/env python3
"""
Continuous load generator for observability testing.

Sends requests to the OpenAI-compatible endpoint to generate real traffic.
Run this in the background while viewing Grafana to see live metrics.

Usage:
    python test/infra/generate_load.py [--requests N] [--delay SECONDS]
"""

import argparse
import requests
import time
import random
from typing import List

SCALARLM_URL = "http://localhost:8000"

PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Tell me a joke about programming.",
    "Explain quantum computing in simple terms.",
    "Write a haiku about coding.",
    "What are the benefits of using Python?",
    "How does machine learning work?",
    "What is the difference between AI and ML?",
    "Tell me about the history of computers.",
    "What is the meaning of life?",
]


def send_chat_completion(prompt: str, max_tokens: int = 50) -> dict:
    """Send a chat completion request to ScalarLM"""
    payload = {
        "model": "tiny-random/qwen3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{SCALARLM_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": response.text[:200]
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_load(num_requests: int, delay: float):
    """Generate continuous load"""
    print(f"Starting load generator...")
    print(f"  Requests: {num_requests}")
    print(f"  Delay: {delay}s between requests")
    print(f"  Target: {SCALARLM_URL}")
    print()

    successes = 0
    errors = 0

    for i in range(num_requests):
        prompt = random.choice(PROMPTS)
        max_tokens = random.randint(20, 100)

        print(f"[{i+1}/{num_requests}] Sending request... ", end='', flush=True)

        start_time = time.time()
        result = send_chat_completion(prompt, max_tokens)
        duration = time.time() - start_time

        if result["status"] == "success":
            successes += 1
            data = result["data"]
            tokens = data.get("usage", {}).get("total_tokens", "?")
            print(f"✓ OK ({duration:.2f}s, {tokens} tokens)")
        else:
            errors += 1
            msg = result.get("message", "Unknown error")[:100]
            print(f"✗ FAIL - {msg}")

        if i < num_requests - 1:
            time.sleep(delay)

    print()
    print("="*60)
    print(f"Load generation complete")
    print(f"  Successes: {successes}")
    print(f"  Errors: {errors}")
    print(f"  Success rate: {successes/(successes+errors)*100:.1f}%")
    print("="*60)
    print()
    print("Check Grafana for metrics:")
    print(f"  {SCALARLM_URL.replace('8000', '3001')}/d/scalarlm-overview/scalarlm-overview")


def main():
    parser = argparse.ArgumentParser(description="Generate load for observability testing")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests to send")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    # Check if ScalarLM is running
    try:
        response = requests.get(f"{SCALARLM_URL}/v1/health", timeout=2)
        print(f"✓ ScalarLM is running")
        print()
    except:
        print(f"✗ ScalarLM is NOT running!")
        print(f"  Run: ./scalarlm up cpu")
        return

    generate_load(args.requests, args.delay)


if __name__ == "__main__":
    main()
