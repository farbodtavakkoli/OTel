import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv("dev.env")

# One request = a state (text, or a JSON object) plus named questions. Question types:
#   noul   - yes/no, answered with a probability ("noul"); optional criteria {"false": ..., "true": ...}
#   choice - pick one key of criteria {option: description}; answered with "choice" and probabilities
#   score  - an ordinal scale, criteria [level0, level1, ...]; answered with an expected "score"
TEAMS = {
    "billing": "charges, invoices, refunds and payment plans",
    "technical_support": "outages, slow speeds, devices and service faults",
    "sales": "new lines, plan upgrades, quotes and pricing",
}

DEFAULT_CASES = [
    {
        "name": "outage",
        "state": "My home fiber has been down since 7 am yesterday and I work from home. I rebooted the router "
        "three times and the light is still red. This is the second outage this week. Send a technician today!",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": TEAMS},
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["calm", "frustrated", "furious"],
            },
        },
        "expect": {"team": "technical_support", "urgent": True},
    },
    {
        "name": "double_charge",
        "state": "Hi, my October bill shows the unlimited plan charged twice, $85 on the 3rd and again on the 5th. "
        "Can you refund the duplicate payment?",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": TEAMS},
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        },
        "expect": {"team": "billing", "refund": True},
    },
    {
        "name": "plan_inquiry",
        "state": "Hello! We are a family of four and are thinking about switching carriers next month. "
        "What 5G plans do you offer with shared data, and is there a discount for adding tablets?",
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": TEAMS},
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        },
        "expect": {"team": "sales", "urgent": False},
    },
    {
        "name": "network_alarm",
        "state": {
            "site": "eNB-4471",
            "alarm": "S1 link down",
            "since": "2026-11-23T08:12:00Z",
            "affected_cells": 3,
            "subscribers_impacted_estimate": 2400,
            "backup_power": "on battery, 41 minutes remaining",
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "What should the network operations center do next?",
                "criteria": {
                    "monitor": "keep watching, no action needed",
                    "dispatch_field_tech": "send a field technician to the site",
                    "open_vendor_ticket": "open a ticket with the equipment vendor",
                },
            },
            "service_impacting": {"type": "noul", "instructions": "Is this alarm impacting customer service?"},
        },
        "expect": {"action": "dispatch_field_tech", "service_impacting": True},
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Client for the Tenstorrent strands-decider System One server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--request_file", type=str, default=None,
                        help="JSON file with one request {state, questions} or a list of them; omit for the built-in cases")
    parser.add_argument("--output", type=str, default=None, help="Path for the reference-output JSON artifact")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def make_request(url, payload):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['API_KEY']}"
    return urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)


def decide(url, case, timeout):
    payload = {"state": case["state"], "questions": case["questions"]}
    started = time.time()
    with urllib.request.urlopen(make_request(url, payload), timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result, time.time() - started


def compact(answer):
    if answer["type"] == "noul":
        return f"noul {answer['noul']:.3f}"
    if answer["type"] == "choice":
        return f"choice {answer['choice']} (confidence {answer['confidence']:.2f})"
    return f"score {answer['score']:.2f} of 0..{len(answer['legend']) - 1}"


def check(answer, expected):
    if answer["type"] == "noul":
        return (answer["noul"] > 0.5) == expected
    if answer["type"] == "choice":
        return answer["choice"] == expected
    return True


def main():
    args = parse_args()
    base = f"http://{args.host}:{args.port}"
    url = f"{base}/v1/systemone"
    if args.request_file:
        with open(args.request_file) as f:
            loaded = json.load(f)
        cases = [{"name": f"request_{i}", **c} for i, c in enumerate(loaded if isinstance(loaded, list) else [loaded])]
    else:
        cases = DEFAULT_CASES

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=args.timeout) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        decide(url, cases[0], args.timeout)  # warm-up
        results = [decide(url, case, args.timeout) for case in cases]
    except urllib.error.URLError as exc:
        print(f"request to {url} failed: {exc}", file=sys.stderr)
        return 1

    print(f"endpoint      : {url}")
    print(f"model         : {results[0][0].get('model')}")
    print(f"workers       : {health.get('workers_ready')}/{health.get('workers')} ready on {health.get('device')}")

    ok = True
    artifact_cases = []
    for case, (result, seconds) in zip(cases, results):
        usage = result.get("usage", {})
        print(f"\n{case['name']}: {seconds * 1000:.1f} ms round trip, server {result.get('latency_ms')} ms, "
              f"{usage.get('input_tokens')} input tokens")
        checks = {}
        for name, answer in result["answers"].items():
            line = f"  {name:<18} {compact(answer)}"
            if name in case.get("expect", {}):
                passed = check(answer, case["expect"][name])
                checks[name] = passed
                ok &= passed
                line += f"  expect {case['expect'][name]} -> {'PASS' if passed else 'FAIL'}"
            print(line)
        artifact_cases.append({"name": case["name"], "state": case["state"], "questions": case["questions"],
                               "answers": result["answers"], "usage": usage, "checks": checks})

    artifact = {"model": results[0][0].get("model"), "endpoint": url, "cases": artifact_cases,
                "sanity_check_passed": bool(ok)}
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nwrote reference artifact: {args.output}")

    if not args.request_file:
        print(f"\nSANITY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
