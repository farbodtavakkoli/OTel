from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# GRPO_JUDGE_RUBRIC=v1 (default, integer 0-10) or v2-expected (continuous, see judge_one).
RUBRIC = os.environ.get("GRPO_JUDGE_RUBRIC", "v1")
RUBRIC_VERSION = "telecom-judge-" + RUBRIC
JUDGE_URL = os.environ.get("GRPO_JUDGE_URL", "http://127.0.0.1:8100/v1")
JUDGE_MODEL = os.environ.get("GRPO_JUDGE_MODEL", "judge")
CONCURRENCY = int(os.environ.get("GRPO_JUDGE_CONC", "64"))
TRUNCATION_PENALTY = -0.2

SYSTEM = (
    "You are a strict grader for telecommunications engineering answers. You are given a "
    "question, a reference answer written by an expert, and a candidate answer. Grade the "
    "candidate ONLY on whether it reaches the same conclusions and key facts as the "
    "reference. Style, length and formatting do not matter unless the question demands a "
    "format. Do not reward hedging, padding or restating the question."
)

USER_TMPL = """[QUESTION]
{question}

[REFERENCE ANSWER]
{reference}

[CANDIDATE ANSWER]
{candidate}

Score the candidate from 0 to 10:
10 = fully correct and complete relative to the reference
7-9 = correct conclusion, minor omissions or small inaccuracies
4-6 = partially correct, a key fact or step is wrong or missing
1-3 = mostly wrong but shows some relevant understanding
0 = wrong, empty, off-topic, or refuses

Reply with exactly one line of JSON and nothing else: {{"score": <integer 0-10>}}"""

# v2: single-digit scale so the whole score is one token and its log-prob distribution can be
# read as a continuous expectation (see _expected_score).
USER_TMPL_V2 = """[QUESTION]
{question}

[REFERENCE ANSWER]
{reference}

[CANDIDATE ANSWER]
{candidate}

Score the candidate from 0 to 9:
9 = fully correct and complete relative to the reference
6-8 = correct conclusion, minor omissions or small inaccuracies
3-5 = partially correct, a key fact or step is wrong or missing
1-2 = mostly wrong but shows some relevant understanding
0 = wrong, empty, off-topic, or refuses

Reply with exactly one line of JSON and nothing else: {{"score": <integer 0-9>}}"""

_JSON_SCORE = re.compile(r'"score"\s*:\s*(\d{1,2})')
_WG = re.compile(r'"WORKING GROUP"\s*:\s*"([^"]+)"', re.IGNORECASE)


def _post(payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        f"{JUDGE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _expected_score(choice) -> float | None:
    content = ((choice.get("logprobs") or {}).get("content")) or []
    for tok in content:
        if tok.get("token", "").strip().isdigit():
            probs = {}
            for alt in tok.get("top_logprobs", []):
                t = alt.get("token", "").strip()
                if t.isdigit() and len(t) == 1:
                    probs[int(t)] = max(probs.get(int(t), 0.0), math.exp(alt["logprob"]))
            z = sum(probs.values())
            if z <= 0:
                return float(tok["token"].strip())
            return sum(k * p for k, p in probs.items()) / z
    return None


def judge_one(question: str, reference: str, candidate: str):
    tmpl = USER_TMPL if RUBRIC == "v1" else USER_TMPL_V2
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": tmpl.format(
                question=question.strip()[:12000], reference=reference.strip()[:8000],
                candidate=(candidate or "").strip()[:8000])},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        # Qwen3-family thinking mode is off for the judge: we want one JSON line, fast.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if RUBRIC == "v2-expected":
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    for attempt in range(3):
        try:
            choice = _post(payload)["choices"][0]
            if RUBRIC == "v2-expected":
                e = _expected_score(choice)
                return None if e is None else e * (10.0 / 9.0)
            m = _JSON_SCORE.search(choice["message"]["content"])
            if m:
                return max(0, min(10, int(m.group(1))))
            return None
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def exact_3gpp(reference: str, candidate: str) -> float:
    ref = _WG.search(reference or "")
    cand = _WG.search(candidate or "")
    if not ref or not cand:
        return 0.0
    return 1.0 if ref.group(1).strip().upper() == cand.group(1).strip().upper() else 0.0


def question_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt or []:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if m.get("role") in ("user", "system"):
            parts.append(str(c))
    return "\n".join(parts)


def score_batch(prompts, completions, references, sources=None, finish_reasons=None) -> dict:
    n = len(completions)
    sources = sources or [None] * n
    finish_reasons = finish_reasons or [None] * n
    questions = [question_text(p) for p in prompts]
    comps = [c[0]["content"] if isinstance(c, list) and c and isinstance(c[0], dict) else (c or "")
             for c in completions]

    t0 = time.time()
    need_judge = [i for i in range(n) if sources[i] != "3gpp_tsg"]
    judge = [None] * n
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, s in zip(need_judge, ex.map(lambda i: judge_one(questions[i], references[i], comps[i]), need_judge)):
            judge[i] = s
    timing = time.time() - t0

    reward = []
    for i in range(n):
        base = exact_3gpp(references[i], comps[i]) if sources[i] == "3gpp_tsg" else ((judge[i] or 0) / 10.0)
        pen = TRUNCATION_PENALTY if finish_reasons[i] == "length" else 0.0
        reward.append(base + pen)
    return {"reward": reward, "judge": judge, "timing_s": timing, "rubric": RUBRIC_VERSION}


if __name__ == "__main__":
    # Self-check against the live judge: a correct and a wrong candidate must be ordered.
    q = "Which 3GPP release introduced 5G NR Phase 1?"
    ref = "Release 15."
    out = score_batch([q, q, q], ["Release 15 introduced NR Phase 1.", "Release 8.", ""],
                      [ref, ref, ref])
    print(json.dumps(out, indent=1))
    assert out["judge"][0] is not None and out["judge"][1] is not None, "judge unreachable"
    assert out["reward"][0] > out["reward"][1] >= out["reward"][2], out["reward"]
    print("judge_reward self-check OK")
