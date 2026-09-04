"""Generic GRPO reward functions — porting guide in readme_unsloth.md."""

import os
import re


def _text(completion):
    """Normalize a TRL completion (conversational list OR plain string) to text."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return completion or ""


def _prompt_text(prompt):
    """Normalize a TRL prompt (message dicts with string or typed-parts content, or a string) to text."""
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt or []:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict)))
        else:
            parts.append(str(c))
    return "\n".join(parts).strip()


# Verifiable-reward scores: large positive for correct, small negative for spread.
CORRECT_REWARD = 4.0
WRONG_REWARD = -0.5


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def reward_verifiable(prompts, completions, answer, **kwargs):
    """Return CORRECT_REWARD when the completion matches the gold `answer`, else WRONG_REWARD."""
    scores = []
    for comp, gold in zip(completions, answer):
        try:
            ok = _normalize(_text(comp)) == _normalize(str(gold))
        except Exception:
            ok = False
        scores.append(CORRECT_REWARD if ok else WRONG_REWARD)
    return scores


# Judge config resolves from args, then env: JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY.
_JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION and a MODEL ANSWER, rate how correct, "
    "relevant, and well-formed the answer is on a scale from 0 to 10. "
    "Reply with ONLY the integer score, nothing else."
)


def make_llm_judge_reward(model=None, base_url=None, api_key=None,
                          max_tokens=8, weight=1.0, timeout=30.0):
    """Return a reward fn that grades each completion 0..10 with an external LLM (OpenAI SDK)."""
    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    base_url = base_url or os.environ.get("JUDGE_BASE_URL")  # None -> OpenAI default endpoint
    api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    def reward_llm_judge(prompts, completions, **kwargs):
        # Lazy import/construct so a rule-only run never needs `openai` or a key.
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)

        scores = []
        for prompt, comp in zip(prompts, completions):
            question = _prompt_text(prompt)
            answer = _text(comp)
            user_msg = f"QUESTION:\n{question}\n\nMODEL ANSWER:\n{answer}\n\nScore (0-10):"
            try:
                # Streamed OpenAI-SDK call; accumulate chunks into the final text.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                              {"role": "user", "content": user_msg}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stream=True,
                    timeout=timeout,
                )
                text = ""
                for chunk in resp:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        text += delta
                m = re.search(r"\d+(?:\.\d+)?", text)
                raw = float(m.group(0)) if m else 0.0
                scores.append(weight * max(0.0, min(raw, 10.0)) / 10.0)
            except Exception:
                scores.append(0.0)  # never crash the training step
        return scores

    reward_llm_judge.__name__ = "reward_llm_judge"
    return reward_llm_judge


def build_reward_funcs(reward_mode="rule", judge_model=None):
    """Return the reward_funcs list GRPOTrainer expects for a given --reward_mode."""
    if reward_mode == "rule":
        return [reward_verifiable]
    if reward_mode == "llm":
        return [make_llm_judge_reward(judge_model)]
    if reward_mode == "hybrid":
        return [reward_verifiable, make_llm_judge_reward(judge_model)]
    raise ValueError(f"unknown reward_mode {reward_mode!r}")
