"""Rule-based reward functions for verl RLVR runs (calling contract documented in readme_verl.md)."""

import re

# Strip everything except alphanumerics and single spaces before comparing.
_NORMALIZE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text):
    """Lowercase, drop punctuation, and collapse runs of whitespace."""
    text = _NORMALIZE.sub(" ", str(text).lower())
    return " ".join(text.split())


def _extract_answer(solution_str):
    """Prefer the contents of the last \\boxed{...}; fall back to the last non-empty line."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", solution_str)
    if boxed:
        return boxed[-1]

    lines = [line.strip() for line in solution_str.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Exact-match reward with a partial-credit fallback (scoring tiers in readme_verl.md)."""
    prediction = _normalize(_extract_answer(solution_str))
    reference = _normalize(ground_truth)

    if not prediction:
        return 0.0
    if prediction == reference:
        return 1.0
    if reference and reference in _normalize(solution_str):
        return 0.5
    return 0.1


def compute_score_format(data_source, solution_str, ground_truth, extra_info=None):
    """Alternative scorer that additionally rewards a <think>...</think> format."""
    correctness = compute_score(data_source, solution_str, ground_truth, extra_info)

    has_think = bool(re.search(r"<think>.*?</think>", solution_str, re.DOTALL))
    format_bonus = 0.2 if has_think else 0.0

    return min(1.0, 0.8 * correctness + format_bonus)
