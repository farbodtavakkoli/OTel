from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.config import Configurable

DEFAULT_PROMPTS = os.environ.get(
    "GRPO_PROMPTS",
    os.path.join(os.environ.get("GRPO_SHARED_DIR", "./shared"), "data", "grpo_prompts.jsonl"),
)


@dataclass(frozen=True, kw_only=True, slots=True)
class TelecomSample:
    id: str
    source: str
    prompt: tuple[dict, ...]  # chat messages; tuple so the sample is hashable
    question: str
    reference: str


def question_from_messages(messages) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if m.get("role") in ("user", "system"):
            parts.append(str(c))
    return "\n".join(parts)


def parse_row(rec: dict, index: int, path: str) -> TelecomSample:
    messages = rec.get("messages")
    gt = rec.get("ground_truth")
    gt = json.loads(gt) if isinstance(gt, str) else dict(gt or {})
    reference = str(gt.get("reference", "") or "")
    if not messages or not reference:
        raise ValueError(f"{path}: row {index}: empty messages or reference")
    return TelecomSample(
        id=str(gt.get("id") or rec.get("id") or index),
        source=str(gt.get("source") or rec.get("source") or "unknown"),
        prompt=tuple({"role": m["role"], "content": m["content"]} for m in messages),
        question=str(gt.get("question") or question_from_messages(messages)),
        reference=reference,
    )


def load_samples(path: str, offset: int = 0, limit: int | None = None) -> list[TelecomSample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if line:
                samples.append(parse_row(json.loads(line), index, path))
    samples = samples[offset:]
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"no prompts read from {path}")
    return samples


class TelecomDataset(Configurable):

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str = DEFAULT_PROMPTS
        seed: int = 42
        shuffle: bool = True
        limit: int | None = None
        offset: int = 0

    def __init__(self, config: Config) -> None:
        self._samples = load_samples(config.path, config.offset, config.limit)
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator[TelecomSample]:
        return self

    def __next__(self) -> TelecomSample:
        if self._position == len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        s = self._samples[self._order[self._position]]
        self._position += 1
        return s

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]
