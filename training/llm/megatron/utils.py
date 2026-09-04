"""Stdlib helpers for the Megatron preprocessing wrapper -- see readme_megatron.md."""

import json
import tempfile
from pathlib import Path


def flatten_chat_record(record) -> str:
    """Concatenate every `messages[].content` into one document string."""
    if record.get("text"):
        return record["text"]
    messages = record.get("messages") or []
    parts = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else ""
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)


def _maybe_flatten_chat_jsonl(path: Path):
    """Write a temp {text} JSONL from a chat JSONL; return its path, or None if already text."""
    first = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                first = json.loads(line)
                break
    if first is None:
        return None
    if first.get("text") or not first.get("messages"):
        return None

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", prefix="megatron_text_", delete=False, encoding="utf-8"
    )
    with path.open("r", encoding="utf-8") as fh, tmp:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            text = flatten_chat_record(record)
            tmp.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return tmp.name
