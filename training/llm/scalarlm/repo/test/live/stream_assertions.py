"""Assertions shared by the live completion-stream smoke test."""


def assert_valid_completion_stream(events: list[dict]) -> None:
    choices = [
        choice
        for event in events
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
    ]
    completion_text = "".join(
        choice["text"] for choice in choices if isinstance(choice.get("text"), str)
    )
    finish_reasons = [
        choice["finish_reason"]
        for choice in choices
        if isinstance(choice.get("finish_reason"), str) and choice["finish_reason"]
    ]

    assert completion_text.strip(), events
    assert finish_reasons, events
