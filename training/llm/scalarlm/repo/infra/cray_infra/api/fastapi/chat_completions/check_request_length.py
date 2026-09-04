"""
Pre-admission length check for the chat completions queue.

vLLM doesn't reject prompts that exceed the per-request KV budget —
the scheduler queues them, can never find a window where they fit,
and the request stalls forever. The /inference browser shows them
as `in_progress` indefinitely; clients see heartbeat whitespace
followed by a timeout. This helper rejects them up front so the
client gets HTTP 400 with a clear reason instead.

The check is a pure threshold computation; the handler is responsible
for getting both `prompt_tokens` and `max_model_length` from vLLM's
authoritative runtime renderer. Splitting concerns this way keeps the
threshold logic trivially unit-testable without a real tokenizer.
"""

from typing import Optional


class RequestTooLongError(Exception):
    """
    Raised by `check_request_length` when prompt + max_tokens exceeds
    `max_model_length`. Carries the structured fields the handler
    uses to compose its 400 detail string — operators can read it
    out of the inference browser's status row directly.
    """

    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        max_model_length: int,
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_model_length = max_model_length
        self.total = prompt_tokens + max_tokens
        super().__init__(self._message())

    def _message(self) -> str:
        return (
            f"Request too long: prompt_tokens={self.prompt_tokens} + "
            f"max_tokens={self.max_tokens} = {self.total} > "
            f"max_model_length={self.max_model_length}. "
            f"Reduce the prompt or max_tokens."
        )


def check_request_length(
    *,
    prompt_tokens: int,
    max_tokens: Optional[int],
    max_model_length: int,
) -> None:
    """
    Raise `RequestTooLongError` if the request is bigger than the
    model can fit. `max_tokens=None` is treated as 0 — the OpenAI
    SDK lets clients omit it, in which case vLLM defaults internally
    and the prompt-side bound is what we actually care about
    blocking on.

    Non-positive `max_model_length` short-circuits to a no-op: the
    config knob hasn't been set, and silently passing every request
    through is safer than rejecting all of them.
    """
    if max_model_length <= 0:
        return
    requested_max = max_tokens or 0
    if prompt_tokens + requested_max > max_model_length:
        raise RequestTooLongError(
            prompt_tokens=prompt_tokens,
            max_tokens=requested_max,
            max_model_length=max_model_length,
        )
