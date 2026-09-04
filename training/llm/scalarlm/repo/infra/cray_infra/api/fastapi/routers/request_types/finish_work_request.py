from pydantic import BaseModel

from typing import Any, Optional, Union


class FinishWorkRequest(BaseModel):
    request_id: str
    response: Optional[Union[str, list[float]]] = None
    error: Optional[str] = None
    token_count: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    flop_count: Optional[int] = None
    # Structured chat-completion fields returned by vLLM's reasoning/tool
    # parsers. Optional so legacy generate workers remain wire-compatible.
    reasoning: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    finish_reason: Optional[str] = None


class FinishWorkRequests(BaseModel):
    requests: list[FinishWorkRequest]
