from pydantic import BaseModel
from cray_infra.api.fastapi.routers.request_types.get_adaptors_response import (
    GetAdaptorsResponse,
)

from typing import Any, Optional, Union

PromptType = Union[str, dict[str, Union[str, list[str]]]]


class GetWorkResponse(BaseModel):
    prompt: PromptType
    request_id: str
    request_type: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    # Present for queue-backed /v1/chat/completions requests. ``prompt``
    # remains available for admission/accounting and request inspection.
    chat_request: Optional[dict[str, Any]] = None


class GetWorkResponses(BaseModel):
    requests: list[GetWorkResponse]
    new_adaptors: GetAdaptorsResponse
