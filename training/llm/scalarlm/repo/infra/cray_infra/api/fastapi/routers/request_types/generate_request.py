from pydantic import BaseModel

from typing import Any, Optional, Union


class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompts: list[Union[str, dict[str, Any]]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.0
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None