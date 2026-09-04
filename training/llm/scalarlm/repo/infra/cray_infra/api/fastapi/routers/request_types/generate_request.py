from pydantic import BaseModel

from typing import Any, Optional, Union


class GenerateRequest(BaseModel):
    model: Optional[str] = None
    # A batch of independent inference requests. Each entry is one of:
    #   - a bare string                              (legacy raw prompt)
    #   - {"prompt": "..."}                          (raw, dict form)
    #   - {"messages": [{"role": ..., "content": ...}, ...]}
    #                                                (chat turns;
    #                                                 rendered via
    #                                                 the model's
    #                                                 chat template
    #                                                 at enqueue time
    #                                                 — see docs/
    #                                                 openai-chat-
    #                                                 completions-
    #                                                 queue.md §10)
    # The dict variant carries the per-entry shape; render_generate_entry
    # is the single point that validates and renders.
    prompts: list[Union[str, dict[str, Any]]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.0
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None