"""Structured chat fields must survive both queue API boundaries."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cray_infra.api.fastapi.generate.finish_work import finish_work
from cray_infra.api.fastapi.generate.get_work import get_work
from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequest,
    FinishWorkRequests,
)
from cray_infra.api.fastapi.routers.request_types.get_adaptors_response import (
    GetAdaptorsResponse,
)
from cray_infra.api.fastapi.routers.request_types.get_work_request import (
    GetWorkRequest,
)


@pytest.mark.asyncio
async def test_get_work_preserves_original_chat_request():
    chat_request = {
        "model": "model-1",
        "messages": [{"role": "user", "content": "hi"}],
        "include_reasoning": False,
        "stream": False,
    }
    queued = {
        "prompt": "rendered prompt",
        "model": "model-1",
        "request_type": "chat_completions",
        "max_tokens": 64,
        "chat_request": chat_request,
    }

    with patch(
        "cray_infra.api.fastapi.generate.get_work.get_inference_work_queue",
        new=AsyncMock(return_value=MagicMock()),
    ), patch(
        "cray_infra.api.fastapi.generate.get_work.get_work_item",
        new=AsyncMock(return_value=(queued, "req-1")),
    ), patch(
        "cray_infra.api.fastapi.generate.get_work.worker_ready", new=AsyncMock()
    ), patch(
        "cray_infra.api.fastapi.generate.get_work.worker_not_ready", new=AsyncMock()
    ), patch(
        "cray_infra.api.fastapi.generate.get_work.get_adaptors",
        new=AsyncMock(return_value=GetAdaptorsResponse(new_adaptors=[])),
    ):
        response = await get_work(GetWorkRequest(batch_size=1, loaded_adaptor_count=0))

    assert response.requests[0].request_id == "req-1"
    assert response.requests[0].chat_request == chat_request


@pytest.mark.asyncio
async def test_finish_work_preserves_structured_chat_response():
    unfinished = {"is_acked": False}
    update = AsyncMock()
    metrics = MagicMock()
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]

    with patch(
        "cray_infra.api.fastapi.generate.finish_work.get_inference_work_queue",
        new=AsyncMock(return_value=MagicMock()),
    ), patch(
        "cray_infra.api.fastapi.generate.finish_work.get_unfinished_result",
        new=AsyncMock(return_value=unfinished),
    ), patch(
        "cray_infra.api.fastapi.generate.finish_work.update_and_ack", new=update
    ), patch(
        "cray_infra.api.fastapi.generate.finish_work.get_metrics",
        return_value=metrics,
    ):
        await finish_work(
            FinishWorkRequests(
                requests=[
                    FinishWorkRequest(
                        request_id="req-1",
                        response="answer",
                        reasoning="private analysis",
                        tool_calls=tool_calls,
                        finish_reason="tool_calls",
                        prompt_tokens=12,
                        completion_tokens=7,
                        token_count=19,
                    )
                ]
            )
        )

    item = update.await_args.kwargs["item"]
    assert item["response"] == "answer"
    assert item["reasoning"] == "private analysis"
    assert item["tool_calls"] == tool_calls
    assert item["finish_reason"] == "tool_calls"
    assert item["prompt_tokens"] == 12
    assert item["completion_tokens"] == 7
    assert item["token_count"] == 19
