from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_template_kwargs_for_render,
)
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

_TOKENIZE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class AdmissionTokenization:
    prompt_tokens: int
    max_model_length: int


class VLLMTokenizeRequestError(Exception):
    def __init__(self, *, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def tokenize_chat_for_admission(
    *, model: str, chat_request: dict[str, Any]
) -> AdmissionTokenization | None:
    payload = _build_tokenize_payload(model=model, chat_request=chat_request)
    url = get_config()["vllm_api_url"] + "/tokenize"

    try:
        session = get_global_session()
        async with asyncio.timeout(_TOKENIZE_TIMEOUT_SECONDS):
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    body = await response.json()
                    parsed = _parse_tokenize_response(body)
                    if parsed is None:
                        logger.warning(
                            "vLLM /tokenize returned an invalid success payload; "
                            "skipping pre-admission length check"
                        )
                    return parsed

                detail = await _response_error_detail(response)
                if 400 <= response.status < 500:
                    raise VLLMTokenizeRequestError(
                        status_code=response.status,
                        detail=detail,
                    )

                logger.warning(
                    "vLLM /tokenize returned %s; skipping pre-admission length check",
                    response.status,
                )
                return None
    except VLLMTokenizeRequestError:
        raise
    except Exception as exc:
        logger.warning(
            "HTTP error querying vLLM /tokenize at %s: %s; skipping "
            "pre-admission length check",
            url,
            exc,
        )
        return None


def _build_tokenize_payload(
    *, model: str, chat_request: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": chat_request["messages"],
        "add_generation_prompt": True,
        "continue_final_message": False,
        "add_special_tokens": False,
    }

    tools = chat_request.get("tools")
    if tools is not None:
        payload["tools"] = tools

    template_kwargs = _chat_template_kwargs_for_render(
        chat_request.get("chat_template_kwargs"),
        chat_request.get("reasoning_effort"),
    )
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs

    return payload


def _parse_tokenize_response(value: Any) -> AdmissionTokenization | None:
    if not isinstance(value, dict):
        return None
    count = value.get("count")
    max_model_len = value.get("max_model_len")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len <= 0
    ):
        return None
    return AdmissionTokenization(
        prompt_tokens=count,
        max_model_length=max_model_len,
    )


async def _response_error_detail(response: Any) -> str:
    try:
        body = await response.json()
    except Exception:
        try:
            text = await response.text()
        except Exception:
            text = ""
        return text or f"vLLM /tokenize returned HTTP {response.status}"

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    return str(body)
