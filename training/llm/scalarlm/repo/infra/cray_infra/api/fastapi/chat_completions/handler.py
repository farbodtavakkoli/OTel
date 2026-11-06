import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.api.fastapi.chat_completions.admission import (
    WaitEstimator,
    is_over_high_water,
)
from cray_infra.api.fastapi.chat_completions.build_chat_completion_response import (
    build_chat_completion_response,
)
from cray_infra.api.fastapi.chat_completions.check_request_length import (
    RequestTooLongError,
    check_request_length,
)
from cray_infra.api.fastapi.chat_completions.coalescer import Coalescer
from cray_infra.api.fastapi.chat_completions.heartbeat import (
    stream_with_heartbeat,
)
from cray_infra.api.fastapi.chat_completions.render_chat_template import (
    render_chat_template,
)
from cray_infra.api.fastapi.chat_completions.tokenize_chat_for_admission import (
    VLLMTokenizeRequestError,
    tokenize_chat_for_admission,
)
from cray_infra.api.fastapi.chat_completions.result_router import (
    ResultRouter,
    get_result_router,
)
from cray_infra.api.fastapi.routers.openai_v1_helpers import (
    _chat_params_from_request,
)
from cray_infra.generate.metrics import get_metrics
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


async def chat_completions_via_queue(request: Any) -> StreamingResponse:
    config = get_config()
    queue_depth = get_queue_depth()
    router = get_result_router()

    if is_over_high_water(
        queue_depth=queue_depth,
        in_flight_count=router.in_flight_count,
        max_num_seqs=config["max_num_seqs"],
        admit_factor=config["chat_admit_factor"],
    ):
        wait = get_wait_estimator().estimate_wait_seconds(
            queue_depth=queue_depth,
            max_num_seqs=config["max_num_seqs"],
        )
        get_metrics().record_chat_rejected_429()
        raise HTTPException(
            status_code=429,
            detail="Server is over capacity; retry after the indicated interval.",
            headers={"Retry-After": str(max(1, int(wait)))},
        )

    model = _resolve_model(getattr(request, "model", None), config)

    chat_request = _chat_params_from_request(request)

    requested_max_completion_tokens = chat_request.get("max_completion_tokens")
    requested_max_tokens = chat_request.get("max_tokens")
    if requested_max_completion_tokens is not None:
        effective_max_tokens = int(requested_max_completion_tokens)
    elif requested_max_tokens is not None:
        effective_max_tokens = int(requested_max_tokens)
    else:
        effective_max_tokens = int(config.get("default_max_output_tokens", 128))
        chat_request["max_tokens"] = effective_max_tokens

    chat_request.update({"model": model, "stream": False})

    try:
        tokenization = await tokenize_chat_for_admission(
            model=model,
            chat_request=chat_request,
        )
    except VLLMTokenizeRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if tokenization is not None:
        try:
            check_request_length(
                prompt_tokens=tokenization.prompt_tokens,
                max_tokens=effective_max_tokens,
                max_model_length=tokenization.max_model_length,
            )
        except RequestTooLongError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    rendered_prompt = _render_legacy_worker_prompt(
        model=model,
        messages=request.messages,
        chat_request=chat_request,
    )

    correlation_id = str(uuid4())
    future = router.register(correlation_id)
    get_metrics().record_chat_admitted(correlation_id)

    backend_request = {
        "prompt": rendered_prompt,
        "model": model,
        "max_tokens": effective_max_tokens,
        "temperature": getattr(request, "temperature", None),
        "chat_request": chat_request,
        "request_type": "generate",
        "correlation_id": correlation_id,
    }

    await get_coalescer().submit(backend_request, correlation_id)

    return StreamingResponse(
        _stream_and_unregister(future, correlation_id, router, model),
        media_type="application/json",
    )


def _render_legacy_worker_prompt(
    *, model: str, messages: list[Any], chat_request: dict[str, Any]
) -> str:
    try:
        return render_chat_template(
            model=model,
            messages=messages,
            prompt=None,
            chat_template_kwargs=chat_request.get("chat_template_kwargs"),
            tools=chat_request.get("tools"),
            reasoning_effort=chat_request.get("reasoning_effort"),
        )
    except Exception as exc:
        logger.warning(
            "Local compatibility prompt render failed for model %s (%s); "
            "using a plain transcript for legacy workers",
            model,
            type(exc).__name__,
        )
        return _plain_chat_transcript(chat_request.get("messages", []))


def _plain_chat_transcript(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            lines.append(f"unknown: {message}")
            continue
        role = message.get("role") or "unknown"
        lines.append(f"{role}: {_plain_message_content(message.get('content'))}")
    lines.append("assistant:")
    return "\n".join(lines)


def _plain_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append("[content]")
            continue
        part_type = part.get("type")
        text = part.get("text")
        if part_type in {"text", "input_text"} and isinstance(text, str):
            parts.append(text)
        else:
            parts.append(f"[{part_type or 'content'}]")
    return " ".join(parts)


def _resolve_model(requested: Any, config: dict) -> str:
    if not requested:
        resolved = config["model"]
        logger.info("Using default model: %s", resolved)
    elif requested == "latest":
        from cray_infra.training.get_latest_model import get_latest_model

        resolved = get_latest_model()
    else:
        resolved = requested

    from cray_infra.training.vllm_model_manager import get_vllm_model_manager

    found = get_vllm_model_manager().find_model(resolved)
    if found is None:
        logger.error("Model %s not found", resolved)
        raise HTTPException(status_code=404, detail=f"Model {resolved} not found")
    return found


def _encode_chat_completion(model: str):
    import json

    def encode(result):
        wrapped = build_chat_completion_response(result=result, model=model)
        return json.dumps(wrapped).encode("utf-8")

    return encode


async def _stream_and_unregister(
    future,
    correlation_id: str,
    router: ResultRouter,
    model: str,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in stream_with_heartbeat(
            future, encode_body=_encode_chat_completion(model)
        ):
            yield chunk
    finally:
        router.unregister(correlation_id)
        get_metrics().record_chat_unregistered(correlation_id)


_coalescer: Coalescer | None = None
_wait_estimator: WaitEstimator | None = None


def get_coalescer() -> Coalescer:
    global _coalescer
    if _coalescer is None:
        from cray_infra.api.fastapi.chat_completions.enqueue_coalesced_batch import (
            enqueue_coalesced_batch,
        )

        config = get_config()
        _coalescer = Coalescer(
            packing_factor=config.get("chat_coalescer_packing_factor", 10),
            window_seconds=config.get("chat_coalescer_window_ms", 50) / 1000.0,
            bypass_threshold=config.get("chat_coalescer_bypass_threshold", 10),
            flush_callback=enqueue_coalesced_batch,
            queue_depth_provider=get_queue_depth,
        )
    return _coalescer


def get_wait_estimator() -> WaitEstimator:
    global _wait_estimator
    if _wait_estimator is None:
        config = get_config()
        _wait_estimator = WaitEstimator(
            default_batch_latency_seconds=config.get(
                "chat_wait_estimator_default_seconds", 5.0
            ),
            padding=config.get("chat_wait_estimator_padding", 1.5),
            sample_size=config.get("chat_wait_estimator_sample_size", 32),
        )
    return _wait_estimator


def get_queue_depth() -> int:
    from cray_infra.generate.metrics import get_metrics

    return get_metrics().queue_depth
