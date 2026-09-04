"""
FastAPI handler for /v1/chat/completions (non-streaming, queue-backed).

Stream=True requests are delegated to the existing direct-to-vLLM
proxy (see openai_v1_router.py); only stream=False flows through the
queue. See docs/openai-chat-completions-queue.md §3.1.

The handler is a thin orchestrator over four foundation pieces:

  1. vLLM /tokenize — performs the authoritative runtime render for admission
  2. is_over_high_water + WaitEstimator — admission decision and 429
     Retry-After hint
  3. ResultRouter — registers a correlation_id before submission so
     the worker has somewhere to deliver the result
  4. Coalescer — accumulates admitted requests; the configured flush
     callback (`enqueue_coalesced_batch`) writes one batch as one
     SQLite row.

The HTTP response is the heartbeat-padded chunked-JSON streamer from
`heartbeat.py`; the body generator wraps it in a try/finally that
unregisters the correlation id on completion or client disconnect.
"""

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
    """
    Non-streaming chat completions handler. The router (see
    openai_v1_router.py) is responsible for branching on
    `request.stream`; this function never sees a streaming request.

    `request` is an OpenAI-compatible ChatCompletionRequest
    (vllm.entrypoints.openai.protocol or equivalent). The shape is
    duck-typed: we read `model`, `messages`, `max_tokens`, and
    `temperature`.
    """
    # Admission check first: rejecting an overload request before
    # rendering the chat template avoids paying the (possibly first-
    # use, multi-second) tokenizer load on requests we're going to
    # 429 anyway.
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

    # Resolve the model name before touching the tokenizer: an
    # unspecified or "latest" model would otherwise crash inside
    # `AutoTokenizer.from_pretrained(None)` with an opaque HuggingFace
    # 401. Mirrors `/v1/generate`'s resolution (generate.py:50-63).
    model = _resolve_model(getattr(request, "model", None), config)

    chat_request = _chat_params_from_request(request)

    # vLLM gives the newer max_completion_tokens field precedence over the
    # deprecated max_tokens field. Resolve that same effective budget before
    # admission and preserve the original field on the queued request.
    requested_max_completion_tokens = chat_request.get("max_completion_tokens")
    requested_max_tokens = chat_request.get("max_tokens")
    if requested_max_completion_tokens is not None:
        effective_max_tokens = int(requested_max_completion_tokens)
    elif requested_max_tokens is not None:
        effective_max_tokens = int(requested_max_tokens)
    else:
        effective_max_tokens = int(config.get("default_max_output_tokens", 128))
        chat_request["max_tokens"] = effective_max_tokens

    # Resolve these fields before either renderer sees the request.
    chat_request.update({"model": model, "stream": False})

    # Ask vLLM to render and tokenize with its actual runtime renderer. The
    # API pod cannot reproduce a --chat-template supplied only through the
    # vLLM pod's SCALARLM_VLLM_ARGS, and specialized renderers may not match
    # transformers.apply_chat_template either. /tokenize uses the configured
    # template and OnlineRenderer in both vLLM 0.26 and 0.27.
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

    # Keep a string prompt for origin/main workers during an API-first rolling
    # upgrade. This is deliberately computed *after* authoritative runtime
    # tokenization: the API tokenizer may lack a template supplied only to the
    # vLLM pod via --chat-template. Such a local mismatch must not reject an
    # otherwise valid request. Current workers ignore this compatibility field
    # and rehydrate ``chat_request`` through vLLM's chat serving path.
    rendered_prompt = _render_legacy_worker_prompt(
        model=model,
        messages=request.messages,
        chat_request=chat_request,
    )

    # Preserve the validated, JSON-safe chat request across the queue
    # boundary so the worker can invoke vLLM's chat-completions serving path.
    # The pre-rendered prompt remains useful for admission and inspection,
    # but feeding it to /v1/completions bypasses vLLM's reasoning and tool
    # parsers and collapses their structured output into visible text.
    correlation_id = str(uuid4())
    future = router.register(correlation_id)
    get_metrics().record_chat_admitted(correlation_id)

    backend_request = {
        "prompt": rendered_prompt,
        "model": model,
        "max_tokens": effective_max_tokens,
        "temperature": getattr(request, "temperature", None),
        "chat_request": chat_request,
        # Keep the origin/main discriminator so a rolling upgrade does not
        # feed an old worker an unknown request type. New workers identify the
        # chat path from the presence of ``chat_request``.
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
    """Best-effort string prompt for an old worker during rolling upgrades.

    The structured ``chat_request`` is authoritative. If the API pod cannot
    reproduce vLLM's server-only template, fall back to a plain text transcript
    instead of preempting a request that the runtime renderer already accepted.
    Non-text multimodal parts become short placeholders so data URLs are not
    duplicated into the compatibility prompt.
    """
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
    """
    Apply the same model-resolution rules as /v1/generate:

      - None / empty → config["model"] (the deployment's default)
      - "latest" → the most recent training job
      - anything else → looked up in the vLLM model manager

    Raises HTTPException(404) for an explicitly-named model that the
    manager doesn't recognise; that's the same shape as the existing
    /v1/generate failure mode, so SDK error handlers stay uniform.
    """
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
    """Encoder closure for stream_with_heartbeat. The worker hands us
    its flat result dict (request_id, response, error, token_count);
    we rewrap into the OpenAI ChatCompletion shape so the SDK can
    parse it without surprises."""
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
    """
    Wrap the heartbeat stream so the cid is always unregistered on
    completion or generator close (the client-disconnect path).
    """
    try:
        async for chunk in stream_with_heartbeat(
            future, encode_body=_encode_chat_completion(model)
        ):
            yield chunk
    finally:
        router.unregister(correlation_id)
        # If the worker resolved already, this is a no-op (the cid's
        # start_time was popped during record_chat_resolved). If the
        # client disconnected before the worker resolved, this is the
        # only place that decrements chat_in_flight.
        get_metrics().record_chat_unregistered(correlation_id)


# ---------------------------------------------------------------------------
# Singletons / wiring. Tests patch these accessors; production wires once.
# ---------------------------------------------------------------------------


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
    """
    Current in-flight inference work (queue + processing). Read from
    the existing Metrics singleton, which already tracks this counter
    for /v1/generate.
    """
    from cray_infra.generate.metrics import get_metrics

    return get_metrics().queue_depth
