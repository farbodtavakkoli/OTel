"""
Unit tests for chat_completions_via_queue.

The handler ties together the four foundation pieces (renderer,
admission, router, coalescer) and returns a StreamingResponse backed
by the heartbeat helper. Tests mock the singletons so the flow can be
exercised without uvicorn / SQLite / a real model.

Contract (see docs/openai-chat-completions-queue.md §5):
- Uses vLLM's runtime /tokenize renderer for admission before constructing the
  old-worker compatibility prompt.
- 429 with Retry-After when admission threshold is exceeded.
- Registers a correlation_id with the result router *before*
  submitting to the coalescer so the worker can never resolve a cid
  the router doesn't yet know about.
- Returns a StreamingResponse with media_type=application/json.
- The streaming generator unregisters the cid on completion or
  cancellation (verified by the disconnect test).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from cray_infra.api.fastapi.chat_completions import handler as h
from cray_infra.api.fastapi.chat_completions.admission import WaitEstimator
from cray_infra.api.fastapi.chat_completions.result_router import ResultRouter
from cray_infra.api.fastapi.chat_completions.tokenize_chat_for_admission import (
    AdmissionTokenization,
    VLLMTokenizeRequestError,
)


_UNSET = object()


def _request(messages=None, prompt_text=None, stream=False, **overrides):
    """Build a minimal ChatCompletionRequest-shaped MagicMock."""
    if messages is None and prompt_text is None:
        messages = [{"role": "user", "content": "hi"}]

    req = MagicMock()
    req.model = overrides.get("model", "test-model")
    req.messages = messages
    req.max_tokens = overrides.get("max_tokens", 64)
    req.max_completion_tokens = overrides.get("max_completion_tokens")
    req.temperature = overrides.get("temperature", 0.7)
    req.stream = stream
    req.tools = overrides.get("tools")
    tool_choice = overrides.get("tool_choice", _UNSET)
    req.tool_choice = None if tool_choice is _UNSET else tool_choice
    req.chat_template_kwargs = overrides.get("chat_template_kwargs")
    req.include_reasoning = overrides.get("include_reasoning")
    req.reasoning_effort = overrides.get("reasoning_effort")
    req.thinking_token_budget = overrides.get("thinking_token_budget")
    req.top_k = overrides.get("top_k")
    req.parallel_tool_calls = overrides.get("parallel_tool_calls")

    raw = {
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "max_completion_tokens": req.max_completion_tokens,
        "temperature": req.temperature,
        "stream": stream,
    }
    for key in (
        "tools",
        "chat_template_kwargs",
        "include_reasoning",
        "reasoning_effort",
        "thinking_token_budget",
        "top_k",
        "parallel_tool_calls",
    ):
        value = getattr(req, key)
        if value is not None:
            raw[key] = value
    if tool_choice is not _UNSET:
        raw["tool_choice"] = tool_choice
    req.model_fields_set = set(raw)
    req.model_dump.return_value = raw
    return req


@pytest.fixture
def fresh_router():
    return ResultRouter()


@pytest.fixture
def fresh_estimator():
    return WaitEstimator(default_batch_latency_seconds=2.0, padding=1.5)


@pytest.fixture
def patched_components(fresh_router, fresh_estimator):
    """
    Patch the four singleton accessors the handler reaches through.
    Returns the mock coalescer and a "queue depth" you can mutate
    per-test.
    """
    coalescer = MagicMock()
    coalescer.submit = AsyncMock()

    queue_depth_holder = {"value": 0}
    fake_config = {
        "max_num_seqs": 256,
        "chat_admit_factor": 4,
    }

    with (
        patch.object(h, "get_result_router", return_value=fresh_router),
        patch.object(h, "get_coalescer", return_value=coalescer),
        patch.object(h, "get_wait_estimator", return_value=fresh_estimator),
        patch.object(
            h, "get_queue_depth", side_effect=lambda: queue_depth_holder["value"]
        ),
        patch.object(h, "get_config", return_value=fake_config),
        patch.object(
            h, "_resolve_model", side_effect=lambda req, cfg: req or "test-model"
        ),
        patch.object(
            h, "tokenize_chat_for_admission", AsyncMock(return_value=None)
        ) as tokenize,
        patch.object(h, "render_chat_template", return_value="rendered-prompt"),
    ):
        yield {
            "coalescer": coalescer,
            "queue_depth": queue_depth_holder,
            "router": fresh_router,
            "estimator": fresh_estimator,
            "config": fake_config,
            "tokenize_chat": tokenize,
        }


@pytest.mark.asyncio
async def test_returns_streaming_response_under_threshold(patched_components):
    response = await h.chat_completions_via_queue(_request())
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "application/json"


@pytest.mark.asyncio
async def test_renders_messages_through_chat_template(patched_components):
    with patch.object(
        h, "render_chat_template", return_value="USER: hi\nASSISTANT: "
    ) as render:
        await h.chat_completions_via_queue(
            _request(messages=[{"role": "user", "content": "hi"}])
        )
    render.assert_called_once()
    kwargs = render.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["prompt"] is None
    assert kwargs["tools"] is None
    assert kwargs["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_safe_template_kwargs_are_used_for_accounting_render(
    patched_components,
):
    with patch.object(h, "render_chat_template", return_value="rendered") as render:
        tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]
        await h.chat_completions_via_queue(
            _request(
                tools=tools,
                reasoning_effort="high",
                chat_template_kwargs={
                    "enable_thinking": False,
                    "reasoning_strength": "low",
                    "chat_template": "untrusted override",
                },
            )
        )

    assert render.call_args.kwargs["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_strength": "low",
    }
    assert render.call_args.kwargs["tools"] == tools
    assert render.call_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_registers_correlation_id_before_submitting(patched_components):
    """
    Submission must happen *after* router.register so the worker can
    never resolve a cid that doesn't exist yet.
    """
    submit_order: list = []

    coalescer = patched_components["coalescer"]

    async def record_submit(req, cid):
        submit_order.append(
            ("submit", cid, patched_components["router"].in_flight_count)
        )

    coalescer.submit = AsyncMock(side_effect=record_submit)

    with patch.object(h, "render_chat_template", return_value="rendered"):
        await h.chat_completions_via_queue(_request())

    assert len(submit_order) == 1
    _, _, in_flight_at_submit = submit_order[0]
    # The router was incremented before the submit ran.
    assert in_flight_at_submit == 1


@pytest.mark.asyncio
async def test_correlation_id_passed_to_coalescer_matches_request_payload(
    patched_components,
):
    """The correlation_id is in the request dict AND the coalescer sees it as the second arg."""
    coalescer = patched_components["coalescer"]

    captured: list = []

    async def capture(req, cid):
        captured.append((req, cid))

    coalescer.submit = AsyncMock(side_effect=capture)

    with patch.object(h, "render_chat_template", return_value="rendered"):
        await h.chat_completions_via_queue(_request())

    assert len(captured) == 1
    req, cid = captured[0]
    assert req["correlation_id"] == cid
    assert req["prompt"] == "rendered"
    assert req["request_type"] == "generate"
    assert req["chat_request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert req["chat_request"]["stream"] is False


@pytest.mark.asyncio
async def test_queue_payload_is_compatible_with_origin_main_worker(
    patched_components,
):
    """Old workers accept only ``generate`` and ignore the new chat payload.

    Keeping a rendered string prompt lets one finish in the legacy completion
    path during an API-first rolling upgrade instead of failing immediately on
    an unknown discriminator. Current workers detect ``chat_request`` first.
    """
    captured = []

    async def capture(request, _correlation_id):
        captured.append(request)

    patched_components["coalescer"].submit = AsyncMock(side_effect=capture)
    await h.chat_completions_via_queue(_request())

    queued = captured[0]
    assert queued["request_type"] == "generate"
    assert isinstance(queued["prompt"], str)
    assert queued["chat_request"]["messages"]


@pytest.mark.asyncio
async def test_vllm_chat_controls_survive_queue_submission(patched_components):
    """Reasoning controls, sampling, and tool definitions reach the worker."""
    coalescer = patched_components["coalescer"]
    captured = []

    async def capture(req, cid):
        captured.append(req)

    coalescer.submit = AsyncMock(side_effect=capture)
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]

    await h.chat_completions_via_queue(
        _request(
            include_reasoning=False,
            reasoning_effort="low",
            thinking_token_budget=512,
            top_k=64,
            parallel_tool_calls=False,
            tools=tools,
            tool_choice="auto",
            chat_template_kwargs={
                "reasoning_strength": "low",
                "untrusted": "drop me",
            },
        )
    )

    chat_request = captured[0]["chat_request"]
    assert chat_request["include_reasoning"] is False
    assert chat_request["reasoning_effort"] == "low"
    assert chat_request["thinking_token_budget"] == 512
    assert chat_request["top_k"] == 64
    assert chat_request["parallel_tool_calls"] is False
    assert chat_request["tools"] == tools
    assert chat_request["tool_choice"] == "auto"
    assert chat_request["chat_template_kwargs"] == {"reasoning_strength": "low"}


# ---------------------------------------------------------------------------
# max_tokens default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_defaults_when_client_omits(patched_components):
    """
    The OpenAI SDK lets clients omit max_tokens. vLLM's
    `request_output_to_completion_response` has a bare
    `assert request.max_tokens is not None` — passing None all the
    way through generates the response, then crashes building it
    with an empty `AssertionError`. The handler must plug in a real
    number from `default_max_output_tokens` so the worker never sees
    None.
    """
    patched_components["config"]["default_max_output_tokens"] = 256
    coalescer = patched_components["coalescer"]
    captured = []

    async def capture(req, cid):
        captured.append(req)

    coalescer.submit = AsyncMock(side_effect=capture)

    await h.chat_completions_via_queue(_request(max_tokens=None))

    assert captured[0]["max_tokens"] == 256
    assert captured[0]["chat_request"]["max_tokens"] == 256


@pytest.mark.asyncio
async def test_max_tokens_passes_through_when_client_provides(
    patched_components,
):
    """A client-supplied max_tokens wins over the default."""
    patched_components["config"]["default_max_output_tokens"] = 256
    coalescer = patched_components["coalescer"]
    captured = []

    async def capture(req, cid):
        captured.append(req)

    coalescer.submit = AsyncMock(side_effect=capture)

    req = _request(max_tokens=42)
    await h.chat_completions_via_queue(req)

    assert captured[0]["max_tokens"] == 42
    assert captured[0]["chat_request"]["max_tokens"] == 42


@pytest.mark.asyncio
async def test_max_completion_tokens_takes_precedence_without_overwrite(
    patched_components,
):
    """The replacement field wins over max_tokens and reaches vLLM intact."""
    coalescer = patched_components["coalescer"]
    captured = []

    async def capture(req, cid):
        captured.append(req)

    coalescer.submit = AsyncMock(side_effect=capture)

    await h.chat_completions_via_queue(
        _request(max_tokens=42, max_completion_tokens=4096)
    )

    queued = captured[0]
    assert queued["max_tokens"] == 4096
    assert queued["chat_request"]["max_completion_tokens"] == 4096
    assert queued["chat_request"]["max_tokens"] == 42


@pytest.mark.asyncio
async def test_max_tokens_default_falls_back_to_128_when_config_missing(
    patched_components,
):
    """If the operator removed `default_max_output_tokens` from the
    cray-config.yaml, fall back to the in-code default of 128 — the
    queue must not pass None even on misconfigured pods."""
    # Fixture's config has no default_max_output_tokens key.
    coalescer = patched_components["coalescer"]
    captured = []

    async def capture(req, cid):
        captured.append(req)

    coalescer.submit = AsyncMock(side_effect=capture)

    await h.chat_completions_via_queue(_request(max_tokens=None))

    assert captured[0]["max_tokens"] == 128


@pytest.mark.asyncio
async def test_429_when_over_high_water(patched_components):
    """queue_depth > 4 × max_num_seqs (1024) trips the threshold."""
    patched_components["queue_depth"]["value"] = 1025

    with pytest.raises(HTTPException) as exc_info:
        await h.chat_completions_via_queue(_request())

    assert exc_info.value.status_code == 429
    retry_after = exc_info.value.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) >= 1


@pytest.mark.asyncio
async def test_429_does_not_register_correlation_id(patched_components):
    """A rejected request must leave no leak in the router."""
    patched_components["queue_depth"]["value"] = 9999

    with pytest.raises(HTTPException):
        await h.chat_completions_via_queue(_request())

    assert patched_components["router"].in_flight_count == 0


# ---------------------------------------------------------------------------
# Pre-admission length check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_400_when_prompt_plus_max_tokens_exceeds_max_model_length(
    patched_components,
):
    """
    A request whose prompt + max_tokens > max_model_length must be
    rejected up front with HTTP 400. Without this check vLLM queues
    it forever — the production stuck-request symptom. Both values come
    from vLLM's runtime /tokenize renderer, not the cray-config knob.
    """
    patched_components["tokenize_chat"].return_value = AdmissionTokenization(
        prompt_tokens=80, max_model_length=100
    )

    with pytest.raises(HTTPException) as exc_info:
        await h.chat_completions_via_queue(_request(max_tokens=50))

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "80" in detail and "50" in detail and "100" in detail


@pytest.mark.asyncio
async def test_length_check_uses_max_completion_tokens_precedence(
    patched_components,
):
    patched_components["tokenize_chat"].return_value = AdmissionTokenization(
        prompt_tokens=80, max_model_length=100
    )

    with pytest.raises(HTTPException) as exc_info:
        await h.chat_completions_via_queue(
            _request(max_tokens=5, max_completion_tokens=30)
        )

    assert exc_info.value.status_code == 400
    assert "max_tokens=30" in exc_info.value.detail


@pytest.mark.asyncio
async def test_too_long_request_does_not_register_correlation_id(
    patched_components,
):
    """The 400 path must leak nothing into the router or coalescer —
    same contract as the 429 over-capacity path."""
    patched_components["tokenize_chat"].return_value = AdmissionTokenization(
        prompt_tokens=200, max_model_length=100
    )
    coalescer = patched_components["coalescer"]

    with pytest.raises(HTTPException):
        await h.chat_completions_via_queue(_request(max_tokens=10))

    assert patched_components["router"].in_flight_count == 0
    coalescer.submit.assert_not_called()


@pytest.mark.asyncio
async def test_length_check_passes_when_within_threshold(patched_components):
    """Boundary case: prompt + max_tokens == max_model_length is fine."""
    patched_components["tokenize_chat"].return_value = AdmissionTokenization(
        prompt_tokens=80, max_model_length=100
    )

    response = await h.chat_completions_via_queue(_request(max_tokens=20))

    # No exception → got the StreamingResponse back.
    assert response is not None


@pytest.mark.asyncio
async def test_length_check_skipped_when_runtime_tokenize_is_unavailable(
    patched_components,
):
    """
    Runtime tokenization fails open so transient vLLM unavailability does not
    make the API reject every request using an inexact local render.
    """
    # Default fixture returns None from tokenize_chat_for_admission.
    await h.chat_completions_via_queue(_request())


@pytest.mark.asyncio
async def test_length_check_uses_vllm_tokenize_count_and_cap_not_local_config(
    patched_components,
):
    """
    The runtime endpoint applies vLLM's configured server template and
    specialized renderer. Its count and context window are authoritative;
    neither the API pod's local tokenizer nor a stale config knob may replace
    them.
    """
    patched_components["config"]["max_model_length"] = 256
    patched_components["tokenize_chat"].return_value = AdmissionTokenization(
        prompt_tokens=4096, max_model_length=65536
    )

    # 4096 + 1000 = 5096 < 65536 → admit, even though config says 256.
    response = await h.chat_completions_via_queue(_request(max_tokens=1000))

    assert response is not None


@pytest.mark.asyncio
async def test_runtime_tokenize_receives_sanitized_chat_request(patched_components):
    await h.chat_completions_via_queue(
        _request(
            reasoning_effort="high",
            chat_template_kwargs={
                "enable_thinking": False,
                "chat_template": "{{ untrusted }}",
            },
        )
    )

    kwargs = patched_components["tokenize_chat"].await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["chat_request"]["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["chat_request"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template" not in kwargs["chat_request"]


@pytest.mark.asyncio
async def test_runtime_tokenize_4xx_is_returned_without_enqueue(
    patched_components,
):
    patched_components["tokenize_chat"].side_effect = VLLMTokenizeRequestError(
        status_code=400,
        detail="template rejected messages",
    )

    with pytest.raises(HTTPException) as exc_info:
        await h.chat_completions_via_queue(_request())

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "template rejected messages"
    patched_components["coalescer"].submit.assert_not_awaited()
    assert patched_components["router"].in_flight_count == 0


@pytest.mark.asyncio
async def test_server_only_template_is_not_preempted_by_local_render(
    patched_components,
):
    """A vLLM-only --chat-template may be absent from the API tokenizer.

    Once the authoritative runtime renderer accepts the request, failure of the
    local rolling-upgrade render must neither reject nor precede it. Current
    workers consume the intact structured request; old workers get a plain text
    fallback with non-text data replaced by a short placeholder.
    """
    events = []
    captured = []

    async def runtime_tokenize(**_kwargs):
        events.append("runtime-tokenize")
        return AdmissionTokenization(prompt_tokens=12, max_model_length=4096)

    def local_render(**_kwargs):
        events.append("local-render")
        raise ValueError("tokenizer has no bundled chat template")

    async def capture(request, _correlation_id):
        captured.append(request)

    patched_components["tokenize_chat"].side_effect = runtime_tokenize
    patched_components["coalescer"].submit = AsyncMock(side_effect=capture)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,secret"},
                },
            ],
        }
    ]

    with patch.object(h, "render_chat_template", side_effect=local_render):
        response = await h.chat_completions_via_queue(_request(messages=messages))

    assert response is not None
    assert events == ["runtime-tokenize", "local-render"]
    assert captured[0]["chat_request"]["messages"] == messages
    assert captured[0]["prompt"] == "user: describe this [image_url]\nassistant:"
    assert "base64" not in captured[0]["prompt"]


# ---------------------------------------------------------------------------
# _resolve_model — None / "latest" / explicit / unknown
# ---------------------------------------------------------------------------


def test_resolve_model_none_falls_back_to_config_default():
    """
    None gets the deployment's default. This was the production crash
    surfaced by the inference browser: a missing-model request used to
    crash inside `AutoTokenizer.from_pretrained(None)` with a
    HuggingFace 401 about a non-existent repo named "None".
    """
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "default-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model(None, cfg) == "default-m"
    fake_manager.find_model.assert_called_once_with("default-m")


def test_resolve_model_empty_string_falls_back_to_config_default():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "default-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model("", cfg) == "default-m"


def test_resolve_model_latest_uses_get_latest_model():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "training-job-abc"
    with (
        patch(
            "cray_infra.training.get_latest_model.get_latest_model",
            return_value="training-job-abc",
        ),
        patch(
            "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
            return_value=fake_manager,
        ),
    ):
        assert h._resolve_model("latest", cfg) == "training-job-abc"


def test_resolve_model_explicit_validates_against_manager():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = "explicit-m"
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        assert h._resolve_model("explicit-m", cfg) == "explicit-m"
    fake_manager.find_model.assert_called_once_with("explicit-m")


def test_resolve_model_unknown_raises_404():
    cfg = {"model": "default-m"}
    fake_manager = MagicMock()
    fake_manager.find_model.return_value = None
    with patch(
        "cray_infra.training.vllm_model_manager.get_vllm_model_manager",
        return_value=fake_manager,
    ):
        with pytest.raises(HTTPException) as exc:
            h._resolve_model("nope-not-here", cfg)
    assert exc.value.status_code == 404
    assert "nope-not-here" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_disconnect_during_stream_unregisters_cid(patched_components):
    """
    The streaming generator's `finally` must unregister the cid even
    if the generator is closed early (client disconnect → FastAPI
    cancels the body iterator).
    """
    with patch.object(h, "render_chat_template", return_value="rendered"):
        response = await h.chat_completions_via_queue(_request())

    router = patched_components["router"]
    assert router.in_flight_count == 1

    # Drive the generator partially then close it (mimics client
    # disconnect: FastAPI calls `aclose()` on the iterator).
    gen = response.body_iterator
    await gen.__anext__()  # consume one heartbeat
    await gen.aclose()

    assert router.in_flight_count == 0
