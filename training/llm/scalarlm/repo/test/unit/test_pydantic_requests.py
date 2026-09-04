"""
Unit tests for FastAPI request / response Pydantic models.

Contract under test: the schemas under
infra/cray_infra/api/fastapi/routers/request_types/ define the wire shape
that SDK, UI, and internal callers depend on. Changes to required fields
and defaults here break clients silently without these assertions.
"""

import pytest
from pydantic import ValidationError

from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequest,
    FinishWorkRequests,
)
from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_adaptors_request import (
    GetAdaptorsRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_results_request import (
    GetResultsRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_work_request import (
    GetWorkRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_work_response import (
    GetWorkResponse,
)
from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse,
)

# ---- GenerateRequest -------------------------------------------------------


def test_generate_request_accepts_string_prompts():
    req = GenerateRequest(prompts=["hello", "world"])

    assert req.prompts == ["hello", "world"]
    assert req.max_tokens == 16
    assert req.temperature == 0.0
    assert req.model is None
    assert req.tools is None
    assert req.tool_choice is None


def test_generate_request_accepts_chat_dict_prompts():
    req = GenerateRequest(
        prompts=[{"role": "user", "content": "hi"}],
        model="tiny-random/gemma-4-dense",
    )

    assert req.model == "tiny-random/gemma-4-dense"
    assert req.prompts[0]["role"] == "user"


def test_generate_request_requires_prompts():
    with pytest.raises(ValidationError):
        GenerateRequest()


def test_generate_request_accepts_explicit_overrides():
    req = GenerateRequest(
        prompts=["hi"],
        max_tokens=128,
        temperature=0.7,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        tool_choice="auto",
    )

    assert req.max_tokens == 128
    assert req.temperature == pytest.approx(0.7)
    assert req.tool_choice == "auto"


# ---- GetWorkRequest --------------------------------------------------------


def test_get_work_request_requires_both_fields():
    with pytest.raises(ValidationError):
        GetWorkRequest(batch_size=4)
    with pytest.raises(ValidationError):
        GetWorkRequest(loaded_adaptor_count=0)


def test_get_work_request_ok_shape():
    req = GetWorkRequest(batch_size=4, loaded_adaptor_count=0)

    assert req.batch_size == 4
    assert req.loaded_adaptor_count == 0


def test_get_work_response_accepts_json_safe_chat_request():
    chat_request = {
        "model": "model-1",
        "messages": [{"role": "user", "content": "hi"}],
        "include_reasoning": False,
    }
    response = GetWorkResponse(
        prompt="rendered",
        request_id="req-1",
        request_type="chat_completions",
        model="model-1",
        max_tokens=64,
        chat_request=chat_request,
    )

    assert response.chat_request == chat_request


# ---- GetResultsRequest -----------------------------------------------------


def test_get_results_request_requires_list():
    with pytest.raises(ValidationError):
        GetResultsRequest()


def test_get_results_request_accepts_empty_list():
    req = GetResultsRequest(request_ids=[])

    assert req.request_ids == []


def test_get_results_request_accepts_ids():
    req = GetResultsRequest(request_ids=["abc_000000001", "abc_000000002"])

    assert len(req.request_ids) == 2


# ---- FinishWorkRequest(s) --------------------------------------------------


def test_finish_work_request_minimal():
    req = FinishWorkRequest(request_id="abc_000000001")

    assert req.request_id == "abc_000000001"
    assert req.response is None
    assert req.error is None
    assert req.token_count is None
    assert req.flop_count is None


def test_finish_work_request_requires_id():
    with pytest.raises(ValidationError):
        FinishWorkRequest()


def test_finish_work_request_accepts_token_split():
    """The worker propagates prompt/completion separately so the SDK
    can surface them in `usage`. Fields are optional for backwards
    compatibility with workers that only know `token_count`."""
    req = FinishWorkRequest(
        request_id="abc",
        response="hi",
        token_count=42,
        prompt_tokens=35,
        completion_tokens=7,
    )
    assert req.prompt_tokens == 35
    assert req.completion_tokens == 7
    assert req.token_count == 42


def test_finish_work_request_accepts_structured_chat_fields():
    req = FinishWorkRequest(
        request_id="abc",
        reasoning="checking",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
        finish_reason="tool_calls",
    )

    assert req.reasoning == "checking"
    assert req.tool_calls[0]["function"]["name"] == "lookup"
    assert req.finish_reason == "tool_calls"


def test_finish_work_request_accepts_embedding_list():
    # response Union[str, list[float]] — embeddings take the list branch.
    req = FinishWorkRequest(request_id="abc", response=[0.1, 0.2, 0.3])

    assert req.response == [0.1, 0.2, 0.3]


def test_finish_work_request_accepts_generate_string():
    req = FinishWorkRequest(request_id="abc", response="hello")

    assert req.response == "hello"


def test_finish_work_requests_empty_list_ok():
    req = FinishWorkRequests(requests=[])

    assert req.requests == []


def test_finish_work_requests_nested_list():
    req = FinishWorkRequests(
        requests=[
            FinishWorkRequest(request_id="a", response="x"),
            FinishWorkRequest(request_id="b", error="oops"),
        ]
    )

    assert len(req.requests) == 2
    assert req.requests[1].error == "oops"


# ---- GetAdaptorsRequest ----------------------------------------------------


def test_get_adaptors_request_requires_loaded_adaptor_count():
    # The worker's get_work response piggybacks adapter delta discovery on
    # `loaded_adaptor_count`. Making it accidentally optional would break
    # the delta contract — so keep this strict.
    with pytest.raises(ValidationError):
        GetAdaptorsRequest()


def test_get_adaptors_request_ok_shape():
    req = GetAdaptorsRequest(loaded_adaptor_count=3)

    assert req.loaded_adaptor_count == 3


# ---- TrainResponse ---------------------------------------------------------


def test_train_response_shape():
    resp = TrainResponse(
        job_status={"status": "QUEUED", "job_id": "5"},
        job_config={"max_steps": 1},
    )

    assert resp.job_status["status"] == "QUEUED"
    assert resp.job_config["max_steps"] == 1
    assert resp.deployed is False


def test_train_response_deployed_overridable():
    resp = TrainResponse(job_status={}, job_config={}, deployed=True)

    assert resp.deployed is True
