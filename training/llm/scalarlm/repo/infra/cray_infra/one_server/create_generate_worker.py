import asyncio
import aiohttp
from itertools import islice
import json
import os
import signal
import sys

import psutil

import logging

from typing import Optional, NoReturn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.chat_completion.api_router import (
    create_chat_completion,
)
from vllm.entrypoints.openai.completion.api_router import (
    create_completion,
)
from vllm.entrypoints.openai.models.api_router import (
    models,
)
from vllm.entrypoints.openai.models.serving import (
    OpenAIServingModels,
)

try:
    from vllm.entrypoints.openai.engine.protocol import (
        ErrorResponse,
    )
except ImportError:
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)

from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
)

try:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
except ImportError:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam


from cray_infra.api.fastapi.routers.request_types.get_work_response import (
    PromptType,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.one_server.decode_error_body import decode_error_body
from cray_infra.one_server.format_exception import format_exception
from cray_infra.one_server.sampling_params import sampling_params_for

from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


class _InflightCounter:
    __slots__ = ("count",)

    def __init__(self):
        self.count = 0


async def create_generate_worker(server_status):
    logger.info("=" * 60)
    logger.info("STARTING GENERATE WORKER")
    logger.info("=" * 60)

    config = get_config()
    api_base = config["api_url"]

    app = await server_status.get_app()

    session: Optional[aiohttp.ClientSession] = None

    tasks = []

    loaded_adaptor_count = 0

    max_inflight = int(
        config.get("max_inflight_requests", config["generate_batch_size"])
    )
    inflight = _InflightCounter()

    try:
        session = aiohttp.ClientSession()

        while True:
            clear_finished_tasks(tasks)

            headroom = max_inflight - inflight.count
            if headroom <= 0:
                logger.debug(
                    "At max_inflight_requests (%d), waiting for drain",
                    max_inflight,
                )
                await asyncio.sleep(0.1)
                continue

            batch_size = await get_batch_size(app)

            if batch_size == 0:
                logger.debug(f"No batch size available, waiting for kv cache space")
                await asyncio.sleep(0.1)
                continue

            batch_size = min(batch_size, headroom)

            logger.debug(
                f"Checking for work with batch size: {batch_size} "
                f"(inflight={inflight.count}/{max_inflight}, "
                f"loaded adaptors: {loaded_adaptor_count})..."
            )

            try:
                get_work_response = await session.post(
                    f"{api_base}/v1/generate/get_work",
                    json={
                        "batch_size": batch_size,
                        "loaded_adaptor_count": loaded_adaptor_count,
                    },
                    timeout=aiohttp.ClientTimeout(
                        total=2 * float(config["inference_work_queue_timeout"])
                    ),
                )

            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for work, retrying...")
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
                continue

            if get_work_response.status != 200:
                logger.error(f"Failed to get work: {get_work_response.status}")
                await asyncio.sleep(1)
                continue

            work_data = await get_work_response.json()

            adaptors = work_data["new_adaptors"]["new_adaptors"]

            loaded_adaptor_count = await add_adaptors(
                app, adaptors, loaded_adaptor_count
            )

            requests = work_data.get("requests", [])

            if not requests:
                logger.debug("No work available")
                await asyncio.sleep(1)
                continue

            new_task = await process_requests(app, requests, inflight)

            tasks.append(new_task)

    except Exception as e:
        logger.error(f"Worker encountered an error: {e}", exc_info=True)
        logger.error("Shutting down worker due to error")
        kill_vllm_container()

    finally:
        if session and not session.closed:
            await session.close()

        logger.info("Generate worker shutting down")


def clear_finished_tasks(tasks):
    for task in tasks:
        if task.done():
            try:
                task.result()
            except Exception as e:
                logger.error(f"Task raised an exception: {e}", exc_info=True)

    tasks[:] = [task for task in tasks if not task.done()]

    logger.debug(f"Remaining tasks: {len(tasks)}")


async def get_batch_size(app):
    vllm_engine_client = app.state.engine_client
    if not hasattr(vllm_engine_client, "get_current_kv_cache_size"):
        return get_config()["generate_batch_size"]
    current_kv_cache_size = await vllm_engine_client.get_current_kv_cache_size()

    config = get_config()
    max_model_length = vllm_engine_client.model_config.max_model_len

    batch_size = current_kv_cache_size // max_model_length

    if batch_size <= 0:
        logger.debug("Batch size is 0, waiting for kv cache space")
        return 0

    current_kv_cache_size = await vllm_engine_client.get_current_kv_cache_size()

    batch_size = current_kv_cache_size // max_model_length

    maximum_batch_size = config["generate_batch_size"]

    return min(batch_size, maximum_batch_size)


async def add_adaptors(app, adaptors, loaded_adaptor_count):
    config = get_config()

    params = {
        "loaded_adaptor_count": loaded_adaptor_count,
    }

    try:
        session = get_global_session()

        for new_adaptor in adaptors:
            logger.info("Loading new adaptor: %s", new_adaptor)
            try:
                await add_new_adaptor(app, new_adaptor)
                loaded_adaptor_count += 1
            except Exception as e:
                logger.error("Error loading adaptor %s: %s", new_adaptor, e)
                continue
    except Exception as e:
        logger.error("Error loading adaptors %s", e)

    return loaded_adaptor_count


async def add_new_adaptor(app: FastAPI, new_adaptor: str):
    config = get_config()
    base_path = config["training_job_directory"]

    new_adaptor_path = os.path.join(base_path, new_adaptor)

    logger.info("Loading new adaptor from path: %s", new_adaptor_path)

    lora_adaptor_request = LoadLoRAAdapterRequest(
        lora_name=new_adaptor, lora_path=new_adaptor_path
    )

    raw_request = Request(
        scope={
            "app": app,
            "type": "http",
            "headers": {},
            "path": "/v1/load_lora_adapter",
        },
        receive=pass_receive,
    )

    response = await load_lora_adapter(lora_adaptor_request, raw_request=raw_request)

    if isinstance(response, JSONResponse):
        if response.status_code != 200:
            error_text = decode_error_body(response)
            logger.error(
                "Failed to load new adaptor %s: %s",
                new_adaptor,
                error_text,
            )
            raise RuntimeError(
                f"Failed to load new adaptor {new_adaptor}: {error_text}"
            )
        else:
            logger.info("Successfully loaded new adaptor: %s", new_adaptor)


async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
    handler: OpenAIServingModels = models(raw_request)
    response = await handler.load_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )

    return Response(status_code=200, content=response)


async def process_requests(app, requests, inflight=None):
    return asyncio.create_task(process_requests_task(app, requests, inflight))


async def process_requests_task(app, requests, inflight=None):
    logger.info(f"Processing {len(requests)} requests")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Got work: %s", truncate_fields({"requests": requests}))

    per_request = [_run_and_finish_one(req, app, inflight) for req in requests]
    await asyncio.gather(*per_request, return_exceptions=True)


async def _run_and_finish_one(request, app, inflight=None):
    if inflight is not None:
        inflight.count += 1
    try:
        try:
            result = await async_generate_task(request, app)
        except Exception as e:
            logger.error(
                "Task for %s raised an exception: %s",
                request.get("request_id"),
                e,
                exc_info=True,
            )
            result = {
                "request_id": request["request_id"],
                "error": format_exception(e),
            }

        config = get_config()
        session = get_global_session()
        try:
            async with session.post(
                config["api_url"] + "/v1/generate/finish_work",
                json={"requests": [result]},
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        "finish_work returned %s for %s",
                        resp.status,
                        result.get("request_id"),
                    )
        except Exception as e:
            logger.error(
                "finish_work POST failed for %s: %s",
                result.get("request_id"),
                e,
                exc_info=True,
            )
    finally:
        if inflight is not None:
            inflight.count -= 1


_LOG_STRING_LIMIT = 100
_LOG_COLLECTION_LIMIT = 20
_LOG_DEPTH_LIMIT = 12
_LOG_REDACTED_KEYS = frozenset(
    {
        "arguments",
        "audio_url",
        "content",
        "image_url",
        "input_audio",
        "prompt",
    }
)


def truncate_fields(data):
    return _sanitize_log_value(data)


def truncate_dict(d):
    sanitized = _sanitize_log_value(d)
    d.clear()
    d.update(sanitized)


def _sanitize_log_value(value, *, key=None, depth=0):
    if key in _LOG_REDACTED_KEYS:
        try:
            size = len(value)
        except TypeError:
            size = 1
        unit = "chars" if isinstance(value, str) else "items"
        return f"<redacted {size} {unit}>"

    if depth >= _LOG_DEPTH_LIMIT and isinstance(value, (dict, list, tuple)):
        return "<truncated nested value>"

    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<redacted {len(value)} chars>"
        if len(value) > _LOG_STRING_LIMIT:
            return value[:_LOG_STRING_LIMIT] + "..."
        return value

    if isinstance(value, dict):
        sanitized = {
            item_key: _sanitize_log_value(
                item_value,
                key=str(item_key),
                depth=depth + 1,
            )
            for item_key, item_value in islice(
                value.items(),
                _LOG_COLLECTION_LIMIT,
            )
        }
        if len(value) > _LOG_COLLECTION_LIMIT:
            sanitized["<truncated>"] = f"{len(value) - _LOG_COLLECTION_LIMIT} keys"
        return sanitized

    if isinstance(value, (list, tuple)):
        items = [
            _sanitize_log_value(item, depth=depth + 1)
            for item in value[:_LOG_COLLECTION_LIMIT]
        ]
        if len(value) > _LOG_COLLECTION_LIMIT:
            items.append(f"<truncated {len(value) - _LOG_COLLECTION_LIMIT} items>")
        return items

    return value


async def pass_receive() -> NoReturn:
    await asyncio.sleep(10.0)
    return {"type": "http.request"}


async def async_generate_task(request, app):
    if request["request_type"] == "chat_completions":
        return await async_chat_completion_task(request, app)
    elif request["request_type"] == "generate":
        if request.get("chat_request") is not None or is_chat_completion_task(request):
            return await async_chat_completion_task(request, app)
        else:
            return await async_completion_task(request, app)
    else:
        raise ValueError(f"Invalid request type: {request['request_type']}")


def is_chat_completion_task(request):
    if isinstance(request["prompt"], str):
        return False

    return True


async def async_chat_completion_task(request, app):
    chat_request = request.get("chat_request")
    if chat_request is not None:
        completion_request = ChatCompletionRequest(**chat_request)
    else:
        completion_request = ChatCompletionRequest(
            model=request["model"],
            messages=convert_prompt_to_openai_format(request["prompt"]),
            max_tokens=request["max_tokens"],
            tools=request.get("tools"),
            tool_choice=request.get("tool_choice"),
            **sampling_params_for(app, request),
        )

    raw_request = Request(
        scope={
            "app": app,
            "type": "http",
            "headers": {},
            "path": "/v1/chat/completions",
        },
        receive=pass_receive,
    )

    response = await create_chat_completion(completion_request, raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {
        "request_id": request["request_id"],
    }

    if "choices" in response_data:
        choice = response_data["choices"][0]
        message = choice["message"]
        response["response"] = message.get("content") or ""
        if message.get("reasoning") is not None:
            response["reasoning"] = message["reasoning"]
        if message.get("tool_calls") is not None:
            response["tool_calls"] = message["tool_calls"]
        if choice.get("finish_reason") is not None:
            response["finish_reason"] = choice["finish_reason"]
    elif "error" in response_data:
        error = response_data["error"]
        response["error"] = (
            error.get("message", repr(error)) if isinstance(error, dict) else str(error)
        )
    elif response_data.get("object") == "error":
        response["error"] = response_data.get("message", repr(response_data))
    else:
        logger.error(
            "Unexpected chat completion response shape for request %s: %s",
            request.get("request_id"),
            response_data,
        )
        response["error"] = f"Unexpected response shape: {response_data!r}"

    if "usage" in response_data:
        usage = response_data["usage"]
        response["token_count"] = usage["total_tokens"]
        response["prompt_tokens"] = usage.get("prompt_tokens")
        response["completion_tokens"] = usage.get("completion_tokens")
        response["flop_count"] = (
            compute_flop_count(app.state.engine_client.model_config)
            * usage["total_tokens"]
        )

    await app.state.engine_client.check_health()

    return response


def convert_prompt_to_openai_format(
    prompt: PromptType,
) -> list[ChatCompletionMessageParam]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": [{"role": "user", "content": prompt}]}]
    elif isinstance(prompt, dict):
        list_of_content = []
        for key, value in prompt.items():
            if key == "text":
                list_of_content.append({"type": "text", "text": value})
            elif key == "images":
                list_of_content.extend(
                    [
                        {"type": "image_url", "image_url": {"url": image}}
                        for image in value
                    ]
                )
            else:
                raise ValueError(
                    f"Invalid prompt sub-field: {key}. Must be 'text' or 'image'."
                )
        return [{"role": "user", "content": list_of_content}]
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")


def compute_flop_count(model_config):
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    head_size = model_config.get_head_size()

    num_layers = getattr(model_config.hf_text_config, "num_hidden_layers", 12)
    num_attention_heads = getattr(
        model_config.hf_text_config, "num_attention_heads", 12
    )
    num_kv_heads = model_config.get_total_num_kv_heads()

    intermediate_size = getattr(
        model_config.hf_text_config, "intermediate_size", 4 * hidden_size
    )

    q_proj_flops = hidden_size * (num_attention_heads * head_size)
    kv_proj_flops = hidden_size * (num_kv_heads * head_size * 2)

    qk_flops = num_attention_heads * head_size

    av_flops = num_attention_heads * head_size

    o_proj_flops = hidden_size * hidden_size

    attention_flops_per_layer = (
        q_proj_flops + kv_proj_flops + qk_flops + av_flops + o_proj_flops
    )
    total_attention_flops = attention_flops_per_layer * num_layers

    fc1_flops = hidden_size * intermediate_size
    fc2_flops = intermediate_size * hidden_size
    mlp_flops_per_layer = fc1_flops + fc2_flops

    total_mlp_flops = mlp_flops_per_layer * num_layers

    output_projection_flops = hidden_size * vocab_size

    embedding_flops = hidden_size * vocab_size

    total_flops = (
        total_attention_flops
        + total_mlp_flops
        + embedding_flops
        + output_projection_flops
    )

    return total_flops


async def async_completion_task(request, app):
    completion_request = CompletionRequest(
        model=request["model"],
        prompt=request["prompt"],
        max_tokens=request["max_tokens"],
        tools=request.get("tools"),
        tool_choice=request.get("tool_choice"),
        **sampling_params_for(app, request),
    )

    raw_request = Request(
        scope={"app": app, "type": "http", "headers": {}, "path": "/v1/completions"},
        receive=pass_receive,
    )

    response = await create_completion(completion_request, raw_request=raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {"request_id": request["request_id"]}

    if "choices" in response_data:
        response["response"] = response_data["choices"][0]["text"]
    elif response_data.get("object") == "error":
        response["error"] = response_data.get("message", repr(response_data))
    else:
        logger.error(
            "Unexpected completion response shape for request %s: %s",
            request.get("request_id"),
            response_data,
        )
        response["error"] = f"Unexpected response shape: {response_data!r}"

    if "usage" in response_data:
        usage = response_data["usage"]
        response["token_count"] = usage["total_tokens"]
        response["prompt_tokens"] = usage.get("prompt_tokens")
        response["completion_tokens"] = usage.get("completion_tokens")
        response["flop_count"] = (
            compute_flop_count(app.state.engine_client.model_config)
            * usage["total_tokens"]
        )

    await app.state.engine_client.check_health()

    return response


def server_process_tree():
    me = psutil.Process()
    root = me
    root_pid = os.environ.get("SCALARLM_SERVER_PID", "")
    if root_pid.isdigit():
        for parent in me.parents():
            if parent.pid == int(root_pid):
                root = parent
                break
    procs = [p for p in root.children(recursive=True) if p.pid != me.pid]
    if root.pid != me.pid:
        procs.append(root)
    return procs


def kill_vllm_container():
    for proc in server_process_tree():
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    os.kill(os.getpid(), signal.SIGKILL)
    sys.exit(1)
