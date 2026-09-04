"""
End-to-end validation of the whitespace-heartbeat technique against a
real `AsyncOpenAI` client over a real localhost socket.

The technique only matters because of how httpx handles the `read`
timeout — it's "max time between bytes," not "max total time" — so
the test has to exercise an actual TCP socket. An ASGI in-memory
transport would skip the layer the heartbeat is designed to defeat.

Layout:
  - The positive case mounts a `/v1/chat/completions` route that
    sleeps 3× the configured read timeout while emitting whitespace
    heartbeats. The OpenAI client is configured with that short read
    timeout. If the heartbeat works, the call completes; if it
    doesn't, the SDK raises `APITimeoutError`.
  - The negative case mounts a `/v1-baseline/chat/completions` route
    that does the same sleep with no heartbeats. It must fail with
    `APITimeoutError` — otherwise the positive case's success doesn't
    actually prove the heartbeat is doing the work.
"""

import asyncio
import socket
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import APITimeoutError, AsyncOpenAI

from cray_infra.api.fastapi.chat_completions.heartbeat import (
    stream_with_heartbeat,
)


READ_TIMEOUT_SECONDS = 1.0
WAIT_SECONDS = 3.0
HEARTBEAT_INTERVAL_SECONDS = 0.2


def _completion_template() -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "refusal": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def with_heartbeat(_payload: dict):
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        loop.call_later(WAIT_SECONDS, future.set_result, _completion_template())
        return StreamingResponse(
            stream_with_heartbeat(
                future,
                heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
            ),
            media_type="application/json",
        )

    @app.post("/v1-baseline/chat/completions")
    async def without_heartbeat(_payload: dict):
        await asyncio.sleep(WAIT_SECONDS)
        return _completion_template()

    return app


def _claim_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@asynccontextmanager
async def _running_server(app: FastAPI):
    port = _claim_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        loop="asyncio",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    if not server.started:
        server.should_exit = True
        await serve_task
        raise RuntimeError("uvicorn did not start within 5 seconds")

    try:
        yield port
    finally:
        server.should_exit = True
        await serve_task


def _build_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url,
        api_key="test-key",
        timeout=httpx.Timeout(
            connect=2.0,
            read=READ_TIMEOUT_SECONDS,
            write=2.0,
            pool=2.0,
        ),
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_heartbeat_keeps_connection_alive_past_read_timeout():
    """
    Heartbeat every 0.2s lets a 3s response complete under a 1s read
    timeout. Asserts the response parses and the wall time was
    actually long enough that the timeout path had to fire.
    """
    async with _running_server(_build_app()) as port:
        client = _build_client(f"http://127.0.0.1:{port}/v1")
        try:
            t0 = time.monotonic()
            response = await client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
            elapsed = time.monotonic() - t0
        finally:
            await client.close()

    assert response.choices[0].message.content == "hello"
    assert response.id == "chatcmpl-test"
    assert elapsed >= WAIT_SECONDS - 0.5, (
        f"expected wait around {WAIT_SECONDS}s but got {elapsed:.2f}s — "
        "the heartbeat path may not have been exercised"
    )
    assert elapsed > READ_TIMEOUT_SECONDS, (
        f"wall time {elapsed:.2f}s did not exceed read timeout "
        f"{READ_TIMEOUT_SECONDS}s — the test isn't proving anything"
    )


@pytest.mark.asyncio
async def test_no_heartbeat_times_out_at_read_timeout():
    """
    Negative control: same wait, no heartbeats. Must hit the read
    timeout, otherwise the positive test's success could be due to
    something other than the heartbeat keeping the connection alive.
    """
    async with _running_server(_build_app()) as port:
        client = _build_client(f"http://127.0.0.1:{port}/v1-baseline")
        try:
            with pytest.raises(APITimeoutError):
                await client.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            await client.close()
