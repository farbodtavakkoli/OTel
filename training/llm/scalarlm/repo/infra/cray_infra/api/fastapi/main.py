from cray_infra.api.fastapi.routers.openai_v1_router import (
    openai_v1_router,
)
from cray_infra.api.fastapi.routers.megatron_router import (
    megatron_router,
)
from cray_infra.api.fastapi.routers.health_router import (
    health_router,
)
from cray_infra.api.fastapi.routers.generate_router import (
    generate_router,
)
from cray_infra.api.fastapi.routers.slurm_router import (
    slurm_router,
)

from cray_infra.api.fastapi.tasks.add_megatron_tasks import (
    add_megatron_tasks,
)
from cray_infra.api.fastapi.setup_ui import add_ui

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import logging
import os
import time
import json

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=add_megatron_tasks)

try:
    from cray_infra.observability.init_tracing import init_tracing
    tracer = init_tracing(app, service_name="scalarlm-api")
    logger.info("Observability: OpenTelemetry tracing initialized")
except Exception as e:
    logger.warning(f"Observability: Failed to initialize tracing: {e}")
    tracer = None

app.include_router(openai_v1_router, prefix="/v1")
app.include_router(megatron_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")
app.include_router(generate_router, prefix="/v1")
app.include_router(slurm_router)

@app.get("/v1/metrics")
async def get_prometheus_metrics():
    try:
        from cray_infra.observability.prometheus_metrics import metrics_endpoint
        return metrics_endpoint()
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Metrics unavailable", status_code=500)

add_ui(app)

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()

    trace_id = None
    traceparent = request.headers.get("traceparent", "")
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 2:
            trace_id = parts[1]

    logger.info(json.dumps({
        "event": "request_start",
        "trace_id": trace_id,
        "method": request.method,
        "path": str(request.url.path),
        "client": request.client.host if request.client else None,
        "timestamp": start_time
    }))

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(json.dumps({
        "event": "request_end",
        "trace_id": trace_id,
        "method": request.method,
        "path": str(request.url.path),
        "status_code": response.status_code,
        "duration_seconds": round(duration, 3),
        "timestamp": time.time()
    }))

    return response
