from cray_infra.api.fastapi.routers.request_types.publish_request import (
    PublishRequest,
)
from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse,
)

from cray_infra.api.fastapi.routers.request_types.get_gpu_count_response import (
    GetGPUCountResponse,
)

from cray_infra.api.fastapi.routers.request_types.get_node_count_response import (
    GetNodeCountResponse,
)

from cray_infra.training.launch_training_job import launch_training_job
from cray_infra.training.upload_training_data import upload_training_data
from cray_infra.training.chunked_upload import init_upload, write_chunk, finalize_upload
from cray_infra.api.fastapi.routers.request_types.upload_chunked_types import (
    UploadInitRequest,
    UploadInitResponse,
    UploadChunkResponse,
    UploadFinalizeRequest,
)
from cray_infra.training.training_logs_generator import training_logs_generator
from cray_infra.training.get_training_job_info import get_training_job_info
from cray_infra.training.get_dataset_slice import get_dataset_slice
from cray_infra.training.download_dataset import download_dataset
from cray_infra.training.launch_publish_job import (
    cancel_publish_job,
    get_publish_status,
    launch_publish_job,
)
from cray_infra.training.publish_logs import publish_logs_generator
from cray_infra.training.list_checkpoints import list_checkpoints
from cray_infra.training.list_models import list_models
from cray_infra.training.squeue import squeue
from cray_infra.training.cancel import cancel
from cray_infra.training.restart import restart
from cray_infra.training.delete import delete
from cray_infra.training.get_gpu_count import get_gpu_count
from cray_infra.training.get_node_count import get_node_count

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

import traceback

import logging

logger = logging.getLogger(__name__)

megatron_router = APIRouter(prefix="/megatron")


@megatron_router.post("/train")
async def train(request: Request):
    logger.info(f"Training request received: {request}")
    training_data_path, params = await upload_training_data(request)

    try:
        job_config = params

        logger.info(f"Training args: {job_config}")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid request body",
        )

    job_status = await launch_training_job(job_config)

    return TrainResponse(job_status=job_status, job_config=job_config, deployed=False)


@megatron_router.post("/upload/init", response_model=UploadInitResponse)
async def upload_init(req: UploadInitRequest):
    return await init_upload(req)


@megatron_router.post("/upload/chunk", response_model=UploadChunkResponse)
async def upload_chunk(request: Request):
    upload_id = request.headers.get("X-Upload-Id")
    chunk_index_str = request.headers.get("X-Chunk-Index")
    chunk_hash = request.headers.get("X-Chunk-Hash")

    if not upload_id or chunk_index_str is None or not chunk_hash:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing required headers: X-Upload-Id, X-Chunk-Index, X-Chunk-Hash",
        )

    try:
        chunk_index = int(chunk_index_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid X-Chunk-Index: {chunk_index_str!r}",
        )

    body = await request.body()
    return await write_chunk(upload_id, chunk_index, chunk_hash, body)


@megatron_router.post("/upload/finalize")
async def upload_finalize(req: UploadFinalizeRequest):
    training_data_path, job_config = await finalize_upload(req)
    job_status = await launch_training_job(job_config)
    return TrainResponse(job_status=job_status, job_config=job_config, deployed=False)


@megatron_router.get("/train/{job_hash}")
async def job_info(job_hash: str):
    return await get_training_job_info(job_hash)


@megatron_router.get("/train/{job_hash}/dataset")
async def job_dataset(
    job_hash: str,
    offset: int = 0,
    limit: int = 50,
    q: str | None = None,
):
    return get_dataset_slice(job_hash, offset=offset, limit=limit, q=q)


@megatron_router.get("/train/{job_hash}/dataset/download")
async def job_dataset_download(job_hash: str, limit: int | None = None):
    """
    Stream the job's dataset.jsonlines verbatim as an attachment.
    Omit `limit` for the full file; pass `limit=N` for the first N rows.
    """
    return download_dataset(job_hash, limit=limit)


@megatron_router.get("/train/{job_hash}/checkpoints")
async def job_checkpoints(job_hash: str):
    return list_checkpoints(job_hash)


@megatron_router.post("/train/{job_hash}/publish")
async def submit_publish_job(job_hash: str, request: PublishRequest):
    """
    Submit a publish-to-HF SLURM job. Returns immediately with the
    publish_job_id; the UI polls /publish/status for progress.
    """
    return launch_publish_job(
        job_hash,
        mode=request.mode,
        repo_id=request.repo_id,
        private=request.private,
        hf_token=request.hf_token,
        checkpoint=request.checkpoint,
        lora_alpha=request.lora_alpha,
        commit_message=request.commit_message,
    )


@megatron_router.get("/train/{job_hash}/publish/status")
async def publish_status(job_hash: str):
    """Latest publish status.json for this training job."""
    return get_publish_status(job_hash)


@megatron_router.post("/train/{job_hash}/publish/cancel")
async def publish_cancel(job_hash: str):
    """scancel the in-flight publish job; returns the resulting status."""
    return cancel_publish_job(job_hash)


@megatron_router.get("/train/{job_hash}/publish/logs")
async def publish_logs(
    job_hash: str,
    starting_line_number: int = 0,
    starting_byte_offset: int | None = None,
    tail: int | None = None,
    limit: int | None = None,
    before_byte_offset: int | None = None,
    before_count: int | None = None,
):
    """
    NDJSON tail of the latest publish job's publish.log. Same wire
    shape as /v1/health/logs/{service}; the UI reuses the same
    streamNdjsonLogOnce consumer.
    """
    try:
        return StreamingResponse(
            content=publish_logs_generator(
                job_hash,
                starting_line_number=starting_line_number,
                starting_byte_offset=starting_byte_offset,
                tail=tail,
                limit=limit,
                before_byte_offset=before_byte_offset,
                before_count=before_count,
            ),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@megatron_router.get("/train/logs/{model_name}")
async def get_training_logs(model_name: str, starting_line_number: int = 0):
    try:
        return StreamingResponse(
            content=training_logs_generator(model_name, starting_line_number),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@megatron_router.post("/cancel/{job_hash}")
async def cancel_job(job_hash: str):
    return await cancel(job_hash)

@megatron_router.post("/restart/{job_hash}")
async def restart_job(job_hash: str):
    return await restart(job_hash)

@megatron_router.post("/delete/{job_hash}")
async def delete_job(job_hash: str):
    return await delete(job_hash)

@megatron_router.get("/list_models")
async def models():
    return await list_models()


@megatron_router.get("/squeue")
async def get_squeue():
    return await squeue()


@megatron_router.get("/gpu_count")
async def gpu_count():
    return GetGPUCountResponse(gpu_count=get_gpu_count())


@megatron_router.get("/node_count")
async def node_count():
    return GetNodeCountResponse(node_count=get_node_count())


@megatron_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in megatron_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
