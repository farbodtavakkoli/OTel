from pydantic import BaseModel
from typing import List


class UploadInitRequest(BaseModel):
    total_size: int
    total_hash: str
    chunk_size: int
    num_chunks: int
    compressed: bool = True
    params: dict


class UploadInitResponse(BaseModel):
    upload_id: str
    received_chunks: List[int] = []


class UploadChunkResponse(BaseModel):
    upload_id: str
    chunk_index: int
    received: bool
    bytes_written: int


class UploadFinalizeRequest(BaseModel):
    upload_id: str
