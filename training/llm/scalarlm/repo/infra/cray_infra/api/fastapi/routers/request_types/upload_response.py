from pydantic import BaseModel

from typing import Optional, Union

class UploadResult(BaseModel):
    request_id: str
    error: Optional[str] = None

