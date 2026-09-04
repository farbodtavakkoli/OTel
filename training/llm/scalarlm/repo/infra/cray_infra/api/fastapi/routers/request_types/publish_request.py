from typing import Optional

from pydantic import BaseModel, Field


class PublishRequest(BaseModel):
    """
    Body of POST /v1/megatron/train/{job_hash}/publish. Drives a SLURM
    publish job; see ui/docs/publish-to-hf.md.

    The HF token is forwarded to sbatch via env-var export and is never
    written to disk or argv on the API pod side.
    """

    mode: str = Field(
        default="merged",
        description="`merged` to fold LoRA into base, `adapter` for a PEFT-format adapter repo.",
    )
    repo_id: str = Field(
        ...,
        description="HuggingFace repo to push to (`owner/name`).",
    )
    private: bool = Field(
        default=False,
        description="Create the repo as private if it doesn't exist.",
    )
    hf_token: str = Field(
        ...,
        description="HuggingFace access token with write permission.",
    )
    checkpoint: Optional[str] = Field(
        default=None,
        description="Basename of checkpoint_<step>.pt to publish. Defaults to latest.",
    )
    lora_alpha: Optional[int] = Field(
        default=None,
        description="Override the lora_alpha used at merge time.",
    )
    commit_message: Optional[str] = Field(
        default=None,
        description="HF commit message; auto-generated if omitted.",
    )
