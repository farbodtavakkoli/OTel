from pydantic import BaseModel, Field, model_validator

from typing import Literal, Optional, Union


class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Union[str, list] = "all-linear"

    use_rslora: bool = False

    target_parameters: Optional[list] = None


class NaraConfig(BaseModel):
    enabled: bool = False
    c_scale: float = 0.1
    fnn_hidden_1: int = 256
    fnn_hidden_2: int = 512
    noise_embed_dim: int = 128
    fourier_scale: float = 16.0


class DiffusionConfig(BaseModel):
    canvas_length: int = 256
    eps: float = 0.001
    self_conditioning_prob: float = 0.5
    anchor_token: bool = False
    supervise_termination: bool = False
    pad_loss_weight: float = 1.0
    nara: Optional[NaraConfig] = None

class GrpoConfig(BaseModel):
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    persistent: bool = False


class ClassificationConfig(BaseModel):
    num_labels: Optional[int] = None
    dropout: float = 0.1
    label_smoothing: float = 0.0
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None


class JobConfig(BaseModel):
    job_directory: str
    training_data_path: str
    dataset_hash: str

    custom_data_path: Optional[str] = None

    llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    max_steps: int = 100
    learning_rate: float = 3e-3
    seed: Optional[int] = None
    batch_size: int = 1
    gradient_clip_value: float = 1.0
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False

    warmup_steps: int = Field(default=0, ge=0)
    scheduler_type: Literal["linear", "cosine"] = "linear"

    @model_validator(mode="after")
    def validate_schedule(self):
        if self.max_steps <= 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("max_steps must be positive and warmup_steps must be less than max_steps")
        return self

    cuda_memory_log_interval: int = 100

    attn_implementation: str = "auto"

    max_token_block_size: int = 16777216

    training_mode: str = "language_model"

    optimizer_type: str = "adamw"
    training_storage_dtype: Literal["float32", "bfloat16"] = "float32"
    pretokenized_dataset: bool = False

    @model_validator(mode="before")
    @classmethod
    def default_language_training_precision(cls, values):
        if isinstance(values, dict) and values.get("training_mode", "language_model") in ("language_model", "grpo"):
            values = dict(values)
            values.setdefault("training_storage_dtype", "bfloat16")
            values.setdefault("optimizer_type", "adamw")
        return values

    freeze_layer_keywords: Optional[Union[str, list]] = None

    classification: Optional[ClassificationConfig] = None

    distribution_strategy: str = "fsdp"
    process_group_timeout_seconds: int = Field(default=3600, gt=0)

    hsdp_shard_size: int = 1
    reshard_after_forward: bool = False

    upload_to_hf: bool = False
    hf_repo_id: str = ""
    hf_upload_token: str = ""

    sampling_seed: int = 42

    steps_per_checkpoint: int = 100
    max_checkpoints_to_keep: int = 3

    gpus: int = 1
    nodes: int = 1

    adapter_type: str = "tokenformer"
    lora_config: Optional[LoraConfig] = LoraConfig()

    init_adapter_from: Optional[str] = None
    init_optimizer_from_adapter: bool = False

    grpo: Optional[GrpoConfig] = None

    diffusion: Optional[DiffusionConfig] = None

    timeout: int = 4 * 60 * 60

    training_history_length: int = 1024

    dtype: str = "auto"

    trust_remote_code: bool = False
