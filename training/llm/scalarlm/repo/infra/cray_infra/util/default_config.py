from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    api_url: str = "http://localhost:8000"

    model: str = "tiny-random/gemma-4-dense"

    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    publish_job_entrypoint: str = "/app/cray/scripts/publish_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    log_directory: str = "/app/cray/nfs/logs"

    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    max_gpus_per_node: int = 1

    max_job_restart_count: int = 3
    job_restart_cooldown_seconds: int = 300
    training_heartbeat_seconds: int = 600

    max_train_time: int = 24 * 60 * 60
    extra_training_seconds: int = 300

    signal_grace_seconds: int = 300
    tensor_parallel_size: int = 1

    slurm_wait_time: int = 30
    node_info_time_limit: int = 3600

    megatron_refresh_period: int = 30

    vllm_api_url: str = "http://localhost:8001"

    generate_batch_size: int = 1024

    max_inflight_requests: int = 1024

    response_timeout: int = 60
    inference_work_queue_timeout: int = 30
    inference_work_queue_idle_time: int = 5
    inference_work_queue_ack_timeout: int = 1800

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"
    upload_base_path: str = "/app/cray/inference_requests"

    gpu_memory_utilization: float = 0.40
    max_model_length: int = 0
    default_max_output_tokens: int = 128
    dtype: str = "auto"
    limit_mm_per_prompt:str = '{"image":2}'

    enable_lora: bool = False

    enable_tokenformer: bool = True

    reasoning_parser: Optional[str] = None

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

    upload_staging_directory: str = "/app/cray/upload_sessions"
    upload_chunk_size_limit: int = 100 * 1024 * 1024
    upload_session_ttl_seconds: int = 6 * 60 * 60

    hf_token: str = ""

    hf_encrypted_token: bytes = b"gAAAAABpyvSQu2QUlUfp-YavLwueXqCU0j2Lhe9Lddij4B-qV3JngfcH4uCtjVGXlWAyM2o91nZXhsS3B3q3zKNiLxnxhFpJd0ddbwWPysez2OpZX4jTFOA9-xjQVk454A_qk6pdJxMv"
    encryption_key: bytes = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

    max_num_seqs: int = 256

    chat_admit_factor: int = 4

    chat_coalescer_packing_factor: int = 10
    chat_coalescer_window_ms: int = 50
    chat_coalescer_bypass_threshold: int = 10

    chat_heartbeat_interval_seconds: float = 4.0

    chat_max_request_seconds: Optional[float] = None

    chat_wait_estimator_default_seconds: float = 5.0
    chat_wait_estimator_padding: float = 1.5
    chat_wait_estimator_sample_size: int = 32

    chat_buffering_check_proxy_timeout_seconds: float = 60.0
    chat_buffering_match_threshold_seconds: float = 0.5
