from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    api_url: str = "http://localhost:8000"

    model: str = "tiny-random/gemma-4-dense"
    #model: str = "google/gemma-3-270m-it"
    #model: str = "yujiepan/qwen3-moe-tiny-random"
    #model: str = "masint/tiny-random-llama"
    #model: str = "masint/tiny-random-qwen2-vl"
    #model: str = "Snowflake/Arctic-Text2SQL-R1-7B"
    #model: str = "Qwen/Qwen2-7B-Instruct"
    #model: str = "Qwen/Qwen2-VL-7B-Instruct"


    # 10GB using 1024 for KB, 1024 for MB, 1024 for GB
    max_upload_file_size: int = 1024 * 1024 * 1024 * 10

    train_job_entrypoint: str = "/app/cray/scripts/train_job_entrypoint.sh"
    publish_job_entrypoint: str = "/app/cray/scripts/publish_job_entrypoint.sh"
    training_job_directory: str = "/app/cray/jobs"

    log_directory: str = "/app/cray/nfs/logs"

    # Per-file size cap before the log rotates and per-server backup
    # count. Three log files × (log_max_bytes × (log_backup_count + 1))
    # is the worst-case disk footprint for the rotated set. Defaults
    # land at ~180 MB total — comfortably under the 974 MB NFS PVC the
    # gemma4 deployment uses, and small enough to tail without trouble.
    # Operators on bigger volumes can raise these in cray-config.yaml.
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_count: int = 5

    max_gpus_per_node: int = 1

    # Job-restart hardening (PR #5). ScalarLM relaunches a QUEUED/TRAINING job
    # whose slurm allocation vanished, which is right for preemption but turns
    # into a tight relaunch loop when the job dies during startup for a
    # deterministic reason (a bad entrypoint, a missing device). Bound the
    # retries and space them out; treat a job that has not updated its status
    # within the heartbeat window as dead.
    max_job_restart_count: int = 3
    job_restart_cooldown_seconds: int = 300
    training_heartbeat_seconds: int = 600

    max_train_time: int = 24 * 60 * 60
    extra_training_seconds: int = 300  # 5 minutes buffer before slurm kills the job

    # Per-slice grace window: SLURM sends SIGTERM this many seconds before
    # hitting the slice's --time limit, giving the trainer room to
    # checkpoint and exit cleanly (see docs/training-lifecycle.md §5.4).
    # Must be smaller than max_train_time and larger than a single
    # training step + checkpoint write. Default 5 min — covers most
    # mid-sized models comfortably; bump higher for big models with
    # slow checkpoint writes.
    signal_grace_seconds: int = 300
    tensor_parallel_size: int = 1

    slurm_wait_time: int = 30 # seconds
    node_info_time_limit: int = 3600 # seconds

    megatron_refresh_period: int = 30 # seconds

    vllm_api_url: str = "http://localhost:8001"

    # vLLM Engine Configuration
    generate_batch_size: int = 1024

    # Hard cap on requests submitted to vLLM but not yet finished.
    # The generate worker refuses to pull more work from the SQLiteAckQueue
    # until the in-flight count drops below this threshold. vLLM's own
    # scheduler doesn't apply admission backpressure, so without this cap
    # the worker's get_batch_size heuristic over-counts available capacity
    # and floods vLLM's waiting queue. Defaults to generate_batch_size so
    # existing deployments keep the same effective ceiling.
    max_inflight_requests: int = 1024

    response_timeout: int = 60 # seconds
    inference_work_queue_timeout: int = 30 # seconds
    inference_work_queue_idle_time: int = 5 # seconds
    # The restart watchdog nacks any request that's been unacked longer
    # than this back into the pending queue. Was 300s, which was shorter
    # than p99 for long generations on saturated GPUs — the watchdog
    # then re-submitted to vLLM while the original was still running,
    # compounding load. Raised to 30 min; operators with fast inference
    # can lower it in cray-config.yaml.
    inference_work_queue_ack_timeout: int = 1800 # seconds

    inference_work_queue_path: str = "/app/cray/inference_work_queue.sqlite"
    upload_base_path: str = "/app/cray/inference_requests"

    gpu_memory_utilization: float = 0.40
    max_model_length: int = 0  # 0 = no cap; resolved dynamically from vLLM at runtime
    default_max_output_tokens: int = 128
    dtype: str = "auto"
    limit_mm_per_prompt:str = '{"image":2}'

    # Whether to pass --enable-lora to vLLM on startup.
    #
    # Defaults to false: with both flags set the fork selects its
    # HybridAdapterManager, which nests every targeted layer under
    # `.base_layer.`, so the attention weights in a ScalarLM Tokenformer
    # checkpoint match nothing and are dropped at activation — the model
    # then serves with attention still at initialization. job_config's
    # adapter_type defaults to "tokenformer", so false matches what the
    # trainer produces.
    #
    # Cost: adapter_type="lora" jobs are ignored in this mode. One server
    # cannot serve both types today. Set SCALARLM_ENABLE_LORA=true for a
    # LoRA-dedicated server.
    enable_lora: bool = False

    # Whether to pass --enable-tokenformer to vLLM on startup. The ScalarLM
    # trainer only ever emits Tokenformer adapters (see
    # ml/cray_megatron/models/get_model_manager.py, which returns
    # TokenformerModelManager unconditionally), so without this flag the vLLM
    # fork picks its LoRA-only adapter manager and rejects every checkpoint
    # this stack produces with "has no LoRA tensors (found only Tokenformer
    # keys)". The adapter then never registers as a served model, and
    # generation against a freshly trained model 404s.
    #
    # With enable_lora false (the default) this selects the Tokenformer-only
    # manager, which leaves the model's parameter names untouched — see the
    # enable_lora comment above for why that matters.
    enable_tokenformer: bool = True

    # vLLM --reasoning-parser value. None enables conservative model-name
    # detection; an explicit parser overrides detection, and an empty string
    # disables it. Unrecognized models remain unchanged unless configured.
    reasoning_parser: Optional[str] = None

    max_log_length: int = 100

    server_list: str = "all"

    tokenformer_r: int = 32
    tokenformer_num_heads: int = 4

    tokenformer_cache_capacity: int = 2

    # Chunked upload settings (docs/chunked-upload.md)
    upload_staging_directory: str = "/app/cray/upload_sessions"
    upload_chunk_size_limit: int = 100 * 1024 * 1024   # 100 MB hard server cap per chunk
    upload_session_ttl_seconds: int = 6 * 60 * 60      # 6 hour session TTL

    hf_token: str = ""

    hf_encrypted_token: bytes = b"gAAAAABpyvSQu2QUlUfp-YavLwueXqCU0j2Lhe9Lddij4B-qV3JngfcH4uCtjVGXlWAyM2o91nZXhsS3B3q3zKNiLxnxhFpJd0ddbwWPysez2OpZX4jTFOA9-xjQVk454A_qk6pdJxMv"
    encryption_key: bytes = b"JAJOZunNSRFeXWXWVVVJfiKSzdzFMw0yFn8_JK50h60="

    # ------------------------------------------------------------------
    # OpenAI chat-completions queue (docs/openai-chat-completions-queue.md)
    # ------------------------------------------------------------------

    # Mirrors vLLM's --max-num-seqs. Chat admission threshold is a
    # multiple of this. Kept in scalarlm config so admission decisions
    # don't round-trip to vLLM at request time.
    max_num_seqs: int = 256

    # Admission high-water mark = chat_admit_factor × max_num_seqs.
    # Beyond this, requests get HTTP 429 + Retry-After. See §5.
    chat_admit_factor: int = 4

    # Coalescer (§6) — packs admitted requests into one queue row to
    # amortize SQLite write cost. packing_factor is the primary
    # throughput knob (§6.3): start here, raise if 429 rate climbs.
    chat_coalescer_packing_factor: int = 10
    chat_coalescer_window_ms: int = 50
    chat_coalescer_bypass_threshold: int = 10

    # Whitespace heartbeat interval (§9). 4 s is below httpx's default
    # 5 s read timeout, leaving margin for slow clocks / event-loop
    # jitter without the caller seeing an idle gap.
    chat_heartbeat_interval_seconds: float = 4.0

    # Optional cap on per-request total time. Unset by default — the
    # queue's existing ack-timeout (inference-queue.md §5.2) provides
    # a backstop for stuck batches.
    chat_max_request_seconds: Optional[float] = None

    # WaitEstimator (§5). Used for the Retry-After hint on 429.
    chat_wait_estimator_default_seconds: float = 5.0
    chat_wait_estimator_padding: float = 1.5
    chat_wait_estimator_sample_size: int = 32

    # Apparent-buffering heuristic (§13.2). Flags requests whose
    # duration falls within `match_threshold_seconds` of a known
    # proxy idle timeout — strong signal that an upstream proxy is
    # buffering responses despite our heartbeats.
    chat_buffering_check_proxy_timeout_seconds: float = 60.0
    chat_buffering_match_threshold_seconds: float = 0.5
