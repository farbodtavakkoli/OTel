from pydantic import BaseModel

from typing import Optional, Union


class LoraConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Union[str, list] = "all-linear"  # or list of module names

    # MUST be declared: undeclared keys are dropped by pydantic extra="ignore".
    # When use_rslora was missing, training used alpha/r while merging used
    # alpha/sqrt(r) — a silent numeric divergence that never raised.
    use_rslora: bool = False

    # Same story: an operator-supplied target_parameters list was dropped, so
    # the union in create_lora_model.py was always just the auto-derived set and
    # a manual override for grouped-expert MoEs silently did nothing.
    target_parameters: Optional[list] = None


class NaraConfig(BaseModel):
    # Noise-aware LoRA (NaRA) for DiffusionGemma — PROTOTYPE. Makes the low-rank
    # update noise-conditioned: dW(t) = B @ C(t) @ A. Rank/alpha/dropout come from
    # lora_config; these are the NaRA-only extras.
    enabled: bool = False
    c_scale: float = 0.1          # residual gain eta: Ceff = c_scale * C(t) + I
    fnn_hidden_1: int = 256       # shared hypernetwork hidden sizes
    fnn_hidden_2: int = 512
    noise_embed_dim: int = 128    # Gaussian-Fourier embedding width for t (even)
    fourier_scale: float = 16.0


class DiffusionConfig(BaseModel):
    # DiffusionGemma canvas denoising knobs (see ADR 0007/0008 and
    # docs/superpowers/specs/2026-07-01-diffusiongemma-design.md). Only consumed
    # when the model is DiffusionGemma (auto-detected via is_diffusion()); ignored
    # otherwise. Nested block like lora_config; MUST be declared here or
    # get_job_config()'s Pydantic model silently DROPS it from train_args (the same
    # footgun that hid trust_remote_code).
    canvas_length: int = 256   # decoder block size = the loader's pad/truncate target
    eps: float = 0.001         # minimum corruption level for t ~ U(eps, 1)
    # Per-step probability of the two-pass self-conditioning scheme (Analog-Bits):
    # feed the model's own no-grad prediction back as the self-conditioning signal
    # to match the iterative serve decode. 0 disables it (single-pass v1 training).
    self_conditioning_prob: float = 0.5
    # Tier-2 reliability lever (see docs/reports/2026-07-16-diffusiongemma-validation-runs.md).
    # Prepend one fixed, never-corrupted, supervised anchor token (BOS) at canvas
    # position 0 so the real output's first token gains a stable left-neighbor. The
    # cccc/fragment-repeat collapses all begin at positions 0-2, where position 0
    # otherwise sits at a boundary with no left-context (output is tokenized with
    # add_special_tokens=False). Default False = byte-identical to prior runs.
    anchor_token: bool = False
    # Supervise the FULL canvas (answer + EOS + pad tail) instead of masking the
    # tail, so training matches serve's noisy-init decode and the model learns to
    # terminate. Default False = byte-identical to prior runs.
    supervise_termination: bool = False
    # Only used when supervise_termination is True and != 1.0: relative CE weight on
    # the pad-tail positions (answer + EOS always weight 1.0). Lower it if the ~230
    # pad targets swamp the ~24 answer targets. 1.0 = uniform (try this first).
    pad_loss_weight: float = 1.0
    # Noise-aware LoRA (NaRA) prototype. None/omitted or enabled=False = plain LoRA
    # (byte-identical to prior runs). See NaraConfig.
    nara: Optional[NaraConfig] = None

class ClassificationConfig(BaseModel):
    # Sequence-classification training (training_mode: "classification"). Only read
    # when training_mode == "classification"; ignored otherwise. Nested block like
    # lora_config -- MUST be declared here or get_job_config()'s Pydantic model
    # silently DROPS it from train_args (the trust_remote_code footgun).
    num_labels: Optional[int] = None       # required for classification; head width
    dropout: float = 0.1                   # classifier-head dropout
    label_smoothing: float = 0.0           # 0 disables; >0 uses smoothed cross-entropy
    id2label: Optional[dict] = None        # optional human-readable label names
    label2id: Optional[dict] = None


class JobConfig(BaseModel):

    job_directory: str
    training_data_path: str
    dataset_hash: str

    # Reuse a dataset that ALREADY exists on the server/VM instead of the tarball
    # uploaded with this job. None = use training_data_path (the normal path).
    custom_data_path: Optional[str] = None

    #llm_name: str = "masint/tiny-random-llama"
    llm_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Training
    max_steps: int = 100
    learning_rate: float = 3e-3
    # Global training RNG seed (applied before the adapter is built, so PEFT's
    # lora_A init, the per-step canvas corruption, and the SC mask are all
    # deterministic). None = historical non-deterministic behavior (byte-identical
    # to prior runs). Set it to make a run reproducible and to seed-sweep for a
    # DiffusionGemma draw that lands in the clean-serving basin (see the 2026-07-16
    # validation report's "State archaeology" section). See determinism.apply_seed.
    seed: Optional[int] = None
    batch_size: int = 1
    gradient_clip_value: float = 1.0
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False

    # Linear warmup before the LinearLR decay; 0 disables it. Recommended 1-5%
    # of max_steps for LoRA on large models, to avoid early-training NaN bursts.
    warmup_steps: int = 0

    # 0 disables; otherwise log torch.cuda.memory_allocated/reserved
    # every N steps. Used to distinguish a real leak (allocated
    # grows) from caching-allocator fragmentation (reserved grows,
    # allocated flat).
    cuda_memory_log_interval: int = 100

    # HF attn_implementation passed to from_pretrained at training time.
    # "auto" (the default) resolves to "sdpa" — flash-attention is no
    # longer supported. Can be forced to "sdpa" or "eager"; eager is the
    # universal fallback for configs that reject sdpa.
    attn_implementation: str = "auto"

    max_token_block_size: int = 16777216 # 16 mega tokens

    training_mode: str = "language_model"  # or "embedding" / "classification"

    # Optimizer. "adamw" (default) matches historical behavior; "sgd" and "rmsprop"
    # are also constructed over trainable parameters only. An unknown value logs a
    # warning and falls back to AdamW rather than failing the job.
    optimizer_type: str = "adamw"

    # Freeze every parameter whose NAME contains any of these substrings, e.g.
    # ["vision_model", "vision_tower", "multi_modal_projector", "visual"] to train a
    # multimodal model's language tower only. Accepts a list or a comma-separated
    # string. None/empty = freeze nothing.
    freeze_layer_keywords: Optional[Union[str, list]] = None

    # Sequence-classification knobs; only consumed when training_mode ==
    # "classification". Nested block like lora_config.
    classification: Optional[ClassificationConfig] = None

    # Distribution strategy
    distribution_strategy: str = "fsdp"

    # PyTorch FSDP2 (`distribution_strategy: pytorch_fsdp`) knobs. Every key MUST
    # be declared here or pydantic's extra="ignore" silently drops it.
    #   hsdp_shard_size: >1 builds a 2D (replicate, shard) mesh; must divide world.
    #   reshard_after_forward: lower peak memory, one extra collective.
    hsdp_shard_size: int = 1
    reshard_after_forward: bool = False

    # Push the trained model to the HuggingFace Hub when the run finishes.
    # All three were undeclared, so upload_to_hf_if_enabled() (main.py:89) read
    # upload_to_hf as absent, hit `if not ...: return` on its first line, and the
    # ENTIRE feature was unreachable — `--upload_to_hf true` was a silent no-op
    # and the job still reported COMPLETED.
    upload_to_hf: bool = False
    hf_repo_id: str = ""
    hf_upload_token: str = ""

    # Seed for dataset packing/sampling and the replay RNG
    # (load_language_model_dataset.py). Undeclared, so it was pinned to its 42
    # fallback and could not be swept. Distinct from `seed` above, which seeds
    # torch/numpy — these are deliberately separate knobs.
    sampling_seed: int = 42

    # Checkpointing
    steps_per_checkpoint: int = 100
    max_checkpoints_to_keep: int = 3

    gpus: int = 1
    nodes: int = 1

    # Adapters
    adapter_type: str = "tokenformer"
    lora_config: Optional[LoraConfig] = LoraConfig()

    # DiffusionGemma-only canvas denoising config (nested, like lora_config).
    # None for every non-diffusion model; is_diffusion() gates whether it is read.
    diffusion: Optional[DiffusionConfig] = None

    # 4 hours in seconds
    timeout: int = 4 * 60 * 60

    training_history_length: int = 1024

    # Override the global cray-config.yaml `dtype` for this job only.
    # "auto" defers to the global config (which itself defaults to
    # "auto" = the model's native dtype). Other values: "float32",
    # "float16", "bfloat16". Per-job control matters on CPU-on-Apple-
    # Silicon where bf16 matmuls SIGILL under Apple's hypervisor while
    # fp32 paths run fine — operators can pass `dtype: float32` in
    # train_args without changing the running deployment's config.
    dtype: str = "auto"

    # Opt-in per job: let AutoConfig/AutoTokenizer run a repo's custom modeling
    # code. Required to TRAIN models that ship it (InternVL3, Molmo, GLM-4).
    # Defaults False. MUST be declared or pydantic drops it.
    trust_remote_code: bool = False

