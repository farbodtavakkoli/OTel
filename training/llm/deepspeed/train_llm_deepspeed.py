"""Generic post-training (SFT / DPO / GRPO) for `messages` models with HF Transformers + DeepSpeed ZeRO; see readme_deepspeed.md."""

import os
# Must be set before `import torch` (read by c10 at import time).
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
import logging
import shutil
import argparse
import warnings
import torch
import torch.distributed as dist
import deepspeed
import datasets
from datetime import datetime, timezone, timedelta
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from transformers.utils import logging as hf_logging
from trl import SFTTrainer
from dotenv import load_dotenv

# DPO/GRPO trainers are imported lazily inside their branches.
from utils import (get_datasets, _log_supervision_sanity, _stable_causal_lm_loss,
                   build_deepspeed_config, build_pref_dataset, build_grpo_prompts)

# Optional callbacks: absent from an older utils.py, the flags become no-ops.
try:
    from utils import CustomEvalCallback
except ImportError:
    CustomEvalCallback = None
try:
    from utils import EmptyCacheCallback
except ImportError:
    EmptyCacheCallback = None

# dev.env (in this folder) supplies HF_TOKEN for gated-model downloads.
load_dotenv('dev.env')

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# CUDA toolkit paths for DeepSpeed JIT builds; a pre-set CUDA_HOME wins.
cuda_path = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ["PATH"] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"


def parse_args():
    parser = argparse.ArgumentParser(description="Generic messages-format fine-tuner")

    # Paths & model
    parser.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl", help="Path to training data (canonical messages JSONL)")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-E4B-it", help="HF model id or local path to fine-tune")
    parser.add_argument("--experiment_root", type=str, default="experiments/", help="Root dir under which run outputs are created")
    parser.add_argument("--hf_home", type=str, default="hf_cache/", help="HF_HOME cache root for models/datasets")
    parser.add_argument("--output_subdir", type=str, default="gemma4_ftaas_uc524_finetuned", help="Subdirectory name appended under experiment_root/<run_id>")
    parser.add_argument("--output_dir", type=str, default=None, help="Full output dir override (skips experiment_root/run_id/output_subdir construction)")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Checkpoint path to resume from (only used if it exists)")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False,
                        help="Load the lowest-eval-loss checkpoint as the final model (off by default; see README)")

    # Data selection & eval sizing
    parser.add_argument("--max_token_length", type=int, default=32768, help="Max tokens per example; longer examples are dropped (never truncated)")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Target absolute eval-set size (floored at world_size for distributed eval)")
    parser.add_argument("--preflight_sample_size", type=int, default=256, help="Number of rows to validate in the fail-fast messages preflight")
    parser.add_argument("--supervision_sample_size", type=int, default=16, help="Number of rows to sample for the supervision sanity check")
    parser.add_argument("--test_mode", action="store_true", help="Limit the dataset to test_mode_count rows for a quick verification run")
    parser.add_argument("--test_mode_count", type=int, default=10000, help="Row cap applied when --test_mode is set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for set_seed and dataset shuffling/splitting")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of the dataset to use (seeded shuffle + subsample)")
    parser.add_argument("--max_samples", type=int, default=None, help="Hard cap on number of samples to load")
    parser.add_argument("--num_proc", type=int, default=8, help="Worker processes for dataset map/filter steps")

    # Trainer / hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train/eval batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs (memorization: start 3, extend to ~8 by the per-epoch eval curve)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate (LoRA default 2e-4; use ~1e-5..2e-5 for full FT)")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=float, default=0.03, help="LR warmup: a value <1 is a fraction of total steps (scales with corpus size); >=1 is an absolute step count")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log training metrics every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd", "rmsprop", "adamw_bnb_8bit"], help="Optimizer (adamw -> adamw_torch, DeepSpeed-native; avoids the bnb-8bit/ZeRO-3 sharp edge)")
    parser.add_argument("--mask_prompt", action="store_true", help="Mask the user prompt during training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--flash_attention", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa"], help="Enable flash attention if available")

    # LoRA / PEFT (default off: full fine-tuning)
    parser.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha (scaling); typically 2x rank")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear", help="PEFT target modules: 'all-linear' or a comma-separated list")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="QLoRA: load the base model in 4-bit (nf4); requires --use_lora and --zero_stage 0")

    # Custom per-epoch eval (pluggable scorer over test/<name>_eval.jsonl)
    parser.add_argument("--custom_eval", action="store_true",
                        help="Run a generation-based eval each epoch using --scorer_module over --test_dir")
    parser.add_argument("--test_dir", type=str, default="test", help="Dir with <name>_eval.jsonl files for --custom_eval")
    parser.add_argument("--scorer_module", type=str, default="step8_score_eval", help="Importable module exposing EVAL_DATASETS/load_eval/score_dataset/macro_average")
    parser.add_argument("--eval_max_new_tokens", type=int, default=768, help="Max new tokens generated per prompt during custom eval")
    parser.add_argument("--empty_cache_steps", type=int, default=0,
                        help="Flush the CUDA cache every N optimizer steps on all ranks (0 = off)")

    # DeepSpeed / sharding
    parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO stage (3 = full sharding; 2 recommended for LoRA; see README)")
    parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer state to CPU pinned memory (more GPU headroom, slower)")

    # Training mode: SFT (default) | DPO | GRPO
    parser.add_argument("--train_mode", type=str, default="sft", choices=["sft", "dpo", "grpo"],
                        help="sft = supervised fine-tuning; dpo = preference optimization on --pref_file; grpo = online RL")
    parser.add_argument("--init_adapter", type=str, default=None,
                        help="Existing LoRA adapter to continue training from (with --use_lora)")
    # DPO
    parser.add_argument("--pref_file", type=str, default=None,
                        help="DPO preference JSONL with fields {system?, prompt, chosen, rejected}; required for --train_mode dpo")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta (KL strength; lower = stays closer to the reference policy)")
    parser.add_argument("--dpo_max_length", type=int, default=4096, help="DPO max total sequence length (prompt+completion)")
    parser.add_argument("--dpo_max_prompt_length", type=int, default=3072, help="DPO max prompt length")
    parser.add_argument("--dpo_precompute_ref_log_probs", action="store_true",
                        help="Precompute and cache reference log-probs, then drop the ref model from the loop (saves memory)")
    # GRPO
    parser.add_argument("--grpo_num_generations", type=int, default=8, help="GRPO: candidate generations sampled per prompt (group size)")
    parser.add_argument("--grpo_max_completion_length", type=int, default=512, help="GRPO: max new tokens per generated candidate")
    parser.add_argument("--grpo_temperature", type=float, default=1.0, help="GRPO sampling temperature for candidate generation")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = args.hf_home

    # Fail fast: Gemma-4's head_dim exceeds FlashAttention-2's 256 limit.
    if args.flash_attention == "flash_attention_2":
        raise SystemExit(
            "flash_attention_2 is unusable for this model: Gemma-4-31B's attention "
            "head_dim exceeds FlashAttention's 256 limit. Use --flash_attention sdpa."
        )

    # Bypass proxies that can 403 huggingface.co.
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(proxy_var, None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # 2h timeout tolerates the slow model AllGather under ZeRO-3.
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
            device_id=torch.device(f"cuda:{local_rank}")
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Logging: rank 0 at INFO, other ranks at WARNING
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[rank=0] %(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Confine transformers/datasets logs and progress bars to rank 0
    if rank == 0:
        hf_logging.set_verbosity_warning()
        datasets.logging.set_verbosity_warning()
    else:
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bar()
        warnings.filterwarnings("ignore")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_id = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        output_dir = os.path.join(args.experiment_root, run_id, args.output_subdir)

    if rank == 0:
        print(f"\n" + "="*50)
        print(f"CHECKPOINT SAVING LOCATION: {output_dir}")
        print("="*50 + "\n")
        logging.info(f"Checkpoints will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    # Model and tokenizer loading
    if rank == 0:
        logging.info(f"Loading tokenizer and model: {args.model_name}")

    try:
        if rank == 0:
            logging.info("Attempting to load tokenizer with AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=True,
            trust_remote_code=True,
        )

    except Exception as e:
        if rank == 0:
            logging.error(f"Tokenizer load failed: {e}")
        raise

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {args.model_name} has no chat_template. This trainer requires a "
            "tokenizer with an official chat template to guarantee train/inference parity."
        )

    tokenizer.model_max_length = args.max_token_length
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA: 4-bit params can't be sharded by ZeRO-3, so guard early
    quantization_config = None
    if args.load_in_4bit:
        if not args.use_lora:
            raise ValueError("--load_in_4bit requires --use_lora (QLoRA).")
        if args.zero_stage == 3:
            raise ValueError("--load_in_4bit is incompatible with ZeRO-3 (4-bit params "
                             "aren't shardable). Use --zero_stage 0.")
        from utils import build_qlora_bnb_config
        quantization_config = build_qlora_bnb_config()
        if rank == 0:
            logging.info("QLoRA: loading base model in 4-bit (nf4).")

    model_load_kwargs = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.flash_attention,
        device_map=None,
    )
    # Never pass an explicit quantization_config=None (crashes some versions)
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config

    try:
        if rank == 0:
            logging.info("Attempting to load model with AutoModelForCausalLM...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_load_kwargs,
        )

    except Exception as e:
        if rank == 0:
            logging.exception(
                f"AutoModel first load attempt failed: {e}. "
                "Retrying with a conservative AutoModel fallback."
            )
        else:
            logging.warning("AutoModel first load attempt failed. Retrying with fallback settings.")

        # Fallback: drop attn_implementation, which some checkpoints reject
        fallback_kwargs = dict(model_load_kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **fallback_kwargs)

        if hasattr(model, "is_quantized"):
            model.is_quantized = False

    if rank == 0:
        logging.info(f"Model weights loaded on all {world_size} ranks.")

    model.config.pad_token_id = tokenizer.pad_token_id

    # GenerationConfig validity: if temperature/top_p are set, do_sample must be True
    if hasattr(model, "generation_config"):
        if model.generation_config.temperature is not None or model.generation_config.top_p is not None:
            model.generation_config.do_sample = True

    # Compat shim: transformers 5.x removed PreTrainedModel.warnings_issued, but
    # TRL 0.24.0's DPOTrainer/GRPOTrainer still write to it in __init__.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # QLoRA: prepare the frozen 4-bit base so gradients flow into the adapters
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # peft_config=None means full FT; with --init_adapter the existing adapter
    # is attached here so training continues on it instead of a fresh one
    peft_config = None
    if args.use_lora and args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        if rank == 0:
            logging.info(f"Loaded existing adapter to continue training: {args.init_adapter}")
    elif args.use_lora:
        from peft import LoraConfig
        target_modules = args.lora_target_modules
        if target_modules != "all-linear" and "," in target_modules:
            target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        if rank == 0:
            logging.info(f"LoRA enabled: r={args.lora_r} alpha={args.lora_alpha} "
                         f"dropout={args.lora_dropout} targets={args.lora_target_modules}")
    else:
        # Full fine-tuning: freeze the vision tower, train the language params
        frozen_params = 0
        trainable_params = 0
        vision_keywords = ["vision_model", "vision_tower", "multi_modal_projector", "visual"]
        for name, param in model.named_parameters():
            if any(key in name for key in vision_keywords):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
                trainable_params += param.numel()

        if rank == 0:
            logging.info(f"Selective Freezing: Frozen {frozen_params/1e6:.1f}M params (Vision), "
                        f"Trainable {trainable_params/1e6:.1f}M params (Language)")

    # Map friendly optimizer names to HF optim identifiers
    optim_map = {"adamw": "adamw_torch", "sgd": "sgd", "rmsprop": "rmsprop",
                 "adamw_bnb_8bit": "adamw_bnb_8bit"}
    hf_optim = optim_map[args.optim]

    # DeepSpeed config built in-memory from the CLI flags (identical on every rank)
    ds_config = build_deepspeed_config(zero_stage=args.zero_stage, offload_optimizer=args.offload_optimizer)
    if rank == 0:
        logging.info(f"Mode={args.train_mode} | DeepSpeed: ZeRO stage={args.zero_stage}, "
                     f"optimizer_offload={'cpu' if args.offload_optimizer else 'none'}")

    # Shared TrainingArguments kwargs (DPOConfig/GRPOConfig subclass TrainingArguments)
    common_targs = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=args.gradient_checkpointing, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        optim=hf_optim,
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        report_to="tensorboard",
        ddp_timeout=7200,
        deepspeed=ds_config,
        ddp_find_unused_parameters=False,
        log_level="info",
        log_level_replica="error",
        log_on_each_node=False,
    )

    # SFT
    if args.train_mode == "sft":
        from accelerate import PartialState
        with PartialState().main_process_first():
            train_dataset, eval_dataset = get_datasets(
                args.train_file,
                tokenizer=tokenizer,
                seed=args.seed,
                max_token_length=args.max_token_length,
                test_mode_count=args.test_mode_count,
                preflight_sample_size=args.preflight_sample_size,
                max_eval_samples=None,
                test_size=max(args.eval_samples, world_size),
                sample_fraction=args.sample_fraction,
                max_samples=args.max_samples,
                test_mode=args.test_mode,
                mask_user_prompt=args.mask_prompt,
                num_proc=args.num_proc,
                min_eval_samples=world_size,
            )

        if len(train_dataset) == 0:
            raise ValueError("Training split is empty after preprocessing and splitting.")
        if len(eval_dataset) < world_size:
            raise ValueError(
                f"Eval split must contain at least one row per rank. "
                f"Found {len(eval_dataset)} eval rows for world size {world_size}.")

        if rank == 0:
            logging.info("--- Tokenization Sanity Check ---")
            _log_supervision_sanity(train_dataset, "train", tokenizer, sample_size=args.supervision_sample_size, decode_example=True)
            _log_supervision_sanity(eval_dataset, "eval", tokenizer, sample_size=args.supervision_sample_size, decode_example=False)
            sample_n = min(2000, len(train_dataset))
            lengths = [len(train_dataset[i]['input_ids']) for i in range(sample_n)]
            over = sum(1 for L in lengths if L > args.max_token_length)
            logging.info(f"Token lengths over first {sample_n}: min={min(lengths)} "
                         f"mean={sum(lengths)/len(lengths):.0f} max={max(lengths)} | "
                         f">{args.max_token_length}: {over} (expected 0)")
            sample_ids = train_dataset[0]['input_ids']
            if len(sample_ids) > 1 and sample_ids[0] == sample_ids[1] == tokenizer.bos_token_id:
                logging.warning("Duplicate BOS token detected!")
            logging.info("--- End Sanity Check ---")

        # loss_type="nll" honors compute_loss_func; max_length=None / packing=False
        # keep the pre-tokenized, length-filtered tensors as-is (see README)
        from trl import SFTConfig
        train_args = SFTConfig(
            eval_strategy="epoch",
            load_best_model_at_end=args.load_best_model_at_end,
            loss_type="nll",
            max_length=None,
            packing=False,
            **common_targs,
        )
        if not hasattr(train_args, 'push_to_hub_token'):
            train_args.push_to_hub_token = None
        sft_callbacks = []
        if args.empty_cache_steps and EmptyCacheCallback is not None:
            sft_callbacks.append(EmptyCacheCallback(every_n_steps=args.empty_cache_steps))
            if rank == 0:
                logging.info(f"EmptyCacheCallback: flushing CUDA cache every "
                             f"{args.empty_cache_steps} step(s) on all ranks.")
        if args.custom_eval:
            if CustomEvalCallback is None:
                if rank == 0:
                    logging.warning("--custom_eval requested but utils.CustomEvalCallback "
                                    "is unavailable; continuing WITHOUT custom eval.")
            elif args.zero_stage == 3:
                # A rank-0-only generate() would hang on the ZeRO-3 param all-gather
                if rank == 0:
                    logging.warning("--custom_eval is disabled with ZeRO-3 (rank-0 generation "
                                    "would hang on the sharded-param all-gather). Use --zero_stage 2 "
                                    "for in-loop custom eval, or eval the saved adapter offline.")
            else:
                sft_callbacks.append(CustomEvalCallback(
                    tokenizer=tokenizer, test_dir=args.test_dir,
                    scorer_module=args.scorer_module,
                    max_new_tokens=args.eval_max_new_tokens,
                    batch_size=args.batch_size, rank=rank,
                ))

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=train_args,
            processing_class=tokenizer,
            compute_loss_func=_stable_causal_lm_loss,
            peft_config=peft_config,
            callbacks=sft_callbacks or None
        )

    # DPO
    elif args.train_mode == "dpo":
        from trl import DPOTrainer, DPOConfig
        if not args.pref_file:
            raise ValueError("--train_mode dpo requires --pref_file (JSONL with prompt/chosen/rejected).")
        train_dataset, eval_dataset = build_pref_dataset(
            args.pref_file, tokenizer, seed=args.seed, eval_size=min(args.eval_samples, 200),
            num_proc=args.num_proc,
        )
        if rank == 0:
            logging.info(f"DPO pairs: train={len(train_dataset)} eval={len(eval_dataset)} | "
                         f"beta={args.dpo_beta} max_len={args.dpo_max_length}")
            ex = train_dataset[0]
            logging.info(f"DPO example — prompt[:120]={ex['prompt'][:120]!r}")
            logging.info(f"  chosen[:100]={ex['chosen'][:100]!r}")
            logging.info(f"  rejected[:100]={ex['rejected'][:100]!r}")
        # This TRL version's DPOConfig takes max_length but not max_prompt_length
        dpo_args = DPOConfig(
            beta=args.dpo_beta,
            max_length=args.dpo_max_length,
            precompute_ref_log_probs=args.dpo_precompute_ref_log_probs,
            eval_strategy="epoch",
            **common_targs,
        )
        if not hasattr(dpo_args, 'push_to_hub_token'):
            dpo_args.push_to_hub_token = None
        # ref_model=None: with PEFT the base model is the implicit reference
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    # GRPO
    elif args.train_mode == "grpo":
        from trl import GRPOTrainer, GRPOConfig
        from utils import grpo_reward_funcs
        from accelerate import PartialState
        with PartialState().main_process_first():
            train_dataset, _ = get_datasets(
                args.train_file, tokenizer=tokenizer, seed=args.seed,
                max_token_length=args.max_token_length, test_mode_count=args.test_mode_count,
                preflight_sample_size=args.preflight_sample_size, max_eval_samples=None,
                test_size=max(args.eval_samples, world_size), sample_fraction=args.sample_fraction,
                max_samples=args.max_samples, test_mode=args.test_mode, mask_user_prompt=True,
                num_proc=args.num_proc, min_eval_samples=world_size,
            )
        # GRPO needs a "prompt" column (the chat-rendered prompt) to generate from
        train_dataset = build_grpo_prompts(args.train_file, tokenizer, seed=args.seed,
                                           max_samples=args.max_samples)
        if rank == 0:
            logging.info(f"GRPO prompts: {len(train_dataset)} | num_gen={args.grpo_num_generations} "
                         f"max_completion={args.grpo_max_completion_length}")
        grpo_args = GRPOConfig(
            num_generations=args.grpo_num_generations,
            max_completion_length=args.grpo_max_completion_length,
            temperature=args.grpo_temperature,
            **common_targs,
        )
        if not hasattr(grpo_args, 'push_to_hub_token'):
            grpo_args.push_to_hub_token = None
        trainer = GRPOTrainer(
            model=model,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=grpo_reward_funcs(),
            peft_config=peft_config,
        )

    if rank == 0:
        logging.info("Starting training...")

    resume_path = args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None
    if rank == 0:
        if resume_path:
            logging.info(f"Resuming training from checkpoint: {resume_path}")
        else:
            logging.info("Starting training from scratch...")

    trainer.train(resume_from_checkpoint=resume_path)

    # Final save
    trainer.accelerator.wait_for_everyone()

    # save_model must run on ALL ranks under ZeRO-3 (collective all-gather); the
    # file write itself is guarded to rank 0 internally
    final_model_dir = os.path.join(output_dir, "final_model")
    if rank == 0:
        logging.info(f"Saving final {'LoRA adapter' if args.use_lora else 'model'} to {final_model_dir}")
    trainer.save_model(final_model_dir)

    if rank == 0:
        shutil.copyfile(__file__, os.path.join(output_dir, "train_script_backup.py"))
        if args.use_lora:
            logging.info("Saved a LoRA ADAPTER — inference needs the base model "
                         f"({args.model_name}) + this adapter (PeftModel.from_pretrained).")
        logging.info(f"Training Complete.")

if __name__ == "__main__":
    # Every rank writes its traceback to train_err_rank<N>.log; only rank 0
    # also prints to the console (the failing rank is often not rank 0)
    import sys, traceback
    _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    try:
        main()
    except SystemExit:
        raise  # argparse / fail-fast guards: message already printed once
    except BaseException:
        err_path = os.path.join(os.getcwd(), f"train_err_rank{_rank}.log")
        try:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        if _rank == 0:
            raise  # full traceback to console on rank 0
        sys.stderr.write(f"[rank{_rank}] FAILED — full traceback in {err_path}\n")
        sys.exit(1)
