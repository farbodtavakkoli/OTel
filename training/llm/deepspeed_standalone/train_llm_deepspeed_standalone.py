"""Standalone single-file SFT trainer (HF Transformers + DeepSpeed ZeRO, full fine-tuning); see readme_standalone.md."""

import os
import logging
import shutil
import argparse
import torch
import torch.distributed as dist
import deepspeed
from datetime import datetime, timezone, timedelta
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from trl import SFTTrainer
from dotenv import load_dotenv

# dev.env (in this folder) supplies HF_TOKEN for gated-model downloads.
load_dotenv('dev.env')

if os.getenv('HF_TOKEN'):
    os.environ["HF_TOKEN"] = os.getenv('HF_TOKEN')
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
# CUDA toolkit paths for DeepSpeed JIT builds; a pre-set CUDA_HOME wins.
cuda_path = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ["PATH"] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone DeepSpeed SFT fine-tuner")

    # Paths & model
    parser.add_argument("--train_file", type=str, default="OTel_LLM_sample_10.jsonl", help="Path to training data (JSONL)")
    parser.add_argument("--model_name", type=str, default="LiquidAI/LFM2.5-1.2B-Instruct", help="HF model id or local path to fine-tune")
    parser.add_argument("--model_type", type=str, default="lfm_ftaas", help="Prompt template selector in format_conversation (qwen3, llama3, gemma3, gemma-4, mistral, olmo3, rnj-1, lfm, lfm_ftaas, phi4, gpt-oss_reasoning, gpt_oss_it)")
    parser.add_argument("--experiment_root", type=str, default="experiments/", help="Root dir under which run outputs are created")
    parser.add_argument("--hf_home", type=str, default="hf_cache/", help="HF_HOME cache root for models/datasets")
    parser.add_argument("--output_subdir", type=str, default="lfm_ftaas_uc524_finetuned", help="Subdirectory name appended under experiment_root/<run_id>")
    parser.add_argument("--output_dir", type=str, default=None, help="Full output dir override (skips experiment_root/run_id/output_subdir construction)")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="Checkpoint path to resume from (only used if it exists)")

    # Data selection & eval sizing
    parser.add_argument("--max_token_length", type=int, default=16192, help="Tokenizer max length; longer examples are truncated")
    parser.add_argument("--test_mode", action="store_true", help="Limit the dataset to test_mode_count rows for a quick verification run")
    parser.add_argument("--test_mode_count", type=int, default=10000, help="Row cap applied when --test_mode is set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for set_seed and dataset shuffling/splitting")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of dataset to use (seeded shuffle + subsample)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load")
    parser.add_argument("--test_size", type=float, default=0.0002, help="Eval split: float fraction of rows held out")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Cap on eval rows (None = no cap)")
    parser.add_argument("--num_proc", type=int, default=8, help="Worker processes for dataset map/filter steps")
    parser.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false", help="Train on the full sequence instead of completion-only loss (default: masked)")

    # Trainer / hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train/eval batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--optim", type=str, default="adamw_bnb_8bit", help="HF optim identifier")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log training metrics every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
    parser.add_argument("--no_load_best_model_at_end", dest="load_best_model_at_end", action="store_false", help="Keep the last checkpoint instead of the lowest-eval-loss one (default: load best)")
    parser.add_argument("--no_save", action="store_true", help="Disable all checkpointing and the final model save (smoke tests: a ZeRO-3 checkpoint of an 8B model is >100GB)")

    # DeepSpeed / sharding
    parser.add_argument("--zero_stage", type=int, default=3, choices=[0, 1, 2, 3], help="DeepSpeed ZeRO stage (3 = full param+grad+optimizer sharding)")
    parser.add_argument("--offload_optimizer", action="store_true", help="Offload optimizer state to CPU pinned memory (more GPU headroom, slower)")

    return parser.parse_args()


def build_deepspeed_config(zero_stage=3, offload_optimizer=False):
    """Build an in-memory DeepSpeed config dict for TrainingArguments(deepspeed=...)."""
    offload_device = "cpu" if offload_optimizer else "none"
    zero_opt = {
        "stage": zero_stage,
        "offload_optimizer": {"device": offload_device, "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if zero_stage == 3:
        zero_opt.update({
            "offload_param": {"device": "none", "pin_memory": True},
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        })
    return {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": zero_opt,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


def format_conversation(example, model_type="gemma3"):
    """Render one row into a single training string using the hand-written template for model_type."""
    prompt = str(example.get('prompt', '')) or ""
    completion = str(example.get('completion', '')) or ""
    reasoning = (example.get('reasoning', '')) or ""
    system_prompt = str(example.get('system', '')) or ""
    user_prompt = str(example.get('user', '')) or ""
    assistant_response = str(example.get('assistant', '')) or ""

    if model_type.lower() == "qwen3" or "qwen" in model_type.lower():
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"

    elif model_type.lower() == "llama3":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}<|eot_id|>"

    elif "gemma-4" in model_type.lower() or "gemma4" in model_type.lower():
        # Gemma 4 format: new turn tokens <|turn> / <turn|> and role name "model".
        # <bos> is added automatically by the tokenizer (add_bos=True for gemma), so it is NOT included here.
        return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n{completion}<turn|>\n"

    elif model_type.lower() == "gemma3" or "gemma" in model_type.lower():
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn>"

    elif model_type.lower() == "rnj-1" or "rnj" in model_type.lower():
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>"

    elif model_type.lower() == "olmo3" or "olmo" in model_type.lower():
        return f"<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|endoftext|>"

    elif model_type.lower() == "mistral":
        return f"<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]{prompt}[/INST]{completion}</s>"

    elif "lfm_ftaas" in model_type.lower():
        return f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_response}<|im_end|>"

    elif "lfm" in model_type.lower():
        return f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>\n"
    
    elif "phi4" in model_type.lower():
        # Phi-4 format: <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>
        return f"<|user|>{prompt}<|end|><|assistant|>{completion}<|end|>"

    elif model_type.lower() == "gpt-oss_reasoning":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nreasoning_language: English\n\nYou are an Open Source Telecom Question answering model that uses provided contexts to answer telecom questions.\n\n<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"

    elif model_type.lower() == "gpt_oss_it":
        return f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{completion}<|return|>"

    else:
        return f"User: {prompt}\nAssistant: {completion}"

def process_and_format(example, tokenizer=None, model_type="gemma3", mask_user_prompt=False):
    """Tokenize one formatted row, optionally masking the prompt prefix so loss falls on the completion only."""
    text = format_conversation(example, model_type=model_type)
    if tokenizer is not None:
        tokenized = tokenizer(text, truncation=True, max_length=tokenizer.model_max_length)
        input_ids = tokenized["input_ids"]

        if mask_user_prompt:
            # Re-render a prompt-only version and mask its matching token prefix
            prompt_only_example = {**example, "completion": "", "reasoning": ""}
            if model_type == "lfm_ftaas":
                prompt_only_example['assistant'] = ""
            prompt_text = format_conversation(prompt_only_example, model_type=model_type)

            prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=tokenizer.model_max_length)
            prompt_ids = prompt_tokenized["input_ids"]

            split_idx = 0
            for i in range(min(len(prompt_ids), len(input_ids))):
                if prompt_ids[i] == input_ids[i]:
                    split_idx = i + 1
                else:
                    break

            labels = [-100] * len(input_ids)
            for i in range(split_idx, len(input_ids)):
                labels[i] = input_ids[i]
                
            tokenized["labels"] = labels
        else:
            tokenized["labels"] = list(input_ids)

        return tokenized
    example['text'] = text
    return example

def _flatten_messages(example):
    """Flatten a `messages` row to flat {system, user, assistant} fields by role, dropping extra columns."""
    messages = example.get('messages', []) or []
    out = {"system": "", "user": "", "assistant": ""}
    for msg in messages:
        role = msg.get('role')
        if role in out:
            out[role] = msg.get('content', '') or ""
    return out


def get_datasets(path, tokenizer, model_type, max_token_length, seed, test_mode_count,
                 max_eval_samples=None, test_size=0.01, sample_fraction=1.0, max_samples=None,
                 test_mode=False, mask_user_prompt=True, num_proc=8):
    """Load, format, tokenize and split a JSONL dataset; returns (train, eval)."""
    full_ds = load_dataset("json", data_files=path, split="train")
    initial_count = len(full_ds)

    if model_type == "lfm_ftaas":
        # remove_columns drops all original columns (incl. extra metadata ones)
        full_ds = full_ds.map(
            _flatten_messages,
            remove_columns=full_ds.column_names,
            num_proc=num_proc,
            desc="Flattening messages",
        )

        full_ds = full_ds.filter(
            lambda x: len(x.get('system', '')) + len(x.get('user', '')) + len(x.get('assistant', '')) < max_token_length * 4,
            num_proc=num_proc,
            desc="Filtering by char length",
        )
        logging.info(f"Filtered dataset from {initial_count} to {len(full_ds)} samples based on character length.")

    else:
        full_ds = full_ds.filter(
            lambda x: len(x.get('prompt', '')) + len(x.get('completion', '')) < max_token_length * 4,
            num_proc=num_proc,
            desc="Filtering by char length",
        )
        logging.info(f"Filtered dataset from {initial_count} to {len(full_ds)} samples based on character length.")

    if test_mode:
        logging.info("TEST MODE: Limiting dataset for verification")
        full_ds = full_ds.select(range(min(test_mode_count, len(full_ds))))

    if sample_fraction < 1.0:
        sample_size = int(len(full_ds) * sample_fraction)
        logging.info(f"Subsampling dataset to {sample_fraction:.2%} ({sample_size} samples)")
        full_ds = full_ds.shuffle(seed=seed).select(range(sample_size))

    if max_samples is not None and max_samples < len(full_ds):
        logging.info(f"Limiting dataset to {max_samples} samples")
        full_ds = full_ds.select(range(max_samples))

    full_ds = full_ds.map(
        lambda x: process_and_format(x, tokenizer=tokenizer, model_type=model_type, mask_user_prompt=mask_user_prompt),
        remove_columns=full_ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    logging.info("Loaded %d training data from %s", len(full_ds), path)

    if max_eval_samples is not None:
        num_test = min(max_eval_samples, int(len(full_ds) * test_size))
        logging.info(f"Using {num_test} samples for evaluation")
    else:
        num_test = test_size

    split_ds = full_ds.train_test_split(test_size=num_test, seed=seed)
    return split_ds["train"], split_ds["test"]

def main():
    args = parse_args()

    set_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = args.hf_home

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

    # Logging: rank 0 at INFO, other ranks at WARNING
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[rank=0] %(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.WARNING)

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
            # Only Gemma needs add_bos=True; other templates add BOS in format_conversation
            add_bos=True if "gemma3" in args.model_type.lower() else False,
        )

    except Exception as e:
        if rank == 0:
            logging.error(f"Tokenizer load failed: {e}")
        raise

    tokenizer.model_max_length = args.max_token_length
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if rank == 0:
            logging.info("Attempting to load model with AutoModelForCausalLM...")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
        )

    except Exception as e:
        if rank == 0:
            logging.warning(f"AutoModel load failed: {e}. Falling back to an explicit model class.")

        # Fallback for Mistral, which needs an explicit model class
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
                args.model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map=None
            )

        if hasattr(model, "is_quantized"):
            model.is_quantized = False

    model.config.pad_token_id = tokenizer.pad_token_id

    # GenerationConfig validity: if temperature/top_p are set, do_sample must be True
    if hasattr(model, "generation_config"):
        if model.generation_config.temperature is not None or model.generation_config.top_p is not None:
            model.generation_config.do_sample = True

    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    from accelerate import PartialState
    with PartialState().main_process_first():
        train_dataset, eval_dataset = get_datasets(
            args.train_file,
            tokenizer=tokenizer,
            model_type=args.model_type,
            max_token_length=args.max_token_length,
            seed=args.seed,
            test_mode_count=args.test_mode_count,
            max_eval_samples=args.max_eval_samples,
            test_size=args.test_size,
            sample_fraction=args.sample_fraction,
            max_samples=args.max_samples,
            test_mode=args.test_mode,
            mask_user_prompt=args.mask_prompt,
            num_proc=args.num_proc,
        )

    if rank == 0:
        logging.info("--- Tokenization Sanity Check ---")
        sample_ids = train_dataset[0]['input_ids']
        sample_labels = train_dataset[0]['labels']
        decoded = tokenizer.decode(sample_ids)

        logging.info(f"Decoded text (first 100 chars): {decoded[:100]}...")
        logging.info(f"First 10 tokens: {sample_ids[:10]}")
        logging.info(f"First 10 labels: {sample_labels[:10]}")
        logging.info(f"Last 5 tokens: {sample_ids[-5:]}")

        # Verify if BOS is duplicated
        if len(sample_ids) > 1 and sample_ids[0] == sample_ids[1] == tokenizer.bos_token_id:
            logging.warning("Duplicate BOS token detected!")

        logging.info("--- End Sanity Check ---")

    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optim,
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="no" if args.no_save else "epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False if args.no_save else args.load_best_model_at_end,
        report_to="tensorboard",
        ddp_timeout=7200,
        deepspeed=build_deepspeed_config(zero_stage=args.zero_stage, offload_optimizer=args.offload_optimizer),
        ddp_find_unused_parameters=False,
    )

    # transformers 5.x removed the push_to_hub_token field, but trl 0.24.0's SFTTrainer
    # still does to_dict().pop("push_to_hub_token"). Setting the attribute is not enough
    # (to_dict() only emits dataclass fields), so shim the dict output as well.
    if not hasattr(train_args, 'push_to_hub_token'):
        train_args.push_to_hub_token = None
    if "push_to_hub_token" not in train_args.to_dict():
        _orig_to_dict = train_args.to_dict
        train_args.to_dict = lambda: {**_orig_to_dict(), "push_to_hub_token": None}


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=train_args,
        processing_class=tokenizer,
    )

    if rank == 0:
        logging.info("Starting training...")

    resume_path = args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None
    if rank == 0:
        logging.info(f"Resuming from checkpoint: {resume_path}" if resume_path
                     else "Starting training from scratch...")

    trainer.train(resume_from_checkpoint=resume_path)

    # Final save
    trainer.accelerator.wait_for_everyone()

    if args.no_save:
        if rank == 0:
            logging.info("--no_save set: skipping checkpoint and final model save.")
    else:
        if rank == 0:
            logging.info(f"Saving final model to {output_dir}")
        # save_model must run on ALL ranks: under ZeRO-3 the 16-bit weight gather is a
        # collective op, and a rank-0-only call deadlocks rank 0 against exited peers.
        # (Trainer still writes files only from the main process.)
        trainer.save_model(os.path.join(output_dir, "final_model"))
    if rank == 0:
        shutil.copyfile(__file__, os.path.join(output_dir, "train_script_backup.py"))
        logging.info(f"Training Complete.")

if __name__ == "__main__":
    main() 