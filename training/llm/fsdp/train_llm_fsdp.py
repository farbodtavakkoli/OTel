"""Chat SFT on PyTorch-native FSDP2 (fully_shard) via HF Trainer + accelerate -- see readme_fsdp.md."""

import os
import json
import logging
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

RANK = int(os.environ.get("RANK", 0))


def parse_args():
    p = argparse.ArgumentParser(description="FSDP2 (fully_shard) chat fine-tuner")

    p.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl", help="Chat JSONL: one {'messages': [...]} per line")
    p.add_argument("--model_name", type=str, required=True, help="HF repo id or local path (tokenizer must have a chat template)")
    p.add_argument("--output_dir", type=str, default="./fsdp_run", help="Where checkpoints and final_model are written")
    p.add_argument("--resume_from_checkpoint", type=str, default="", help="Checkpoint dir to resume from (used only if it exists)")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Max tokens per example; longer rows are dropped, never truncated")
    p.add_argument("--max_samples", type=int, default=None, help="Hard cap on rows loaded (quick smoke runs)")
    p.add_argument("--eval_samples", type=int, default=0, help="Rows held out for eval (0 = no eval split)")
    p.add_argument("--mask_prompt", action="store_true", default=True, help="Completion-only loss: supervise assistant turns only (default on)")
    p.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false", help="Train on the full rendered sequence instead")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device train/eval batch size")
    p.add_argument("--grad_acc_steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Peak LR (full FT ~1e-5..2e-5; LoRA ~1e-4..2e-4)")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    p.add_argument("--warmup_ratio", type=float, default=0.03, help="Fraction of total steps spent warming up")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N steps")
    p.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    p.add_argument("--no_save", action="store_true", help="Disable all checkpointing AND the final save_model (smoke tests: an 8-way FULL_STATE_DICT of an 8B model is tens of GB per epoch)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Recompute activations to save memory (do NOT combine with FSDP activation_checkpointing)")
    p.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention kernel; sdpa is the portable default (works on CUDA, ROCm and XPU)")
    p.add_argument("--fsdp_config", type=str, default=None, help="Trainer FSDP config JSON (raw torchrun route, e.g. fsdp2_torchrun.json). Leave unset with accelerate launch, which supplies the FSDP2 plugin itself")
    p.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning")
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (scaling)")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str, default="all-linear", help="'all-linear' or a comma-separated module list")

    return p.parse_args()


def build_example(messages, tokenizer, mask_prompt):
    """Render one conversation with the chat template; mask non-assistant tokens to -100 unless mask_prompt is off."""
    # return_dict=False: transformers v5 changed apply_chat_template(tokenize=True) to
    # return a BatchEncoding dict by default; we need the flat token list (v4 behavior).
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)

    if not mask_prompt:
        return {"input_ids": input_ids, "labels": list(input_ids)}

    labels = [-100] * len(input_ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        start = len(tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=True, return_dict=False))
        end = len(tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False, return_dict=False))
        labels[start:end] = input_ids[start:end]
    return {"input_ids": input_ids, "labels": labels}


def load_chat_jsonl(path, tokenizer, max_seq_len, mask_prompt, max_samples=None):
    """Read a chat JSONL, tokenize it, and drop rows that are over-length or unsupervised."""
    rows, dropped_long, dropped_empty = [], 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages = json.loads(line)["messages"]
            example = build_example(messages, tokenizer, mask_prompt)
            if len(example["input_ids"]) > max_seq_len:
                dropped_long += 1
                continue
            if all(label == -100 for label in example["labels"]):
                dropped_empty += 1
                continue
            rows.append(example)
            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No usable rows in {path} (over-length: {dropped_long}, unsupervised: {dropped_empty}).")
    if RANK == 0:
        logger.info(
            "Loaded %d rows from %s (dropped %d over %d tokens, %d with no supervised tokens)",
            len(rows), path, dropped_long, max_seq_len, dropped_empty,
        )
    return rows


def make_collator(pad_token_id):
    """Pad a batch to its longest row: input_ids with pad, labels with -100."""
    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    return collate


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.environ.get("HF_TOKEN")

    if RANK == 0:
        accelerator_type = getattr(getattr(torch, "accelerator", None), "current_accelerator", lambda: None)()
        logger.info("Accelerator: %s | model: %s | LoRA: %s", accelerator_type, args.model_name, args.use_lora)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError(f"{args.model_name} has no chat_template; this trainer requires one for train/inference parity.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_chat_jsonl(args.train_file, tokenizer, args.max_seq_len, args.mask_prompt, args.max_samples)
    eval_dataset = None
    if args.eval_samples > 0 and len(rows) > args.eval_samples:
        eval_dataset = Dataset.from_list(rows[: args.eval_samples])
        rows = rows[args.eval_samples:]
    train_dataset = Dataset.from_list(rows)

    # device_map stays None: FSDP2 places and shards the parameters.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None,
        token=hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        targets = args.lora_target_modules
        if targets != "all-linear":
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ))
        if RANK == 0:
            model.print_trainable_parameters()

    # Set only on the raw-torchrun route; under accelerate launch the plugin comes from the YAML.
    fsdp_kwargs = {"fsdp": True, "fsdp_config": args.fsdp_config} if args.fsdp_config else {}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="no" if args.no_save else "epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="tensorboard",
        seed=args.seed,
        ddp_timeout=7200,
        log_on_each_node=False,
        **fsdp_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=make_collator(tokenizer.pad_token_id),
    )

    resume = args.resume_from_checkpoint if os.path.isdir(args.resume_from_checkpoint) else None
    if RANK == 0:
        logger.info("Starting training%s", f" (resuming from {resume})" if resume else "")
    trainer.train(resume_from_checkpoint=resume)

    # save_model must run on ALL ranks: under FSDP it is a collective. --no_save skips
    # it on ALL ranks (symmetric, so no rank is left waiting in the collective).
    final_dir = os.path.join(args.output_dir, "final_model")
    if not args.no_save:
        trainer.save_model(final_dir)
    trainer.accelerator.wait_for_everyone()

    if args.no_save:
        if RANK == 0:
            logger.info("--no_save set: skipped checkpointing and final save_model.")
            logger.info("Training complete.")
    elif RANK == 0:
        tokenizer.save_pretrained(final_dir)
        if args.use_lora:
            logger.info("Saved a LoRA ADAPTER to %s -- inference needs the base model plus this adapter.", final_dir)
        else:
            logger.info("Saved consolidated weights to %s", final_dir)
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
