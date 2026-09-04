"""Chat SFT with standalone Hugging Face PEFT (LoRA / QLoRA / DoRA / rsLoRA) -- see readme_peft.md."""

import os
import json
import logging
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    p = argparse.ArgumentParser(description="Standalone PEFT (LoRA / QLoRA / DoRA / rsLoRA) chat fine-tuner")

    p.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl", help="Chat JSONL: one {'messages': [...]} per line")
    p.add_argument("--model_name", type=str, required=True, help="HF repo id or local path (tokenizer must have a chat template)")
    p.add_argument("--output_dir", type=str, default="./peft_run", help="Where checkpoints and final_adapter are written")
    p.add_argument("--resume_from_checkpoint", type=str, default="", help="Checkpoint dir to resume from (used only if it exists)")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Max tokens per example; longer rows are dropped, never truncated")
    p.add_argument("--max_samples", type=int, default=None, help="Hard cap on rows loaded (quick smoke runs)")
    p.add_argument("--eval_samples", type=int, default=0, help="Rows held out for eval (0 = no eval split)")
    p.add_argument("--mask_prompt", action="store_true", default=True, help="Completion-only loss: supervise assistant turns only (default on)")
    p.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false", help="Train on the full rendered sequence instead")
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (scaling); 2x rank is a good default")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str, default="all-linear", help="'all-linear' (QLoRA-style) or a comma-separated module list")
    p.add_argument("--modules_to_save", type=str, default=None, help="Comma-separated extra modules trained in full and saved with the adapter (e.g. embed_tokens,lm_head)")
    p.add_argument("--use_dora", action="store_true", help="DoRA: decompose the update into magnitude + direction (better at low rank, slower)")
    p.add_argument("--use_rslora", action="store_true", help="rsLoRA: scale by lora_alpha/sqrt(r) instead of lora_alpha/r (stabler at high rank)")
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA: load the frozen base in 4-bit nf4 via bitsandbytes")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"], help="4-bit data type; nf4 is the QLoRA default")
    p.add_argument("--no_double_quant", dest="double_quant", action="store_false", default=True, help="Disable nested (double) quantization of the quantization constants")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device train/eval batch size")
    p.add_argument("--grad_acc_steps", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Peak LR (LoRA/QLoRA typically 1e-4..2e-4)")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    p.add_argument("--warmup_ratio", type=float, default=0.03, help="Fraction of total steps spent warming up")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--optim", type=str, default="adamw_torch", help="HF optimizer id (e.g. adamw_torch, paged_adamw_8bit for QLoRA)")
    p.add_argument("--logging_steps", type=int, default=10, help="Log training metrics every N steps")
    p.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Recompute activations to save memory")
    p.add_argument("--ddp_find_unused_parameters", action="store_true", help="Multi-GPU (DDP) only: allow parameters that produce no gradient this step. Required when 'all-linear' puts adapters on submodules a text-only batch never runs (e.g. the vision/audio towers of a multimodal base). Costs a little DDP overhead.")
    p.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"], help="Attention kernel")

    return p.parse_args()


def build_example(messages, tokenizer, mask_prompt):
    """Render one conversation with the chat template; mask non-assistant tokens to -100 unless mask_prompt is off."""
    # return_dict=False: transformers v5 returns a BatchEncoding by default; we need the flat id list.
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

    load_kwargs = dict(
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        token=hf_token,
    )
    if args.load_in_4bit:
        # QLoRA: the frozen base is 4-bit, the adapter stays bf16.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if RANK == 0:
            logger.info("QLoRA: loading base in 4-bit (%s, double_quant=%s)", args.bnb_4bit_quant_type, args.double_quant)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.load_in_4bit:
        # Upcasts norms and lets gradients flow through the frozen 4-bit base.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    targets = args.lora_target_modules
    if targets != "all-linear":
        targets = [t.strip() for t in targets.split(",") if t.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=[m.strip() for m in args.modules_to_save.split(",")] if args.modules_to_save else None,
        use_dora=args.use_dora,
        use_rslora=args.use_rslora,
    )
    model = get_peft_model(model, lora_config)
    if RANK == 0:
        logger.info("Adapter: r=%d alpha=%d dropout=%.3f targets=%s dora=%s rslora=%s",
                    args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
                    args.use_dora, args.use_rslora)
        model.print_trainable_parameters()

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
        optim=args.optim,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="tensorboard",
        seed=args.seed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        log_on_each_node=False,
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

    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    if RANK == 0:
        tokenizer.save_pretrained(final_dir)
        logger.info("Saved LoRA adapter to %s (base model: %s)", final_dir, args.model_name)
        logger.info("Merge it into base weights with: python merge_adapter.py --base_model %s "
                    "--adapter %s --output_dir <merged_dir>", args.model_name, final_dir)
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
