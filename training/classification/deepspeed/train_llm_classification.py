"""Sequence-classification fine-tuner (HF Transformers + DeepSpeed ZeRO-2) — see readme_classification.md."""
import argparse
import importlib.util
import torch
import sys
import numpy as np
import evaluate
import time
import logging
import warnings
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)
from transformers.utils import logging as hf_logging

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Sequence-classification fine-tuner (DeepSpeed ZeRO-2)")
    parser.add_argument("--model_id", type=str, default="EssentialAI/rnj-1", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="models/rnj-1-classifier",
                        help="Where checkpoints and the best model are written")
    parser.add_argument("--train_file", type=str, default="data/classification_sample.csv",
                        help="Training CSV with a text column (text/question) and a label column (label/answer)")
    parser.add_argument("--test_file", type=str, default="data/classification_sample.csv",
                        help="Evaluation CSV with the same columns as --train_file")
    parser.add_argument("--learning_rate", type=float, default=7e-6, help="Optimizer learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size; effective batch = batch_size * grad_accum * num_GPUs")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=150, help="LR warmup steps")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping norm")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor; softens targets to reduce overconfidence")
    parser.add_argument("--max_length", type=int, default=5000, help="Max input tokens")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--eval_steps", type=int, default=19, help="Eval cadence; tune to your steps-per-epoch")
    parser.add_argument("--save_steps", type=int, default=19, help="Checkpoint cadence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_deepspeed_config(zero_stage=2):
    """Build an in-memory DeepSpeed config dict for TrainingArguments(deepspeed=...)."""
    return {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


# Metrics are loaded individually because they take different arguments
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """Compute accuracy plus macro F1/precision/recall and log the prediction distribution."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Debug logging on rank 0 — show all classes including zeros
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        num_classes = logits.shape[-1]
        dist_dict = {i: int(np.sum(predictions == i)) for i in range(num_classes)}
        logger.info(f"\nPred Distribution (all {num_classes} classes): {dist_dict}")

    acc = accuracy_metric.compute(predictions=predictions, references=labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        prec = precision_metric.compute(predictions=predictions, references=labels, average="macro")
        rec = recall_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        **acc,
        **f1,
        **prec,
        **rec
    }


def load_classification_dataset(train_file, test_file):
    """Load the train/test CSVs and map the label column (answer or label) to integer ids."""
    train_file = Path(train_file)
    test_file = Path(test_file)

    if not train_file.exists():
        raise FileNotFoundError(f"Could not find training file at {train_file}")

    logger.info(f"Loading data from: {train_file}")
    dataset = load_dataset('csv', data_files={'train': str(train_file), 'test': str(test_file)})

    cols = dataset["train"].column_names
    logger.info(f"Columns: {cols}")

    if "answer" in cols:
        target_col = "answer"
    elif "label" in cols:
        target_col = "label"
    else:
        raise ValueError(f"No 'answer' or 'label' column found in {cols}.")

    # Unique labels
    unique_labels = sorted(list(set(dataset["train"][target_col])))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    logger.info(f"Labels: {label2id}")
    
    def map_labels(examples):
        return {"labels": [label2id[l] for l in examples[target_col]]}
        
    dataset = dataset.map(map_labels, batched=True)
    
    # Rename text column if needed for uniformity
    if "question" in cols and "text" not in cols:
        dataset = dataset.rename_column("question", "text")
        
    return dataset["train"], dataset["test"], label2id, id2label

def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info(f"Loading data...")
    train_dataset, eval_dataset, label2id, id2label = load_classification_dataset(args.train_file, args.test_file)

    logger.info(f"Train size: {len(train_dataset)}, Test size: {len(eval_dataset)}")
    
    num_labels = len(label2id)
    # Ensure consistent label ordering
    sorted_labels = sorted(label2id.items(), key=lambda x: x[1])
    logger.info(f"Classes: {num_labels} => {dict(sorted_labels)}")

    logger.info(f"Loading model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Left padding is required for decoder-based sequence classification.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16,
        # flash-attn is CUDA-only as pinned; on ROCm (AMD) it is absent, so fall back to SDPA.
        attn_implementation="flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa",
        trust_remote_code=True
    )
    model = model.to(torch.bfloat16)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # required for gradient checkpointing

    # Classification head: Dropout before the linear to reduce overfitting.
    import torch.nn as nn
    model.score = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(model.config.hidden_size, num_labels, bias=False)
    ).to(torch.bfloat16)
    torch.nn.init.normal_(model.score[1].weight, mean=0.0, std=0.01)
    logger.info(f"Head: Dropout(0.1) -> Linear({model.config.hidden_size}, {num_labels}, bias=False)")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Full fine-tuning: all parameters trainable (single phase, no freeze/unfreeze).
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # Tokenization: raw text, no chat template
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    logger.info("Tokenizing...")
    cols_to_drop = [c for c in train_dataset.column_names if c not in ("labels",)]
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)
    tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        label_smoothing_factor=args.label_smoothing,
        num_train_epochs=args.num_epochs,

        bf16=True,
        # TF32 is an NVIDIA-only tensor format; on ROCm TrainingArguments(tf32=True) raises
        # "--tf32 requires Ampere or a newer GPU arch". Enable it only on CUDA builds.
        tf32=torch.version.cuda is not None,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        deepspeed=build_deepspeed_config(zero_stage=args.zero_stage),

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        logging_steps=1,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[]
    )
    
    logger.info("=== Starting full fine-tuning ===")
    trainer.train()

    # Save the best model + tokenizer.
    folder = Path(args.output_dir)
    trainer.save_model(folder)
    tokenizer.save_pretrained(folder)
    logger.info(f"Best model saved to {folder}")


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting classification training...")
    main()
    end_time = time.time()
    duration = end_time - start_time
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(duration % 60)}s")
 