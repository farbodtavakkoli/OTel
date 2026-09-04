"""Chat fine-tuning / continued pre-training of an HF causal LM with PyTorch Lightning -- see readme_lightning.md."""

import os
import json
import logging
import argparse

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch Lightning chat fine-tuner for HF causal LMs")

    p.add_argument("--train_file", default="data/OTel_LLM_sample_10.jsonl", help="Chat JSONL: one {'messages': [...]} per line")
    p.add_argument("--model_name", required=True, help="HF repo id or local path (tokenizer must have a chat template)")
    p.add_argument("--output_dir", default="./lightning_run", help="Lightning logs and .ckpt files land here")
    p.add_argument("--export_hf_dir", default="", help="If set, write a Hugging Face folder here after training")
    p.add_argument("--resume_ckpt", default="", help="Lightning .ckpt to resume from (used only if it exists)")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Max tokens per example; longer rows are dropped, never truncated")
    p.add_argument("--max_samples", type=int, default=None, help="Hard cap on rows loaded (quick smoke runs)")
    p.add_argument("--eval_samples", type=int, default=0, help="Rows held out for validation (0 = no val loop)")
    p.add_argument("--mask_prompt", action="store_true", default=True, help="Completion-only loss: assistant turns only (default on)")
    p.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false", help="Train on the full rendered sequence (continued pre-training style)")
    p.add_argument("--num_workers", type=int, default=4, help="Dataloader workers per rank")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device micro-batch size")
    p.add_argument("--grad_acc_steps", type=int, default=8, help="Trainer accumulate_grad_batches")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5, help="Peak LR (full fine-tune ~1e-5..2e-5)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0, help="Global grad-norm clip; 0 disables")
    p.add_argument("--devices", type=int, default=-1, help="GPUs per node; -1 = all visible")
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed", choices=["bf16-mixed", "bf16-true", "16-mixed", "32-true"])
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--strategy", default="fsdp",
                   choices=["fsdp", "deepspeed", "ddp", "ddp_find_unused_parameters_true", "auto"],
                   help="ddp_find_unused_parameters_true is needed for multimodal checkpoints trained on "
                        "text-only data (the vision/audio towers get no gradient and plain DDP errors out)")
    p.add_argument("--sharding_strategy", default="FULL_SHARD", help="FSDP: FULL_SHARD | SHARD_GRAD_OP | NO_SHARD | HYBRID_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
    p.add_argument("--state_dict_type", default="full", choices=["full", "sharded"],
                   help="FSDP: 'full' consolidates on rank 0, 'sharded' writes one file per rank")
    p.add_argument("--zero_stage", type=int, default=3, choices=[1, 2, 3], help="DeepSpeed only")
    p.add_argument("--cpu_offload", action="store_true", help="Offload optimizer (and ZeRO-3 params) to CPU")
    p.add_argument("--activation_checkpointing", action="store_true", help="Recompute decoder-layer activations (FSDP path)")
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Disable Lightning checkpoint writing entirely (smoke runs: a .ckpt of a 8B model "
                        "with AdamW state is ~100 GB and Lightning writes one every epoch)")
    return p.parse_args()


def build_example(messages, tokenizer, mask_prompt):
    """Render one conversation with the chat template; mask non-assistant tokens to -100 unless mask_prompt is off."""
    # return_dict=False: transformers >= 5.x returns a BatchEncoding from
    # apply_chat_template(tokenize=True) by default; we need the flat token list.
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
            example = build_example(json.loads(line)["messages"], tokenizer, mask_prompt)
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
    logger.info("Loaded %d rows from %s (dropped %d over %d tokens, %d with no supervised tokens)",
                len(rows), path, dropped_long, max_seq_len, dropped_empty)
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


class LitCausalLM(L.LightningModule):
    """The whole model definition: forward, loss, optimizer, LR schedule."""

    def __init__(self, model_name, learning_rate, weight_decay, warmup_ratio,
                 attn_implementation, hf_token=None):
        super().__init__()
        # hf_token is excluded so it is never serialized into a .ckpt.
        self.save_hyperparameters(ignore=["hf_token"])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation=attn_implementation,
            device_map=None, token=hf_token)
        self.model.config.use_cache = False

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay, betas=(0.9, 0.95))
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * self.hparams.warmup_ratio),
            num_training_steps=total_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def decoder_layer_classes(model):
    """Resolve the transformer block class(es) to wrap from HF's _no_split_modules."""
    names = set(getattr(model, "_no_split_modules", None) or [])
    classes = {type(m) for m in model.modules() if type(m).__name__ in names}
    if not classes:
        raise ValueError("Could not resolve a decoder-layer class from _no_split_modules; "
                         "pass an explicit auto_wrap_policy for this architecture.")
    return classes


def build_strategy(args, model):
    if args.strategy in ("ddp", "ddp_find_unused_parameters_true", "auto"):
        return args.strategy
    if args.strategy == "deepspeed":
        return DeepSpeedStrategy(stage=args.zero_stage, offload_optimizer=args.cpu_offload,
                                 offload_parameters=args.cpu_offload and args.zero_stage == 3)

    layers = decoder_layer_classes(model)
    logger.info("FSDP will wrap: %s", sorted(c.__name__ for c in layers))
    return FSDPStrategy(
        auto_wrap_policy=layers,
        activation_checkpointing_policy=layers if args.activation_checkpointing else None,
        sharding_strategy=args.sharding_strategy,
        state_dict_type=args.state_dict_type,
        cpu_offload=args.cpu_offload,
        limit_all_gathers=True,
    )


def export_hf(trainer, lit_module, tokenizer, out_dir):
    """Write a plain Hugging Face folder from the trained (possibly sharded) model."""
    if isinstance(trainer.strategy, DeepSpeedStrategy):
        logger.warning("ZeRO checkpoints are sharded; consolidate with the zero_to_fp32.py "
                       "script DeepSpeed writes into the checkpoint folder. Skipping export.")
        return
    # Must run on EVERY rank: under FSDP this is the all-gather that rebuilds full params.
    state_dict = trainer.strategy.lightning_module_state_dict()
    if trainer.is_global_zero:
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        lit_module.model.save_pretrained(out_dir, state_dict=state_dict)
        tokenizer.save_pretrained(out_dir)
        logger.info("Exported Hugging Face weights to %s", out_dir)


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError(f"{args.model_name} has no chat_template; this trainer requires one for train/inference parity.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_chat_jsonl(args.train_file, tokenizer, args.max_seq_len, args.mask_prompt, args.max_samples)
    collate = make_collator(tokenizer.pad_token_id)

    val_loader = None
    if args.eval_samples > 0 and len(rows) > args.eval_samples:
        val_rows, rows = rows[: args.eval_samples], rows[args.eval_samples:]
        val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate)

    # Lightning injects the DistributedSampler itself, so shuffle=True is correct here.
    train_loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    lit_module = LitCausalLM(args.model_name, args.learning_rate, args.weight_decay,
                             args.warmup_ratio, args.attn_implementation, hf_token=hf_token)
    lit_module.model.config.pad_token_id = tokenizer.pad_token_id

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if not args.no_checkpoint:
        # With no val split there is no monitored metric, so keep only the latest checkpoint.
        callbacks.insert(0, ModelCheckpoint(dirpath=os.path.join(args.output_dir, "checkpoints"),
                                            save_last=True,
                                            monitor="val_loss" if val_loader is not None else None,
                                            save_top_k=2 if val_loader is not None else 1))

    trainer = L.Trainer(
        accelerator="gpu",
        enable_checkpointing=not args.no_checkpoint,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=build_strategy(args, lit_module.model),
        precision=args.precision,
        max_epochs=args.num_train_epochs,
        accumulate_grad_batches=args.grad_acc_steps,
        gradient_clip_val=args.grad_clip or None,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.output_dir,
        enable_progress_bar=False,
        limit_val_batches=1.0 if val_loader is not None else 0,
        callbacks=callbacks,
    )

    resume = args.resume_ckpt if os.path.exists(args.resume_ckpt) else None
    logger.info("Starting training%s", f" (resuming from {resume})" if resume else "")
    trainer.fit(lit_module, train_loader, val_loader, ckpt_path=resume)

    if args.export_hf_dir:
        export_hf(trainer, lit_module, tokenizer, args.export_hf_dir)

    if trainer.is_global_zero:
        logger.info("Training complete. Checkpoints under %s", os.path.join(args.output_dir, "checkpoints"))


if __name__ == "__main__":
    main()
