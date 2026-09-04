"""Fine-tune or continue pre-training a Hugging Face causal LM with MosaicML Composer — see readme_composer.md."""

import os
import json
import logging
import argparse

from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import GradientClipping, LowPrecisionLayerNorm, SeqLengthWarmup
from composer.callbacks import LRMonitor, MemoryMonitor, RuntimeEstimator, SpeedMonitor
from composer.metrics import LanguageCrossEntropy, LanguagePerplexity
from composer.models import HuggingFaceModel
from composer.optim import CosineAnnealingWithWarmupScheduler, DecoupledAdamW
from composer.utils import dist, get_device, reproducibility

from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MosaicML Composer chat fine-tuner for HF causal LMs")

    # Paths and model
    p.add_argument("--train_file", default="data/OTel_LLM_sample_10.jsonl",
                   help="Chat JSONL: one {'messages': [...]} per line (default: the shipped 10-row sample)")
    p.add_argument("--model_name", required=True, help="HF repo id or local path (tokenizer must have a chat template)")
    p.add_argument("--save_folder", default="./composer_run", help="Where Composer .pt checkpoints are written")
    p.add_argument("--save_interval", default="1ep", help="Checkpoint cadence, e.g. 1ep or 500ba")
    p.add_argument("--load_path", default="", help="Composer checkpoint to resume from (used only if it exists)")
    p.add_argument("--run_name", default="composer-sft")

    # Data
    p.add_argument("--max_seq_len", type=int, default=4096, help="Max tokens per example; longer rows are dropped, never truncated")
    p.add_argument("--max_samples", type=int, default=None, help="Hard cap on rows loaded (quick smoke runs)")
    p.add_argument("--eval_samples", type=int, default=0, help="Rows held out for eval (0 = no eval loop)")
    p.add_argument("--mask_prompt", action="store_true", default=True, help="Completion-only loss: supervise assistant turns only (default on)")
    p.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false", help="Train on the full rendered sequence (continued pre-training style)")
    p.add_argument("--num_workers", type=int, default=4)

    # Optimization
    p.add_argument("--global_train_batch_size", type=int, default=64,
                   help="Batch size across ALL ranks; Composer derives grad accumulation from it")
    p.add_argument("--device_train_microbatch_size", default="auto",
                   help="Per-device microbatch, or 'auto' to let Composer find the largest that fits")
    p.add_argument("--max_duration", default="3ep", help="Composer Time string: 3ep, 2000ba, 10000000tok")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--t_warmup", default="0.03dur", help="Warmup as a fraction of training ('0.03dur') or a Time string ('100ba')")
    p.add_argument("--alpha_f", type=float, default=0.1, help="Final LR as a fraction of peak")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Global grad-norm clip; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])

    # Parallelism
    p.add_argument("--no_fsdp", action="store_true", help="Disable FSDP and fall back to DDP (single GPU / small models)")
    p.add_argument("--sharding_strategy", default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
    p.add_argument("--fsdp_mixed_precision", default="PURE", choices=["FULL", "DEFAULT", "PURE"])
    p.add_argument("--state_dict_type", default="full", choices=["full", "sharded"])
    p.add_argument("--activation_checkpointing", action="store_true",
                   help="Recompute decoder-layer activations to save memory")

    # Speedup algorithms
    p.add_argument("--low_precision_layernorm", action="store_true",
                   help="Run LayerNorm in the autocast dtype instead of upcasting to fp32")
    p.add_argument("--seq_length_warmup", action="store_true",
                   help="Ramp sequence length up over the first 30%% of training (throughput win, see readme caveat)")

    return p.parse_args()


def build_example(messages, tokenizer, mask_prompt):
    """Render one conversation with the chat template; mask non-assistant tokens when mask_prompt."""
    # return_dict=False: transformers 5.x returns a BatchEncoding by default, whose len() is
    # its key count (2) — that silently breaks the length-diff masking below. Force a flat list.
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
    if dist.get_global_rank() == 0:
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


def build_dataloader(rows, batch_size, collate, num_workers, shuffle):
    """Composer requires an explicit DistributedSampler for map-style datasets."""
    sampler = dist.get_sampler(rows, shuffle=shuffle)
    return DataLoader(rows, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate, num_workers=num_workers, drop_last=shuffle)


def tag_blocks_for_fsdp(model, activation_checkpointing):
    """Set _fsdp_wrap on each transformer block so Composer's auto-wrap policy shards them."""
    names = set(getattr(model, "_no_split_modules", None) or [])
    if not names:
        raise ValueError("Model does not declare _no_split_modules; set _fsdp_wrap manually for this architecture.")

    tagged = 0
    for module in model.modules():
        if type(module).__name__ in names:
            module._fsdp_wrap = True
            if activation_checkpointing:
                module._activation_checkpointing = True
            tagged += 1

    if tagged == 0:
        raise ValueError(f"Found no modules matching _no_split_modules={sorted(names)}.")
    if dist.get_global_rank() == 0:
        logger.info("Tagged %d %s blocks for FSDP wrapping", tagged, sorted(names))


def main():
    args = parse_args()
    # Init the process group first: the samplers below need world size and rank
    dist.initialize_dist(get_device(None), timeout=1800)
    reproducibility.seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # HF_TOKEN comes from dev.env / the environment
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError(f"{args.model_name} has no chat_template; this trainer requires one for train/inference parity.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_chat_jsonl(args.train_file, tokenizer, args.max_seq_len, args.mask_prompt, args.max_samples)
    eval_rows = []
    if args.eval_samples > 0 and len(rows) > args.eval_samples:
        eval_rows, rows = rows[: args.eval_samples], rows[args.eval_samples:]

    collate = make_collator(tokenizer.pad_token_id)
    # Per-device minibatch; Composer splits it further into microbatches
    device_batch_size = max(1, args.global_train_batch_size // dist.get_world_size())
    train_loader = build_dataloader(rows, device_batch_size, collate, args.num_workers, shuffle=True)
    eval_loader = build_dataloader(eval_rows, device_batch_size, collate, args.num_workers, shuffle=False) if eval_rows else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None,
        token=hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    use_fsdp = not args.no_fsdp and dist.get_world_size() > 1
    if use_fsdp:
        tag_blocks_for_fsdp(model, args.activation_checkpointing)

    composer_model = HuggingFaceModel(
        model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=[LanguageCrossEntropy(), LanguagePerplexity()],
        shift_labels=True,
    )

    optimizer = DecoupledAdamW(
        composer_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup, alpha_f=args.alpha_f)

    algorithms = []
    if args.grad_clip:
        algorithms.append(GradientClipping(clipping_type="norm", clipping_threshold=args.grad_clip))
    if args.low_precision_layernorm:
        algorithms.append(LowPrecisionLayerNorm())
    if args.seq_length_warmup:
        algorithms.append(SeqLengthWarmup(duration=0.3, min_seq_length=64,
                                          max_seq_length=args.max_seq_len, step_size=64, truncate=True))

    parallelism_config = None
    if use_fsdp:
        parallelism_config = {
            "fsdp": {
                "sharding_strategy": args.sharding_strategy,
                "mixed_precision": args.fsdp_mixed_precision,
                "activation_checkpointing": args.activation_checkpointing,
                "activation_checkpointing_reentrant": False,
                "activation_cpu_offload": False,
                "state_dict_type": args.state_dict_type,
                "limit_all_gathers": True,
                "sync_module_states": True,
            }
        }

    microbatch = args.device_train_microbatch_size
    if microbatch != "auto":
        microbatch = int(microbatch)

    trainer = Trainer(
        run_name=args.run_name,
        model=composer_model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=args.max_duration,
        eval_interval="1ep" if eval_loader is not None else 0,
        precision="amp_bf16",
        device_train_microbatch_size=microbatch,
        parallelism_config=parallelism_config,
        algorithms=algorithms,
        callbacks=[SpeedMonitor(window_size=10), LRMonitor(), MemoryMonitor(), RuntimeEstimator()],
        save_folder=args.save_folder,
        save_interval=args.save_interval,
        save_num_checkpoints_to_keep=2,
        save_overwrite=True,
        load_path=args.load_path if os.path.exists(args.load_path) else None,
        seed=args.seed,
        progress_bar=False,
        log_to_console=True,
        console_log_interval="10ba",
    )

    if dist.get_global_rank() == 0:
        logger.info("Starting training: %s, fsdp=%s, algorithms=%s",
                    args.max_duration, use_fsdp, [type(a).__name__ for a in algorithms])
    trainer.fit()

    if dist.get_global_rank() == 0:
        logger.info("Training complete. Composer checkpoints are in %s", args.save_folder)
        logger.info("Convert to Hugging Face format with HuggingFaceModel.hf_from_composer_checkpoint "
                    "(or llm-foundry's scripts/inference/convert_composer_to_hf.py). See readme_composer.md.")


if __name__ == "__main__":
    main()
