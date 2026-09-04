"""Generate a NeMo AutoModel SFT/LoRA recipe YAML and launch it via the `automodel` CLI -- see readme_nemo.md."""

import os
import sys
import json
import shlex
import random
import logging
import argparse
import subprocess

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

HF_TOKEN = os.getenv("HF_TOKEN")

RECIPE = "TrainFinetuneRecipeForNextTokenPrediction"
CHAT_DATASET = "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a NeMo AutoModel SFT/LoRA recipe YAML and launch it via the `automodel` CLI"
    )

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HF repo id or local path of the base checkpoint to post-train")
    parser.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl",
                        help="Chat JSONL, one {'messages': [...]} per line. Extra columns are ignored.")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Optional held-out chat JSONL. If omitted, --val_fraction is split off --train_file")
    parser.add_argument("--val_fraction", type=float, default=0.02,
                        help="Fraction of --train_file held out for validation when --val_file is not given")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Where AutoModel writes checkpoints")
    parser.add_argument("--config_out", type=str, default="generated_recipe.yaml",
                        help="Path the generated recipe YAML is written to")

    parser.add_argument("--seq_length", type=int, default=2048, help="Max sequence length (tokens)")
    parser.add_argument("--global_batch_size", type=int, default=64, help="Global batch size across all ranks")
    parser.add_argument("--local_batch_size", type=int, default=1, help="Per-GPU micro batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Hard cap on optimizer steps")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Peak LR. Default: 1e-4 with --use_lora, else 5e-6 for full SFT")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=10, help="LR warmup steps")
    parser.add_argument("--val_every_steps", type=int, default=100, help="Run validation every N steps")
    parser.add_argument("--ckpt_every_steps", type=int, default=200, help="Write a checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"],
                        help="Attention kernel. sdpa is the safe default; FA2 needs flash-attn built")

    parser.add_argument("--use_lora", action="store_true",
                        help="Train a LoRA adapter instead of full fine-tuning")
    parser.add_argument("--lora_dim", type=int, default=16, help="LoRA rank (AutoModel calls this `dim`)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="*_proj",
                        help="Glob or comma-separated module names to adapt")

    parser.add_argument("--nproc_per_node", type=int, default=8, help="GPUs per node (passed to the automodel CLI)")
    parser.add_argument("--dp_size", type=int, default=None, help="Data-parallel size (default: nproc_per_node)")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor-parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline-parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context-parallel size")

    parser.add_argument("--dry_run", action="store_true",
                        help="Write the recipe YAML and print the launch command, but do not run it")

    return parser.parse_args()


def split_train_val(train_file, val_fraction, seed):
    """Hold out `val_fraction` of a chat JSONL as a validation file; return (train_path, val_path)."""
    # Normalize line endings: a missing final newline would merge two rows after the shuffle.
    with open(train_file) as f:
        rows = [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"{train_file} is empty.")

    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction))
    if n_val >= len(rows):
        raise SystemExit(f"--val_fraction {val_fraction} leaves no training rows.")

    base = os.path.splitext(train_file)[0]
    train_path, val_path = f"{base}_split_train.jsonl", f"{base}_split_val.jsonl"
    with open(val_path, "w") as f:
        f.writelines(rows[:n_val])
    with open(train_path, "w") as f:
        f.writelines(rows[n_val:])

    logging.info("Split %s -> %d train / %d val rows", train_file, len(rows) - n_val, n_val)
    return train_path, val_path


def preflight(path, sample_size=64):
    """Fail fast if the JSONL is not this repo's {'messages': [...]} contract."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise SystemExit(f"{path} line {i + 1}: expected a non-empty 'messages' list.")
            for m in msgs:
                if "role" not in m or "content" not in m:
                    raise SystemExit(f"{path} line {i + 1}: each message needs 'role' and 'content'.")
    logging.info("Preflight OK: %s looks like chat `messages` JSONL", path)


def build_recipe(args, train_path, val_path):
    """Build the AutoModel recipe dict; key names follow upstream examples/llm_finetune recipes."""
    lr = args.learning_rate
    if lr is None:
        lr = 1e-4 if args.use_lora else 5e-6

    targets = args.lora_target_modules
    if "," in targets:
        targets = [t.strip() for t in targets.split(",") if t.strip()]

    def dataset_block(path):
        return {
            "_target_": CHAT_DATASET,
            "path_or_dataset_id": path,
            "seq_length": args.seq_length,
            "padding": "do_not_pad",
            "truncation": "longest_first",
        }

    recipe = {
        "recipe": RECIPE,
        "dist_env": {"backend": "nccl", "timeout_minutes": 30},
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": args.seed,
            "ranked": True,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": args.model_name,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "attn_implementation": args.attn_implementation,
        },
        "distributed": {
            "_target_": "nemo_automodel.components.distributed.fsdp2.FSDP2Manager",
            "dp_size": args.dp_size if args.dp_size is not None else args.nproc_per_node,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "cp_size": args.cp_size,
            "ep_size": None,
            "sequence_parallel": False,
        },
        "step_scheduler": {
            "global_batch_size": args.global_batch_size,
            "local_batch_size": args.local_batch_size,
            "max_steps": args.max_steps,
            "num_epochs": args.num_epochs,
            "val_every_steps": args.val_every_steps,
            "ckpt_every_steps": args.ckpt_every_steps,
        },
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": lr,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1.0e-08,
        },
        "lr_scheduler": {"lr_decay_style": "cosine", "lr_warmup_steps": args.warmup_steps},
        "checkpoint": {
            "enabled": True,
            "model_save_format": "safetensors",
            "checkpoint_dir": args.checkpoint_dir,
            "save_consolidated": True,
        },
        "dataset": dataset_block(train_path),
        "validation_dataset": dataset_block(val_path),
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "shuffle": True,
        },
        "validation_dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
        },
        "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
    }

    if args.use_lora:
        recipe["peft"] = {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "dim": args.lora_dim,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": targets,
            "use_triton": True,
        }

    return recipe


def main():
    args = parse_args()

    if not HF_TOKEN:
        logging.warning("HF_TOKEN is not set (dev.env missing?). Gated models will fail to download.")

    if not os.path.exists(args.train_file):
        raise SystemExit(f"--train_file not found: {args.train_file}")
    preflight(args.train_file)

    if args.val_file:
        preflight(args.val_file)
        train_path, val_path = args.train_file, args.val_file
    else:
        train_path, val_path = split_train_val(args.train_file, args.val_fraction, args.seed)

    # PyYAML ships with nemo-automodel; a clear failure beats an obscure one.
    try:
        import yaml
    except ImportError:
        raise SystemExit("PyYAML is missing. Install requirements_nemo.txt, or run inside the NGC container.")

    recipe = build_recipe(args, train_path, val_path)
    with open(args.config_out, "w") as f:
        yaml.safe_dump(recipe, f, sort_keys=False, default_flow_style=False)

    mode = f"LoRA (dim={args.lora_dim}, alpha={args.lora_alpha})" if args.use_lora else "full SFT"
    logging.info("Wrote recipe: %s | model=%s | mode=%s", args.config_out, args.model_name, mode)
    logging.info("Checkpoints will be written under: %s", args.checkpoint_dir)

    cmd = ["automodel", args.config_out, "--nproc-per-node", str(args.nproc_per_node)]
    logging.info("Launch command: %s", " ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        logging.info("--dry_run set; not launching.")
        return

    # The automodel CLI owns the torchrun launch; propagate its exit code.
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
