"""Thin launcher for LLaMA-Factory post-training: validate the YAML config and dataset registry, then shell out to llamafactory-cli (see readme_llamafactory.md)."""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys

import yaml
from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("training/llm/llamafactory")

REQUIRED_KEYS = ("model_name_or_path", "stage", "finetuning_type", "template", "output_dir")
PAIR_DATA_STAGES = ("dpo", "rm")           # stages whose datasets need "ranking": true
DATA_CONFIG = "dataset_info.json"          # upstream constant; name is not configurable


def parse_args():
    """Parse launcher flags; unknown arguments are forwarded verbatim to llamafactory-cli."""
    parser = argparse.ArgumentParser(description="Launch LLaMA-Factory training from a YAML config")
    parser.add_argument("--config", required=True,
                        help="Path to the LLaMA-Factory YAML config (e.g. config_sft_lora.yaml)")
    parser.add_argument("--task", default="train", choices=["train", "export", "chat", "api"],
                        help="Upstream subcommand: train, export (merge LoRA), chat, api (default: train)")
    parser.add_argument("--gpus", default=None,
                        help="Value for CUDA_VISIBLE_DEVICES, e.g. '0,1,2,3'. Default: all visible GPUs")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="GPU count used only for the printed summary and the torchrun decision (default: 8)")
    parser.add_argument("--force-torchrun", action="store_true",
                        help="Always set FORCE_TORCHRUN=1 (already automatic for multi-GPU full FT)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the config and registry, print the command, exit without running")
    return parser.parse_known_args()


def load_config(path):
    """Parse the YAML config, failing loudly if it is missing or malformed."""
    if not os.path.isfile(path):
        raise SystemExit(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"config did not parse to a mapping: {path}")
    return cfg


def check_datasets(cfg, config_path):
    """Verify every dataset name is registered and its backing file exists."""
    names = cfg.get("dataset")
    if not names:
        raise SystemExit(f"{config_path}: no `dataset:` - nothing to train on")
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",")]

    base = os.path.dirname(os.path.abspath(config_path))
    dataset_dir = cfg.get("dataset_dir", "data")
    dataset_dir = dataset_dir if os.path.isabs(dataset_dir) else os.path.join(base, dataset_dir)

    registry_path = os.path.join(dataset_dir, DATA_CONFIG)
    if not os.path.isfile(registry_path):
        raise SystemExit(f"{registry_path} not found - dataset_dir must contain {DATA_CONFIG}")
    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)

    for name in names:
        if name not in registry:
            raise SystemExit(
                f"dataset {name!r} is not registered in {registry_path} "
                f"(known: {', '.join(sorted(registry)) or 'none'})"
            )
        entry = registry[name]
        remote = any(key in entry for key in ("hf_hub_url", "ms_hub_url", "script_url", "cloud_file_name"))
        file_name = entry.get("file_name")
        if not remote:
            if not file_name:
                raise SystemExit(f"dataset {name!r} in {registry_path} has no file_name and no *_url")
            if not os.path.exists(os.path.join(dataset_dir, file_name)):
                raise SystemExit(f"dataset {name!r} points at missing file: {os.path.join(dataset_dir, file_name)}")
        if cfg.get("stage") in PAIR_DATA_STAGES and not entry.get("ranking"):
            raise SystemExit(
                f"stage {cfg['stage']!r} needs preference data, but dataset {name!r} is not "
                f'marked "ranking": true in {registry_path}'
            )
    return names


def validate(cfg, config_path):
    """Fail fast on the mistakes that otherwise surface minutes into a run."""
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise SystemExit(f"{config_path}: missing required key(s): {', '.join(missing)}")

    names = check_datasets(cfg, config_path)

    if cfg.get("quantization_bit") and cfg.get("finetuning_type") != "lora":
        logger.warning("quantization_bit with finetuning_type=%r - QLoRA expects finetuning_type: lora",
                       cfg.get("finetuning_type"))
    if cfg.get("quantization_bit") and cfg.get("deepspeed"):
        logger.warning("a quantized base cannot be sharded by DeepSpeed ZeRO-3; expect this to fail")
    if cfg.get("val_size") and cfg.get("eval_dataset"):
        logger.warning("val_size and eval_dataset are mutually exclusive upstream")

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set - gated models on the Hub will fail to download")

    return names


def summarize(cfg, config_path, names, num_gpus):
    """One compact block so the log says exactly what is about to run."""
    method = cfg.get("finetuning_type")
    if cfg.get("quantization_bit"):
        method = f"{method} + {cfg['quantization_bit']}-bit {cfg.get('quantization_method', 'bnb')}"

    logger.info("config          : %s", config_path)
    logger.info("model           : %s", cfg.get("model_name_or_path"))
    logger.info("stage / method  : %s / %s", cfg.get("stage"), method)
    logger.info("template        : %s", cfg.get("template"))
    logger.info("dataset(s)      : %s (dataset_dir=%s)", ", ".join(names), cfg.get("dataset_dir", "data"))
    logger.info("cutoff_len      : %s", cfg.get("cutoff_len"))
    logger.info("batch           : %s per device x grad_accum %s x %s GPU(s)",
                cfg.get("per_device_train_batch_size"), cfg.get("gradient_accumulation_steps"), num_gpus)
    logger.info("learning_rate   : %s", cfg.get("learning_rate"))
    logger.info("output_dir      : %s", cfg.get("output_dir"))
    if cfg.get("deepspeed"):
        logger.info("deepspeed       : %s", cfg["deepspeed"])


def build_env(cfg, args, num_gpus):
    """Environment for the child: HF_TOKEN (from dev.env) plus the launch switches."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Upstream launches full-parameter multi-GPU runs through torchrun; the CLI only does
    # that when FORCE_TORCHRUN is set.
    if args.force_torchrun or (cfg.get("finetuning_type") == "full" and num_gpus > 1):
        env["FORCE_TORCHRUN"] = "1"
        logger.info("FORCE_TORCHRUN=1 (multi-GPU full fine-tuning)")
    return env


def main():
    """Validate the config and registry, print a summary, and launch llamafactory-cli."""
    args, passthrough = parse_args()

    num_gpus = len(args.gpus.split(",")) if args.gpus else args.num_gpus

    cfg = load_config(args.config)
    names = validate(cfg, args.config)
    summarize(cfg, args.config, names, num_gpus)

    # Anything extra is forwarded verbatim; upstream accepts `key=value` overrides,
    # e.g. learning_rate=1e-5 logging_steps=1
    cmd = ["llamafactory-cli", args.task, args.config] + list(passthrough)
    logger.info("command         : %s", " ".join(cmd))

    if args.validate_only:
        logger.info("--validate-only set; not launching")
        return

    if shutil.which("llamafactory-cli") is None:
        raise SystemExit("`llamafactory-cli` is not on PATH - activate the venv (see readme_llamafactory.md)")

    result = subprocess.run(cmd, env=build_env(cfg, args, num_gpus), check=False)
    if result.returncode != 0:
        logger.error("llamafactory-cli exited with code %s", result.returncode)
        sys.exit(result.returncode)
    logger.info("llamafactory-cli finished; artifacts under %s", cfg.get("output_dir"))


if __name__ == "__main__":
    main()
