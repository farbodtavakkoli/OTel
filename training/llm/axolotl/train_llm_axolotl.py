"""Thin launcher for Axolotl post-training: validate the YAML config, then shell out to the axolotl CLI (see readme_axolotl.md)."""

import argparse
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
logger = logging.getLogger("training/llm/axolotl")

# Keys Axolotl requires in every config (see docs.axolotl.ai/docs/config-reference.html).
REQUIRED_KEYS = ("base_model", "learning_rate")


def parse_args():
    """Parse launcher flags; unknown arguments are forwarded verbatim to the axolotl CLI."""
    parser = argparse.ArgumentParser(description="Launch Axolotl training from a YAML config")
    parser.add_argument("--config", required=True,
                        help="Path to the Axolotl YAML config (e.g. config_sft_lora.yaml)")
    parser.add_argument("--task", default="train", choices=["train", "preprocess", "merge-lora"],
                        help="Upstream subcommand to run (default: train)")
    parser.add_argument("--num-processes", type=int, default=8,
                        help="Training processes = GPUs on this node (default: 8)")
    parser.add_argument("--deepspeed", default=None,
                        help="Override the config's deepspeed json, e.g. deepspeed_configs/zero3_bf16.json")
    parser.add_argument("--lora-model-dir", default=None,
                        help="Adapter directory to merge (only used with --task merge-lora)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check the config and print the command, then exit without running")
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


def local_dataset_files(cfg):
    """Collect local file paths referenced by the `datasets` blocks."""
    files = []
    for entry in cfg.get("datasets") or []:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        # A Hub repo id ("org/dataset") also contains a slash, so only treat a path as
        # local when it is explicitly relative/absolute or names a data file.
        local_like = (".", "/", "~")
        data_ext = (".json", ".jsonl", ".csv", ".parquet", ".arrow", ".txt")
        if isinstance(path, str) and (path.startswith(local_like) or path.endswith(data_ext)):
            if not path.startswith(("s3://", "gs://", "hf://")):
                files.append(path)
        data_files = entry.get("data_files")
        if isinstance(data_files, str):
            files.append(data_files)
        elif isinstance(data_files, list):
            files.extend([f for f in data_files if isinstance(f, str)])
    return files


def validate(cfg, config_path):
    """Fail fast on the mistakes that otherwise surface minutes into a run."""
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise SystemExit(f"{config_path}: missing required key(s): {', '.join(missing)}")

    if not cfg.get("datasets"):
        raise SystemExit(f"{config_path}: no `datasets:` block - nothing to train on")

    for path in local_dataset_files(cfg):
        candidate = path if os.path.isabs(path) else os.path.join(os.path.dirname(config_path) or ".", path)
        if not os.path.exists(candidate):
            raise SystemExit(f"{config_path}: dataset file does not exist: {path}")

    adapter = cfg.get("adapter")
    if cfg.get("load_in_4bit") and adapter != "qlora":
        logger.warning("load_in_4bit is set but adapter is %r - upstream expects adapter: qlora", adapter)
    if adapter == "qlora" and not cfg.get("load_in_4bit"):
        logger.warning("adapter: qlora without load_in_4bit: true - the base model stays unquantized")
    if cfg.get("rl") and cfg.get("sample_packing"):
        logger.warning("sample_packing with rl: %s is not a supported combination", cfg["rl"])

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set - gated models on the Hub will fail to download")


def summarize(cfg, config_path, num_processes):
    """One compact block so the log says exactly what is about to run."""
    mode = cfg.get("rl") or "sft"
    if cfg.get("adapter"):
        method = cfg["adapter"]
    elif cfg.get("load_in_4bit") or cfg.get("load_in_8bit"):
        method = "quantized (no adapter)"
    else:
        method = "full fine-tune"

    logger.info("config          : %s", config_path)
    logger.info("base_model      : %s", cfg.get("base_model"))
    logger.info("objective       : %s", mode)
    logger.info("method          : %s", method)
    logger.info("sequence_len    : %s", cfg.get("sequence_len"))
    logger.info("micro_batch_size: %s x grad_accum %s x %s GPU(s)",
                cfg.get("micro_batch_size"), cfg.get("gradient_accumulation_steps"), num_processes)
    logger.info("learning_rate   : %s", cfg.get("learning_rate"))
    logger.info("output_dir      : %s", cfg.get("output_dir"))
    if cfg.get("deepspeed"):
        logger.info("deepspeed       : %s", cfg["deepspeed"])
    if cfg.get("fsdp_version"):
        logger.info("fsdp_version    : %s", cfg["fsdp_version"])


def build_command(args, passthrough):
    """Assemble the upstream CLI invocation."""
    cmd = ["axolotl", args.task, args.config]
    if args.task == "train":
        cmd += ["--num-processes", str(args.num_processes)]
        if args.deepspeed:
            cmd += ["--deepspeed", args.deepspeed]
    if args.task == "merge-lora":
        if not args.lora_model_dir:
            raise SystemExit("--lora-model-dir is required for --task merge-lora")
        cmd += [f"--lora-model-dir={args.lora_model_dir}"]
    return cmd + list(passthrough)


def main():
    """Validate the config, print a summary, and launch the axolotl CLI."""
    args, passthrough = parse_args()

    cfg = load_config(args.config)
    validate(cfg, args.config)
    summarize(cfg, args.config, args.num_processes)

    cmd = build_command(args, passthrough)
    logger.info("command         : %s", " ".join(cmd))

    if args.validate_only:
        logger.info("--validate-only set; not launching")
        return

    if shutil.which("axolotl") is None:
        raise SystemExit("`axolotl` is not on PATH - activate the venv (see readme_axolotl.md)")

    # HF_TOKEN comes from dev.env via load_dotenv and is inherited by the child process.
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        logger.error("axolotl exited with code %s", result.returncode)
        sys.exit(result.returncode)
    logger.info("axolotl finished; artifacts under %s", cfg.get("output_dir"))


if __name__ == "__main__":
    main()
