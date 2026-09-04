"""Continued pre-training launcher for torchtune (legacy — upstream wound down 2025) -- see readme_torchtune.md."""

import argparse
import logging
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLE = os.path.join(HERE, "data", "OTel_LLM_sample_10.jsonl")
DEFAULT_FLAT = os.path.join(HERE, "data", "otel_cpt.jsonl")
DEFAULT_CONFIG = os.path.join(HERE, "config_cpt_lora.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="torchtune continued-pretraining launcher")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="torchtune YAML config")
    parser.add_argument(
        "--recipe",
        default="lora_finetune_distributed",
        help="tune recipe. Use lora_finetune_single_device on one GPU.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--input",
        default=DEFAULT_SAMPLE,
        help="chat JSONL to flatten (default: OTel_LLM_sample_10.jsonl)",
    )
    parser.add_argument(
        "--flat-file",
        default=DEFAULT_FLAT,
        help="destination {text} JSONL written by prepare_data_torchtune.py",
    )
    parser.add_argument("--skip-prepare", action="store_true",
                        help="do not flatten; assume --flat-file already exists")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="forwarded to `tune run` (e.g. epochs=1 batch_size=1)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_data(args):
    if args.skip_prepare:
        if not os.path.exists(args.flat_file):
            raise SystemExit(f"--skip-prepare set but {args.flat_file} does not exist")
        return
    prep = os.path.join(HERE, "prepare_data_torchtune.py")
    cmd = [sys.executable, prep, "--input", args.input, "--output", args.flat_file]
    logger.info("flattening %s -> %s", args.input, args.flat_file)
    if args.dry_run:
        logger.info("would run: %s", " ".join(cmd))
        return
    raise_if = subprocess.call(cmd)
    if raise_if != 0:
        raise SystemExit(raise_if)


def main():
    args = parse_args()

    if not os.getenv("HF_TOKEN"):
        logger.warning("HF_TOKEN not set - gated downloads will fail")

    if shutil.which("tune") is None and not args.dry_run:
        raise SystemExit("`tune` not on PATH. pip install -r requirements_torchtune.txt")

    prepare_data(args)

    cmd = ["tune", "run"]
    if args.nproc_per_node > 1 and "distributed" in args.recipe:
        cmd += ["--nnodes", "1", "--nproc_per_node", str(args.nproc_per_node)]
    cmd += [args.recipe, "--config", args.config]
    cmd += [f"dataset.data_files={os.path.abspath(args.flat_file)}"]
    cmd += args.extra

    logger.info("launching: %s", " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
