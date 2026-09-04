"""Thin LLM Foundry launcher — picks a recipe YAML, applies overrides, execs `composer`; see readme_llmfoundry.md."""

import argparse
import logging
import os
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

RECIPES = {
    # instruction finetuning: local chat JSONL through the `finetuning` dataloader
    "sft": "yamls/finetune_chat_sft.yaml",
    # continued pre-training: MDS-converted raw text through the `text` dataloader
    "pretrain": "yamls/continued_pretrain.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Foundry train launcher (SFT / continued pre-training)")

    parser.add_argument("--recipe", default="sft", choices=sorted(RECIPES),
                        help="sft = instruction finetuning; pretrain = continued pre-training")
    parser.add_argument("--config", default=None,
                        help="explicit path to a YAML; overrides --recipe")
    parser.add_argument("--foundry-dir", default="./llm-foundry",
                        help="path to an llm-foundry checkout (needs scripts/train/train.py)")
    parser.add_argument("--gpus", type=int, default=None,
                        help="processes to launch; omit to let the composer launcher autodetect")

    # The overrides that actually change between runs. Everything else lives in the YAML.
    parser.add_argument("--model", default=None, help="HF repo id or local path -> variables.model_name")
    parser.add_argument("--data-local", default=None,
                        help="data dir (sft) or MDS root (pretrain) -> variables.data_local; "
                             "omit to use the recipe YAML default (sft: ./data, pretrain: ./my-mds-data)")
    parser.add_argument("--data-remote", default=None, help="MDS object-store URI -> variables.data_remote (pretrain)")
    parser.add_argument("--max-seq-len", type=int, default=None, help="-> variables.max_seq_len")
    parser.add_argument("--max-duration", default=None, help="Composer Time string: 3ep, 2000ba, 10000000tok")
    parser.add_argument("--global-batch-size", type=int, default=None, help="-> global_train_batch_size")
    parser.add_argument("--device-microbatch-size", default=None, help="int or 'auto' -> device_train_microbatch_size")
    parser.add_argument("--lr", type=float, default=None, help="-> optimizer.lr")
    parser.add_argument("--save-folder", default=None, help="-> save_folder")
    parser.add_argument("--run-name", default=None, help="-> run_name")

    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="everything after --extra is forwarded verbatim as key=value overrides")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    return parser.parse_args()


def resolve_config(args):
    """Honor an explicit --config, else use the recipe YAML shipped in this folder."""
    if args.config:
        return args.config

    local = RECIPES[args.recipe]
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), local)
    return here if os.path.exists(here) else local


def check_env(args, config, train_script):
    """Fail early on the things that always go wrong: no launcher, no repo, no token."""
    ok = True

    if shutil.which("composer") is None:
        logger.error("`composer` launcher not found on PATH -- pip install mosaicml (or the llm-foundry extras)")
        ok = False

    if not os.path.exists(train_script):
        logger.error("train script not found: %s", train_script)
        logger.error("clone the repo and install it: "
                     "git clone https://github.com/mosaicml/llm-foundry.git && cd llm-foundry && pip install -e '.[gpu]'")
        ok = False

    if not os.path.exists(config):
        logger.error("config not found: %s", config)
        ok = False

    if not os.getenv("HF_TOKEN"):
        logger.warning("HF_TOKEN not set -- gated checkpoints (Llama, Gemma) will fail to download. "
                       "Put HF_TOKEN=hf_... in dev.env next to this script.")

    return ok


def build_overrides(args):
    """Map the CLI flags onto YAML keys that exist in the shipped configs or TrainConfig."""
    overrides = {
        "variables.model_name": args.model,
        "variables.data_local": args.data_local,
        "variables.data_remote": args.data_remote,
        "variables.max_seq_len": args.max_seq_len,
        "max_duration": args.max_duration,
        "global_train_batch_size": args.global_batch_size,
        "device_train_microbatch_size": args.device_microbatch_size,
        "optimizer.lr": args.lr,
        "save_folder": args.save_folder,
        "run_name": args.run_name,
    }
    return [f"{key}={value}" for key, value in overrides.items() if value is not None]


def main():
    args = parse_args()
    config = os.path.abspath(resolve_config(args))
    train_script = os.path.join(args.foundry_dir, "scripts", "train", "train.py")

    if not check_env(args, config, train_script) and not args.dry_run:
        raise SystemExit(1)

    cmd = ["composer"]
    if args.gpus:
        cmd += ["-n", str(args.gpus)]
    cmd += [train_script, config]
    cmd += build_overrides(args)
    cmd += args.extra

    logger.info("recipe=%s config=%s", args.recipe, config)
    logger.info("launching: %s", " ".join(cmd))

    if args.dry_run:
        return

    # HF_TOKEN is in os.environ via load_dotenv; composer passes it to every rank
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
