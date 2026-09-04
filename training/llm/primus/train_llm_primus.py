"""AMD Primus post-training launcher (SFT / LoRA on ROCm) -- see readme_primus.md."""

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

# gfx942 = MI300X / MI325X, gfx950 = MI350X / MI355X; the tuned reference configs differ per arch.
ARCH_ALIASES = {
    "MI300X": "MI300X", "MI325X": "MI300X",
    "MI350X": "MI355X", "MI355X": "MI355X",
}


def parse_args():
    parser = argparse.ArgumentParser(description="AMD Primus SFT/LoRA post-training launcher")

    parser.add_argument("--config", default=None,
                        help="explicit path to a Primus YAML; overrides --method/--arch")
    parser.add_argument("--method", default="sft", choices=["sft", "lora"])
    parser.add_argument("--arch", default="MI300X", choices=sorted(ARCH_ALIASES))

    parser.add_argument("--mode", default="container", choices=["container", "direct", "slurm"],
                        help="container = run inside the ROCm image (recommended); "
                             "direct = bare metal or already inside a container")
    parser.add_argument("--image", default="rocm/primus:v26.5",
                        help="ROCm training image, used when --mode container")
    parser.add_argument("--cli", default="./primus-cli",
                        help="path to primus-cli (a clone often exposes ./runner/primus-cli)")

    # Common key=value overrides; anything else goes through --extra.
    parser.add_argument("--data-path", default="data/OTel_LLM_sample_10.jsonl",
                        help="training data path passed to the config (see readme_primus.md for the dataset-key caveat)")
    parser.add_argument("--train-iters", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--log-file", default=None, help="write the training log here")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="everything after --extra is forwarded verbatim to primus-cli")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_config(args):
    """Pick the shipped config for this method, or honor an explicit --config."""
    if args.config:
        return args.config

    local = f"configs/{ARCH_ALIASES[args.arch]}/qwen3_32b_{args.method}_posttrain.yaml"
    if os.path.exists(local):
        return local

    # Fall back to the reference configs inside a Primus checkout.
    return (f"./examples/megatron_bridge/configs/{ARCH_ALIASES[args.arch]}/"
            f"qwen3_32b_{args.method}_posttrain.yaml")


def check_env(args):
    """Fail early on the two things that always go wrong: no ROCm, no token."""
    if shutil.which("rocm-smi") is None:
        logger.warning("rocm-smi not found - Primus targets AMD Instinct GPUs on ROCm >= 7.0")
    else:
        subprocess.call(["rocm-smi", "--showproductname"])

    if not os.getenv("HF_TOKEN"):
        logger.warning("HF_TOKEN not set - gated model/tokenizer downloads will fail")


def build_command(args, config):
    """Assemble the primus-cli invocation."""
    cmd = [args.cli, args.mode]

    if args.mode == "container":
        cmd += ["--image", args.image]
        # The container does not inherit the host environment; pass the token in.
        if os.getenv("HF_TOKEN"):
            cmd += ["--env", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
        if args.data_path:
            # Mount the data directory so the in-container config can see it.
            cmd += ["--volume", f"{os.path.dirname(os.path.abspath(args.data_path))}:/data"]

    if args.log_file:
        cmd += ["--", "--log_file", args.log_file]

    cmd += ["--", "train", "posttrain", "--config", config]

    overrides = {
        "train_iters": args.train_iters,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "seq_length": args.seq_length,
        "finetune_lr": args.finetune_lr,
    }
    # The Megatron-Bridge SFT trainer's dataset key is not pinned in public docs; forward
    # --data-path under both names seen in reference configs and drop the rejected one.
    if args.data_path:
        data_in_container = (
            f"/data/{os.path.basename(args.data_path)}"
            if args.mode == "container"
            else os.path.abspath(args.data_path)
        )
        overrides["data_path"] = data_in_container
        overrides["dataset"] = data_in_container
    for key, value in overrides.items():
        if value is not None:
            cmd.append(f"{key}={value}")

    if args.method == "lora":
        cmd.append("peft=lora")

    cmd += args.extra
    return cmd


def main():
    args = parse_args()
    config = resolve_config(args)

    check_env(args)

    if not os.path.exists(config) and args.mode != "container":
        logger.warning("config not found on this host: %s", config)
        logger.warning("run from a Primus checkout, or pass --config explicitly")

    cmd = build_command(args, config)
    logger.info("method=%s arch=%s mode=%s", args.method, args.arch, args.mode)
    logger.info("config=%s", config)
    logger.info("launching: %s", " ".join(cmd))

    if args.dry_run:
        return

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
