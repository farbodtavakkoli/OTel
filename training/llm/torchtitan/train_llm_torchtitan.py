"""Thin launcher for torchtitan continued pre-training / SFT -- see readme_torchtitan.md."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("training/llm/torchtitan")

HERE = Path(__file__).resolve().parent
RECIPE_MODULE = "recipe_torchtitan"


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch torchtitan continued-pretraining / SFT from a checked-out torchtitan repo.",
    )
    p.add_argument("--mode", default="train",
                   choices=["train", "convert-from-hf", "convert-to-hf", "download-assets"],
                   help="train: run torchtitan.train. convert-*: run the upstream "
                        "scripts/checkpoint_conversion helpers. download-assets: fetch a "
                        "tokenizer/HF assets folder. (default: train)")
    p.add_argument("--titan-repo", required=True,
                   help="Path to a checked-out https://github.com/pytorch/torchtitan clone. "
                        "Commands are run with this as the working directory, which is what "
                        "upstream's run_train.sh and CI do.")

    # Recipe selection (torchtitan --module / --config).
    p.add_argument("--module", default=RECIPE_MODULE,
                   help=f"Python module holding the config function (default: {RECIPE_MODULE}, "
                        "the recipe shipped in this folder). Any importable module works, "
                        "e.g. 'qwen3' or 'torchtitan_recipes.llama3'.")
    p.add_argument("--config", default="cpt_qwen3_8b",
                   help="Name of the config function inside --module (default: cpt_qwen3_8b).")
    p.add_argument("--ngpu", type=int, default=8,
                   help="Processes per node for torchrun (default: 8). Must equal the product "
                        "of the parallelism degrees set in the recipe.")
    p.add_argument("--log-rank", default="0",
                   help="Ranks whose stdout is shown (torchrun --local-ranks-filter). Default: 0.")

    # Conversion / asset arguments.
    p.add_argument("--input-dir", help="convert-*: source checkpoint directory.")
    p.add_argument("--output-dir", help="convert-*: destination checkpoint directory.")
    p.add_argument("--model-name", default="qwen3",
                   help="convert-*: torchtitan model package name, e.g. qwen3 or llama3.")
    p.add_argument("--model-flavor", default="8B",
                   help="convert-*: model flavor registered for that model, e.g. 8B.")
    p.add_argument("--hf-assets-path",
                   help="convert-to-hf: directory holding the HF config.json / tokenizer files "
                        "that describe the target architecture.")
    p.add_argument("--repo-id", help="download-assets: Hugging Face repo id to pull assets from.")
    p.add_argument("--assets", default="tokenizer",
                   help="download-assets: which assets to fetch (default: tokenizer).")

    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is forwarded verbatim to the underlying "
                        "command, e.g. --extra --training.steps 200 --checkpoint.interval 100.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command and the environment it would run under, then exit.")
    return p.parse_args()


def validate_repo(titan_repo: Path) -> None:
    """Fail fast if --titan-repo is not actually a torchtitan checkout."""
    if not titan_repo.is_dir():
        raise SystemExit(f"--titan-repo does not exist: {titan_repo}")
    train_entry = titan_repo / "torchtitan" / "train.py"
    if not train_entry.is_file():
        raise SystemExit(
            f"{titan_repo} does not look like a torchtitan clone (missing torchtitan/train.py). "
            "Clone it with: git clone https://github.com/pytorch/torchtitan"
        )


def build_env(titan_repo: Path) -> dict:
    """Environment for the child process: recipe on PYTHONPATH, allocator tuned."""
    env = os.environ.copy()
    pythonpath = [str(HERE), str(titan_repo)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # Same allocator setting upstream's run_train.sh exports.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def build_train_cmd(args) -> list:
    """Mirror upstream run_train.sh: torchrun -m torchtitan.train --module M --config C."""
    return [
        "torchrun",
        f"--nproc_per_node={args.ngpu}",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:0",
        "--local-ranks-filter", args.log_rank,
        "--role", "rank",
        "--tee", "3",
        "-m", "torchtitan.train",
        "--module", args.module,
        "--config", args.config,
    ] + args.extra


def build_convert_cmd(args, direction: str) -> list:
    if not args.input_dir or not args.output_dir:
        raise SystemExit(f"--mode {args.mode} requires --input-dir and --output-dir")
    script = f"scripts/checkpoint_conversion/convert_{direction}_hf.py"
    cmd = [sys.executable, script, args.input_dir, args.output_dir,
           "--model_name", args.model_name, "--model_flavor", args.model_flavor]
    if direction == "to":
        if not args.hf_assets_path:
            raise SystemExit("--mode convert-to-hf requires --hf-assets-path")
        cmd += ["--hf_assets_path", args.hf_assets_path]
    return cmd + args.extra


def build_download_cmd(args) -> list:
    if not args.repo_id:
        raise SystemExit("--mode download-assets requires --repo-id")
    cmd = [sys.executable, "scripts/download_hf_assets.py",
           "--repo_id", args.repo_id, "--assets", args.assets]
    token = os.environ.get("HF_TOKEN")
    if token:
        cmd += [f"--hf_token={token}"]
    return cmd + args.extra


def main():
    args = parse_args()
    titan_repo = Path(args.titan_repo).expanduser().resolve()
    validate_repo(titan_repo)

    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN is not set. Put HF_TOKEN=hf_... in dev.env next to this script "
                    "if you need gated Hugging Face weights or tokenizers.")

    if args.mode == "train":
        recipe = HERE / f"{RECIPE_MODULE}.py"
        if args.module == RECIPE_MODULE and not recipe.is_file():
            raise SystemExit(f"Expected the shipped recipe at {recipe} but it is missing.")
        cmd = build_train_cmd(args)
    elif args.mode == "convert-from-hf":
        cmd = build_convert_cmd(args, "from")
    elif args.mode == "convert-to-hf":
        cmd = build_convert_cmd(args, "to")
    else:
        cmd = build_download_cmd(args)

    env = build_env(titan_repo)
    printable = " ".join(cmd)
    log.info("mode=%s", args.mode)
    log.info("cwd=%s", titan_repo)
    log.info("PYTHONPATH=%s", env["PYTHONPATH"])
    log.info("command: %s", printable)

    if args.dry_run:
        log.info("--dry-run set; not executing.")
        return 0

    proc = subprocess.run(cmd, cwd=str(titan_repo), env=env)
    if proc.returncode != 0:
        log.error("command exited with code %s", proc.returncode)
    else:
        log.info("command completed successfully")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
