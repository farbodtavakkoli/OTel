"""Thin launcher for Megatron-LM continued pre-training from a TOML config -- see readme_megatron.md."""

import argparse
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("training/llm/megatron")

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "megatron_cpt_config.toml"

# Keys in [launcher] are torchrun arguments, not Megatron flags.
LAUNCHER_KEYS = ("nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port")

# Path-valued flags, resolved against this folder because the child process
# runs with the Megatron clone as its working directory.
PATH_KEYS = ("data_path", "load", "save", "tensorboard_dir")


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch Megatron-LM continued pre-training from a TOML config.",
    )
    p.add_argument("--megatron-repo", required=True,
                   help="Path to a checked-out https://github.com/NVIDIA/Megatron-LM clone "
                        "(or the /workspace/megatron path inside the NGC container). "
                        "pretrain_gpt.py is run with this as the working directory.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"TOML config to translate into flags (default: {DEFAULT_CONFIG.name}).")
    p.add_argument("--entrypoint", default="pretrain_gpt.py",
                   help="Megatron training entrypoint (default: pretrain_gpt.py). "
                        "pretrain_mamba.py / pretrain_hybrid.py also live at the repo root.")

    # Convenience overrides for the fields that change between runs.
    p.add_argument("--nproc-per-node", type=int,
                   help="Override [launcher].nproc_per_node (GPUs on this node).")
    p.add_argument("--data-path", help="Override [data].data_path (indexed dataset prefix).")
    p.add_argument("--load", help="Override [checkpoint].load (checkpoint to start from).")
    p.add_argument("--save", help="Override [checkpoint].save (where to write checkpoints).")
    p.add_argument("--train-iters", type=int, help="Override [training].train_iters.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Override or add any config key, e.g. --set lr=5e-6 --set finetune=true. "
                        "Repeatable. Values are parsed as TOML scalars.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Everything after --extra is appended verbatim to the Megatron "
                        "command line.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved command and exit without running it.")
    return p.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def flatten(config: dict) -> tuple:
    """Split the TOML into (launcher settings, Megatron key/value pairs); duplicate keys are an error."""
    launcher, flags = {}, {}
    for table, body in config.items():
        if not isinstance(body, dict):
            raise SystemExit(f"top-level key '{table}' must be a TOML table, e.g. [{table}]")
        for key, value in body.items():
            if table == "launcher":
                if key not in LAUNCHER_KEYS:
                    raise SystemExit(f"[launcher].{key} is not a torchrun setting {LAUNCHER_KEYS}")
                launcher[key] = value
            elif key in flags:
                raise SystemExit(f"duplicate config key '{key}' (second occurrence in [{table}])")
            else:
                flags[key] = value
    return launcher, flags


def apply_overrides(launcher: dict, flags: dict, args) -> None:
    """CLI beats TOML. --set values are parsed as TOML so types stay honest."""
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip().replace("-", "_")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # treat an unquoted bare word as a plain string
        (launcher if key in LAUNCHER_KEYS else flags)[key] = value

    if args.nproc_per_node is not None:
        launcher["nproc_per_node"] = args.nproc_per_node
    for cli_value, key in ((args.data_path, "data_path"), (args.load, "load"),
                           (args.save, "save"), (args.train_iters, "train_iters")):
        if cli_value is not None:
            flags[key] = cli_value


def resolve_paths(flags: dict) -> None:
    """Absolutize relative path flags against this folder."""
    for key in PATH_KEYS:
        value = flags.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            flags[key] = str((HERE / value).resolve())


def to_cli(flags: dict) -> list:
    """`num_layers = 32` -> `--num-layers 32`; `swiglu = true` -> `--swiglu`."""
    argv = []
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)          # false means "do not pass the flag"
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def validate(repo: Path, entrypoint: str, launcher: dict, flags: dict) -> None:
    if not repo.is_dir():
        raise SystemExit(f"--megatron-repo does not exist: {repo}")
    if not (repo / entrypoint).is_file():
        raise SystemExit(
            f"{repo} does not look like a Megatron-LM clone (missing {entrypoint}). "
            "Clone it with: git clone https://github.com/NVIDIA/Megatron-LM.git"
        )

    # Megatron reads its data as an indexed dataset: <prefix>.bin plus <prefix>.idx.
    data_path = flags.get("data_path")
    if isinstance(data_path, str) and data_path:
        missing = [s for s in (".bin", ".idx") if not Path(data_path + s).is_file()]
        if missing:
            log.warning("data_path prefix %s is missing %s. Megatron needs the indexed "
                        "format - run preprocess_megatron_data.py first.",
                        data_path, " and ".join(missing))

    load_dir = flags.get("load")
    if isinstance(load_dir, str) and load_dir and not Path(load_dir).is_dir():
        log.warning("checkpoint.load=%s does not exist yet. For continued pre-training this "
                    "must be a Megatron-format checkpoint (convert the HF one with "
                    "Megatron-Bridge first).", load_dir)

    world = int(launcher.get("nproc_per_node", 8)) * int(launcher.get("nnodes", 1))
    model_parallel = (int(flags.get("tensor_model_parallel_size", 1))
                      * int(flags.get("pipeline_model_parallel_size", 1))
                      * int(flags.get("context_parallel_size", 1)))
    if world % model_parallel != 0:
        raise SystemExit(
            f"world size {world} is not divisible by tp*pp*cp = {model_parallel}; "
            "Megatron cannot build the process groups."
        )
    log.info("world size %d, tp*pp*cp = %d, data-parallel degree = %d",
             world, model_parallel, world // model_parallel)

    if flags.get("sequence_parallel") and int(flags.get("tensor_model_parallel_size", 1)) < 2:
        log.warning("sequence_parallel is set but tensor_model_parallel_size < 2; "
                    "sequence parallelism only does something alongside tensor parallelism.")


def main():
    args = parse_args()
    repo = Path(args.megatron_repo).expanduser().resolve()

    config = load_config(Path(args.config).expanduser())
    launcher, flags = flatten(config)
    apply_overrides(launcher, flags, args)
    resolve_paths(flags)
    validate(repo, args.entrypoint, launcher, flags)

    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN is not set. Put HF_TOKEN=hf_... in dev.env next to this script "
                    "if your tokenizer or base checkpoint is gated on the Hugging Face Hub.")

    cmd = [
        "torchrun",
        f"--nproc_per_node={launcher.get('nproc_per_node', 8)}",
        f"--nnodes={launcher.get('nnodes', 1)}",
        f"--node_rank={launcher.get('node_rank', 0)}",
        f"--master_addr={launcher.get('master_addr', 'localhost')}",
        f"--master_port={launcher.get('master_port', 29500)}",
        args.entrypoint,
    ] + to_cli(flags) + args.extra

    env = os.environ.copy()
    # Both upstream example scripts export this before torchrun.
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log.info("config: %s", args.config)
    log.info("cwd: %s", repo)
    log.info("command: %s", " ".join(cmd))

    if args.dry_run:
        log.info("--dry-run set; not executing.")
        return 0

    proc = subprocess.run(cmd, cwd=str(repo), env=env)
    if proc.returncode != 0:
        log.error("command exited with code %s", proc.returncode)
    else:
        log.info("command completed successfully")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
