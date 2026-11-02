"""Run-directory, environment, logging, hashing, and result helpers."""

import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from embedding_training_config import COMPILE_CACHE_DIRECTORIES, DEPENDENCY_NAMES, QUIET_LOGGERS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def runtime_preflight():
    """Validate the native environment before downloading or compiling a model."""
    metadata = Path(sys.prefix) / "recipe-environment.json"
    executable_paths = [str(Path(sys.executable).parent)]
    if metadata.exists():
        executable_paths.append(json.loads(metadata.read_text())["base_bin"])
    os.environ["PATH"] = os.pathsep.join(
        executable_paths + [os.environ.get("PATH", "")]
    )
    if not shutil.which("neuronx-cc"):
        raise RuntimeError(
            "neuronx-cc is not on PATH. Run setup_embedding_environment.py and use its environment."
        )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    import torch
    import torchvision
    from PIL import Image  # noqa: F401 - verifies the processor dependency

    if not hasattr(torch, "neuron") or not torch.neuron.is_available():
        raise RuntimeError(
            "Native torch.neuron is unavailable; use the validated native environment."
        )
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "compiler": shutil.which("neuronx-cc"),
    }


def validate_arguments(args, default_model):
    if args.preflight and (args.prepare_only or args.self_test):
        raise ValueError(
            "--preflight cannot be combined with --prepare_only or --self_test."
        )
    if args.model_name != default_model:
        raise ValueError(
            "This release validates only google/embeddinggemma-2; "
            "other architectures need a tested profile."
        )
    numeric_values = (
        args.batch_size,
        args.eval_batch_size,
        args.retrieval_batch_size,
        args.epochs,
        args.max_seq_length,
        args.eval_max_seq_length,
    )
    if (
        min(numeric_values) < 1
        or args.eval_corpus_size < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or args.max_steps == 0
        or args.max_steps < -1
    ):
        raise ValueError(
            "Invalid batch, length, epoch, learning-rate or step setting."
        )


def prepare_run(args, recipe_dir):
    args.train_file = args.train_file.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    args.model_cache_dir = args.model_cache_dir.resolve()

    compile_cache = (args.compile_cache_dir or output / "compiled").resolve()
    for variable, directory_name in COMPILE_CACHE_DIRECTORIES:
        cache_directory = compile_cache / directory_name
        cache_directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache_directory)

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    write_json(output / "config.json", vars(args))
    source_paths = [path for path in recipe_dir.iterdir() if path.is_file()]
    tests_directory = recipe_dir / "tests"
    if tests_directory.is_dir():
        source_paths.extend(
            path
            for path in tests_directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    write_json(
        output / "source_hashes.json",
        {
            str(path.relative_to(recipe_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_paths
            if path.name != ".DS_Store"
        },
    )
    return output


def collect_environment(device):
    versions = {}
    for name in DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "python": sys.version,
        "python_executable": sys.executable,
        "sys_path": sys.path,
        "device": device,
    }


def hash_artifact(directory):
    artifact_hash = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            artifact_hash.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    artifact_hash.update(block)
    return artifact_hash.hexdigest()
