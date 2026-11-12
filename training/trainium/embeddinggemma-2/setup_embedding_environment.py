#!/usr/bin/env python3
"""Create the validated native TorchNeuron 2.14 environment without changing the AMI."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-env",
        type=Path,
        required=True,
        help="Existing validated native TorchNeuron 2.14 environment",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path(__file__).resolve().parent / ".venv",
    )
    args = parser.parse_args()
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 12)
    ):
        parser.error("Setup requires Linux x86_64 and Python 3.12.")
    base = args.base_env.resolve(strict=True)
    target = args.venv.resolve()
    if target.exists():
        parser.error(
            "Destination already exists. Choose a new --venv; "
            "existing environments are preserved."
        )
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    probe_script = (
        "import importlib.metadata as m,json,sysconfig;"
        "print(json.dumps({"
        "'site':sysconfig.get_paths()['purelib'],"
        "'versions':{n:m.version(n) for n in "
        "['torch','torch-neuronx','neuronx-cc','nki']}"
        "}))"
    )
    probe = subprocess.check_output(
        [str(base / "bin/python"), "-c", probe_script],
        env=env,
        text=True,
    )
    profile = json.loads(probe)
    recipe = Path(__file__).resolve().parent
    constraints = recipe / "constraints_neuron_2_14.txt"
    pinned = dict(
        line.split("==", 1)
        for line in constraints.read_text().splitlines()
        if line and not line.startswith("#")
    )
    expected = {
        name: pinned[name]
        for name in ("torch", "torch-neuronx", "neuronx-cc", "nki")
    }
    for name, version in expected.items():
        actual = profile["versions"][name]
        normalized_actual = (
            actual.split("+")[0] if name == "torch" else actual
        )
        if normalized_actual != version:
            parser.error(
                f"Unsupported {name}: {actual}; "
                f"validated profile requires {version}."
            )
    subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        check=True,
    )
    site = target / "lib/python3.12/site-packages"
    # addsitedir also executes the AMI's editable-backend registration .pth files.
    (site / "neuron-base.pth").write_text(
        f"import site; site.addsitedir({profile['site']!r})\n"
    )
    (target / "recipe-environment.json").write_text(
        json.dumps({"base_bin": str(base / "bin"), **profile}, indent=2)
        + "\n"
    )
    requirements = [
        line
        for line in (
            recipe / "requirements_embedding.txt"
        ).read_text().splitlines()
        if line and not line.startswith("#")
    ]
    manifest = target / "requirements.host.txt"
    manifest.write_text("\n".join(requirements) + "\n")
    installed_constraints = target / "constraints.txt"
    installed_constraints.write_text(constraints.read_text())
    python = str(target / "bin/python")
    with (target / "setup.log").open("w") as log:
        for command in (
            [
                python,
                "-m",
                "pip",
                "install",
                "-r",
                str(manifest),
                "-c",
                str(installed_constraints),
            ],
            [python, "-m", "pip", "check"],
            [
                python,
                str(recipe / "train_embedding_standalone.py"),
                "--self_test",
            ],
        ):
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    resolved = subprocess.check_output(
        [python, "-m", "pip", "freeze"],
        text=True,
        env=env,
    )
    (target / "requirements.resolved.txt").write_text(resolved)
    print(
        f"Environment ready: {target}. "
        "Run run_embedding_training.sh --preflight before training."
    )


if __name__ == "__main__":
    main()
