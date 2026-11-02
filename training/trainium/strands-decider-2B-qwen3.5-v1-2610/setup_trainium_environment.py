"""Create an application environment without modifying the native Neuron base."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from decider_recipe_utils import UPSTREAM_REVISION, require, write_json


def main(args):
    require(sys.platform == "linux", "Setup requires a Linux Trainium host")
    require(
        os.geteuid() != 0,
        "Run setup as an ordinary user in a private writable work directory, not root",
    )
    os.umask(0o077)
    work = args.work.expanduser().resolve()
    package = Path(__file__).resolve().parent
    require(
        not work.is_relative_to(package),
        "Keep the writable work directory outside the source package",
    )
    require(
        not work.exists(), "Work directory already exists; choose a fresh directory"
    )
    base_python = args.base_python.expanduser().absolute()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    probe = (
        "import json,site,sys,importlib.metadata as m; import torch,torch_neuronx; "
        "print(json.dumps(dict(version=list(sys.version_info[:2]),"
        "sites=site.getsitepackages(),torch=m.version('torch'),"
        "neuron=m.version('torch-neuronx'))))"
    )
    info = json.loads(
        subprocess.check_output(
            [str(base_python), "-c", probe], env=env, text=True
        ).splitlines()[-1]
    )
    require(
        info["version"] == [3, 12]
        and info["torch"] == "2.14.0"
        and info["neuron"] == "2.14.3.0.1+b1b6c66",
        "Expected Python 3.12, torch 2.14.0, native torch-neuronx 2.14.3.0.1+b1b6c66; "
        "use the documented AWS Neuron image",
    )
    work.mkdir(parents=True)

    def run(*cmd):
        subprocess.run([str(x) for x in cmd], env=env, check=True, cwd=work)

    run(base_python, "-m", "venv", work / "venv")
    python = work / "venv/bin/python"
    site_dir = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        env=env,
    ).strip()
    Path(site_dir, "neuron-base.pth").write_text(
        "import site; "
        + "; ".join("site.addsitedir(" + repr(p) + ")" for p in info["sites"])
        + "\n"
    )
    upstream = work / "upstream"
    run("git", "init", upstream)
    run(
        "git",
        "-C",
        upstream,
        "remote",
        "add",
        "origin",
        "https://github.com/strands-labs/strands-decider.git",
    )
    run("git", "-C", upstream, "fetch", "--depth=1", "origin", UPSTREAM_REVISION)
    run("git", "-C", upstream, "checkout", "--detach", UPSTREAM_REVISION)
    run(
        python,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--index-url",
        "https://pypi.org/simple",
        "-c",
        package / "constraints_neuron.txt",
        "-r",
        package / "requirements_decider_training.txt",
    )
    run(
        python,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-deps",
        "--no-build-isolation",
        upstream,
    )
    run(python, "-m", "pip", "check")
    for sub in ("cache", "compiled", "data", "results"):
        (work / sub).mkdir()
    write_json(
        work / "environment.json",
        dict(
            base_python=str(base_python), base=info, upstream_revision=UPSTREAM_REVISION
        ),
    )
    with (work / "environment.freeze.txt").open("x") as out:
        subprocess.run(
            [str(python), "-m", "pip", "freeze"], env=env, stdout=out, check=True
        )
    print("Application environment ready:", work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    main(parser.parse_args())
