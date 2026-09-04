"""Pytest hooks for distributed tests launched via torchrun."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CRAY_ROOT = Path(__file__).resolve().parents[3]

_TORCHRUN_CHILD_ENV = "_TORCHRUN_CHILD"
_ARCH_ENV = "DISTRIBUTED_TEST_ARCH"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "torchrun(nproc, arch): run test under torch.distributed.run",
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("torchrun") is not None:
            item.add_marker(pytest.mark.xdist_group("distributed_torchrun"))


def _is_torchrun_child() -> bool:
    return os.environ.get(_TORCHRUN_CHILD_ENV) == "1"


@pytest.fixture(scope="session", autouse=True)
def distributed_process_group():
    if not _is_torchrun_child():
        yield
        return

    from distributed_benchmarks import setup_distributed, teardown_distributed

    setup_distributed()
    yield
    teardown_distributed()


@pytest.fixture
def distributed_arch(request):
    arch = os.environ.get(_ARCH_ENV)
    if arch is None:
        marker = request.node.get_closest_marker("torchrun")
        if marker is not None:
            arch = marker.kwargs.get("arch")
    if arch is None:
        pytest.fail("DISTRIBUTED_TEST_ARCH is not set for distributed test worker")
    return arch


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    marker = pyfuncitem.get_closest_marker("torchrun")
    if marker is None:
        return None
    if _is_torchrun_child():
        return None

    nproc = marker.kwargs.get("nproc", 2)
    arch = marker.kwargs.get("arch", "cpu")
    env = {
        **os.environ,
        _TORCHRUN_CHILD_ENV: "1",
        _ARCH_ENV: arch,
    }

    # nodeid is relative to pytest's rootdir, but the child is launched with
    # cwd=CRAY_ROOT — so a nodeid collected from anywhere else ("file or
    # directory not found"). Build an absolute file::name selector instead, which
    # resolves no matter where pytest was invoked from.
    node_selector = f"{pyfuncitem.path}::{pyfuncitem.name}"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={nproc}",
        "-m",
        "pytest",
        node_selector,
        "-xvs",
        "-p",
        "no:xdist",
        "-p",
        "no:forked",
    ]
    result = subprocess.run(cmd, env=env, cwd=CRAY_ROOT)
    if result.returncode != 0:
        pytest.fail(
            f"torchrun pytest failed (exit {result.returncode}): {' '.join(cmd)}"
        )
    return True
