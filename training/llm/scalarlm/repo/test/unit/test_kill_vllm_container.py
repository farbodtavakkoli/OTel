import json
import os
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

SLEEPER = [sys.executable, "-c", "import time; time.sleep(300)"]

WORKER = textwrap.dedent(
    """
    import json, subprocess, sys
    child = subprocess.Popen({sleeper!r})
    with open({pids!r}, "w") as f:
        json.dump({{"worker": __import__("os").getpid(), "grandchild": child.pid}}, f)
    from cray_infra.one_server.create_generate_worker import kill_vllm_container
    kill_vllm_container()
    """
)

ROOT = textwrap.dedent(
    """
    import os, subprocess, sys
    os.environ["SCALARLM_SERVER_PID"] = {root_pid}
    subprocess.Popen([sys.executable, "-c", {worker!r}]).wait()
    import time
    time.sleep(300)
    """
)


def _gone(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _wait_gone(pid, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.2)
    return False


@pytest.fixture
def bystander():
    proc = subprocess.Popen(SLEEPER)
    yield proc
    proc.kill()
    proc.wait()


def _start_root(tmp_path, root_pid_expr):
    pids = str(tmp_path / "pids.json")
    worker = WORKER.format(sleeper=SLEEPER, pids=pids)
    root = subprocess.Popen(
        [sys.executable, "-c", ROOT.format(root_pid=root_pid_expr, worker=worker)]
    )
    return root, pids


def _read_pids(path, root, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path) as f:
                return json.load(f)
        time.sleep(0.2)
    root.kill()
    pytest.fail("worker did not start")


def test_kills_supervisor_worker_and_children_but_not_bystanders(tmp_path, bystander):
    root, pids_path = _start_root(tmp_path, "str(os.getpid())")
    pids = _read_pids(pids_path, root)

    assert root.wait(timeout=120) == -9
    assert _wait_gone(pids["worker"])
    assert _wait_gone(pids["grandchild"])
    assert bystander.poll() is None
    assert psutil.pid_exists(os.getpid())


def test_ignores_server_pid_that_is_not_an_ancestor(tmp_path, bystander):
    root, pids_path = _start_root(tmp_path, repr(str(bystander.pid)))
    pids = _read_pids(pids_path, root)
    try:
        assert _wait_gone(pids["worker"])
        assert _wait_gone(pids["grandchild"])
        assert root.poll() is None
        assert bystander.poll() is None
    finally:
        root.kill()
        root.wait()
