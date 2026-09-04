"""Regression tests for the live profile's shell orchestration."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

TEST_COMMAND = Path(__file__).resolve().parents[2] / "cmd" / "test_command.sh"
LIVE_HARNESS = Path(__file__).resolve().parents[1] / "live" / "run_live_server_tests.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_live_profile_stops_when_image_build_fails(tmp_path):
    repo = tmp_path / "repo"
    (repo / "cmd").mkdir(parents=True)
    (repo / "test" / "live").mkdir(parents=True)
    (repo / "cmd" / "test_command.sh").write_text(TEST_COMMAND.read_text())

    _write_executable(repo / "scalarlm", "#!/bin/bash\nexit 23\n")
    _write_executable(
        repo / "test" / "live" / "run_live_server_tests.sh",
        '#!/bin/bash\ntouch "$HARNESS_MARKER"\n',
    )

    runner = tmp_path / "run-test-command.sh"
    _write_executable(
        runner,
        """#!/bin/bash
inspect_args() { :; }
red_bold() { printf '%s\\n' "$*"; }
green_bold() { printf '%s\\n' "$*"; }
blue_bold() { printf '%s\\n' "$*"; }
declare -A args=(
  [test-path]=""
  [--level]="live"
  [--tag]="cray:latest"
  [--coverage-path]="/tmp/coverage"
  [--verbose]="no"
  [--workers]="1"
  [--no-build]="no"
  [--model]="tiny-random/gemma-4-dense"
  [--live-target]="cpu"
  [--live-timeout]="30"
)
source "$1"
""",
    )

    harness_marker = tmp_path / "harness-ran"
    env = os.environ.copy()
    env["HARNESS_MARKER"] = str(harness_marker)
    result = subprocess.run(
        ["bash", str(runner), str(repo / "cmd" / "test_command.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert not harness_marker.exists()
    assert "Live server (cpu, tiny-random/gemma-4-dense) FAILED" in result.stdout


def test_live_profile_forwards_pytest_filters(tmp_path):
    repo = tmp_path / "repo"
    (repo / "cmd").mkdir(parents=True)
    (repo / "test" / "live").mkdir(parents=True)
    (repo / "cmd" / "test_command.sh").write_text(TEST_COMMAND.read_text())

    _write_executable(repo / "scalarlm", "#!/bin/bash\nexit 91\n")
    _write_executable(
        repo / "test" / "live" / "run_live_server_tests.sh",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["HARNESS_ARGS"]).write_text(json.dumps(sys.argv[1:]))
""",
    )

    runner = tmp_path / "run-test-command.sh"
    _write_executable(
        runner,
        """#!/bin/bash
inspect_args() { :; }
red_bold() { printf '%s\\n' "$*"; }
green_bold() { printf '%s\\n' "$*"; }
blue_bold() { printf '%s\\n' "$*"; }
declare -A args=(
  [test-path]=""
  [--level]="live"
  [--keyword]="streaming or queued"
  [--mark]="not slow"
  [--tag]="image:test"
  [--coverage-path]="/tmp/coverage"
  [--verbose]="no"
  [--workers]="1"
  [--no-build]="yes"
  [--model]="org/model name"
  [--live-target]="spark"
  [--live-timeout]="37"
)
source "$1"
""",
    )

    harness_args = tmp_path / "harness-args.json"
    env = os.environ.copy()
    env["HARNESS_ARGS"] = str(harness_args)
    result = subprocess.run(
        ["bash", str(runner), str(repo / "cmd" / "test_command.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(harness_args.read_text()) == [
        "--tag",
        "image:test",
        "--model",
        "org/model name",
        "--target",
        "spark",
        "--timeout",
        "37",
        "--keyword",
        "streaming or queued",
        "--mark",
        "not slow",
    ]


@pytest.mark.parametrize(
    ("signal_number", "exit_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_live_harness_pid_only_signal_stops_named_test_container(
    tmp_path, signal_number, exit_code
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.jsonl"
    test_runner_ready = tmp_path / "test-runner-ready"
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log_path = Path(os.environ["FAKE_DOCKER_LOG"])


def log(entry):
    with log_path.open("a") as stream:
        stream.write(json.dumps(entry) + "\\n")


log(args)
if args[:2] == ["image", "inspect"]:
    raise SystemExit(0)
if args[:2] == ["network", "create"]:
    print("network-id")
    raise SystemExit(0)
if args[:2] == ["network", "rm"] or args[:2] == ["rm", "-f"]:
    raise SystemExit(0)
if args and args[0] == "inspect":
    print("true")
    raise SystemExit(0)
if args and args[0] == "exec":
    print('{"api":"up","vllm":"up"}')
    raise SystemExit(0)
if args and args[0] == "logs":
    raise SystemExit(0)
if args and args[0] == "run" and "--detach" in args:
    print("server-id")
    raise SystemExit(0)
if args and args[0] == "run":
    def stop(signum, _frame):
        log(["SIGNAL", signum])
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    Path(os.environ["FAKE_TEST_RUNNER_READY"]).write_text(str(os.getpid()))
    while True:
        time.sleep(0.05)
raise SystemExit(9)
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(docker_log)
    env["FAKE_TEST_RUNNER_READY"] = str(test_runner_ready)
    env["SCALARLM_MODEL_CACHE"] = str(tmp_path / "model-cache")
    process = subprocess.Popen(
        [
            str(LIVE_HARNESS),
            "--tag",
            "image:test",
            "--model",
            "org/model name",
            "--target",
            "cpu",
            "--timeout",
            "10",
            "--keyword",
            "streaming or queued",
            "--mark",
            "not slow",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    deadline = time.monotonic() + 5
    while not test_runner_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not test_runner_ready.exists():
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(stdout + stderr)

    process.send_signal(signal_number)  # Deliberately signal only the harness PID.
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == exit_code, stdout + stderr
    calls = [json.loads(line) for line in docker_log.read_text().splitlines()]
    test_run = next(
        call for call in calls if call and call[0] == "run" and "--detach" not in call
    )
    test_name = test_run[test_run.index("--name") + 1]
    server_run = next(
        call for call in calls if call and call[0] == "run" and "--detach" in call
    )
    server_name = server_run[server_run.index("--name") + 1]
    network_name = server_run[server_run.index("--network") + 1]

    assert test_name.startswith("scalarlm-live-tests-")
    assert ["-k", "streaming or queued"] == test_run[-4:-2]
    assert ["-m", "not slow"] == test_run[-2:]
    assert ["SIGNAL", signal_number] in calls
    assert ["rm", "-f", test_name] in calls
    assert ["rm", "-f", server_name] in calls
    assert ["network", "rm", network_name] in calls
