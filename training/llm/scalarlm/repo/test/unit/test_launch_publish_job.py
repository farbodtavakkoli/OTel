import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.training import launch_publish_job as mod


_ENTRYPOINT_TEMPLATE = """\
#!/bin/bash
set -Eeuoxa pipefail

LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

export PYTHONPATH="$LOCAL_DIRECTORY/ml:$LOCAL_DIRECTORY/infra:${PYTHONPATH:-}"

python -m adapters.merge_lora_and_push "$@"
"""


@pytest.fixture
def fake_entrypoint(tmp_path):
    ep = tmp_path / "scripts" / "publish_job_entrypoint.sh"
    ep.parent.mkdir(parents=True)
    ep.write_text(_ENTRYPOINT_TEMPLATE)
    return ep


@pytest.fixture
def mock_config(fake_entrypoint):
    with patch(
        "cray_infra.training.launch_publish_job.get_config",
        return_value={"publish_job_entrypoint": str(fake_entrypoint)},
    ):
        yield


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "deadbeef"
    d.mkdir()
    (d / "config.yaml").write_text("llm_name: foo/bar\n")
    (d / "checkpoint_5.pt").write_bytes(b"")
    (d / "checkpoint_100.pt").write_bytes(b"")
    return d


@pytest.fixture
def resolve_dir(job_dir):
    with patch(
        "cray_infra.training.launch_publish_job.get_job_directory_for_hash",
        return_value=str(job_dir),
    ):
        yield


def _ok(stdout="Submitted batch job 4242\n"):
    return subprocess.CompletedProcess(
        args=["sbatch"], returncode=0, stdout=stdout, stderr=""
    )


def _fail(stderr="boom"):
    return subprocess.CompletedProcess(
        args=["sbatch"], returncode=1, stdout="", stderr=stderr
    )


def test_sbatch_cmd_is_script_only(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    cmd = mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=["--job-dir", "/data/job", "--repo-id", "org/model"],
    )
    assert cmd == ["sbatch", str(publish_dir / "job.sh")]


def test_job_sh_has_sbatch_directives(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    log_path = publish_dir / "publish.log"
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=log_path,
        cli_args=["--job-dir", "/data/job"],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "#!/bin/bash" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --nodes=1" in script
    assert f"#SBATCH --output={log_path}" in script
    assert "#SBATCH --export=ALL,HF_TOKEN,HUGGING_FACE_HUB_TOKEN" in script


def test_job_sh_bakes_cli_args_via_set(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=["--job-dir", "/data/job", "--repo-id", "org/model"],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "set -- " in script
    assert "--job-dir" in script
    assert "/data/job" in script
    assert "--repo-id" in script
    assert "org/model" in script


def test_job_sh_quotes_args_with_special_chars(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=["--commit-message", "has spaces & special chars"],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "'has spaces & special chars'" in script


def test_job_sh_token_never_written_to_script(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=["--job-dir", "/data/job"],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "hf_secret" not in script
    assert "HF_TOKEN=" not in script


def test_job_sh_local_directory_is_statically_resolved(
    tmp_path, fake_entrypoint, mock_config
):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=[],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "BASH_SOURCE" not in script
    expected = str(fake_entrypoint.resolve().parent.parent)
    assert expected in script


def test_job_sh_preserves_entrypoint_body(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=[],
    )
    script = (publish_dir / "job.sh").read_text()
    assert "PYTHONPATH" in script
    assert 'python -m adapters.merge_lora_and_push "$@"' in script


def test_job_sh_is_executable(tmp_path, fake_entrypoint, mock_config):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    mod._build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=publish_dir / "publish.log",
        cli_args=[],
    )
    assert os.access(publish_dir / "job.sh", os.X_OK)


def test_missing_entrypoint_raises_500(tmp_path):
    publish_dir = tmp_path / "publish_123"
    publish_dir.mkdir()
    with patch(
        "cray_infra.training.launch_publish_job.get_config",
        return_value={"publish_job_entrypoint": "/does/not/exist.sh"},
    ):
        with pytest.raises(HTTPException) as exc:
            mod._build_sbatch_argv(
                publish_dir=publish_dir,
                log_path=publish_dir / "publish.log",
                cli_args=[],
            )
    assert exc.value.status_code == 500


def test_submits_with_latest_checkpoint(job_dir, resolve_dir, mock_config):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _ok()

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=False,
            hf_token="hf_secret_xyz",
        )

    assert result["publish_job_id"] == "4242"
    publish_dir = Path(result["publish_dir"])
    assert publish_dir.parent == job_dir
    assert publish_dir.name.startswith("publish_")

    status = json.loads((publish_dir / "status.json").read_text())
    assert status["mode"] == "merged"
    assert status["repo_id"] == "myorg/my-model"
    assert status["phase"] == "queued"
    assert status["publish_job_id"] == "4242"

    cmd = captured["cmd"]
    assert cmd[0] == "sbatch"
    assert cmd[1] == str(publish_dir / "job.sh")
    assert len(cmd) == 2

    job_sh = (publish_dir / "job.sh").read_text()
    assert "hf_secret_xyz" not in " ".join(cmd)
    assert "hf_secret_xyz" not in job_sh

    assert captured["env"]["HF_TOKEN"] == "hf_secret_xyz"
    assert captured["env"]["HUGGING_FACE_HUB_TOKEN"] == "hf_secret_xyz"

    assert "#SBATCH --export=ALL,HF_TOKEN,HUGGING_FACE_HUB_TOKEN" in job_sh
    assert "--checkpoint" in job_sh
    assert "checkpoint_100.pt" in job_sh


def test_submits_with_explicit_checkpoint(job_dir, resolve_dir, mock_config):
    with patch("subprocess.run", return_value=_ok()):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=False,
            hf_token="t",
            checkpoint="checkpoint_5.pt",
        )
    assert result["publish_job_id"] == "4242"
    job_sh = (Path(result["publish_dir"]) / "job.sh").read_text()
    assert "checkpoint_5.pt" in job_sh


def test_passes_through_optional_flags(job_dir, resolve_dir, mock_config):
    with patch("subprocess.run", return_value=_ok()):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=True,
            hf_token="t",
            lora_alpha=64,
            commit_message="release v1",
        )
    job_sh = (Path(result["publish_dir"]) / "job.sh").read_text()
    assert "--private" in job_sh
    assert "--lora-alpha" in job_sh
    assert "64" in job_sh
    assert "--commit-message" in job_sh
    assert "release v1" in job_sh


def test_rejects_unknown_mode(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef", mode="bogus", repo_id="x/y", private=False, hf_token="t"
        )
    assert exc.value.status_code == 400


def test_rejects_missing_token(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef", mode="merged", repo_id="x/y", private=False, hf_token=""
        )
    assert exc.value.status_code == 400


def test_rejects_repo_id_without_slash(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef", mode="merged", repo_id="bad", private=False, hf_token="t"
        )
    assert exc.value.status_code == 400


def test_rejects_checkpoint_path_traversal(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
            checkpoint="../../../etc/passwd",
        )
    assert exc.value.status_code == 400


def test_rejects_unknown_checkpoint(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
            checkpoint="checkpoint_999.pt",
        )
    assert exc.value.status_code == 404


def test_rejects_when_publish_already_in_flight(job_dir, resolve_dir):
    in_flight = job_dir / "publish_111"
    in_flight.mkdir()
    (in_flight / "status.json").write_text(
        json.dumps({"phase": "uploading", "publish_job_id": "9001"})
    )
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef", mode="merged", repo_id="x/y", private=False, hf_token="t"
        )
    assert exc.value.status_code == 409


def test_terminal_status_does_not_block_new_publish(
    job_dir, resolve_dir, mock_config
):
    done = job_dir / "publish_001"
    done.mkdir()
    (done / "status.json").write_text(
        json.dumps({"phase": "done", "publish_job_id": "1"})
    )
    with patch("subprocess.run", return_value=_ok()):
        result = mod.launch_publish_job(
            "deadbeef", mode="merged", repo_id="x/y", private=False, hf_token="t"
        )
    assert result["publish_job_id"] == "4242"


def test_sbatch_failure_writes_error_and_raises(job_dir, resolve_dir, mock_config):
    with patch("subprocess.run", return_value=_fail("queue is full")):
        with pytest.raises(HTTPException) as exc:
            mod.launch_publish_job(
                "deadbeef", mode="merged", repo_id="x/y", private=False, hf_token="t"
            )
    assert exc.value.status_code == 500

    publish_dirs = list(job_dir.glob("publish_*"))
    assert len(publish_dirs) == 1
    state = json.loads((publish_dirs[0] / "status.json").read_text())
    assert state["phase"] == "error"
    assert "queue is full" in state["error"]


def test_get_publish_status_returns_latest(job_dir, resolve_dir):
    older = job_dir / "publish_100"
    newer = job_dir / "publish_200"
    older.mkdir()
    newer.mkdir()
    (older / "status.json").write_text(json.dumps({"phase": "done"}))
    (newer / "status.json").write_text(json.dumps({"phase": "uploading"}))

    state = mod.get_publish_status("deadbeef")
    assert state["phase"] == "uploading"
    assert state["publish_dir"] == str(newer)


def test_get_publish_status_404_when_none(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.get_publish_status("deadbeef")
    assert exc.value.status_code == 404
