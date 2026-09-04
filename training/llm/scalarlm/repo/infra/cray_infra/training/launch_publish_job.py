"""
Submit a publish-to-HF SLURM job for a completed training run.

The publish job runs `adapters.merge_lora_and_push` on a megatron pod
because that's where the base model is already cached and where there's
GPU + RAM headroom for `merge_and_unload`. The API pod's job is just
to construct the sbatch invocation, hand it the HF token via an
env-var export, and return the submission's job id.

See ui/docs/publish-to-hf.md for the protocol this drives.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from cray_infra.training.get_training_job_info import get_job_directory_for_hash
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


# Phases reported in status.json that mean the publish is still in flight
# — used by the "one publish at a time per training job" guard.
_NON_TERMINAL_PHASES = frozenset(
    {"queued", "validating", "loading_base", "merging", "saving", "uploading"}
)


def launch_publish_job(
    job_hash: str,
    *,
    mode: str,
    repo_id: str,
    private: bool,
    hf_token: str,
    checkpoint: Optional[str] = None,
    lora_alpha: Optional[int] = None,
    commit_message: Optional[str] = None,
) -> dict:
    """
    Submit the SLURM publish job. Returns the publish_job_id, the
    publish-dir path the UI can poll, and the initial QUEUED status.

    Raises HTTPException on validation failure or 409 if a publish for
    this job_hash is already mid-flight.
    """
    if mode not in ("merged", "adapter"):
        raise HTTPException(
            status_code=400, detail=f"unknown mode '{mode}'"
        )
    if not repo_id or "/" not in repo_id:
        raise HTTPException(
            status_code=400, detail="repo_id must be 'owner/name'"
        )
    if not hf_token:
        raise HTTPException(
            status_code=400, detail="hf_token is required"
        )

    job_directory = get_job_directory_for_hash(job_hash)
    if not os.path.isdir(job_directory):
        raise HTTPException(status_code=404, detail="job directory not found")

    # Resolve checkpoint relative to the job dir; fall back to latest.
    checkpoint_path = _resolve_checkpoint(Path(job_directory), checkpoint)

    _reject_if_publish_in_flight(Path(job_directory))

    publish_dir = Path(job_directory) / f"publish_{int(time.time())}"
    publish_dir.mkdir(parents=True, exist_ok=False)
    status_path = publish_dir / "status.json"
    log_path = publish_dir / "publish.log"

    # Write the initial status before sbatch lands so the UI's poller
    # can find the file even before SLURM has scheduled the job.
    _write_initial_status(status_path, mode=mode, repo_id=repo_id)

    cli_args = _build_cli_args(
        job_directory=job_directory,
        publish_dir=publish_dir,
        status_path=status_path,
        mode=mode,
        repo_id=repo_id,
        private=private,
        checkpoint_path=checkpoint_path,
        lora_alpha=lora_alpha,
        commit_message=commit_message,
    )

    sbatch_cmd = _build_sbatch_argv(
        publish_dir=publish_dir,
        log_path=log_path,
        cli_args=cli_args,
    )

    # Token enters the SLURM job via env export only — never on argv.
    # `--export=ALL,HF_TOKEN=...` extends the calling env rather than
    # replacing it, so PATH and friends survive.
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    logger.info("sbatch (publish) cmd: %s", _scrub_argv_for_log(sbatch_cmd))
    result = subprocess.run(
        sbatch_cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=job_directory,
    )
    if result.returncode != 0:
        # Surface the failure into status.json so the UI sees it.
        msg = (result.stdout + result.stderr).strip() or "sbatch failed"
        _update_status(status_path, phase="error", error=msg, completed_at=time.time())
        raise HTTPException(
            status_code=500,
            detail=f"sbatch failed: {msg[:200]}",
        )

    publish_job_id = _parse_sbatch_jobid(result.stdout)
    _update_status(
        status_path,
        publish_job_id=publish_job_id,
        phase="queued",
        started_at=time.time(),
    )

    return {
        "publish_job_id": publish_job_id,
        "publish_dir": str(publish_dir),
        "status": "QUEUED",
    }


def _resolve_checkpoint(job_dir: Path, requested: Optional[str]) -> Path:
    if requested:
        # Refuse path traversal — only a basename is accepted, resolved
        # under the job directory.
        if "/" in requested or requested in (".", ".."):
            raise HTTPException(
                status_code=400,
                detail="checkpoint must be a basename within the job directory",
            )
        candidate = job_dir / requested
        if not candidate.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"checkpoint not found: {requested}",
            )
        return candidate

    candidates = []
    for p in job_dir.iterdir():
        name = p.name
        if not (name.startswith("checkpoint_") and name.endswith(".pt")):
            continue
        try:
            step = int(name[len("checkpoint_") : -len(".pt")])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"no checkpoints in {job_dir}",
        )
    candidates.sort()
    return candidates[-1][1]


def _reject_if_publish_in_flight(job_dir: Path) -> None:
    for child in sorted(job_dir.glob("publish_*"), reverse=True):
        status_file = child / "status.json"
        if not status_file.is_file():
            continue
        try:
            with status_file.open() as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("phase") in _NON_TERMINAL_PHASES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"a publish is already in flight at {child.name} "
                    f"(phase={state.get('phase')}); cancel it first"
                ),
            )


def _build_cli_args(
    *,
    job_directory: str,
    publish_dir: Path,
    status_path: Path,
    mode: str,
    repo_id: str,
    private: bool,
    checkpoint_path: Path,
    lora_alpha: Optional[int],
    commit_message: Optional[str],
) -> list[str]:
    args: list[str] = [
        "--job-dir",
        job_directory,
        "--repo-id",
        repo_id,
        "--mode",
        mode,
        "--checkpoint",
        str(checkpoint_path),
        "--status-file",
        str(status_path),
        "--output-dir",
        str(publish_dir / ("merged" if mode == "merged" else "adapter")),
    ]
    if private:
        args.append("--private")
    if lora_alpha is not None:
        args += ["--lora-alpha", str(lora_alpha)]
    if commit_message:
        args += ["--commit-message", commit_message]
    return args


def _build_sbatch_argv(
    *,
    publish_dir: Path,
    log_path: Path,
    cli_args: list[str],
) -> list[str]:
    config = get_config()
    entrypoint = config.get(
        "publish_job_entrypoint",
        "/app/cray/scripts/publish_job_entrypoint.sh",
    )

    try:
        entrypoint_content = Path(entrypoint).read_text()
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"could not read publish entrypoint {entrypoint}: {e}",
        )

    # The entrypoint uses `dirname "${BASH_SOURCE[0]}"` to locate the repo
    # root, which breaks when job.sh lives inside publish_dir instead of
    # scripts/. Resolve the correct path at generation time and substitute it.
    local_directory = shlex.quote(str(Path(entrypoint).resolve().parent.parent))
    entrypoint_content = re.sub(
        r"LOCAL_DIRECTORY=.*\n",
        f"LOCAL_DIRECTORY={local_directory}\n",
        entrypoint_content,
    )

    # Inject #SBATCH directives after the shebang and bake cli_args in via
    # `set --` so the entrypoint body's "$@" still works. This produces a
    # single self-contained job.sh — no separate entrypoint copy needed.
    # sbatch is then called with no positional args beyond the script path,
    # avoiding a Slurm bug where args after the script name are silently dropped.
    lines = entrypoint_content.splitlines(keepends=True)
    shebang = lines[0]
    rest = "".join(lines[1:])

    job_name = f"publish-{publish_dir.parent.name[:8]}"
    sbatch_block = (
        f"#SBATCH --ntasks-per-node=1\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --output={log_path}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --export=ALL,HF_TOKEN,HUGGING_FACE_HUB_TOKEN\n"
    )
    set_args = "set -- " + " ".join(shlex.quote(a) for a in cli_args) + "\n"

    job_script = shebang + sbatch_block + "\n" + set_args + rest

    job_script_path = publish_dir / "job.sh"
    job_script_path.write_text(job_script)
    os.chmod(job_script_path, 0o755)

    return ["sbatch", str(job_script_path)]


_SBATCH_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def _parse_sbatch_jobid(stdout: str) -> str:
    match = _SBATCH_JOBID_RE.search(stdout)
    if not match:
        raise HTTPException(
            status_code=500,
            detail=f"could not parse sbatch output: {stdout!r}",
        )
    return match.group(1)


def _write_initial_status(path: Path, *, mode: str, repo_id: str) -> None:
    _update_status(
        path,
        mode=mode,
        repo_id=repo_id,
        phase="queued",
        publish_job_id=None,
        started_at=None,
        completed_at=None,
        error=None,
        repo_url=None,
    )


def _update_status(path: Path, **fields) -> None:
    """Merge `fields` into status.json and rewrite atomically."""
    state: dict = {}
    if path.is_file():
        try:
            with path.open() as f:
                state = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            state = {}
    state.update(fields)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _scrub_argv_for_log(argv: list[str]) -> list[str]:
    """Defensive — token shouldn't be on argv at all, but redact any
    arg that looks like it just in case."""
    out = []
    for a in argv:
        if "HF_TOKEN=" in a:
            out.append(a.split("HF_TOKEN=")[0] + "HF_TOKEN=***")
        else:
            out.append(a)
    return out


def get_publish_status(job_hash: str) -> dict:
    """Return the freshest publish status.json for this job, or 404."""
    job_directory = Path(get_job_directory_for_hash(job_hash))
    candidates = sorted(
        job_directory.glob("publish_*"), key=lambda p: p.name, reverse=True
    )
    for child in candidates:
        status_file = child / "status.json"
        if status_file.is_file():
            try:
                with status_file.open() as f:
                    state = json.load(f) or {}
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Bad status file %s: %s", status_file, e)
                continue
            state.setdefault("publish_dir", str(child))
            return state
    raise HTTPException(
        status_code=404, detail="no publish has been submitted for this job"
    )


def cancel_publish_job(job_hash: str) -> dict:
    """
    `scancel` the latest publish job for this training run and update
    its status.json to phase=error / error="cancelled by user" so the
    UI's poller sees the transition. No-ops cleanly when the publish
    has already finished — returns the current state, not a 404.
    """
    state = get_publish_status(job_hash)
    publish_dir = Path(state["publish_dir"])
    status_path = publish_dir / "status.json"

    if state.get("phase") not in _NON_TERMINAL_PHASES:
        return state

    publish_job_id = state.get("publish_job_id")
    if publish_job_id is not None:
        try:
            subprocess.run(
                ["scancel", str(publish_job_id)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("scancel failed for publish %s: %s", publish_job_id, e)

    _update_status(
        status_path,
        phase="error",
        error="cancelled by user",
        completed_at=time.time(),
    )
    return get_publish_status(job_hash)
