"""
List the `checkpoint_<step>.pt` files in a training-job directory.

Powers the checkpoint dropdown on the Publish-to-HF modal. Newest
checkpoint first; mtimes are float seconds since epoch, matching what
the rest of the API uses for time fields.
"""

import logging
import os

from fastapi import HTTPException

from cray_infra.training.get_training_job_info import get_job_directory_for_hash

logger = logging.getLogger(__name__)


def list_checkpoints(job_hash: str) -> dict:
    job_directory = get_job_directory_for_hash(job_hash)
    if not os.path.isdir(job_directory):
        raise HTTPException(status_code=404, detail="job directory not found")

    checkpoints = []
    for name in os.listdir(job_directory):
        if not (name.startswith("checkpoint_") and name.endswith(".pt")):
            continue
        try:
            step = int(name[len("checkpoint_") : -len(".pt")])
        except ValueError:
            # Skip files that match the prefix/suffix but don't have an
            # integer step — keeps stray editor swap files from breaking
            # the dropdown.
            continue
        path = os.path.join(job_directory, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        checkpoints.append({"name": name, "step": step, "mtime": mtime})

    checkpoints.sort(key=lambda c: c["step"], reverse=True)
    return {"checkpoints": checkpoints}
