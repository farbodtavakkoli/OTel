from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.training.get_training_job_info import get_job_directory_for_hash

from cray_infra.util.get_config import get_config

import aiofiles
import os
import json
import logging


logger = logging.getLogger(__name__)


def _slurm_sort_key(log_file_path: str):
    """Order slurm-<job_id>.out files by their numeric job ID.

    Returns a (bucket, value) tuple so the sort is total even when a filename
    doesn't carry a parseable integer ID (e.g. some stray slurm-*.out file):
    parseable IDs sort first in numeric order (bucket 0), anything else falls
    to the end in stable name order (bucket 1) instead of crashing the whole
    log stream.
    """
    name = os.path.basename(log_file_path)
    middle = name[len("slurm-") : -len(".out")]
    try:
        return (0, int(middle))
    except ValueError:
        logger.warning(
            "Log file %s has a non-numeric job id; ordering it after numbered "
            "slices",
            name,
        )
        return (1, name)


def training_logs_generator(model_name: str, starting_line_number: int):
    config = get_config()

    if model_name == "latest":
        latest_model_name = get_latest_model()
        if latest_model_name is None:
            raise FileNotFoundError("Could not find any models")
        model_name = latest_model_name

    job_directory = get_job_directory_for_hash(model_name)

    # Find the log file inside the job directory, it will be named "slurm-<job_id>.out, but we don't know the job_id yet
    log_files = []

    for file in os.listdir(job_directory):
        if file.startswith("slurm-") and file.endswith(".out"):
            log_file = os.path.join(job_directory, file)
            log_files.append(log_file)

    # Each resume/slice writes its own slurm-<job_id>.out, and job IDs grow
    # monotonically with time — so ordering by job ID is chronological order.
    # Sort by the *parsed integer* ID, not the string: a plain lexicographic
    # sort puts "slurm-1000.out" before "slurm-999.out" (because '1' < '9'),
    # which would stitch a later slice ahead of an earlier one across any
    # digit-width boundary (9->10, 99->100, ...). Continuous line numbering is
    # assigned by position in this concatenation, so a wrong order also makes
    # the UI's line-number resume skip the misordered slice.
    log_files.sort(key=_slurm_sort_key)

    logger.info(f"Found log files: {log_files}")

    if len(log_files) == 0:
        raise FileNotFoundError(f"Could not find log file in {job_directory}")

    async def generate():
        line_number = 0
        for log_file in log_files:
            async with aiofiles.open(log_file, mode="r") as f:
                async for line in f:
                    if line_number < starting_line_number:
                        line_number += 1
                        continue

                    yield json.dumps({"line": line.rstrip(), "line_number": line_number}) + "\n"
                    line_number += 1

    return generate()
