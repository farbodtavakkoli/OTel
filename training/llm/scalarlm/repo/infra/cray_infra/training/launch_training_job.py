from cray_infra.util.get_config import get_config

from fastapi import HTTPException, status

import json
import os
import yaml
import time
import subprocess
import datetime
import re
import shutil
import logging

from typing import Dict

logger = logging.getLogger(__name__)


async def launch_training_job(train_args: Dict):
    await wait_for_slurm()

    validate_gpu_request(train_args)

    if job_already_exists(train_args):
        logging.info(f"Job already exists: {train_args['job_directory']}")
        return get_existing_job_info(train_args)

    make_training_directory(train_args)

    start_slurm_job(train_args)

    return get_existing_job_info(train_args)


def validate_gpu_request(train_args: Dict):
    """
    Reject a GPU-requesting job up front when no SLURM node has a GPU.
    Without this, sbatch happily queues the job and it sits PENDING
    forever (or worse, runs on CPU silently with the
    `--gres=gpu` flag dropped). Raising here surfaces the mismatch
    to the SDK caller as a 400 instead of a phantom-pending job.

    `gpus: 0` in the job config opts out of the check for legit
    CPU-only training.
    """
    requested_gpus = train_args.get("gpus", 1)
    if requested_gpus < 1:
        return
    if not is_gpu_job(train_args):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Training job requested gpus={requested_gpus}, but no SLURM "
                "node advertises a GPU (no `Gres=gpu` in `scontrol show nodes`). "
                "Submit on a cluster with GPU nodes, or set `gpus: 0` in the "
                "job config to run on CPU."
            ),
        )


async def wait_for_slurm():
    config = get_config()

    # wait up to 30 seconds for slurm to be ready
    start_time = time.time()

    while time.time() - start_time < config.get("slurm_wait_time", 30):
        try:
            subprocess.run(["squeue"], check=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(1)


def job_already_exists(train_args: Dict):
    config = get_config()

    train_args_path = os.path.join(train_args["job_directory"], "status.json")

    return os.path.exists(train_args_path)


def make_training_directory(train_args: Dict):
    config = get_config()

    if not "llm_name" in train_args:
        train_args["llm_name"] = config["model"]

    job_directory = get_training_job_directory(train_args)

    os.makedirs(job_directory, exist_ok=True)

    with open(os.path.join(job_directory, "config.yaml"), "w") as f:
        yaml.dump(train_args, f)


def get_training_job_directory(train_args: Dict):
    return train_args["job_directory"]


def start_slurm_job(train_args):
    run_command = create_slurm_run_command(train_args)

    run_sbatch(run_command, train_args)


def create_slurm_run_command(train_args):

    run_command = ["sbatch"]

    tasks_per_node = get_tasks_per_node(train_args)
    run_command += [f"--ntasks-per-node={tasks_per_node}"]
    logger.info(f"tasks_per_node: {tasks_per_node}")

    if is_gpu_job(train_args):
        run_command += [f"--gres=gpu:{tasks_per_node}"]

    node_count = get_node_count(train_args)
    run_command += [f"--nodes={node_count}"]
    logger.info(f"node_count: {node_count}")

    # PR #5: give the job the whole node. A training job already asks for every
    # GPU it intends to use, and letting an unrelated job land on the same node
    # perturbs both the collectives and any timing measured from them.
    run_command += ["--exclusive"]

    cpu_per_task = get_cpu_per_task(train_args)
    run_command += [f"--cpus-per-task={cpu_per_task}"]
    logger.info(f"cpu_per_task: {cpu_per_task}")

    train_time_limit = get_train_time_limit(train_args)
    run_command += [f"--time={train_time_limit}"]
    logger.info(f"train_time_limit: {train_time_limit}")

    # `--signal=B:TERM@grace` makes SLURM send SIGTERM to the batch
    # shell `grace` seconds before the slice's --time runs out. The
    # bash entrypoint `exec`s mpirun, so this signal lands on mpirun
    # directly; mpirun forwards to ranks; the trainer's SIGTERM handler
    # sets stop_flag, the loop drains and checkpoints cleanly.
    # restart_megatron_jobs (the API server's periodic reconciler)
    # then sees the job missing from squeue with status=TRAINING and
    # queues the next slice — see docs/training-lifecycle.md §5.4.
    signal_grace = get_config().get("signal_grace_seconds", 300)
    run_command += [f"--signal=B:TERM@{signal_grace}"]

    slurm_log_file = os.path.join(
        get_training_job_directory(train_args), "slurm-%j.out"
    )
    run_command += [f"--output={slurm_log_file}"]

    run_command += [f"--job-name", os.path.basename(train_args["job_directory"])]
    logger.info(f"job_name: {os.path.basename(train_args['job_directory'])}")

    config_path = os.path.join(get_training_job_directory(train_args), "config.yaml")
    train_job_entrypoint = get_train_job_entrypoint(train_args)

    run_command += [train_job_entrypoint]

    return run_command


def get_train_job_entrypoint(train_args: Dict):
    config = get_config()
    train_job_entrypoint_script = config["train_job_entrypoint"]

    # Copy the script to the job directory
    job_directory = get_training_job_directory(train_args)

    train_job_entrypoint = os.path.join(job_directory, "train_job_entrypoint.sh")

    shutil.copyfile(train_job_entrypoint_script, train_job_entrypoint)

    # Replace the REPLACE_CONFIG_PATH with the config path within the job directory
    with open(train_job_entrypoint, "r") as f:
        entrypoint_script = f.read()

    entrypoint_script = entrypoint_script.replace(
        "REPLACE_CONFIG_PATH", os.path.join(job_directory, "config.yaml")
    )

    with open(train_job_entrypoint, "w") as f:
        f.write(entrypoint_script)

    return train_job_entrypoint


def get_tasks_per_node(train_args: Dict):
    requested_gpu_count = train_args.get("gpus", 1)

    max_task_count = train_args.get("max_gpus", get_max_gpu_count_from_slurm())

    return max(1, min(requested_gpu_count, max_task_count))


def get_max_gpu_count_from_slurm():
    scontrol_command = ["scontrol", "show", "nodes"]

    scontrol_output = subprocess.check_output(scontrol_command).decode()

    max_gpu_count = 1

    for line in scontrol_output.split("\n"):
        if "Gres=gpu" in line:
            max_gpu_count = max(
                int(re.search(r"Gres=gpu:(\d+)", line).group(1)), max_gpu_count
            )

    return max_gpu_count


def is_gpu_job(train_args: Dict):
    scontrol_command = ["scontrol", "show", "nodes"]

    scontrol_output = subprocess.check_output(scontrol_command).decode()

    return "Gres=gpu" in scontrol_output


def get_node_count(train_args: Dict):
    requested_node_count = train_args.get("nodes", 1)

    max_node_count = get_max_node_count_from_slurm()

    return min(requested_node_count, max_node_count)


def get_max_node_count_from_slurm():
    scontrol_command = ["scontrol", "show", "nodes"]

    scontrol_output = subprocess.check_output(scontrol_command).decode()

    max_node_count = 0

    for line in scontrol_output.split("\n"):
        if "NodeName" in line:
            max_node_count += 1

    assert max_node_count > 0, f"No nodes found in slurm config: {scontrol_output}"

    return max_node_count


def get_cpu_per_task(train_args: Dict):
    tasks_per_node = max(get_tasks_per_node(train_args), get_max_gpu_count_from_slurm())

    total_cpu_count = get_total_cpu_count_from_slurm()

    return total_cpu_count // tasks_per_node


def get_total_cpu_count_from_slurm():
    scontrol_command = ["scontrol", "show", "nodes"]

    scontrol_output = subprocess.check_output(scontrol_command).decode()

    total_cpu_count = 1

    for line in scontrol_output.split("\n"):
        if "CPUTot" in line:
            total_cpu_count = max(
                int(re.search(r"CPUTot=(\d+)", line).group(1)), total_cpu_count
            )

    return total_cpu_count


def get_train_time_limit(train_args: Dict):
    # Returns the per-SLURM-slice --time, NOT the user's total training
    # budget. `train_args["timeout"]` is the total time the user wants
    # the job to run for; each SLURM slice is capped at max_train_time
    # (cluster's hard cap on a single job). On per-slice timeout the
    # trainer checkpoints, exits, and restart_megatron_jobs requeues
    # (docs/training-lifecycle.md §5.4). TimeoutCallback stops
    # training once accumulated elapsed across slices reaches
    # `train_args["timeout"]`.
    train_time = train_args.get("timeout", None)

    config = get_config()

    extra_training_seconds = config.get("extra_training_seconds", 0)
    max_train_time = config.get("max_train_time", 14400)

    if train_time is None:
        train_time = max_train_time
    else:
        train_time = min(train_time, max_train_time)

    # convert seconds to DD:HH:MM:SS
    return format_timedelta(
        datetime.timedelta(seconds=train_time + extra_training_seconds)
    )


def format_timedelta(delta):
    days = delta.days
    hours, remaining_seconds = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"


def run_sbatch(run_command, train_args):

    clean_environs = os.environ.copy()
    clean_environs = {
        k: v for k, v in clean_environs.items() if not k.startswith("PMI")
    }

    write_job_status("QUEUED", train_args, {"start_time": time.time()})

    logger.info(f"sbatch run_command: {' '.join(run_command)}")

    logger.info(f"cwd: {get_training_job_directory(train_args)}")

    result = subprocess.run(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=get_training_job_directory(train_args),
        env=clean_environs,
    )

    if result.returncode != 0:
        result_output = result.stdout.decode("utf-8") + result.stderr.decode("utf-8")
        write_job_status("FAILED", train_args, {"output": result_output})

    job_id = get_job_id_from_sbatch_output(result.stdout.decode("utf-8"))

    write_job_status("QUEUED", train_args, {"job_id": job_id})


def get_job_id_from_sbatch_output(sbatch_output):
    logger.info(f"sbatch_output: {sbatch_output}")
    return re.search(r"Submitted batch job (\d+)", sbatch_output).group(1)


def write_job_status(status, train_args, extra_info):
    job_status = {
        "status": status,
        **extra_info,
    }

    status_path = os.path.join(get_training_job_directory(train_args), "status.json")

    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            existing_job_status = json.load(f)
            job_status = {**existing_job_status, **job_status}

    with open(status_path, "w") as f:
        json.dump(job_status, f)


def get_existing_job_info(train_args):
    with open(
        os.path.join(get_training_job_directory(train_args), "status.json"), "r"
    ) as f:
        job_info = json.load(f)
        job_info["job_directory"] = get_training_job_directory(train_args)
        job_info["model_name"] = os.path.basename(
            get_training_job_directory(train_args)
        )
        return job_info
