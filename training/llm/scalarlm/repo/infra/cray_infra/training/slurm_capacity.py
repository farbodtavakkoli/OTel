import logging
import re
import subprocess
from typing import List

import torch

logger = logging.getLogger(__name__)

_GPU_RE = re.compile(r"Gres=gpu:(\d+)")
_NODENAME_RE = re.compile(r"^NodeName=(\S+)")
_STATE_RE = re.compile(r"\bState=(\S+)")

_HEALTHY_PRIMARY_STATES = frozenset(
    {"IDLE", "MIXED", "ALLOCATED", "COMPLETING", "RESERVED", "RESUMING"}
)
_UNHEALTHY_FLAGS = frozenset({"DOWN", "DRAIN", "FAIL", "NOT_RESPONDING", "INVALID"})


def get_slurm_nodes() -> List[dict]:
    try:
        result = subprocess.run(
            ["scontrol", "show", "nodes"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug("scontrol unavailable: %s", e)
        return []

    if result.returncode != 0:
        logger.debug(
            "scontrol show nodes failed (rc=%s): %s",
            result.returncode,
            result.stderr.strip(),
        )
        return []

    nodes: List[dict] = []
    current: dict | None = None
    for line in result.stdout.splitlines():
        m = _NODENAME_RE.match(line.strip())
        if m:
            if current is not None:
                nodes.append(current)
            current = {"name": m.group(1), "gpus": 0, "state": ""}
        if current is None:
            continue
        gm = _GPU_RE.search(line)
        if gm:
            current["gpus"] = int(gm.group(1))
        sm = _STATE_RE.search(line)
        if sm and not current["state"]:
            current["state"] = sm.group(1)
    if current is not None:
        nodes.append(current)

    return nodes


def is_healthy_state(state: str) -> bool:
    if not state:
        return False
    if "*" in state:
        return False
    segments = state.split("+")
    primary = segments[0]
    if primary not in _HEALTHY_PRIMARY_STATES:
        return False
    for flag in segments[1:]:
        if flag in _UNHEALTHY_FLAGS:
            return False
    return True


def count_slurm_nodes() -> int:
    nodes = get_slurm_nodes()
    if nodes:
        return len(nodes)
    return 1


def count_slurm_gpus() -> int:
    nodes = get_slurm_nodes()
    if nodes:
        return sum(n["gpus"] for n in nodes)

    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception as e:
        logger.debug("torch.cuda.device_count() failed: %s", e)
    return 0
