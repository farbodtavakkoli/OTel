"""
Build the rotating file handlers used by the API/vllm/megatron
processes. Lives in its own module (not in main.py) so unit tests
can exercise rotation without paying for `import torch` and the
rest of the vLLM bootstrap chain that main.py drags in.
"""

import logging
import logging.handlers
import os
from typing import List

from cray_infra.util.get_config import get_config


def get_log_file_handlers() -> List[logging.Handler]:
    """
    One RotatingFileHandler per server in `server_list`. Each file
    rotates at `log_max_bytes` and keeps `log_backup_count` backups.

    Without rotation the API process writes
    /app/cray/nfs/logs/{vllm,megatron,api}.log forever; once the small
    NFS PVC operators allocate for slurm config + logs (974 MB on the
    gemma4 cluster) hits 100%, slurm.conf writes start failing and
    the pod loses node config. The defaults in default_config.py
    (10 MB × 5 backups × 3 files = 180 MB worst case) are sized for
    that volume.
    """
    config = get_config()

    log_base_path = config["log_directory"]
    os.makedirs(log_base_path, exist_ok=True)

    server_list = config["server_list"]
    server_names = [s.strip() for s in server_list.split(",")]
    if server_names[0] == "all":
        server_names = ["vllm", "megatron", "api"]

    max_bytes = int(config.get("log_max_bytes", 10 * 1024 * 1024))
    backup_count = int(config.get("log_backup_count", 5))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handlers: List[logging.Handler] = []
    for server_name in server_names:
        log_file_path = os.path.join(log_base_path, f"{server_name}.log")
        handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        handlers.append(handler)

    return handlers
