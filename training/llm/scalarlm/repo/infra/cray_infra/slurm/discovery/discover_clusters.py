from cray_infra.util.get_config import get_config

import torch

from atomicwrites import atomic_write

import subprocess
import time
import socket
import os
import json
import re

import logging

logger = logging.getLogger(__name__)

slurm_config_path = "/app/cray/infra/slurm_configs/slurm.conf"
cluster_info_path = "/app/cray/infra/slurm_configs/cluster_info.json"
cgroup_config_path = "/app/cray/infra/slurm_configs/cgroup.conf"

shared_slurm_config_path = "/app/cray/nfs/slurm.conf"
shared_gres_config_path = "/app/cray/nfs/gres.conf"
shared_cgroup_config_path = "/app/cray/nfs/cgroup.conf"
shared_node_config_directory = "/app/cray/nfs/nodes"

meminfo_path = "/proc/meminfo"
cgroup_v2_memory_limit_path = "/sys/fs/cgroup/memory.max"
cgroup_v1_memory_limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

_bytes_per_kib = 1024
_bytes_per_mib = 1024 * 1024
_max_memory_bytes = (1 << 63) - 1
_cgroup_v1_unlimited_threshold = 1 << 60
_memtotal_pattern = re.compile(r"MemTotal:[ \t]+([1-9][0-9]*)[ \t]+kB")


def main():
    setup_logging()
    discover_clusters()


def discover_clusters():
    clean_old_node_info()

    node_info = get_node_info()

    save_node_info(node_info)

    cluster_info = get_cluster_info(node_info)

    save_cluster_info(cluster_info)


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Logging setup complete.")


def clean_old_node_info():
    config = get_config()

    time_limit = config["node_info_time_limit"]

    current_time = time.time()

    if not os.path.exists(shared_node_config_directory):
        return

    for filename in os.listdir(shared_node_config_directory):
        file_path = os.path.join(shared_node_config_directory, filename)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if current_time - file_mtime > time_limit:
                logging.debug(f"Removing old node info file: {file_path}")
                os.remove(file_path)


def get_node_info() -> dict:
    hostname = get_hostname()
    cpu_count = get_cpu_count()
    gpu_count = get_gpu_count()
    machine_id = get_machine_id()

    return {
        "machine_id": machine_id,
        "hostname": hostname,
        "cpu_count": cpu_count,
        "memory_mb": get_memory_mb(),
        "gpu_count": gpu_count,
        "gpu_type": get_gpu_type(),
        "gpu_indexes": get_gpu_indexes(),
    }


def get_machine_id():
    try:
        return get_board_serial()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Error reading machine ID: {e}")
        return None


def get_board_serial() -> str | None:
    result = subprocess.run(
        ["dmidecode", "-s", "baseboard-serial-number"], capture_output=True, text=True
    )
    serial = result.stdout.strip()
    return serial if serial else None


def get_hostname():
    return socket.gethostname()


def get_cpu_count() -> int | None:
    return os.cpu_count()


def get_memory_mb() -> int | None:
    total_bytes = _read_memtotal_bytes()
    if total_bytes is None:
        return None

    limit_is_known, limit_bytes = _read_cgroup_memory_limit_bytes()
    if not limit_is_known:
        return None

    effective_bytes = (
        min(total_bytes, limit_bytes) if limit_bytes is not None else total_bytes
    )
    memory_mb = effective_bytes // _bytes_per_mib
    return memory_mb if memory_mb > 0 else None


def _read_memtotal_bytes() -> int | None:
    try:
        with open(meminfo_path) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    match = _memtotal_pattern.fullmatch(line.rstrip("\r\n"))
                    if match is None:
                        logger.debug("Malformed MemTotal line in %s", meminfo_path)
                        return None
                    total_kib = int(match.group(1))
                    if total_kib > _max_memory_bytes // _bytes_per_kib:
                        logger.debug("MemTotal in %s is out of range", meminfo_path)
                        return None
                    return total_kib * _bytes_per_kib
    except OSError as e:
        logger.debug(f"Error reading total memory: {e}")
    return None


def _read_cgroup_memory_limit_bytes() -> tuple[bool, int | None]:
    try:
        with open(cgroup_v2_memory_limit_path) as f:
            value = f.read().strip()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug(f"Error reading cgroup v2 memory limit: {e}")
        return False, None
    else:
        if value == "max":
            return False, None
        return _parse_cgroup_limit(value, "v2")

    try:
        with open(cgroup_v1_memory_limit_path) as f:
            value = f.read().strip()
    except FileNotFoundError:
        return True, None
    except OSError as e:
        logger.debug(f"Error reading cgroup v1 memory limit: {e}")
        return False, None

    parsed, limit_bytes = _parse_cgroup_limit(value, "v1")
    if parsed and limit_bytes is not None:
        if limit_bytes >= _cgroup_v1_unlimited_threshold:
            return False, None
    return parsed, limit_bytes


def _parse_cgroup_limit(value: str, version: str) -> tuple[bool, int | None]:
    if not value.isascii() or not value.isdecimal():
        logger.debug("Malformed cgroup %s memory limit", version)
        return False, None

    limit_bytes = int(value)
    if not 0 < limit_bytes <= _max_memory_bytes:
        logger.debug("Cgroup %s memory limit is out of range", version)
        return False, None
    return True, limit_bytes


def get_gpu_count():
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
    elif get_neuron_devices():
        gpu_count = 4 * len(get_neuron_devices())
    return gpu_count


def write_neuron_core_links(hostname, core_indexes):
    links_dir = os.path.join(os.path.dirname(shared_gres_config_path), "neuron_cores")
    os.makedirs(links_dir, exist_ok=True)
    devices = sorted(get_neuron_devices(), key=lambda name: int(name[len("neuron"):]))
    lines = ""
    for index in core_indexes:
        link = os.path.join(links_dir, f"core{index}")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join("/dev", devices[index // 4]), link)
        lines += f"NodeName={hostname} Name=gpu File={link}\n"
    return lines


def get_neuron_devices():
    if not os.path.exists("/dev"):
        return []
    return [f for f in os.listdir("/dev") if f.startswith("neuron") and f[len("neuron"):].isdigit()]


def save_node_info(node_info):
    node_config_path = os.path.join(
        shared_node_config_directory, f"{node_info['hostname']}.json"
    )

    os.makedirs(shared_node_config_directory, exist_ok=True)

    with open(node_config_path, "w") as f:
        json.dump(node_info, f, indent=4)


def get_cluster_info(node_info):
    all_nodes = load_all_nodes()

    controller_info = elect_controller(all_nodes)

    return {
        "controller_info": controller_info,
        "all_nodes": all_nodes,
        "partitions": [{"name": "short", "nodes": all_nodes}],
    }


def load_all_nodes():
    all_nodes = []
    for filename in os.listdir(shared_node_config_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(shared_node_config_directory, filename)
            with open(file_path, "r") as f:
                node_info = json.load(f)
                all_nodes.append(node_info)
    return all_nodes


def elect_controller(all_nodes):
    if not all_nodes:
        raise ValueError("No nodes found in the cluster.")

    all_nodes.sort(key=lambda x: (x["gpu_count"], x["hostname"]))

    controller_node = all_nodes[0]

    logger.info(
        f"Controller node elected: {controller_node['hostname']} with {controller_node['gpu_count']} GPUs"
    )

    return controller_node


def is_controller(node_info, controller_info):
    is_controller = node_info["hostname"] == controller_info["hostname"]
    return is_controller


def save_cluster_info(cluster_info):
    old_cluster_info = load_cluster_info_file()

    if old_cluster_info:
        if old_cluster_info == cluster_info:
            logger.info("Cluster info is unchanged, skipping write.")
            return

    write_slurm_config(cluster_info)
    write_gres_config(cluster_info)
    write_cgroup_config(cluster_info)
    write_cluster_info_file(cluster_info)
    reload_slurm_configs()


def load_cluster_info_file():
    if not os.path.exists(cluster_info_path):
        return None

    with open(cluster_info_path, "r") as f:
        try:
            cluster_info = json.load(f)
            return cluster_info
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {cluster_info_path}: {e}")
            return None


def write_slurm_config(cluster_info):
    node_info = cluster_info["controller_info"]

    slurm_conf_values = load_slurm_conf_values()

    slurm_conf_values["SlurmctldHost"] = node_info["hostname"]

    if has_any_gpus(cluster_info):
        slurm_conf_values["GresTypes"] = "gpu"
    else:
        if "GresTypes" in slurm_conf_values:
            del slurm_conf_values["GresTypes"]

    if len(cluster_info["all_nodes"]) <= 1:
        slurm_conf_values["MpiDefault"] = "none"
    else:
        slurm_conf_values["MpiDefault"] = "pmix"

    new_config = save_slurm_conf_values(slurm_conf_values)

    for node in cluster_info["all_nodes"]:
        new_config += write_node_config(node)

    for partition in cluster_info["partitions"]:
        new_config += write_partition_config(partition)

    with atomic_write(shared_slurm_config_path, overwrite=True) as f:
        f.write(new_config)


def load_slurm_conf_values():
    slurm_conf_values = {}
    with open(slurm_config_path, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue

            if "=" not in line:
                continue

            key_and_value = line.split("=")

            if len(key_and_value) != 2:
                continue

            key, value = key_and_value[0], key_and_value[1]
            slurm_conf_values[key] = value.strip()
    return slurm_conf_values


def has_any_gpus(cluster_info):
    for node in cluster_info["all_nodes"]:
        if node["gpu_count"] > 0:
            return True
    return False


def save_slurm_conf_values(slurm_conf_values):
    config = ""
    with open(shared_slurm_config_path, "w") as f:
        for key, value in slurm_conf_values.items():
            config += f"{key}={value}\n"

    return config


def write_node_config(node: dict) -> str:
    max_gpus_per_node = get_config()["max_gpus_per_node"]
    gres_string = (
        f"Gres=gpu:{min(max_gpus_per_node, node['gpu_count'])}"
        if node["gpu_count"] > 0
        else ""
    )
    memory_mb = node.get("memory_mb")
    memory_is_valid = (
        isinstance(memory_mb, int)
        and not isinstance(memory_mb, bool)
        and 0 < memory_mb <= _max_memory_bytes // _bytes_per_mib
    )
    memory_string = f"RealMemory={memory_mb}" if memory_is_valid else ""
    fields = [
        f"NodeName={node['hostname']}",
        f"CPUs={node['cpu_count']}",
        memory_string,
        gres_string,
        "State=UNKNOWN",
    ]
    node_config = " ".join(field for field in fields if field)
    return node_config + "\n"


def write_partition_config(partition):
    config = get_config()

    max_training_time = (
        config["max_train_time"] + config["extra_training_seconds"]
    ) // 60

    node_names = ",".join([node["hostname"] for node in partition["nodes"]])
    partition_config = f"PartitionName={partition['name']} Nodes={node_names} Default=YES MaxTime={max_training_time} State=UP"

    return partition_config + "\n"


def write_gres_config(cluster_info):
    max_gpus_per_node = get_config()["max_gpus_per_node"]

    gres_config = ""
    for node in cluster_info["all_nodes"]:
        gpu_indexes = node["gpu_indexes"][: min(max_gpus_per_node, node["gpu_count"])]
        if node["gpu_type"] == "neuron":
            gres_config += write_neuron_core_links(node["hostname"], gpu_indexes)
            continue
        for index in gpu_indexes:
            if node["gpu_type"] == "amd":
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/dri/renderD{index}\n"
                )
            else:
                gres_config += (
                    f"NodeName={node['hostname']} Name=gpu File=/dev/nvidia{index}\n"
                )

    with atomic_write(shared_gres_config_path, overwrite=True) as f:
        f.write(gres_config)


def write_cgroup_config(cluster_info):
    with open(cgroup_config_path) as config:
        with atomic_write(shared_cgroup_config_path, overwrite=True) as shared_config:
            shared_config.write(config.read())


def get_amd_render_node_indexes():
    indexes = []
    drm_root = "/sys/class/drm"

    if not os.path.exists(drm_root):
        return indexes

    for entry in os.listdir(drm_root):
        if not entry.startswith("renderD"):
            continue

        suffix = entry[len("renderD") :]
        if not suffix.isdigit():
            continue

        try:
            device_link = os.path.join(drm_root, entry, "device")
            target = os.path.basename(os.path.realpath(device_link))
        except OSError:
            continue

        if target.count(":") < 2:
            continue

        if not os.path.exists(os.path.join("/dev/dri", entry)):
            logger.debug(
                "Skipping %s: present in sysfs but /dev/dri/%s is not accessible "
                "from this node", entry, entry
            )
            continue

        indexes.append(int(suffix))

    return sorted(indexes)


def get_gpu_indexes():
    if torch.version.hip:
        return get_amd_render_node_indexes()
    elif not torch.cuda.is_available() and get_neuron_devices():
        return list(range(get_gpu_count()))
    else:
        prefix = "/dev"
        card_name = "nvidia"

    indexes = []

    if os.path.exists(prefix):
        for file in os.listdir(prefix):
            if file.startswith(card_name):
                try:
                    index_str = file[len(card_name) :]
                    if index_str.isdigit():
                        print(file[len(card_name) :])
                        index_as_int = int(file[len(card_name) :])
                        indexes.append(index_as_int)
                except Exception as e:
                    continue

    return indexes


def get_gpu_type():
    if torch.version.hip:
        return "amd"
    elif not torch.cuda.is_available() and get_neuron_devices():
        return "neuron"
    else:
        return "nvidia"

    return "none"


def write_cluster_info_file(cluster_info):
    with open(cluster_info_path, "w") as f:
        json.dump(cluster_info, f, indent=4)
    logger.info(f"Cluster info saved to {cluster_info_path}")


def reload_slurm_configs():
    try:
        result = subprocess.run(["scontrol", "ping"], capture_output=True, timeout=5)
        if result.returncode != 0:
            logger.debug("Slurm controller not available, skipping reconfigure")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug("Slurm not installed or not responding, skipping reconfigure")
        return

    try:
        subprocess.run(["scontrol", "reconfigure"], check=True)
        logger.info("Slurm configurations reloaded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to reload Slurm configurations: {e}")


if __name__ == "__main__":
    main()
