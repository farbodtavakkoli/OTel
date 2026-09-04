"""Probe whether this host can run TensorRT-LLM (NVIDIA-only) and report why not."""

import argparse
import ctypes
import glob
import json
import os
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv("dev.env")

NVIDIA_SONAMES = ["libcuda.so.1", "libnvidia-ml.so.1", "libcudart.so.13", "libcudart.so.12"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device_id", type=int, default=6)
    parser.add_argument("--check_import", dest="check_import", action="store_true", default=True)
    parser.add_argument("--no_check_import", dest="check_import", action="store_false")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def probe_nvidia_smi():
    path = shutil.which("nvidia-smi")
    if not path:
        return {"present": False, "path": None, "output": "nvidia-smi not found on PATH"}
    try:
        out = subprocess.run(
            [path, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return {"present": True, "path": path, "output": (out.stdout or out.stderr).strip()}
    except Exception as exc:
        return {"present": True, "path": path, "output": f"{type(exc).__name__}: {exc}"}


def probe_device_nodes():
    return {
        "nvidia_nodes": sorted(glob.glob("/dev/nvidia*")),
        "kfd_present": os.path.exists("/dev/kfd"),
    }


def probe_sonames():
    found = {}
    for soname in NVIDIA_SONAMES:
        try:
            ctypes.CDLL(soname)
            found[soname] = True
        except OSError:
            found[soname] = False
    return found


def probe_torch():
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
    info = {
        "installed": True,
        "version": torch.__version__,
        "version_cuda": torch.version.cuda,
        "version_hip": getattr(torch.version, "hip", None),
        "cuda_is_available": None,
        "device_count": 0,
        "device_name": None,
    }
    try:
        info["cuda_is_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["device_count"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_trtllm_import():
    try:
        import tensorrt_llm
    except BaseException as exc:
        return {"imports": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"imports": True, "version": getattr(tensorrt_llm, "__version__", "unknown")}


def main():
    args = parse_args()
    report = {"device_id": args.device_id}

    smi = probe_nvidia_smi()
    report["nvidia_smi"] = smi
    print(f"nvidia-smi      : {'FOUND at ' + smi['path'] if smi['present'] else 'ABSENT'}")
    if smi["present"]:
        print(f"  output        : {smi['output']}")

    nodes = probe_device_nodes()
    report["device_nodes"] = nodes
    print(f"/dev/nvidia*    : {nodes['nvidia_nodes'] or 'none'}")
    print(f"/dev/kfd (ROCm) : {'present' if nodes['kfd_present'] else 'absent'}")

    sonames = probe_sonames()
    report["sonames"] = sonames
    for soname, ok in sonames.items():
        print(f"  {soname:<20} -> {'loadable' if ok else 'MISSING'}")

    torch_info = probe_torch()
    report["torch"] = torch_info
    if torch_info["installed"]:
        print(f"torch           : {torch_info['version']}")
        print(f"  version.cuda  : {torch_info['version_cuda']}")
        print(f"  version.hip   : {torch_info['version_hip']}")
        alias_note = " (torch.cuda is aliased to HIP -- NOT proof of CUDA)" if torch_info["version_hip"] else ""
        print(f"  is_available  : {torch_info['cuda_is_available']}{alias_note}")
        print(f"  device 0      : {torch_info['device_name']}")
    else:
        print(f"torch           : not installed ({torch_info['error']})")

    real_cuda = bool(torch_info.get("version_cuda")) and sonames.get("libcuda.so.1", False)
    report["real_cuda"] = real_cuda

    if args.check_import:
        trt = probe_trtllm_import()
        report["tensorrt_llm"] = trt
        if trt["imports"]:
            print(f"tensorrt_llm    : imports OK (version {trt['version']})")
        else:
            print(f"tensorrt_llm    : IMPORT FAILED -- {trt['error']}")
    else:
        report["tensorrt_llm"] = {"imports": None, "error": "skipped"}

    usable = real_cuda and smi["present"] and bool(report["tensorrt_llm"].get("imports"))
    report["usable"] = usable

    print()
    if usable:
        print("VERDICT: TensorRT-LLM appears usable on this host.")
    else:
        print("VERDICT: TensorRT-LLM is NOT usable on this host.")
        if not smi["present"] and not nodes["nvidia_nodes"]:
            print("  No NVIDIA GPU, driver, or CUDA runtime is present.")
            print("  TensorRT-LLM is NVIDIA-only by design; ROCm/AMD is not a supported target.")
            print("  Use inference/vllm/llm or inference/sglang/llm on this AMD host.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nreport written to {args.out}")

    raise SystemExit(0 if usable else 1)


if __name__ == "__main__":
    main()
