import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

for _p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))


def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def is_multimodal(cfg: dict) -> bool:
    return any("ForConditionalGeneration" in a for a in cfg.get("architectures") or [])


def default_lora_regex(cfg: dict) -> str:
    return (r".*language_model.*" if is_multimodal(cfg) else r".*") + LORA_PROJECTIONS


def auto_model_class(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_multimodal(cfg) else AutoModelForCausalLM


def write_versions(path: str, training_hub_dir: str = "/opt/training_hub", verl_dir: str = "/opt/verl"):
    import importlib.metadata as md
    import peft, ray, torch, transformers, verl, vllm  # noqa: E401

    def git_head(d):
        r = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or "unknown"

    with open(path, "w") as f:
        f.write(f"training_hub {md.version('training_hub')} git {git_head(training_hub_dir)}\n"
                f"verl {verl.__version__} git {git_head(verl_dir)}\nvllm {vllm.__version__}\n"
                f"torch {torch.__version__}\ntransformers {transformers.__version__}\n"
                f"peft {peft.__version__}\nray {ray.__version__}\npython {sys.version.split()[0]}\n")


def _override_key(ov: str) -> str:
    return ov.lstrip("+~").split("=", 1)[0]


def apply_overrides(cmd: list, overrides: list) -> list:
    cmd = list(cmd)
    index = {_override_key(c): i for i, c in enumerate(cmd) if "=" in c and not c.startswith("-")}
    for ov in overrides:
        k = _override_key(ov)
        if k in index:
            cmd[index[k]] = ov
        else:
            index[k] = len(cmd)
            cmd.append(ov)
    return cmd


def patch_training_hub_popen(overrides: list, command_log: str):
    import types

    import training_hub.algorithms.lora_grpo_verl as thv

    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        cmd = apply_overrides(cmd, overrides)
        with open(command_log, "a") as f:
            f.write(" ".join(cmd) + "\n")
        return real_popen(cmd, **kw)

    thv.subprocess = types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)


_JUDGE_SEMAPHORE = threading.BoundedSemaphore(int(os.environ.get("GRPO_JUDGE_CONC_PER_WORKER", "8")))
_LOG_LOCK = threading.Lock()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    import judge_reward

    extra_info = extra_info or {}
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    except Exception:
        gt = {"reference": str(ground_truth)}
    question = gt.get("question") or extra_info.get("prompt_text") or ""
    reference = gt.get("reference", "")
    source = gt.get("source", data_source)
    truncated = bool(extra_info.get("truncated", False))

    t0 = time.time()
    with _JUDGE_SEMAPHORE:
        t_start = time.time()
        out = judge_reward.score_batch([question], [solution_str or ""], [reference], [source],
                                       ["length" if truncated else "stop"])
    t_end = time.time()
    reward = float(out["reward"][0])
    judge = out["judge"][0]

    log_path = extra_info.get("reward_log_path") or os.environ.get("GRPO_REWARD_LOG")
    if log_path:
        rec = {"start": t_start, "end": t_end, "queued_s": round(t_start - t0, 4),
               "judge_latency": round(out["timing_s"], 4), "source": source, "id": gt.get("id"),
               "reward": reward, "judge": judge, "truncated": truncated,
               "response_length": extra_info.get("response_length"), "worker_pid": os.getpid()}
        with _LOG_LOCK, open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    penalty = judge_reward.TRUNCATION_PENALTY if truncated else 0.0
    judge_norm = (judge or 0) / 10.0 if source != "3gpp_tsg" else reward - penalty
    return {"score": reward, "judge_norm": judge_norm, "truncated": float(truncated),
            "judge_failed": float(source != "3gpp_tsg" and judge is None)}


try:
    import numpy as np
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

    class TruncationAwareRewardManager(NaiveRewardManager):

        async def run_single(self, data):
            item = data[-1:]
            d = item[0]
            response_length = d.batch["responses"].shape[-1]
            valid_len = int(d.batch["attention_mask"][-response_length:].sum())
            max_len = int(self.config.data.max_response_length)
            prompt_ids = d.batch["prompts"]
            prompt_mask = d.batch["attention_mask"][:-response_length]

            extra = d.non_tensor_batch.get("extra_info", None)
            extra = dict(extra) if extra is not None else {}
            extra.update(
                response_length=valid_len,
                max_response_length=max_len,
                truncated=bool(valid_len >= max_len),
                prompt_text=self.tokenizer.decode(prompt_ids[prompt_mask.bool()], skip_special_tokens=True),
                reward_log_path=os.path.join(
                    os.path.dirname(str(self.config.trainer.default_local_dir).rstrip("/")), "reward_calls.jsonl"),
            )
            arr = np.empty(1, dtype=object)
            arr[0] = extra
            item.non_tensor_batch["extra_info"] = arr
            return await super().run_single(item)
except ImportError:  # verl not importable outside the training container
    TruncationAwareRewardManager = None


def latest_actor_checkpoint(run_dir: str) -> str:
    steps = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "global_step_*", "actor")),
                   key=lambda p: int(re.search(r"global_step_(\d+)", p).group(1)))
    if not steps:
        raise FileNotFoundError(f"no checkpoints/global_step_*/actor under {run_dir}")
    return steps[-1]


def export_lora_adapter(ckpt: str, out: str, base: str, r: int, alpha: int, target_regex: str) -> dict:
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file

    if not dist.is_initialized():  # ShardedTensor pickles need a process group
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        dist.init_process_group("gloo", rank=0, world_size=1)

    files = sorted(glob.glob(os.path.join(ckpt, "model_world_size_*_rank_*.pt")),
                   key=lambda p: int(re.search(r"_rank_(\d+)\.pt$", p).group(1)))
    if not files:
        raise FileNotFoundError(f"no model_world_size_*_rank_*.pt under {ckpt}")

    pieces = {}
    for rank, f in enumerate(files):
        sd = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in sd.items():
            if "lora_" not in k:
                continue
            if hasattr(v, "local_shards"):
                for sh in v.local_shards():
                    pieces.setdefault(k, []).append((tuple(sh.metadata.shard_offsets), sh.tensor.detach().clone()))
            elif hasattr(v, "to_local"):
                pieces.setdefault(k, []).append(((rank,), v.to_local().detach().clone()))
            else:
                pieces.setdefault(k, []).append(((rank,), v.detach().clone()))
        del sd

    out_sd = {}
    for k, parts in pieces.items():
        parts.sort(key=lambda x: x[0])
        t = torch.cat([p[1] for p in parts], dim=0) if len(parts) > 1 else parts[0][1]
        nk = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        nk = re.sub(r"\.lora_(A|B)\.default\.weight$", r".lora_\1.weight", nk)
        if not nk.startswith("base_model.model."):
            nk = "base_model.model." + nk
        out_sd[nk] = t.to(torch.bfloat16).contiguous()

    r_seen = {v.shape[0] for k, v in out_sd.items() if k.endswith("lora_A.weight")}
    if r_seen != {r}:
        raise ValueError(f"lora_A rows {r_seen} != r={r}")

    os.makedirs(out, exist_ok=True)
    save_file(out_sd, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
                   "r": r, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
                   "target_modules": target_regex, "fan_in_fan_out": False, "inference_mode": True,
                   "init_lora_weights": True, "use_rslora": False, "use_dora": False,
                   "exported_from": ckpt}, f, indent=1)
    n_modules = len({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in out_sd})
    print(f"[export] {len(out_sd)} tensors, {n_modules} LoRA modules -> {out}", flush=True)
    return out_sd


def verify_adapter(base: str, adapter: str, prompts: list, max_new_tokens: int = 160) -> str:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = auto_model_class(load_model_config(base)).from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter)
    params = dict(model.named_parameters())
    exported = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    unmatched = [k for k, v in exported.items()
                 if (pk := k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")) not in params
                 or not torch.equal(params[pk].detach().to("cpu", torch.bfloat16), v)]
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[verify] LoRA modules attached: {n_lora}; exported tensors: {len(exported)}; unmatched: {len(unmatched)}", flush=True)

    lines = [f"base={base}\nadapter={adapter}\nlora_modules_attached={n_lora}\nunmatched={len(unmatched)}\n"]
    for i, prompt in enumerate(prompts, 1):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[verify] prompt {i}: {prompt}\n[verify] adapter ON: {text}", flush=True)
        lines.append(f"\n=== prompt {i} ===\n{prompt}\n=== adapter ON ===\n{text}\n")
        if i == 1:
            with model.disable_adapter(), torch.no_grad():
                out0 = model.generate(**ids, max_new_tokens=48, do_sample=False)
            text0 = tok.decode(out0[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"[verify] adapter OFF (48 tok): {text0}", flush=True)
            lines.append(f"=== adapter OFF (first 48 tok) ===\n{text0}\n")
    proof = os.path.join(adapter, "load_and_generate_proof.txt")
    with open(proof, "w") as f:
        f.write("".join(lines))
    return proof


def parse_args():
    p = argparse.ArgumentParser(description="GRPO LoRA training with training_hub (verl backend)")
    p.add_argument("--model", required=True, help="HF checkpoint directory")
    p.add_argument("--data", required=True, help="JSONL with `messages` and `ground_truth` per row")
    p.add_argument("--output-dir", required=True, help="checkpoints, logs and the exported adapter go here")
    p.add_argument("--steps", type=int, default=22)
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=3072)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-regex", default=None, help="verl target_modules regex (default: derived from the model class)")
    p.add_argument("--n-gpus", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ray-port", type=int, default=6391, help="Ray head port; ignored when RAY_ADDRESS is set")
    p.add_argument("--override", action="append", default=[], help="extra verl override key=value (repeatable)")
    p.add_argument("--no-export", action="store_true", help="skip exporting the final LoRA adapter")
    p.add_argument("--export-only", action="store_true", help="skip training; export (and --verify) from the checkpoints already in --output-dir")
    p.add_argument("--verify", action="store_true", help="after export, load base+adapter on one GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the lora_grpo kwargs and injected overrides, then exit")
    return p.parse_args()


# A console script next to the running interpreter. On the venv route it is not on PATH, and a
# bare name either fails with FileNotFoundError or silently picks up another environment's.
def venv_bin(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else name


def start_ray(port: int, n_gpus: int):
    if os.environ.get("RAY_ADDRESS"):
        return
    subprocess.run([venv_bin("ray"), "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Ray sizes its object store from /dev/shm. Where the shared dir is a large tmpfs, the default
    # claim collides with the weights living there and the raylet is OOM-killed mid-run; GRPO_RAY_OBJECT_STORE_GB
    # caps it (default 32 GB, ample for rollout batches).
    store_gb = float(os.environ.get("GRPO_RAY_OBJECT_STORE_GB", "32"))
    # Ray actors inherit the head's environment, not the launcher's, and a JIT kernel build inside an
    # actor shells out to `ninja`, so the venv's bin/ must be on the head's PATH.
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    subprocess.run([venv_bin("ray"), "start", "--head", f"--port={port}", f"--num-gpus={n_gpus}",
                    f"--object-store-memory={int(store_gb * 1024**3)}",
                    "--include-dashboard=false", "--disable-usage-stats", "--temp-dir", "/tmp/ray_redhat"],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    os.environ["RAY_ADDRESS"] = f"127.0.0.1:{port}"
