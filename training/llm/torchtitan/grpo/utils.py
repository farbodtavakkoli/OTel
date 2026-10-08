import argparse
import glob
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# torchtitan qwen3 flavors by (hidden_size, num_hidden_layers) of the HF config.
QWEN3_FLAVORS = {(1024, 28): "0.6B", (2048, 28): "1.7B", (2560, 36): "4B", (4096, 36): "8B",
                 (5120, 40): "14B", (5120, 64): "32B"}
METRIC_KEYS = ["rollout_s", "judge_s", "train_step_s", "sync_s", "cycle_s", "tokens_per_s"]
HF_SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja")


def common_dir() -> str:
    return os.path.abspath(os.environ.get("GRPO_COMMON_DIR") or os.path.join(HERE, "..", "..", "grpo_common"))


if common_dir() not in sys.path:
    sys.path.insert(0, common_dir())


def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def qwen3_flavor(cfg: dict) -> str:
    if cfg.get("model_type") != "qwen3":
        raise SystemExit(f"model_type {cfg.get('model_type')!r}: torchtitan has no model definition "
                         "for it (this stack runs dense Qwen3 only; pass --flavor to force one)")
    key = (int(cfg["hidden_size"]), int(cfg["num_hidden_layers"]))
    if key not in QWEN3_FLAVORS:
        raise SystemExit(f"no torchtitan qwen3 flavor for hidden_size/layers {key}; pass --flavor")
    return QWEN3_FLAVORS[key]


def count_rows(path: str, titan_repo: str) -> int:
    for p in (titan_repo, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from titanrl_telecom.data import parse_row

    n = 0
    with open(path, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if line.strip():
                parse_row(json.loads(line), index, path)
                n += 1
    if n == 0:
        raise SystemExit(f"{path}: no rows")
    return n


def _git_sha(path: str) -> str:
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def probe_versions(titan_repo: str) -> str:
    code = ("import sys, torch, importlib.metadata as md\n"
            "def ver(d):\n"
            "    try: return md.version(d)\n"
            "    except Exception: return 'unknown'\n"
            "print('versions python', sys.version.split()[0], 'torch', torch.__version__, "
            "'hip', torch.version.hip, 'vllm', ver('vllm'), 'transformers', ver('transformers'), "
            "'torchmonarch', ver('torchmonarch'), 'torchstore', ver('torchstore'), "
            "'renderers', ver('renderers'))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or ["versions unknown"])[-1]
    line += f" torchtitan_commit {_git_sha(titan_repo)}"
    vllm_src = os.environ.get("VLLM_SRC", os.path.join(os.path.dirname(titan_repo), "vllm_src"))
    if os.path.isdir(vllm_src):
        line += f" vllm_commit {_git_sha(vllm_src)}"
    return line


def write_versions(path: str, titan_repo: str) -> str:
    line = probe_versions(titan_repo)
    with open(path, "w") as f:
        f.write(line + "\n")
    return line


def parse_versions(line: str) -> dict:
    toks = line.strip().split()
    if toks and toks[0] == "versions":
        toks = toks[1:]
    return dict(zip(toks[0::2], toks[1::2]))


def run_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "command.txt")
    if not os.path.exists(path):
        return {}
    cfg = {}
    with open(path) as f:
        for line in f:
            if line.startswith("==="):
                for tok in shlex.split(line):
                    if tok.startswith("TITAN_GRPO_CONFIG="):
                        cfg = json.loads(tok.partition("=")[2])
    return cfg


def load_tb(run_dir: str) -> dict:
    from tensorboard.backend.event_processing import event_accumulator

    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        ea = event_accumulator.EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                out[tag][e.step] = e.value
    return out


def load_trace(run_dir: str) -> list:
    rows = []
    for f in glob.glob(os.path.join(run_dir, "rollout_trace", "*.jsonl")):
        with open(f) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def per_step(run_dir: str) -> list:
    tb = load_tb(run_dir)
    by_pv = defaultdict(list)
    for r in load_trace(run_dir):
        if r.get("policy_version") is not None and r.get("t_gen_start") is not None:
            by_pv[r["policy_version"]].append(r)
    rows = []
    for k in sorted(tb.get("timing/step/total/mean", {})):
        g = by_pv.get(k - 1, [])
        rec = {
            "step": k,
            "cycle_s": tb["timing/step/total/mean"].get(k),
            "train_step_s": (tb.get("timing/step/forward_backward/mean", {}).get(k, 0.0)
                             + tb.get("timing/step/optim/mean", {}).get(k, 0.0)),
            "fwd_bwd_s": tb.get("timing/step/forward_backward/mean", {}).get(k),
            "optim_s": tb.get("timing/step/optim/mean", {}).get(k),
            "sync_s": (tb.get("timing/weight_sync/trainer_push_model_state_dict", {}).get(k, 0.0)
                       + tb.get("timing/weight_sync/generator_pull_model_state_dict", {}).get(k, 0.0)),
            "sync_push_s": tb.get("timing/weight_sync/trainer_push_model_state_dict", {}).get(k),
            "sync_pull_s": tb.get("timing/weight_sync/generator_pull_model_state_dict", {}).get(k),
            "wait_for_batch_s": tb.get("timing/step/wait_for_training_batch/mean", {}).get(k),
            "reward_mean": tb.get("rollout_reward/_mean", {}).get(k),
            "loss": tb.get("loss/mean", {}).get(k),
            "response_len_mean": tb.get("train/response_length/mean", {}).get(k),
            "zero_std_groups_dropped": tb.get("training_sample_builder/num_groups_dropped_zero_std", {}).get(k),
            "peak_mem_gb": tb.get("trainer/peak_mem_gb/max", {}).get(k),
            "groups_generated": len(g),
        }
        if g:
            t0 = min(r["t_gen_start"] for r in g)
            t1 = max(r["t_gen_end"] for r in g)
            tj = max(r["t_judge_end"] for r in g)
            toks = sum(sum(r["completion_tokens"]) for r in g)
            rec.update(
                rollout_s=t1 - t0,
                judge_s=max(0.0, tj - t1),
                judge_sum_s=sum(r["judge_timing_s"] for r in g),
                completion_tokens=toks,
                tokens_per_s=toks / (t1 - t0) if t1 > t0 else None,
                truncated_frac=(sum(s == "truncated_length" for r in g for s in r["status"])
                                / sum(len(r["status"]) for r in g)),
                reward_mean_all_generated=statistics.fmean(x for r in g for x in r["rewards"]),
                zero_spread_groups=sum(1 for r in g if len(set(r["rewards"])) == 1),
            )
        else:
            rec.update(rollout_s=None, judge_s=None, judge_sum_s=None, tokens_per_s=None)
        rows.append(rec)
    return rows


def summarize(rows: list, warmup: int) -> dict:
    meas = [r for r in rows if r["step"] > warmup]
    summ = {}
    for k in METRIC_KEYS:
        v = [r[k] for r in meas if r.get(k) is not None]
        summ[k] = {"n": len(v), "median": statistics.median(v), "mean": statistics.fmean(v),
                   "min": min(v), "max": max(v)} if v else None
    summ["steps_measured"] = len(meas)
    summ["peak_mem_gb"] = max((r["peak_mem_gb"] for r in rows if r.get("peak_mem_gb")), default=None)
    summ["reward_first"] = meas[0]["reward_mean"] if meas else None
    summ["reward_last"] = meas[-1]["reward_mean"] if meas else None
    return summ


def run_metrics(run_dir: str, warmup: int) -> dict:
    if run_dir.endswith(".json"):
        with open(run_dir) as f:
            return json.load(f)
    rows = per_step(run_dir)
    if len(rows) <= warmup:  # a run shorter than the warm-up (a preset smoke) summarises every step
        warmup = 0
    return {"per_step": rows, "summary": summarize(rows, warmup), "warmup": warmup}


def format_steps(rows: list) -> str:
    f = lambda x, w=8, p=1: (f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")
    lines = [f"{'step':>4} {'cycle':>8} {'rollout':>8} {'judge':>7} {'train':>7} {'sync':>6} "
             f"{'tok/s':>8} {'reward':>7} {'groups':>6} {'peakGB':>7}"]
    for r in rows:
        lines.append(f"{r['step']:>4} {f(r['cycle_s'])} {f(r['rollout_s'])} {f(r['judge_s'], 7)} "
                     f"{f(r['train_step_s'], 7)} {f(r['sync_s'], 6, 2)} {f(r['tokens_per_s'], 8, 0)} "
                     f"{f(r['reward_mean'], 7, 3)} {r['groups_generated']:>6} {f(r['peak_mem_gb'], 7)}")
    return "\n".join(lines)


def latest_hf_checkpoint(run_dir: str) -> str:
    steps = sorted(glob.glob(os.path.join(run_dir, "checkpoint", "step-*")),
                   key=lambda p: int(re.search(r"step-(\d+)$", p).group(1)))
    steps = [s for s in steps if glob.glob(os.path.join(s, "*.safetensors"))]
    if not steps:
        raise FileNotFoundError(f"no checkpoint/step-N with safetensors under {run_dir}")
    return steps[-1]


def finalize_hf_checkpoint(ckpt: str, base: str) -> list:
    copied = []
    for name in HF_SIDE_FILES:
        src, dst = os.path.join(base, name), os.path.join(ckpt, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def _safetensor_index(d: str) -> dict:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return json.load(f)["weight_map"]
    files = glob.glob(os.path.join(d, "*.safetensors"))
    from safetensors import safe_open

    out = {}
    for fn in files:
        with safe_open(fn, "pt") as sf:
            for k in sf.keys():
                out[k] = os.path.basename(fn)
    return out


def compare_tensors(base: str, ckpt: str, n: int = 60, seed: int = 0) -> dict:
    import random

    import torch
    from safetensors import safe_open

    bmap, cmap = _safetensor_index(base), _safetensor_index(ckpt)
    keys = sorted(set(bmap) & set(cmap))
    random.Random(seed).shuffle(keys)
    keys = keys[:n]
    changed, max_diff = 0, 0.0
    for k in keys:
        with safe_open(os.path.join(base, bmap[k]), "pt") as fb, safe_open(os.path.join(ckpt, cmap[k]), "pt") as fc:
            d = float((fc.get_tensor(k).float() - fb.get_tensor(k).float()).abs().max())
        changed += d > 0
        max_diff = max(max_diff, d)
    return {"checked": len(keys), "changed": changed, "max_abs_diff": max_diff, "common_tensors": len(set(bmap) & set(cmap))}


def verify_checkpoint(base: str, ckpt: str, prompts: list, max_new_tokens: int = 160) -> str:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map={"": 0})
    n_params = sum(p.numel() for p in model.parameters())
    lines = [f"LOADED {ckpt} in {time.time() - t0:.1f}s params={n_params}"]
    cmp = compare_tensors(base, ckpt)
    lines.append(f"tensors checked={cmp['checked']} changed_vs_base={cmp['changed']} "
                 f"max_abs_diff={cmp['max_abs_diff']:.3e} (common tensors {cmp['common_tensors']})")
    print("[verify]", lines[0], "\n[verify]", lines[1], flush=True)
    for i, prompt in enumerate(prompts, 1):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, enable_thinking=False).to(0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[verify] prompt {i}: {prompt}\n[verify] answer: {text}", flush=True)
        lines.append(f"PROMPT {i}: {prompt}")
        lines.append(f"GENERATION: {text}")
    ok = cmp["checked"] > 0 and cmp["changed"] > 0
    lines.append("LOADPROOF_OK" if ok else "LOADPROOF_FAIL (no tensor differs from the base)")
    proof = os.path.join(ckpt, "load_and_generate_proof.txt")
    with open(proof, "w") as f:
        f.write("\n".join(lines) + "\n")
    if not ok:
        raise SystemExit(lines[-1])
    return proof


DEFAULT_TITAN = os.environ.get("TITAN_SRC", os.path.join(os.environ.get("GRPO_ENV_DIR", os.path.join(HERE, "..", "..", "envs")),
                                                         "torchtitan_src"))
GPU_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def parse_args():
    p = argparse.ArgumentParser(description="GRPO full-parameter training with TitanRL (torchtitan/experiments/rl)")
    p.add_argument("--model", required=True, help="HF checkpoint directory of a dense Qwen3 model")
    p.add_argument("--data", required=True, help="train file: JSONL with `messages` + `ground_truth`")
    p.add_argument("--output-dir", required=True, help="TitanRL dump folder: logs, metrics, checkpoint/ and the proof")
    p.add_argument("--gpus", required=True, help="physical GPU list, e.g. 1,2: trainer GPUs first, then generators")
    p.add_argument("--gpu-env-var", default="ROCR_VISIBLE_DEVICES",
                   help="variable that pins --gpus for the whole run (CUDA_VISIBLE_DEVICES on NVIDIA)")
    p.add_argument("--config", default="rl_telecom",
                   help="titanrl_telecom config: rl_telecom (built from these args) or a recorded preset "
                        "(rl_telecom_qwen3_4b_smoke, ..._bench_2gpu, ..._bench_4gpu, rl_telecom_qwen3_32b_*)")
    p.add_argument("--flavor", default=None, help="torchtitan qwen3 flavor (default: from the model's config.json)")
    p.add_argument("--steps", type=int, default=22)
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt")
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=1024, help="max response tokens")
    p.add_argument("--learning-rate", type=float, default=1e-6, help="AdamW lr, constant (full-parameter)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-generators", type=int, default=1, help="vLLM generator instances")
    p.add_argument("--generator-tp", type=int, default=1, help="GPUs per generator (tensor parallel)")
    p.add_argument("--trainer-dp-shard", type=int, default=1, help="FSDP shard degree of the trainer")
    p.add_argument("--trainer-tp", type=int, default=1, help="tensor parallel degree of the trainer")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (aot_eager) in the trainer")
    p.add_argument("--no-save", action="store_true", help="do not write the HF checkpoint at the last step")
    p.add_argument("--warmup-steps", type=int, default=2,
                   help="steps excluded from the metrics summary (0 when --steps is not larger)")
    p.add_argument("--titan-repo", default=DEFAULT_TITAN, help="torchtitan clone (cwd and PYTHONPATH of the run)")
    p.add_argument("--judge-url", default=None, help="GRPO_JUDGE_URL for judge_reward (default: its own)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after --extra goes verbatim to torchtitan (tyro overrides)")
    p.add_argument("--export-only", action="store_true",
                   help="skip training; finalise (and --verify) the checkpoint already in --output-dir")
    p.add_argument("--verify", action="store_true", help="after training, load the checkpoint on the first GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the torchtitan command and environment, then exit")
    return p.parse_args()


def titan_repo_path(args) -> str:
    path = os.path.abspath(args.titan_repo)
    if not os.path.isfile(os.path.join(path, "torchtitan", "experiments", "rl", "train.py")):
        raise SystemExit(f"{path} is not a torchtitan clone with experiments/rl (--titan-repo)")
    return path


def gpu_list(args) -> list:
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"--gpus {args.gpus}: a GPU is listed twice")
    # The judge holds one GPU, GPU 0 unless JUDGE_GPU moves it. This guard matters on CUDA because
    # TitanRL's PerHostProvisioner overwrites CUDA_VISIBLE_DEVICES with an absolute range starting
    # at 0 (torchtitan/experiments/rl/train.py), so a parent-level mask cannot keep actors off it.
    judge_gpu = os.environ.get("JUDGE_GPU", "0").strip()
    if judge_gpu in gpus:
        raise SystemExit(f"--gpus must not include GPU {judge_gpu} (the judge; override with JUDGE_GPU)")
    need = args.trainer_dp_shard * args.trainer_tp + args.num_generators * args.generator_tp
    if args.config == "rl_telecom" and len(gpus) != need:
        raise SystemExit(f"--gpus lists {len(gpus)} GPUs but trainer ({args.trainer_dp_shard}x{args.trainer_tp}) "
                         f"+ generators ({args.num_generators}x{args.generator_tp}) need {need}")
    return gpus


def pin_gpus(env: dict, gpu_env_var: str, gpus) -> dict:
    env[gpu_env_var] = gpus if isinstance(gpus, str) else ",".join(gpus)
    for var in GPU_ENV_VARS:
        if var != gpu_env_var:
            env.pop(var, None)
    return env


def titan_command(args, run_dir: str) -> list:
    return [sys.executable, "-m", "torchtitan.experiments.rl.train",
            "--module", "titanrl_telecom", "--config", args.config,
            "--dump-folder", run_dir,
            "--metrics.no-enable-wandb", "--metrics.enable-tensorboard", *args.extra]


def subprocess_env(args, run_dir: str, gpus: list, cfg: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(p for p in (os.path.abspath(args.titan_repo), HERE, common_dir(),
                                             os.environ.get("PYTHONPATH")) if p)
    env["GRPO_COMMON_DIR"] = common_dir()
    env["TITAN_GRPO_CONFIG"] = json.dumps(cfg)
    env["TITAN_DUMP_ROOT"], env["TITAN_RUN_NAME"] = os.path.split(run_dir)
    env["GRPO_PROMPTS"] = cfg["data"]
    pin_gpus(env, args.gpu_env_var, gpus)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HYPERACTOR_CODEC_MAX_FRAME_LENGTH", "8589934592")  # Monarch large-RPC frames
    if args.judge_url:
        env["GRPO_JUDGE_URL"] = args.judge_url
    return env


def run_titan(cmd: list, env: dict, cwd: str, run_dir: str) -> int:
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", start_new_session=True)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            rc = proc.wait()
        log.write(f"=== TRAINING EXIT rc={rc}\n")
    return rc


def print_dry_run(cmd: list, env: dict, cwd: str, gpu_env_var: str):
    keys = ("PYTHONPATH", "GRPO_COMMON_DIR", gpu_env_var, "TITAN_DUMP_ROOT", "TITAN_RUN_NAME",
            "GRPO_PROMPTS", "GRPO_JUDGE_URL", "TITAN_GRPO_CONFIG")
    print("env:", " ".join(f"{k}={shlex.quote(env[k])}" for k in keys if k in env))
    print("cwd:", cwd)
    print(shlex.join(cmd))


def record_command(cmd: list, run_dir: str, gpu_env_var: str, gpus: list, titan_config: str):
    with open(os.path.join(run_dir, "command.txt"), "a") as f:
        f.write(f"=== {os.path.basename(run_dir)} {gpu_env_var}={','.join(gpus)} "
                f"TITAN_GRPO_CONFIG={shlex.quote(titan_config)}\n{shlex.join(cmd)}\n")


def write_metrics(run_dir: str, warmup_steps: int, steps: int) -> dict:
    warmup = warmup_steps if steps > warmup_steps else 0
    metrics = run_metrics(run_dir, warmup)
    if metrics["per_step"]:
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
        print(format_steps(metrics["per_step"]), flush=True)
        print("[grpo] summary:", json.dumps(metrics["summary"]), flush=True)
    return metrics


def run_verify(args, gpu: str, model: str, ckpt: str, prompts: list) -> int:
    env = pin_gpus(dict(os.environ), args.gpu_env_var, gpu)
    code = ("import json, sys, utils\n"
            "print('[verify] proof written to', utils.verify_checkpoint(*json.loads(sys.argv[1])), flush=True)")
    return subprocess.run([sys.executable, "-c", code, json.dumps([model, ckpt, prompts])], cwd=HERE, env=env).returncode
