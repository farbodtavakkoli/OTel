import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
FROZEN_TOWERS = ["vision_model", "vision_tower", "multi_modal_projector", "visual",
                 "embed_vision", "embed_audio", "audio_tower"]

for _p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "grpo_common")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

# Selects how GPUs are granted to the container and how --render-nodes entries are read; see the
# GPU selection table in README.md. SCALARLM_LOOP_TRAIN_ROCR_DEVICES indexes into the granted set,
# which the container sees renumbered from 0 on either platform.
PLATFORM = os.environ.get("GRPO_PLATFORM", "rocm")


def _is_cuda():
    return PLATFORM == "cuda"


def _gpus_flag(gpus):
    return f'"device={gpus}"'


def _gpu_args(nodes):
    if _is_cuda():
        return ["--gpus", _gpus_flag(",".join(str(n).replace("/dev/dri/renderD", "") for n in nodes))]
    args = ["--device=/dev/kfd"]
    for node in nodes:
        args.append(f"--device={node}")
    return args + ["--group-add", "video"]


def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def is_multimodal(cfg: dict) -> bool:
    return any("ForConditionalGeneration" in a for a in cfg.get("architectures") or [])


def default_lora_regex(cfg: dict) -> str:
    return (r".*language_model.*" if is_multimodal(cfg) else r".*") + LORA_PROJECTIONS


def served_model_name(model_dir: str) -> str:
    base = os.path.basename(os.path.abspath(model_dir).rstrip("/"))
    return base.replace("--", "/", 1)


def load_rows(path: str) -> list:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" not in r or "ground_truth" not in r:
                raise ValueError(f"{path}:{n}: expected keys `messages` and `ground_truth`, got {sorted(r)}")
            gt = r["ground_truth"]
            gt = json.loads(gt) if isinstance(gt, str) else dict(gt)
            rows.append({"id": gt.get("id") or r.get("id") or f"row{n}", "messages": r["messages"],
                         "question": gt.get("question"), "reference": gt.get("reference", ""),
                         "source": gt.get("source")})
    return rows


def render_prompts(rows: list, tokenizer_dir: str) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    for r in rows:
        r["rendered"] = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
    return turn_end_string(tok)


def turn_end_string(tok) -> str:
    marker = "ASSISTANT_TURN_MARKER"
    try:
        text = tok.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": marker}],
                                       tokenize=False, add_generation_prompt=False)
        tail = text.split(marker, 1)[1].strip()
        return tail or (tok.eos_token or "")
    except Exception:  # noqa: BLE001
        return tok.eos_token or ""


def http_json(url, payload=None, timeout=600.0, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    if body.startswith("data:"):  # the OpenAI proxy answers with an SSE media type but a plain JSON body
        body = body[len("data:"):].strip()
    return json.loads(body)


def health(api_url: str, timeout: float = 30) -> dict:
    return http_json(f"{api_url}/v1/health", timeout=timeout)


def wait_health(api_url: str, timeout_s: float, poll_s: float = 5.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            last = health(api_url, timeout=5)
            if last.get("all") == "up":
                return last
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(poll_s)
    raise TimeoutError(f"ScalarLM at {api_url} not healthy after {timeout_s:.0f}s: {last}")


def completion_request(api_url, model, prompt, max_tokens, seed, want_logprobs, temperature=1.0):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 1.0, "seed": seed}
    if want_logprobs:
        body.update(logprobs=0, return_token_ids=True)
    out = http_json(f"{api_url}/v1/completions", body)
    if "choices" not in out:
        raise RuntimeError(f"completion failed: {str(out)[:500]}")
    ch = out["choices"][0]
    lp = ch.get("logprobs") or {}
    return {"text": ch["text"], "finish_reason": ch.get("finish_reason"),
            "completion_tokens": out["usage"]["completion_tokens"],
            "prompt_token_ids": ch.get("prompt_token_ids"),
            "completion_token_ids": ch.get("token_ids"),
            "old_logprobs": lp.get("token_logprobs")}


def probe_logprobs(api_url, model, prompt):
    g = completion_request(api_url, model, prompt, 4, 0, want_logprobs=True)
    ok = (g["old_logprobs"] is not None and g["completion_token_ids"] is not None
          and g["prompt_token_ids"] is not None and len(g["old_logprobs"]) == len(g["completion_token_ids"]))
    return ok, g


def rollout(api_url, model, rendered_prompts, G, max_tokens, seed_base, concurrency, want_logprobs, temperature=1.0):
    jobs = [(pi, gi) for pi in range(len(rendered_prompts)) for gi in range(G)]

    def one(job):
        pi, gi = job
        err = None
        for attempt in range(3):
            try:
                return completion_request(api_url, model, rendered_prompts[pi], max_tokens,
                                          seed_base + pi * 1000 + gi, want_logprobs, temperature)
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2.0 * (attempt + 1))
        return {"text": "", "finish_reason": "error", "completion_tokens": 0, "error": repr(err),
                "prompt_token_ids": None, "completion_token_ids": None, "old_logprobs": None}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, jobs))
    return results, time.time() - t0


def submit_training_job(api_url, dataset, train_args):
    import scalarlm
    llm = scalarlm.SupermassiveIntelligence(api_url=api_url)
    return llm.train(dataset, train_args=train_args)


def wait_for_job(api_url, model_name, poll_s=2.0, timeout_s=7200):
    t_start = time.time()
    last = None
    while time.time() - t_start < timeout_s:
        try:
            status = http_json(f"{api_url}/v1/megatron/train/{model_name}", timeout=30).get("job_status", {})
            if status.get("status") != last:
                last = status.get("status")
                print(f"    job {model_name[:12]} -> {last}", flush=True)
            if last == "COMPLETED":
                return status, time.time()
            if last == "FAILED":
                raise RuntimeError(f"training job FAILED: {json.dumps(status)[:2000]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # transient: the API server stalls while a job logs; the timeout_s budget still bounds us
            print(f"    poll error {e}", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"job {model_name} not COMPLETED after {timeout_s}s")


def generate(api_url, model_name, prompt, max_tokens=8, temperature=0.0, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = http_json(f"{api_url}/v1/generate", {"prompts": [prompt], "model": model_name,
                                                       "max_tokens": max_tokens, "temperature": temperature}, timeout=300)
            res = out["results"][0]
            if res.get("error"):
                raise RuntimeError(res["error"])
            if res.get("response") is not None:
                return time.time() - t0, res["response"]
            ids = [r["request_id"] for r in out["results"]]
            while time.time() - t0 < timeout_s:  # the API returns after its response_timeout with None
                r0 = http_json(f"{api_url}/v1/generate/get_results", {"request_ids": ids}, timeout=300)["results"][0]
                if r0.get("error"):
                    raise RuntimeError(r0["error"])
                if r0.get("response") is not None:
                    return time.time() - t0, r0["response"]
                time.sleep(0.5)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code == 404:  # not registered yet
                time.sleep(1.0)
                continue
            raise RuntimeError(f"generate against {model_name}: HTTP {e.code} {body}")
    raise TimeoutError(f"{model_name} did not answer within {timeout_s}s")


sync_adapter = generate


def parse_job_log(job_dir):
    out = {"peak_mem_gib": None, "step_times_s": [], "from_pretrained_s": None, "gpu_env": None,
           "trainer_device": None, "log": None, "grpo_steps": [], "init_adapter": None,
           "init_optimizer": None, "on_policy_warning": False, "microbatch_times_s": []}
    logs = sorted(glob.glob(os.path.join(job_dir, "slurm-*.out")))
    if not logs:
        return out
    out["log"] = logs[-1]
    with open(logs[-1], errors="replace") as f:
        txt = f.read()
    mem = re.findall(r"max_allocated=([0-9.]+) GiB", txt)
    if mem:
        out["peak_mem_gib"] = max(float(m) for m in mem)
    out["step_times_s"] = [float(s) for s in re.findall(r"step time ([0-9.]+) seconds", txt)]
    out["microbatch_times_s"] = [float(s) for s in re.findall(r"grpo micro-batch .* time ([0-9.]+)s", txt)]
    m = re.search(r"from_pretrained latency: ([0-9.]+)s", txt)
    if m:
        out["from_pretrained_s"] = float(m.group(1))
    m = re.search(r"\[launcher\] slurm-provided GPU env: (.*)", txt)
    if m:
        out["gpu_env"] = m.group(1).strip()
    m = re.search(r"\[launcher\] trainer pinned to (.*)", txt)
    if m:
        out["trainer_device"] = m.group(1).strip()
    for s in re.finditer(r"GRPO step (\d+): loss (\S+) mean_ratio (\S+) clip_frac (\S+) tokens (\d+) "
                         r"sequences (\d+) micro_batches (\d+)", txt):
        out["grpo_steps"].append({"step": int(s.group(1)), "loss": float(s.group(2)),
                                  "mean_ratio": float(s.group(3)), "clip_frac": float(s.group(4)),
                                  "tokens": int(s.group(5)), "sequences": int(s.group(6)),
                                  "micro_batches": int(s.group(7))})
    m = re.search(r"init_adapter_from: loaded (\d+)/(\d+) tensors from (\S+)", txt)
    if m:
        out["init_adapter"] = {"loaded": int(m.group(1)), "total": int(m.group(2)), "from": m.group(3)}
    m = re.search(r"init_optimizer_from_adapter: carried AdamW state for (\d+) params", txt)
    if m:
        out["init_optimizer"] = int(m.group(1))
    out["on_policy_warning"] = "rows without old_logprobs present" in txt
    return out


def group_advantages(rewards, G, keep_zero_spread=False):
    advs, zero = [], 0
    for pi in range(len(rewards) // G):
        grp = rewards[pi * G:(pi + 1) * G]
        mean = statistics.fmean(grp)
        std = statistics.pstdev(grp)
        if std <= 1e-9:
            zero += 1
            advs.extend([0.0 if keep_zero_spread else None] * G)
        else:
            advs.extend([(r - mean) / (std + 1e-6) for r in grp])
    return advs, zero


def model_revision(model_dir: str) -> str:
    # The trainer resolves the model through snapshot_download with HF_HUB_OFFLINE=1, so the cache
    # entry needs a revision id. Any stable 40-hex string works: it only names a directory and is
    # written to refs/main. The derived id must stay deterministic, or the same checkpoint remaps
    # to a new cache entry on every run.
    rev_file = os.path.join(model_dir, "HF_REVISION")
    if os.path.exists(rev_file):
        with open(rev_file) as f:
            rev = f.read().strip()
        if rev:
            return rev
    listing = sorted((n, os.path.getsize(os.path.join(model_dir, n)))
                     for n in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, n)))
    return hashlib.sha1(repr(listing).encode()).hexdigest()


def ensure_hf_cache_entry(repo_id: str, model_dir: str, mount: str, hf_cache: str) -> str:
    rev = model_revision(model_dir)
    entry = os.path.join(hf_cache, "hub", "models--" + repo_id.replace("/", "--"))
    snap = os.path.join(entry, "snapshots", rev)
    os.makedirs(snap, exist_ok=True)
    os.makedirs(os.path.join(entry, "refs"), exist_ok=True)
    os.makedirs(os.path.join(entry, "blobs"), exist_ok=True)
    with open(os.path.join(entry, "refs", "main"), "w") as f:
        f.write(rev)
    n = 0
    for name in sorted(os.listdir(model_dir)):
        if name == "HF_REVISION":
            continue
        link = os.path.join(snap, name)
        target = f"{mount}/{name}"
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
        n += 1
    print(f"[server] HF cache entry {entry}: {n} files -> {mount} (revision {rev})", flush=True)
    return entry


def server_command(name, image, model_name, model_dir, mount, hf_cache, jobs_dir, render_nodes,
                   max_gpus, train_rocr, gpu_mem_util, vllm_args) -> list:
    cmd = ["docker", "run", "-d", "--name", name, "--init", "--network", "host",
           "--shm-size=64g", "--ulimit", "memlock=-1"]
    cmd += _gpu_args(render_nodes)
    cmd += ["--security-opt", "seccomp=unconfined",
            "--cap-add", "SYS_PTRACE", "--cap-add", "IPC_LOCK",
            "-e", f"SCALARLM_MODEL={model_name}",
            "-e", f"SCALARLM_MAX_GPUS_PER_NODE={max_gpus}",
            "-e", "SCALARLM_ENABLE_LORA=true",
            "-e", f"SCALARLM_GPU_MEMORY_UTILIZATION={gpu_mem_util}",
            "-e", 'SCALARLM_LIMIT_MM_PER_PROMPT={"image":0,"audio":0}',
            "-e", f"SCALARLM_VLLM_ARGS={vllm_args}",
            "-e", f"SCALARLM_LOOP_TRAIN_ROCR_DEVICES={train_rocr}",
            # read by train_job_entrypoint_gpu_pinned.sh to choose ROCR_ vs CUDA_VISIBLE_DEVICES
            "-e", f"GRPO_PLATFORM={PLATFORM}",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_DATASETS_CACHE=/tmp/hf_datasets",
            "-v", f"{os.path.abspath(model_dir)}:{mount}:ro",
            "-v", f"{os.path.abspath(hf_cache)}:/root/.cache/huggingface",
            "-v", f"{os.path.abspath(jobs_dir)}:/app/cray/jobs",
            "-v", f"{os.path.join(HERE, 'train_job_entrypoint_gpu_pinned.sh')}:/app/cray/scripts/train_job_entrypoint.sh:ro",
            image]
    return cmd


def start_server(cmd: list, name: str):
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"[server] started container {name}", flush=True)


def stop_server(name: str, log_path: str | None = None):
    if log_path:
        with open(log_path, "w") as f:
            subprocess.run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT)
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[server] stopped container {name}", flush=True)


def run_in_image(image, script_args, mounts, render_node=None, env=None) -> int:
    cmd = ["docker", "run", "--rm", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "PYTHONDONTWRITEBYTECODE=1"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    if render_node:
        cmd += _gpu_args([render_node]) + ["--security-opt", "seccomp=unconfined"]
    for host, cont, mode in mounts:
        cmd += ["-v", f"{os.path.abspath(host)}:{cont}:{mode}"]
    cmd += ["-v", f"{HERE}/utils.py:/work/utils.py:ro", "--entrypoint", "python", image, "/work/utils.py", *script_args]
    print("[verify] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


_CKPT = re.compile(r"checkpoint_(\d+)\.pt$")


def latest_checkpoint(path):
    if path.endswith(".pt"):
        return path
    files = [p for p in glob.glob(os.path.join(path, "checkpoint_*.pt")) if _CKPT.search(p)]
    if not files:
        raise FileNotFoundError(f"no checkpoint_*.pt under {path}")
    return max(files, key=lambda p: int(_CKPT.search(p).group(1)))


def continuation_check(job_a: str, job_b: str, steps_per_job: int = 1) -> dict:
    import torch

    pa, pb = latest_checkpoint(job_a), latest_checkpoint(job_b)
    a = torch.load(pa, map_location="cpu", weights_only=False)
    b = torch.load(pb, map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    out = {"checkpoint_a": pa, "checkpoint_b": pb, "step_a": a.get("step"), "step_b": b.get("step"),
           "n_tensors_a": len(sa), "n_tensors_b": len(sb), "same_keys": set(sa) == set(sb),
           "same_shapes": set(sa) == set(sb) and all(sa[k].shape == sb[k].shape for k in sa)}
    for part in ("lora_A", "lora_B"):
        keys = [k for k in sa if part in k and k in sb]
        num = sum(float((sb[k].float() - sa[k].float()).norm() ** 2) for k in keys) ** 0.5
        den = sum(float(sa[k].float().norm() ** 2) for k in keys) ** 0.5
        out[part] = {"tensors": len(keys), "bit_identical": sum(int(torch.equal(sa[k], sb[k])) for k in keys),
                     "rel_change_B_vs_A": (num / den) if den > 0 else None,
                     "nonzero_in_a": sum(int(bool(sa[k].abs().sum() > 0)) for k in keys),
                     "nonzero_in_b": sum(int(bool(sb[k].abs().sum() > 0)) for k in keys)}

    def adam_steps(ckpt):
        st = (ckpt.get("optimizer_state_dict") or {}).get("state") or {}
        return sorted({int(s["step"]) for s in st.values() if "step" in s})

    out["adam_steps_a"], out["adam_steps_b"] = adam_steps(a), adam_steps(b)
    out["adam_step_advanced_by_job_b"] = bool(out["adam_steps_a"] and out["adam_steps_b"]
                                              and out["adam_steps_b"][0] == out["adam_steps_a"][0] + steps_per_job)
    lb, la = out["lora_B"], out["lora_A"]
    out["b_is_continuation_of_a"] = bool(
        out["same_shapes"] and lb["nonzero_in_a"] == lb["tensors"] and lb["nonzero_in_b"] == lb["tensors"]
        and lb["bit_identical"] < lb["tensors"] and la["rel_change_B_vs_A"] is not None
        and la["rel_change_B_vs_A"] < 0.1 and out["adam_step_advanced_by_job_b"])
    return out


def export_peft_adapter(job_dir: str, out_dir: str) -> dict:
    import torch
    import yaml
    from peft import LoraConfig
    from safetensors.torch import save_file

    ckpt_path = latest_checkpoint(job_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(os.path.join(job_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    lc = cfg["lora_config"]
    sd = {}
    for k, v in ckpt["model_state_dict"].items():
        if ".lora_A." not in k and ".lora_B." not in k:
            continue
        k = k.replace(".default.", ".")
        if not k.startswith("base_model.model."):
            k = "base_model.model." + k
        sd[k] = v.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))
    LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc.get("lora_dropout", 0.0),
               target_modules=lc["target_modules"], use_rslora=lc.get("use_rslora", False),
               base_model_name_or_path=cfg["llm_name"], task_type=None).save_pretrained(out_dir)
    n_modules = len({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in sd})
    print(f"[export] {ckpt_path} (step {ckpt.get('step')}) -> {out_dir}: {len(sd)} tensors, {n_modules} LoRA modules, "
          f"{sum(v.numel() for v in sd.values()):,} params", flush=True)
    return sd


def peft_check(job_dir: str, out_dir: str, base: str, prompts: list, max_new_tokens: int = 96) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    ref_sd = export_peft_adapter(job_dir, out_dir)
    cfg = load_model_config(base)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    cls = AutoModelForImageTextToText if is_multimodal(cfg) else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base)
    t0 = time.time()
    model = cls.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    print(f"[verify] base loaded in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    model = PeftModel.from_pretrained(model, out_dir)
    live = {k.replace(".default.", "."): v for k, v in model.named_parameters() if "lora_" in k}
    matched = sum(1 for k, v in ref_sd.items() if k in live and torch.equal(live[k].detach().cpu().to(v.dtype), v))
    n_b = sum(1 for k in ref_sd if ".lora_B." in k)
    nonzero_b = sum(1 for k, v in ref_sd.items() if ".lora_B." in k and v.abs().sum() > 0)
    n_modules = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[verify] tensors matched {matched}/{len(ref_sd)}; lora_B non-zero {nonzero_b}/{n_b}; modules {n_modules}", flush=True)
    result = {"job_dir": job_dir, "adapter_dir": out_dir, "tensors": len(ref_sd), "modules": n_modules,
              "matched": matched, "lora_B_nonzero": nonzero_b, "lora_B_total": n_b,
              "device": torch.cuda.get_device_name(0), "answers": []}
    model.eval()
    for i, prompt in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        entry = {"prompt": prompt, "adapter_on": ans}
        if i == 0:
            with model.disable_adapter(), torch.no_grad():
                out0 = model.generate(**ids, max_new_tokens=48, do_sample=False)
            entry["adapter_off"] = tok.decode(out0[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[verify] prompt: {prompt}\n[verify] adapter ON: {ans}", flush=True)
        if "adapter_off" in entry:
            print(f"[verify] adapter OFF (48 tok): {entry['adapter_off']}", flush=True)
        result["answers"].append(entry)
    result["passed"] = matched == len(ref_sd) and nonzero_b == n_b
    with open(os.path.join(out_dir, "peft_check.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("PEFT ADAPTER CHECK " + ("OK" if result["passed"] else "FAILED"), flush=True)
    return result


def _cli():
    # Image-only, not host: these checks need torch and peft, which the client venv does not carry.
    ap = argparse.ArgumentParser(description="Checkpoint-level adapter checks, run inside the ScalarLM image.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("continuation", help="checkpoint-level proof that job B continued job A")
    c.add_argument("--job-a", required=True)
    c.add_argument("--job-b", required=True)
    c.add_argument("--steps-per-job", type=int, default=1)
    c.add_argument("--out", default=None, help="write the JSON result here as well")
    p = sub.add_parser("peft", help="export the job's adapter to PEFT layout, load it and generate")
    p.add_argument("--job-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base", required=True, help="base model directory")
    p.add_argument("--prompt", action="append", default=None)
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    if args.cmd == "continuation":
        out = continuation_check(args.job_a, args.job_b, args.steps_per_job)
        print(json.dumps(out, indent=1))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
        sys.exit(0 if out["b_is_continuation_of_a"] else 1)
    prompts = args.prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
    res = peft_check(args.job_dir, args.out_dir, args.base, prompts, args.max_new_tokens)
    sys.exit(0 if res["passed"] else 1)


def parse_args():
    p = argparse.ArgumentParser(description="GRPO (or rejection-sampling SFT) loop on a ScalarLM server")
    p.add_argument("--model", required=True, help="HF checkpoint directory `<org>--<name>` (tokenizer; mounted into the server)")
    p.add_argument("--model-name", default=None, help="name the server serves the base under (default: from --model's directory name)")
    p.add_argument("--data", required=True, help="JSONL with `messages` and `ground_truth` per row")
    p.add_argument("--output-dir", required=True, help="cycles.jsonl, summary.json, rollouts/, verify/ (and jobs/ by default)")
    p.add_argument("--label", default="run")
    p.add_argument("--objective", choices=("grpo", "rft"), default="grpo")
    p.add_argument("--steps", type=int, default=20, help="measured cycles")
    p.add_argument("--warmup-steps", type=int, default=2, help="untimed cycles before the measured ones")
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per cycle")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=256, help="parallel /v1/completions requests")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-regex", default=None, help="PEFT target_modules regex (default: derived from the model class)")
    p.add_argument("--block-size", type=int, default=8192, help="max_token_block_size (prompt+completion cap)")
    p.add_argument("--trainer-gpus", type=int, default=1, help="GPUs per training job (data-parallel ranks)")
    g = p.add_argument_group("grpo objective")
    g.add_argument("--learning-rate", type=float, default=1e-5)
    g.add_argument("--micro-batch", type=int, default=8, help="completions per forward/backward")
    g.add_argument("--ppo-epochs", type=int, default=1, help="optimizer steps per cycle (max_steps)")
    g.add_argument("--clip-eps", type=float, default=0.2)
    g.add_argument("--keep-zero-spread", action="store_true", help="train on zero-spread groups too (advantage 0)")
    g.add_argument("--no-carry-optimizer", action="store_true", help="fresh AdamW state every job")
    g.add_argument("--no-old-logprobs", action="store_true", help="force the on-policy ratio==1 path")
    r = p.add_argument_group("rft objective")
    r.add_argument("--rft-max-steps", type=int, default=8, help="optimizer steps per SFT job")
    r.add_argument("--rft-learning-rate", type=float, default=1e-4)
    r.add_argument("--rft-batch-size", type=int, default=1, help="token blocks per micro-batch")
    r.add_argument("--rft-grad-accum", type=int, default=1)
    r.add_argument("--rft-no-cumulative", action="store_true", help="train each job on the current batch's winners only")
    s = p.add_argument_group("server")
    s.add_argument("--api-url", default=os.environ.get("SCALARLM_API_URL", "http://127.0.0.1:8000"))
    s.add_argument("--start-server", action="store_true", help="start the ScalarLM container, wait for health, stop it on exit")
    s.add_argument("--image", default=os.environ.get("SCALARLM_IMAGE", "farbodatdocker/scalarlm:mi355-grpo-v1.1"))
    s.add_argument("--container-name", default="scalarlm-grpo")
    s.add_argument("--render-nodes", nargs="+", default=None,
                   help="/dev/dri/renderD* of the GPUs to pass in: vLLM first, training GPU(s) after (see README)")
    s.add_argument("--max-gpus", type=int, default=None, help="SCALARLM_MAX_GPUS_PER_NODE (default: number of render nodes)")
    s.add_argument("--train-rocr", default=None, help="ROCr index list for training jobs (default: 1.. ; 0 with one render node)")
    s.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM gpu_memory_utilization (0.35 when it shares the training GPU)")
    s.add_argument("--vllm-args", default="--max-model-len=8192 --max-num-seqs=256 --seed=42 --max-lora-rank=16 --max-loras=2 --max-cpu-loras=4")
    s.add_argument("--hf-cache", default=os.path.join(os.environ.get("GRPO_ENV_DIR", "./envs"), "hf_cache_scalarlm"),
                   help="HF cache dir mounted at /root/.cache/huggingface (a symlink entry for --model is created)")
    s.add_argument("--jobs-dir", default=None, help="host dir bound at /app/cray/jobs (default <output-dir>/jobs)")
    s.add_argument("--server-timeout", type=float, default=1200, help="seconds to wait for /v1/health all=up")
    s.add_argument("--keep-server", action="store_true", help="leave the container running on exit")
    v = p.add_argument_group("verification")
    v.add_argument("--verify", action="store_true",
                   help="after the loop: continuation proof (last two jobs), PEFT export + load + generate, generate through the served adapter")
    v.add_argument("--verify-device", default=None, help="render node for the PEFT check (default: the last of --render-nodes)")
    v.add_argument("--verify-prompt", action="append", default=None)
    p.add_argument("--dry-run", action="store_true", help="print the server command, train_args and plan, then exit")
    args = p.parse_args()
    args.model = os.path.abspath(args.model)
    args.model_name = args.model_name or served_model_name(args.model)
    args.jobs_dir = os.path.abspath(args.jobs_dir or os.path.join(args.output_dir, "jobs"))
    if args.start_server and not args.render_nodes:
        p.error("--start-server needs --render-nodes")
    if args.render_nodes:
        args.max_gpus = args.max_gpus or len(args.render_nodes)
        args.train_rocr = args.train_rocr or ("0" if len(args.render_nodes) == 1 else
                                              ",".join(str(i) for i in range(1, len(args.render_nodes))))
    return args


def mount_path(args) -> str:
    return "/models/" + os.path.basename(args.model)


def server_cmd(args) -> list:
    return server_command(args.container_name, args.image, args.model_name, args.model, mount_path(args),
                                args.hf_cache, args.jobs_dir, args.render_nodes, args.max_gpus, args.train_rocr,
                                args.gpu_mem_util, args.vllm_args)


def append_jsonl(path: str, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def verify(args, summary: dict, run_dir: str) -> dict:
    out_dir = os.path.join(run_dir, "verify")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [m for m in summary["models"] if m]
    proof = {"jobs": jobs}
    prompts = args.verify_prompt or ["In 5G NR, what is the difference between SSB and CSI-RS, and what is each used for?"]
    if not jobs:
        proof["error"] = "no training job completed"
        return proof

    if len(jobs) >= 2 and args.objective == "grpo":
        rc = run_in_image(args.image, ["continuation", "--job-a", f"/jobs/{jobs[-2]}", "--job-b", f"/jobs/{jobs[-1]}",
                                             "--steps-per-job", str(args.ppo_epochs), "--out", "/out/continuation_check.json"],
                                [(args.jobs_dir, "/jobs", "ro"), (out_dir, "/out", "rw")])
        with open(os.path.join(out_dir, "continuation_check.json")) as f:
            proof["continuation"] = json.load(f)
        proof["continuation"]["rc"] = rc
        print(f"[verify] continuation {jobs[-2][:12]} -> {jobs[-1][:12]}: "
              f"{'OK' if proof['continuation'].get('b_is_continuation_of_a') else 'FAILED'}", flush=True)

    device = args.verify_device or (args.render_nodes[-1] if args.render_nodes else None)
    if device:
        mount = mount_path(args)
        rc = run_in_image(args.image, ["peft", "--job-dir", f"/jobs/{jobs[-1]}", "--out-dir", f"/jobs/{jobs[-1]}/peft_adapter",
                                             "--base", mount, *sum((["--prompt", p] for p in prompts), [])],
                                [(args.jobs_dir, "/jobs", "rw"), (args.model, mount, "ro")], render_node=device)
        pc = os.path.join(args.jobs_dir, jobs[-1], "peft_adapter", "peft_check.json")
        proof["peft"] = json.load(open(pc)) if os.path.exists(pc) else {"error": f"peft check rc={rc}"}
        proof["peft"]["rc"] = rc
    else:
        proof["peft"] = {"skipped": "no --verify-device / --render-nodes"}

    served = []
    for p in prompts:
        row = {"messages": [{"role": "user", "content": p}]}
        render_prompts([row], args.model)
        secs, ans = generate(args.api_url, jobs[-1], row["rendered"], max_tokens=160, temperature=0.0)
        served.append({"prompt": p, "answer": ans, "seconds": secs})
        print(f"[verify] served adapter {jobs[-1][:12]} ({secs:.1f}s): {ans[:200]}", flush=True)
    proof["served_generate"] = served
    with open(os.path.join(out_dir, "proof.json"), "w") as f:
        json.dump(proof, f, indent=1)
    print(f"[verify] proof written to {os.path.join(out_dir, 'proof.json')}", flush=True)
    return proof


if __name__ == "__main__":
    _cli()
