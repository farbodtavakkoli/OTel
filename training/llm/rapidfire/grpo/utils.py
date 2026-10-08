import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

for _p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common")):
    if _p and os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))


def load_model_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def is_multimodal(cfg):
    return any("ForConditionalGeneration" in a for a in cfg.get("architectures") or [])


def default_lora_regex(cfg):
    return (r".*language_model.*" if is_multimodal(cfg) else r".*") + LORA_PROJECTIONS


def auto_model_class(cfg):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_multimodal(cfg) else AutoModelForCausalLM


def eos_id_set(model_dir):
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else None
    ids = set()
    for v in (tok.eos_token_id, getattr(cfg, "eos_token_id", None), getattr(text_cfg, "eos_token_id", None)):
        if v is not None:
            ids |= set(v if isinstance(v, list) else [v])
    if "<end_of_turn>" in tok.get_vocab():
        ids.add(tok.convert_tokens_to_ids("<end_of_turn>"))
    return ids


def grpo_versions():
    import torch
    import transformers
    import peft
    import trl
    import vllm
    import rapidfireai
    return {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
            "peft": peft.__version__, "vllm": vllm.__version__, "rapidfireai": rapidfireai.__version__,
            "python": sys.version.split()[0]}


def load_grpo_dataset(path):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=path, split="train")
    missing = [c for c in ("messages", "ground_truth") if c not in ds.column_names]
    if missing:
        raise ValueError(f"{path}: rows need {missing}; found columns {ds.column_names}")

    def to_trl(row):
        msgs = row["messages"]
        last = msgs[-1].get("role") if msgs else None
        if last != "user":
            raise ValueError(f"last message must be a user turn (the policy answers it), got {last!r}")
        gt = row["ground_truth"]
        return {"prompt": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "ground_truth": gt if isinstance(gt, str) else json.dumps(gt)}

    return ds.map(to_trl, remove_columns=ds.column_names, desc="messages -> prompt")


def parse_ground_truth(gt):
    try:
        d = json.loads(gt) if isinstance(gt, str) else dict(gt)
    except Exception:
        d = {"reference": str(gt)}
    return d if isinstance(d, dict) else {"reference": str(gt)}


# Per-process state shared by the reward function, the callbacks and the generate/sync wrappers in the worker.
# Controller -> worker settings travel through the environment (RF_BENCH_OUT, RF_BENCH_MAX_COMPLETION,
# RF_BENCH_EOS_IDS, RF_BENCH_DUMP_COMPLETIONS): RapidFire spawns the worker as a fresh process, so
# they must be set before Experiment.run_fit and are read back through the accessors below.
STATE = {"step": {}, "judge_s": [], "reward_s": [], "records": []}


def out_dir():
    return os.environ.get("RF_BENCH_OUT")


def eos_ids():
    raw = os.environ.get("RF_BENCH_EOS_IDS", "")
    return {int(x) for x in raw.split(",") if x.strip()}


def event(name, **fields):
    out = out_dir()
    if not out:
        return
    rec = {"event": name, "t": time.time(), "pid": os.getpid(), **fields}
    with open(os.path.join(out, "events.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def wlog(msg):
    line = f"[worker {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    out = out_dir()
    if out:
        with open(os.path.join(out, "worker.log"), "a") as f:
            f.write(line + "\n")


def _callbacks():
    import torch
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            event("train_begin", max_steps=state.max_steps)
            wlog(f"train begin: max_steps={state.max_steps}")

        def on_train_end(self, a, state, control, **kw):
            event("train_end", global_step=state.global_step)
            wlog(f"train end: global_step={state.global_step}")

        def on_step_begin(self, a, state, control, **kw):
            STATE["step"] = {"t_begin": time.perf_counter()}
            STATE["judge_s"].clear()
            STATE["reward_s"].clear()

        def on_step_end(self, a, state, control, **kw):
            s = STATE["step"]
            rec = {
                "step": state.global_step,
                "cycle_s": time.perf_counter() - s["t_begin"],
                "rollout_s": s.get("rollout_s"),
                "sync_s": s.get("sync_s"),
                "gen_score_s": s.get("gen_score_s"),
                "judge_s": sum(STATE["judge_s"]) if STATE["judge_s"] else None,
                "reward_s": sum(STATE["reward_s"]) if STATE["reward_s"] else None,
                "completion_tokens": s.get("completion_tokens"),
                "num_completions": s.get("num_completions"),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "t_wall_end": time.time(),
            }
            rec["train_step_s"] = rec["cycle_s"] - rec["gen_score_s"] if rec["gen_score_s"] is not None else None
            rec["tokens_per_s"] = (rec["completion_tokens"] / rec["rollout_s"]
                                   if rec["rollout_s"] and rec["completion_tokens"] else None)
            s["pending"] = rec

    class LogAttach(TrainerCallback):
        # on_log fires after on_step_end for the same step; attach TRL's metrics, then flush.
        def on_log(self, a, state, control, logs=None, **kw):
            rec = STATE["step"].get("pending")
            if rec is None:
                return
            rec["trl"] = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            STATE["records"].append(rec)
            out = out_dir()
            if out:
                with open(os.path.join(out, "steps.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            wlog(f"step {rec['step']}: cycle={rec['cycle_s']:.1f}s rollout={rec['rollout_s']} judge={rec['judge_s']} "
                 f"sync={rec['sync_s']} train={rec['train_step_s']} tok/s={rec['tokens_per_s']} "
                 f"reward={rec['trl'].get('reward')} mean_len={rec['trl'].get('completions/mean_length')} "
                 f"peak={rec['peak_mem_gb']:.1f}GB")
            STATE["step"]["pending"] = None

    return [StepTimer(), LogAttach()]


def attach_timers(trainer):
    import torch

    for cb in _callbacks():
        trainer.add_callback(cb)
    vg = trainer.vllm_generation
    orig_sync, orig_gen, orig_gas = vg.sync_weights, vg.generate, trainer._generate_and_score_completions

    def timed_sync(*a, **kw):
        t = time.perf_counter()
        r = orig_sync(*a, **kw)
        STATE["step"]["sync_s"] = STATE["step"].get("sync_s", 0.0) + (time.perf_counter() - t)
        return r

    def timed_generate(*a, **kw):
        t = time.perf_counter()
        r = orig_gen(*a, **kw)
        s = STATE["step"]
        s["rollout_s"] = s.get("rollout_s", 0.0) + (time.perf_counter() - t)
        completion_ids = r[1]
        s["completion_tokens"] = s.get("completion_tokens", 0) + sum(len(c) for c in completion_ids)
        s["num_completions"] = s.get("num_completions", 0) + len(completion_ids)
        return r

    def timed_gas(*a, **kw):
        t = time.perf_counter()
        r = orig_gas(*a, **kw)
        STATE["step"]["gen_score_s"] = STATE["step"].get("gen_score_s", 0.0) + (time.perf_counter() - t)
        return r

    vg.sync_weights, vg.generate = timed_sync, timed_generate
    trainer._generate_and_score_completions = timed_gas
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    lora_dtypes = sorted({str(p.dtype) for n, p in trainer.model.named_parameters() if "lora_" in n})
    base = trainer.model.get_base_model() if hasattr(trainer.model, "get_base_model") else trainer.model
    event("timers_attached", trainable_params=trainable, total_params=total, lora_dtypes=lora_dtypes,
          model_class=type(trainer.model).__name__, base_model_class=type(base).__name__,
          peft_config=str(getattr(trainer.model, "peft_config", None)))
    wlog(f"timers attached: trainable={trainable / 1e6:.1f}M of {total / 1e9:.2f}B, lora dtypes={lora_dtypes}")
    torch.cuda.reset_peak_memory_stats()


_INSTALLED = {}


def install_worker_hooks():
    import torch
    import trl.generation.vllm_client as vc
    from trl import GRPOTrainer

    vc._HAS_WEIGHT_UPDATE_LIFECYCLE = True
    if _INSTALLED:
        return
    orig_init = GRPOTrainer.__init__

    def patched_init(self, *args, **kwargs):
        a = kwargs.get("args")
        event("trainer_init_start", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              cuda_device_count=torch.cuda.device_count(),
              device_name=torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
              vllm_mode=getattr(a, "vllm_mode", None), vllm_port=getattr(a, "vllm_server_port", None),
              max_steps=getattr(a, "max_steps", None), grad_accum=getattr(a, "gradient_accumulation_steps", None),
              steps_per_generation=getattr(a, "steps_per_generation", None))
        t0 = time.perf_counter()
        orig_init(self, *args, **kwargs)
        event("trainer_init_end", init_s=time.perf_counter() - t0)
        if out_dir():
            attach_timers(self)

    GRPOTrainer.__init__ = patched_init
    _INSTALLED["grpo_init"] = orig_init


# RapidFire dill-serialises these into its SQLite DB and the spawned worker unpickles them. Everything they need is
# imported inside the body: a function whose module is not importable in the worker is pickled by value and gets an
# empty globals dict there (dill is imported while the worker is still bootstrapping its __main__), and settings only
# reach the worker through the environment it inherits.
def create_model(model_config):
    import time as _time
    import torch as _torch
    from transformers import AutoTokenizer as _AutoTokenizer
    import utils as U

    U.install_worker_hooks()
    U.event("create_model_start", model=model_config["model_name"], model_kwargs=model_config.get("model_kwargs"))
    t0 = _time.perf_counter()
    kw = dict(model_config.get("model_kwargs") or {})
    dtype = kw.pop("dtype", "bfloat16")
    kw["dtype"] = getattr(_torch, dtype) if isinstance(dtype, str) else dtype
    kw.setdefault("attn_implementation", "sdpa")
    model = U.auto_model_class(U.load_model_config(model_config["model_name"])).from_pretrained(model_config["model_name"], **kw)
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = True
    tokenizer = _AutoTokenizer.from_pretrained(model_config["model_name"], **dict(model_config.get("tokenizer_kwargs") or {}))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_s = _time.perf_counter() - t0
    dev = str(next(model.parameters()).device)
    n = sum(p.numel() for p in model.parameters())
    U.event("create_model_done", load_s=load_s, model_class=type(model).__name__, device=dev, params=n,
            cuda_device_count=_torch.cuda.device_count())
    U.wlog(f"create_model: {type(model).__name__} {n / 1e9:.2f}B on {dev} in {load_s:.1f}s "
           f"(torch.cuda.device_count={_torch.cuda.device_count()})")
    return model, tokenizer


def judge_reward(prompts, completions, completion_ids=None, ground_truth=None, **kwargs):
    import json as _json
    import os as _os
    import time as _time
    import judge_reward as _jr
    import utils as U

    max_len = int(_os.environ.get("RF_BENCH_MAX_COMPLETION", "1024"))
    eos = U.eos_ids()
    t = _time.perf_counter()
    gts = [U.parse_ground_truth(g) for g in (ground_truth or [{}] * len(completions))]
    questions = [g.get("question") or _jr.question_text(p) for g, p in zip(gts, prompts)]
    references = [g.get("reference", "") for g in gts]
    sources = [g.get("source") for g in gts]
    finish = []
    for ids in completion_ids or [[]] * len(completions):
        truncated = len(ids) >= max_len and (not ids or ids[-1] not in eos)
        finish.append("length" if truncated else "stop")
    out = _jr.score_batch(questions, completions, references, sources, finish)
    reward_s = _time.perf_counter() - t
    U.STATE["judge_s"].append(out["timing_s"])
    U.STATE["reward_s"].append(reward_s)
    failed = sum(1 for j, s in zip(out["judge"], sources) if j is None and s != "3gpp_tsg")
    ids_list = completion_ids or []
    call_no = U.STATE["reward_calls"] = U.STATE.get("reward_calls", 0) + 1
    finish_length_frac = sum(f == "length" for f in finish) / max(1, len(finish))
    o = U.out_dir()
    if o:
        with open(_os.path.join(o, "judge_calls.jsonl"), "a") as f:
            f.write(_json.dumps({"call": call_no, "t_start": t, "t_end": _time.perf_counter(), "n": len(completions),
                                 "timing_s": out["timing_s"], "reward_s": reward_s, "judge_failed": failed,
                                 "finish_length_frac": finish_length_frac,
                                 "mean_reward": sum(out["reward"]) / max(1, len(out["reward"])),
                                 "rubric": out.get("rubric")}) + "\n")
        # RF_BENCH_DUMP_COMPLETIONS completions per generation batch as text, strided over the batch so the sample
        # spans different prompts (consecutive completions belong to the same prompt).
        n_dump = int(_os.environ.get("RF_BENCH_DUMP_COMPLETIONS", "8"))
        if n_dump > 0:
            stride = max(1, len(completions) // n_dump)
            with open(_os.path.join(o, "completions_sample.jsonl"), "a") as f:
                for i in range(0, len(completions), stride)[:n_dump]:
                    c = completions[i]
                    text = c[0]["content"] if isinstance(c, list) and c and isinstance(c[0], dict) else c
                    f.write(_json.dumps({"call": call_no, "i": i, "source": sources[i], "finish": finish[i],
                                         "reward": out["reward"][i], "judge": out["judge"][i],
                                         "n_tokens": len(ids_list[i]) if i < len(ids_list) else None,
                                         "completion": text}) + "\n")
    log_metric = kwargs.get("log_metric")
    if log_metric is not None:
        log_metric("judge_s", out["timing_s"])
        log_metric("judge_failed", float(failed))
        log_metric("finish_length_frac", finish_length_frac)
        log_metric("last_tok_eos_frac", sum(1 for ids in ids_list if ids and ids[-1] in eos) / max(1, len(ids_list)))
    return out["reward"]


def _http_ok(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start_mlflow(port, run_dir, timeout=60):
    url = f"http://127.0.0.1:{port}"
    if _http_ok(url + "/health"):
        return None
    mlflow_dir = os.path.join(run_dir, "mlflow")
    os.makedirs(mlflow_dir, exist_ok=True)
    log = open(os.path.join(mlflow_dir, "mlflow_server.log"), "wb")
    proc = subprocess.Popen(["mlflow", "server", "--host", "127.0.0.1", "--port", str(port),
                             "--backend-store-uri", "sqlite:///" + _mlflow_db(),
                             "--default-artifact-root", os.path.join(mlflow_dir, "artifacts")],
                            stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _http_ok(url + "/health"):
            return proc
        if proc.poll() is not None:
            raise RuntimeError("mlflow server exited; see mlflow/mlflow_server.log")
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"mlflow server not healthy after {timeout}s")


def _mlflow_db():
    rf_home = os.environ.get("RF_HOME", os.path.join(os.path.expanduser("~"), "rapidfireai"))
    os.makedirs(rf_home, exist_ok=True)
    return os.path.join(rf_home, "mlflow.db")


def stop_mlflow(proc, run_dir):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if os.path.exists(_mlflow_db()):
        os.makedirs(os.path.join(run_dir, "mlflow"), exist_ok=True)
        shutil.copy2(_mlflow_db(), os.path.join(run_dir, "mlflow", "mlflow.db"))


def collect_adapter(experiment, adapter_dir):
    exp_dir = os.path.join(str(experiment.experiment_path), experiment.experiment_name)
    final = [root for root, _, files in os.walk(exp_dir)
             if os.path.basename(root) == "final_checkpoint" and "adapter_config.json" in files]
    if not final:
        raise RuntimeError(f"no final_checkpoint with adapter_config.json under {exp_dir}")
    src = sorted(final)[-1]
    if os.path.isdir(adapter_dir):
        shutil.rmtree(adapter_dir)
    shutil.copytree(src, adapter_dir)
    print(f"[grpo] adapter copied {src} -> {adapter_dir}: {sorted(os.listdir(adapter_dir))}", flush=True)
    return src


def verify_adapter(base, adapter, prompts, max_new_tokens=160):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(base)
    model = auto_model_class(load_model_config(base)).from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    exported = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    n_match = n_lora = n_b_nonzero = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            n_lora += 1
            key = n.replace(".default", "")
            if key in exported and torch.equal(exported[key].to(torch.bfloat16), p.detach().to("cpu", torch.bfloat16)):
                n_match += 1
            if "lora_B" in n and p.detach().abs().sum().item() > 0:
                n_b_nonzero += 1
    n_modules = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[verify] LoRA modules attached: {n_modules}; tensors: {n_match}/{n_lora} match the file "
          f"({len(exported)} exported); lora_B non-zero: {n_b_nonzero}", flush=True)
    if not (n_lora > 0 and n_match == n_lora == len(exported)):
        raise RuntimeError(f"adapter reload mismatch: {n_match}/{n_lora} attached, {len(exported)} exported")

    lines = [f"base={base}\nadapter={adapter}\nlora_modules_attached={n_modules}\nexported_tensors={len(exported)}\n"
             f"matched={n_match}\n"]
    generated = []
    for i, prompt in enumerate(prompts, 1):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(0)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        generated.append(text)
        print(f"[verify] prompt {i}: {prompt}\n[verify] adapter ON: {text}", flush=True)
        lines.append(f"\n=== prompt {i} ===\n{prompt}\n=== adapter ON ===\n{text}\n")
        if i == 1:
            with model.disable_adapter(), torch.no_grad():
                out0 = model.generate(**ids, max_new_tokens=48, do_sample=False)
            text0 = tok.decode(out0[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"[verify] adapter OFF (48 tok): {text0}", flush=True)
            lines.append(f"=== adapter OFF (first 48 tok) ===\n{text0}\n")
    proof_path = os.path.join(adapter, "load_and_generate_proof.txt")
    with open(proof_path, "w") as f:
        f.write("".join(lines))
    proof = {"adapter_dir": adapter, "lora_tensors": n_lora, "lora_tensors_matching_file": n_match,
             "exported_tensors": len(exported), "lora_modules_attached": n_modules, "lora_B_nonzero_tensors": n_b_nonzero,
             "prompts": list(prompts), "generated_text": generated, "proof_wall_s": time.perf_counter() - t0,
             "model_class": type(model.get_base_model()).__name__, "proof_file": proof_path}
    del model
    torch.cuda.empty_cache()
    return proof


def rf_overhead(events, run_fit_wall):
    ev = {}
    for e in events:
        ev.setdefault(e["event"], e)

    def dt(a, b):
        return (ev[b]["t"] - ev[a]["t"]) if a in ev and b in ev else None

    return ev, {
        "rf_spawn_to_create_model_s": dt("run_fit_start", "create_model_start"),
        "model_load_s": ev.get("create_model_done", {}).get("load_s"),
        "rf_shm_transfer_and_peft_s": dt("create_model_done", "trainer_init_start"),
        "trainer_init_s": ev.get("trainer_init_end", {}).get("init_s"),
        "trainer_init_to_train_begin_s": dt("trainer_init_end", "train_begin"),
        "train_s": dt("train_begin", "train_end"),
        "rf_train_end_to_run_fit_end_s": dt("train_end", "run_fit_end"),
        "run_fit_s": run_fit_wall,
        "experiment_end_s": ev.get("experiment_end", {}).get("end_s"),
        "experiment_init_s": ev.get("experiment_created", {}).get("init_s"),
    }


def write_grpo_summary(path, run_dir, config, warmup_steps, run_fit_wall, adapter_dir, adapter_src, rubric):
    records = [json.loads(l) for l in open(os.path.join(run_dir, "steps.jsonl")) if l.strip()]
    events = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl")) if l.strip()]
    ev, overhead = rf_overhead(events, run_fit_wall)
    attached = ev.get("timers_attached", {})
    measured = [r for r in records if r["step"] > warmup_steps]
    config = dict(config, model_class=attached.get("base_model_class") or ev.get("create_model_done", {}).get("model_class"),
                  total_params=attached.get("total_params"))
    summary = {
        "label": config.get("label"), "config": config, "versions": grpo_versions(),
        "rf_events": ev, "rf_overhead": overhead,
        "steps_total": len(records), "steps_measured": len(measured),
        "peak_mem_gb": max((r["peak_mem_gb"] for r in records), default=None),
        "adapter_dir": adapter_dir, "adapter_src": adapter_src, "rubric": rubric,
        "trainable_params": attached.get("trainable_params"), "lora_dtypes": attached.get("lora_dtypes"),
    }
    for k in ("rollout_s", "judge_s", "reward_s", "train_step_s", "sync_s", "cycle_s", "gen_score_s", "tokens_per_s",
              "completion_tokens"):
        summary[k] = [r.get(k) for r in measured]
    summary["reward_per_step"] = [r["trl"].get("reward") for r in records]
    summary["mean_length_per_step"] = [r["trl"].get("completions/mean_length") for r in records]
    summary["clipped_ratio_per_step"] = [r["trl"].get("completions/clipped_ratio") for r in records]
    summary["reward_first"] = measured[0]["trl"].get("reward") if measured else None
    summary["reward_last"] = measured[-1]["trl"].get("reward") if measured else None
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# The trainer container re-execs the launcher. Name it explicitly rather than via __file__,
# which from this module would resolve to utils.py.
LAUNCHER = os.path.join(HERE, "train_llm_grpo_rapidfire.py")
SERVER_NAME = "grpo-rapidfire-vllm"
DOCKER_GPU_ARGS = ["--network", "host", "--device", "/dev/kfd", "--group-add", "video",
                   "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE", "--ipc", "host"]
DOCKER_GPU_ARGS_CUDA = ["--network", "host", "--security-opt", "seccomp=unconfined",
                        "--cap-add", "SYS_PTRACE", "--ipc", "host"]
# rocm (default) pins a GPU with ROCR_VISIBLE_DEVICES and exposes /dev/kfd + the render nodes;
# cuda pins it with `--gpus "device=<n>"` and CUDA_VISIBLE_DEVICES, and has no kfd/render nodes.
PLATFORM = os.environ.get("GRPO_PLATFORM", "rocm")


def _is_cuda():
    return PLATFORM == "cuda"


def _visible_devices_var():
    return "CUDA_VISIBLE_DEVICES" if _is_cuda() else "ROCR_VISIBLE_DEVICES"


def _gpus_flag(gpus):
    return f'"device={gpus}"'


# RapidFire's own ports (its default MLflow port is 8852); off the defaults so two stacks can share a node.

def parse_args():
    p = argparse.ArgumentParser(description="GRPO LoRA training with RapidFire AI (TRL GRPOTrainer) + vLLM server")
    p.add_argument("--model", required=True, help="HF checkpoint directory (absolute path)")
    p.add_argument("--data", required=True, help="JSONL with `messages` and `ground_truth` per row")
    p.add_argument("--output-dir", required=True, help="steps.jsonl, events.jsonl, summary.json, adapter/, RapidFire's experiment and log dirs")
    p.add_argument("--steps", type=int, default=22, help="optimizer steps in total")
    p.add_argument("--warmup-steps", type=int, default=2, help="leading steps excluded from summary.json lists")
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt")
    p.add_argument("--per-device-batch", type=int, default=8, help="completions per micro-batch")
    p.add_argument("--max-tokens", type=int, default=1024, help="max completion tokens")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-regex", default=None, help="PEFT target_modules regex (default: derived from the model class)")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="bench")
    p.add_argument("--experiment-name", default=None, help="RapidFire experiment name, unique per RapidFire DB (default: rf-grpo-<output-dir name>)")
    p.add_argument("--num-chunks", type=int, default=1, help="RapidFire chunk count (1 = no chunk swaps)")
    p.add_argument("--dump-completions", type=int, default=8, help="completions per generation batch written to completions_sample.jsonl (0 = off)")
    p.add_argument("--server-mode", default="docker", choices=["docker", "process", "external"],
                   help="docker: server + trainer containers from the host; process: vllm-serve child process; "
                        "external: server already running at --server-host:--server-port")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8106)
    p.add_argument("--server-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--server-timeout", type=int, default=900, help="seconds to wait for the server to become healthy")
    p.add_argument("--server-gpu", default="0",
                   help="ROCR_VISIBLE_DEVICES (rocm) / CUDA_VISIBLE_DEVICES (cuda) for the server "
                        "(index into the visible GPUs)")
    p.add_argument("--train-gpu", default="1",
                   help="ROCR_VISIBLE_DEVICES (rocm) / CUDA_VISIBLE_DEVICES (cuda) for the RapidFire "
                        "controller + worker (exactly one GPU)")
    p.add_argument("--cuda-devices", default=os.environ.get("GRPO_CUDA_DEVICES", "0,1"),
                   help="docker, cuda: physical GPU indices exposed to BOTH containers (comma separated; "
                        "both GPUs, for the NCCL push). --server-gpu/--train-gpu index into this list.")
    p.add_argument("--dri-devices", default=os.environ.get("GRPO_DRI_DEVICES", "/dev/dri"),
                   help="docker: render nodes exposed to BOTH containers (space separated; both GPUs, for the RCCL push)")
    p.add_argument("--image", default=os.environ.get("GRPO_IMAGE", "grpo-rapidfire:mi355"))
    p.add_argument("--shm-size", default="100g", help="docker: /dev/shm of the trainer container (RapidFire copies the bf16 base model into it)")
    p.add_argument("--mlflow-port", type=int, default=int(os.environ.get("RF_MLFLOW_PORT", "8862")), help="RapidFire's MLflow server port")
    p.add_argument("--verify", action="store_true", help="after the adapter check, generate with base + adapter (adapter on and off)")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--verify-only", action="store_true", help="skip the server and training; --verify the adapter already in --output-dir")
    p.add_argument("--dry-run", action="store_true", help="print the commands / config that would run, then exit")
    args = p.parse_args()
    args.experiment_name = args.experiment_name or "rf-grpo-" + os.path.basename(os.path.abspath(args.output_dir).rstrip("/"))
    return args


def server_command(args):
    return ["-m", "trl.scripts.vllm_serve", "--model", args.model, "--host", args.server_host,
            "--port", str(args.server_port), "--dtype", "bfloat16",
            "--gpu_memory_utilization", str(args.server_gpu_memory_utilization),
            "--max_model_len", str(args.max_model_len), "--enable_prefix_caching", "true",
            "--max-num-seqs", str(args.max_num_seqs), "--limit-mm-per-prompt", '{"image":0,"audio":0}',
            "--seed", str(args.seed)]


def server_args_record(args):
    return {"gpu_memory_utilization": args.server_gpu_memory_utilization, "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs, "dtype": "bfloat16", "enable_prefix_caching": True,
            "weight_transfer_config": {"backend": "nccl"}, "logprobs_mode": "processed_logprobs",
            "limit_mm_per_prompt": {"image": 0, "audio": 0}, "seed": args.seed}


def _http_ok(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_server(args, still_alive):
    base = f"http://{args.server_host}:{args.server_port}"
    t0 = time.time()
    while time.time() - t0 < args.server_timeout:
        if _http_ok(base + "/health") and _http_ok(base + "/v1/models"):
            print(f"[server] healthy after {time.time() - t0:.0f}s at {base}", flush=True)
            return
        if not still_alive():
            raise RuntimeError("vLLM server exited before becoming healthy (see vllm_server.log)")
        time.sleep(5)
    raise RuntimeError(f"vLLM server not healthy after {args.server_timeout}s")


def worker_argv():
    argv, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--server-mode":
            skip = True
            continue
        if a.startswith("--server-mode=") or a == "--dry-run":
            continue
        argv.append(a)
    return argv + ["--server-mode", "external"]


def _bind_mounts(args):
    wanted = []
    shared = os.environ.get("GRPO_SHARED_DIR")
    if shared:
        wanted.append((os.path.abspath(shared), "rw"))
    wanted += [(REPO_ROOT, "ro"), (os.path.abspath(args.model), "ro"),
               (os.path.dirname(os.path.abspath(args.data)), "ro"), (os.path.abspath(args.output_dir), "rw")]
    mounts = []
    for path, mode in wanted:
        if any(path == m or path.startswith(m.rstrip("/") + "/") for m, _ in mounts):
            continue
        mounts.append((path, mode))
    return [x for path, mode in mounts for x in ("-v", f"{path}:{path}:{mode}")]


def docker_commands(args):
    # Both containers get BOTH GPUs (TRL 1.12 pushes the full merged model over NCCL/RCCL); the
    # visibility variable then selects which one each process trains or serves on.
    if _is_cuda():
        devices = []
        gpu_args = DOCKER_GPU_ARGS_CUDA + ["--gpus", _gpus_flag(args.cuda_devices)]
    else:
        devices = [x for d in args.dri_devices.split() for x in ("--device", d)]
        gpu_args = DOCKER_GPU_ARGS
    vis = _visible_devices_var()
    server = (["docker", "run", "-d", "--name", SERVER_NAME] + gpu_args + devices +
              ["--shm-size", "32g", "-e", f"{vis}={args.server_gpu}", "-e", "HF_HUB_OFFLINE=1",
               "-e", "VLLM_USE_V1=1", "-v", f"{args.model}:{args.model}:ro", "--entrypoint", "python", args.image]
              + server_command(args))
    env = ["-e", f"{vis}={args.train_gpu}", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "TOKENIZERS_PARALLELISM=false"]
    for k in ("GRPO_SHARED_DIR", "GRPO_JUDGE_URL", "GRPO_JUDGE_MODEL", "GRPO_JUDGE_RUBRIC", "RF_MLFLOW_PORT"):
        if os.environ.get(k):
            env += ["-e", f"{k}={os.environ[k]}"]
    train_name = "grpo-rapidfire-train-" + os.path.basename(os.path.abspath(args.output_dir).rstrip("/"))
    # --ulimit nofile: torch's file_descriptor sharing needs one fd per tensor of the shared-memory base-model copy.
    trainer = (["docker", "run", "--rm", "--name", train_name] + gpu_args + devices +
               ["--shm-size", args.shm_size, "--ulimit", "nofile=1048576:1048576"] + env + _bind_mounts(args) +
               ["-w", HERE, "--entrypoint", "python", args.image, LAUNCHER]
               + worker_argv())
    return server, trainer, train_name


def run_docker(args, grpo_config_kwargs):
    server_cmd, trainer_cmd, train_name = docker_commands(args)
    if args.dry_run:
        print("[dry-run] server container:\n  " + shlex.join(server_cmd))
        print("[dry-run] trainer container:\n  " + shlex.join(trainer_cmd))
        print("[dry-run] RFGRPOConfig kwargs:\n" + json.dumps(grpo_config_kwargs(args), indent=1, default=str))
        return
    os.makedirs(args.output_dir, exist_ok=True)
    started = False
    try:
        if not args.verify_only:
            subprocess.run(["docker", "rm", "-f", SERVER_NAME], capture_output=True)
            subprocess.run(server_cmd, check=True, stdout=subprocess.DEVNULL)
            started = True
            print(f"[server] container {SERVER_NAME} started on GPU index {args.server_gpu}; waiting for health", flush=True)
            wait_for_server(args, lambda: subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", SERVER_NAME], capture_output=True, text=True).stdout.strip() == "true")
        print("[trainer] " + shlex.join(trainer_cmd), flush=True)
        subprocess.run(trainer_cmd, check=True)
    finally:
        if started:
            with open(os.path.join(args.output_dir, "vllm_server.log"), "wb") as f:
                subprocess.run(["docker", "logs", SERVER_NAME], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(["docker", "rm", "-f", SERVER_NAME, train_name], capture_output=True)
        print("[server] containers removed", flush=True)


def run_local(args, train, grpo_config_kwargs):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dry_run:
        if args.server_mode == "process":
            print("[dry-run] server process: %s=%s python %s" % (_visible_devices_var(), args.server_gpu, shlex.join(server_command(args))))
        print("[dry-run] RFGRPOConfig kwargs:\n" + json.dumps(grpo_config_kwargs(args), indent=1, default=str))
        return
    server = None
    try:
        if args.server_mode == "process" and not args.verify_only:
            env = dict(os.environ, VLLM_USE_V1="1"); env[_visible_devices_var()] = args.server_gpu
            log = open(os.path.join(args.output_dir, "vllm_server.log"), "wb")
            server = subprocess.Popen([sys.executable] + server_command(args), env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[server] vllm-serve pid {server.pid} on GPU index {args.server_gpu}; waiting for health", flush=True)
            wait_for_server(args, lambda: server.poll() is None)
        if args.server_mode == "process":
            os.environ[_visible_devices_var()] = args.train_gpu
        train(args)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
            print("[server] vllm-serve stopped", flush=True)
