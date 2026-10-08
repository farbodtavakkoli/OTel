import argparse
import ast
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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def common_dir() -> str:
    for p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common")):
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    return os.path.abspath(os.path.join(HERE, "..", "..", "grpo_common"))


if common_dir() not in sys.path:
    sys.path.insert(0, common_dir())


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


def write_versions(path: str) -> str:
    code = ("import torch, vllm, openrlhf, deepspeed, transformers, peft, ray; from importlib.metadata import version; "
            "print('versions torch', torch.__version__, 'vllm', vllm.__version__, 'openrlhf', version('openrlhf'), "
            "'deepspeed', deepspeed.__version__, 'transformers', transformers.__version__, "
            "'peft', peft.__version__, 'ray', ray.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or ["versions unknown"])[-1]
    with open(path, "w") as f:
        f.write(line + "\n")
    return line


def jsonl_to_openrlhf(src: str, dst: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    n = 0
    with open(src, encoding="utf-8") as fh, open(dst, "w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "prompt" in rec and "label" in rec:
                row = rec
            else:
                if "messages" not in rec or "ground_truth" not in rec:
                    raise ValueError(f"{src}:{n + 1}: not the GRPO contract (keys {sorted(rec)}); need `messages` "
                                     f"+ `ground_truth` — see the Data section of README.md")
                messages, gt = rec["messages"], rec["ground_truth"]
                gt_d = json.loads(gt) if isinstance(gt, str) else dict(gt)
                if not messages or not gt_d.get("reference"):
                    raise ValueError(f"{src}:{n + 1}: need non-empty `messages` and `ground_truth.reference`")
                row = {"prompt": messages, "label": gt if isinstance(gt, str) else json.dumps(gt_d, ensure_ascii=False)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def apply_overrides(cmd: list, overrides: list) -> list:
    cmd = list(cmd)

    def span(key):
        if key not in cmd:
            return None
        i = cmd.index(key)
        j = i + 1
        while j < len(cmd) and not cmd[j].startswith("--"):
            j += 1
        return i, j

    for ov in overrides:
        toks = shlex.split(ov)
        if not toks:
            raise ValueError(f"empty override: {ov!r}")
        remove = toks[0].startswith("~")
        key = "--" + toks[0].lstrip("~").lstrip("-")
        s = span(key)
        if remove:
            if s:
                del cmd[s[0]:s[1]]
        elif s:
            cmd[s[0]:s[1]] = [key] + toks[1:]
        else:
            cmd += [key] + toks[1:]
    return cmd


EOS_TOKENS = ("<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>")
MODEL_TURN_MARKERS = ("<|turn>model\n", "<start_of_turn>model\n", "<|im_start|>assistant\n",
                      "<|start_header_id|>assistant<|end_header_id|>\n\n")
_TEMPLATE_NOISE = re.compile(
    r"<bos>|<\|begin_of_text\|>|<\|turn>(user|system|model)\n|<turn\|>|<\|channel>thought\n<channel\|>|"
    r"<start_of_turn>(user|model)\n|<end_of_turn>|<\|start_header_id\|>(user|system|assistant)<\|end_header_id\|>|<\|eot_id\|>"
)


def split_completion(query: str, prompt: str) -> str:
    if prompt and query.startswith(prompt):
        return query[len(prompt):]
    for marker in MODEL_TURN_MARKERS:
        idx = query.rfind(marker)
        if idx >= 0:
            tail = query[idx + len(marker):]
            if tail.startswith("<|channel>thought\n<channel|>"):
                tail = tail[len("<|channel>thought\n<channel|>"):]
            return tail
    return query


def finish_and_clean(completion: str):
    c = completion.rstrip()
    for eos in EOS_TOKENS:
        if c.endswith(eos):
            return c[: -len(eos)].rstrip(), "stop"
    return c, "length"


def parse_label(label) -> dict:
    if isinstance(label, dict):
        return label
    if isinstance(label, str):
        s = label.strip()
        if s.startswith("{"):
            try:
                d = json.loads(s)
                if isinstance(d, dict) and "reference" in d:
                    return d
            except json.JSONDecodeError:
                pass
        return {"reference": label}
    return {"reference": str(label)}


class RewardServer:

    def __init__(self, port: int, log_path: str, concurrency: int = 64, host: str = "0.0.0.0"):
        self.port, self.host, self.log_path = port, host, log_path
        self.sem = threading.BoundedSemaphore(concurrency)
        self.lock = threading.Lock()
        self.n = self.judge_none = 0
        self.log = self.srv = self.thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/get_reward"

    def score_one(self, query, prompt, label):
        import judge_reward
        gt = parse_label(label)
        completion, finish = finish_and_clean(split_completion(query, prompt))
        question = gt.get("question") or _TEMPLATE_NOISE.sub("", prompt or "").strip()
        source = gt.get("source")
        t0 = time.time()
        with self.sem:
            out = judge_reward.score_batch([question], [completion], [gt.get("reference", "")], [source], [finish])
        t1 = time.time()
        rec = {"t_start": t0, "t_end": t1, "wait_s": round(t1 - t0, 4), "judge_s": round(out["timing_s"], 4),
               "id": gt.get("id"), "source": source, "finish": finish, "reward": out["reward"][0],
               "judge": out["judge"][0], "completion_chars": len(completion)}
        failed = out["judge"][0] is None and source != "3gpp_tsg"
        with self.lock:
            self.n += 1
            self.judge_none += int(failed)
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        return out["reward"][0], {"judge_norm": (out["judge"][0] or 0) / 10.0,
                                  "truncated": 1.0 if finish == "length" else 0.0,
                                  "judge_failed": 1.0 if failed else 0.0}

    def start(self):
        import judge_reward
        server = self
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        self.log = open(self.log_path, "a", encoding="utf-8")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with server.lock:
                    self._send(200, {"requests": server.n, "judge_none": server.judge_none,
                                     "rubric": judge_reward.RUBRIC_VERSION})

            def do_POST(self):
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    queries = data.get("query") or data.get("queries") or []
                    prompts = data.get("prompts") or [""] * len(queries)
                    labels = data.get("labels") or [""] * len(queries)
                    rewards, extra = [], {}
                    for q, p, l in zip(queries, prompts, labels):
                        r, e = server.score_one(q, p, l)
                        rewards.append(r)
                        for k, v in e.items():
                            extra.setdefault(k, []).append(v)
                    extra = {k: sum(v) / len(v) for k, v in extra.items()}  # OpenRLHF reads scalars per request
                    self._send(200, {"rewards": rewards, "scores": rewards, "extra_logs": extra})
                except Exception as e:  # report, never hang the trainer
                    self._send(500, {"error": repr(e)})

        # A fast policy finishes all samples in a burst; the default listen backlog of 5 drops handshakes.
        ThreadingHTTPServer.request_queue_size = 1024
        self.srv = ThreadingHTTPServer((self.host, self.port), Handler)
        self.srv.daemon_threads = True
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True, name="reward-server")
        self.thread.start()
        self.wait_ready()
        print(f"[reward] serving {self.url} rubric={judge_reward.RUBRIC_VERSION} judge={judge_reward.JUDGE_URL} "
              f"log={self.log_path}", flush=True)

    def wait_ready(self, timeout: float = 30.0):
        import urllib.request
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    json.loads(r.read())
                    return
            except Exception:
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"reward server on {self.url} did not answer within {timeout}s")
                time.sleep(0.2)

    def stop(self):
        if self.srv is not None:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None
        if self.log is not None:
            self.log.close()
            self.log = None
        print(f"[reward] stopped after {self.n} requests ({self.judge_none} judge failures)", flush=True)


# Ray prefixes lines and tqdm may glue "\r..." progress output after the dict, so match the dict
# up to its closing brace only.
STEP_RE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) ppo_trainer.py:\d+\] .*?Global step (\d+): (\{[^{}]*\})")
METRIC_KEYS = ("rollout_s", "judge_s", "make_experience_s", "train_step_s", "sync_s", "cycle_s", "tokens_per_s")


def parse_steps(train_log: str) -> list:
    steps = []
    with open(train_log, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw
            for chunk in raw.split("\r"):
                if "Global step" in chunk:
                    line = chunk
                    break
            m = STEP_RE.search(line)
            if not m:
                continue
            d = ast.literal_eval(m.group(3))
            d["_step"], d["_logged_at"] = int(m.group(2)), m.group(1)
            steps.append(d)
    return steps


def _log_ts_to_epoch(ts: str, year: int) -> float:
    return time.mktime(time.strptime(f"{year}-{ts}", "%Y-%m-%d %H:%M:%S"))


def step_windows(steps: list, year: int, t_run_start: float) -> list:
    wins, prev_end = [], t_run_start
    for s in steps:
        end = _log_ts_to_epoch(s["_logged_at"], year)
        gen_end = end - (s.get("timing/make_experience", 0) + s.get("timing/ppo_train", 0) + s.get("timing/broadcast", 0))
        gen_start = gen_end - s.get("timing/generation", 0)
        wins.append((max(prev_end, gen_start - 2.0), gen_end + 2.0))
        prev_end = end
    return wins


def union_busy(intervals: list) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for a, b in intervals[1:]:
        if a > ce:
            total += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return total + (ce - cs)


def judge_per_step(reward_log: str, wins: list) -> list:
    recs = []
    if reward_log and os.path.exists(reward_log):
        with open(reward_log, encoding="utf-8") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    out = []
    for a, b in wins:
        inside = [r for r in recs if a <= r["t_start"] <= b]
        judged = [r for r in inside if r.get("source") != "3gpp_tsg"]
        out.append({
            "n_requests": len(inside), "n_judged": len(judged),
            "judge_busy_s": round(union_busy([(r["t_start"], r["t_end"]) for r in judged]), 3) if judged else 0.0,
            "judge_span_s": round(max(r["t_end"] for r in judged) - min(r["t_start"] for r in judged), 3) if judged else 0.0,
            "judge_sum_latency_s": round(sum(r["judge_s"] for r in judged), 3),
            "judge_none": sum(1 for r in judged if r.get("judge") is None),
            "truncated": sum(1 for r in inside if r.get("finish") == "length"),
        })
    return out


def vram_peaks(vram_log: str, t0: float, t1: float):
    if not vram_log or not os.path.exists(vram_log):
        return None
    peaks = [0, 0]
    with open(vram_log) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and t0 <= float(p[0]) <= t1:
                peaks[0], peaks[1] = max(peaks[0], int(p[1])), max(peaks[1], int(p[2]))
    return [round(v / 2**30, 1) for v in peaks]


def per_step_table(run_dir: str, reward_log: str = None, vram_log: str = None, warmup: int = 2,
                   measured: int = 20, year: int = None) -> dict:
    steps = parse_steps(os.path.join(run_dir, "train.log"))
    t_start = float(open(os.path.join(run_dir, "t_start")).read().strip())
    t_end_path = os.path.join(run_dir, "t_end")
    t_end = float(open(t_end_path).read().strip()) if os.path.exists(t_end_path) else time.time()
    year = year or time.localtime(t_start).tm_year
    wins = step_windows(steps, year, t_start)
    judge = judge_per_step(reward_log, wins)
    rows = []
    for s, j, w in zip(steps, judge, wins):
        n, gen, rl = s.get("rollout/num_samples"), s.get("timing/generation"), s.get("response_length")
        rows.append({
            "step": s["_step"],
            "rollout_s": round(gen, 2) if gen is not None else None,
            "judge_s": j["judge_busy_s"] if j["n_judged"] else None,
            "judge_span_s": j["judge_span_s"],
            "judge_requests": j["n_requests"],
            "judge_failed": j["judge_none"],
            "make_experience_s": round(s.get("timing/make_experience", 0), 2),
            "train_step_s": round(s.get("timing/ppo_train", 0), 2),
            "sync_s": round(s.get("timing/broadcast", 0), 2),
            "cycle_s": round(s.get("timing/step_total", 0), 2),
            "tokens_per_s": round(rl * n / gen, 1) if (gen and rl is not None and n) else None,
            "response_length": rl,
            "num_samples": n,
            "reward": s.get("reward"),
            "judge_norm": s.get("judge_norm"),
            "truncated_rate": s.get("rollout/truncated_rate", s.get("truncated")),
            "group_reward_std": s.get("group_reward_std"),
            "policy_loss": s.get("policy_loss"),
            "actor_grad_norm": s.get("actor_grad_norm"),
            "ppo_clip_ratio": s.get("ppo_clip_ratio"),
            "gen_window": [round(w[0] - t_start, 1), round(w[1] - t_start, 1)],
        })
    meas = rows[warmup:warmup + measured]
    summary = {
        "steps_total": len(rows),
        "steps_measured": len(meas),
        "medians": {k: (statistics.median([r[k] for r in meas if r[k] is not None]) if any(r[k] is not None for r in meas) else None)
                    for k in METRIC_KEYS},
        "reward_first": meas[0]["reward"] if meas else None,
        "reward_last": meas[-1]["reward"] if meas else None,
        "vram_peak_gb_vllm_actor": vram_peaks(vram_log, t_start, t_end),
        "wall_s_total": round(t_end - t_start, 1),
    }
    return {"summary": summary, "steps": rows}


def latest_adapter(run_dir: str) -> str:
    final = os.path.join(run_dir, "lora_adapter")
    if os.path.exists(os.path.join(final, "adapter_model.safetensors")):
        return final
    steps = sorted(glob.glob(os.path.join(run_dir, "ds_ckpt", "global_step*_hf")),
                   key=lambda p: int(re.search(r"global_step(\d+)_hf", p).group(1)))
    if steps:
        return steps[-1]
    legacy = os.path.join(run_dir, "hf_ckpt")
    if os.path.exists(os.path.join(legacy, "adapter_model.safetensors")):
        return legacy
    raise FileNotFoundError(f"no PEFT adapter (lora_adapter/, ds_ckpt/global_step*_hf, hf_ckpt/) under {run_dir}")


def export_lora_adapter(src: str, out: str) -> dict:
    from safetensors.torch import load_file

    src, out = os.path.abspath(src), os.path.abspath(out)
    if src != out:
        if os.path.isdir(out):
            shutil.rmtree(out)
        shutil.copytree(src, out)
    sd = load_file(os.path.join(out, "adapter_model.safetensors"))
    modules = {re.sub(r"\.lora_[AB]\.weight$", "", k) for k in sd}
    nonzero_b = sum(1 for k, v in sd.items() if "lora_B" in k and float(v.abs().max()) > 0)
    with open(os.path.join(out, "adapter_config.json")) as f:
        cfg = json.load(f)
    stats = {"tensors": len(sd), "modules": len(modules), "lora_B_nonzero": nonzero_b, "r": cfg.get("r"),
             "lora_alpha": cfg.get("lora_alpha"), "target_modules": cfg.get("target_modules"), "source": src}
    print(f"[export] {stats['tensors']} tensors, {stats['modules']} LoRA modules, lora_B non-zero "
          f"{nonzero_b}/{stats['modules']} -> {out}", flush=True)
    return stats


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
    p = argparse.ArgumentParser(description="GRPO LoRA training with OpenRLHF train_ppo_ray (group_norm)")
    p.add_argument("--model", required=True, help="HF checkpoint directory")
    p.add_argument("--data", required=True, help="train file: JSONL (`messages` + `ground_truth`) or a ready OpenRLHF prompt/label JSONL")
    p.add_argument("--output-dir", required=True, help="checkpoints, logs, data shim and the adapter go here")
    p.add_argument("--steps", type=int, default=22, help="GRPO optimizer steps (= prompts consumed / --prompt-batch-size)")
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step (rollout.batch_size)")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (rollout.n_samples_per_prompt)")
    p.add_argument("--max-tokens", type=int, default=1024, help="max response tokens (rollout.max_new_tokens)")
    p.add_argument("--max-len", type=int, default=4096, help="prompt + response tokens (data.max_len)")
    p.add_argument("--save-freq", type=int, default=None, help="PEFT checkpoint every N steps (default: only at --steps; -1 never)")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--actor-gpus", type=int, default=1, help="DeepSpeed actor GPUs")
    p.add_argument("--vllm-engines", type=int, default=1, help="vLLM rollout engines (each on --vllm-tp GPUs)")
    p.add_argument("--vllm-tp", type=int, default=1, help="vLLM tensor parallel size")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true", help="vLLM without HIP/CUDA graphs")
    p.add_argument("--micro-train-bs", type=int, default=4, help="sequences per GPU per forward/backward in the actor update")
    p.add_argument("--micro-rollout-bs", type=int, default=8, help="sequences per forward in the old-log-prob pass")
    p.add_argument("--attn", default="sdpa", help="ds.attn_implementation")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-regex", default=None, help="PEFT target_modules regex (default: derived from the model class)")
    p.add_argument("--judge-rubric", default=None, help="GRPO_JUDGE_RUBRIC for the training reward (default: judge_reward's)")
    p.add_argument("--reward-port", type=int, default=8340, help="port of the in-process reward server")
    p.add_argument("--reward-concurrency", type=int, default=64, help="max concurrent judge calls")
    p.add_argument("--ray-port", type=int, default=6390, help="Ray head port; ignored when RAY_ADDRESS is set")
    p.add_argument("--override", action="append", default=[], help="OpenRLHF option `key value` / `flag` (replaces or appends; `~key` removes), repeatable")
    p.add_argument("--no-export", action="store_true", help="skip the adapter report/verify after training")
    p.add_argument("--export-only", action="store_true", help="skip training; report (and --verify) the adapter already in --output-dir")
    p.add_argument("--verify", action="store_true", help="after training, load base+adapter on one GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the OpenRLHF command and environment, then exit")
    return p.parse_args()


def data_file(path: str, run_dir: str, dry_run: bool) -> str:
    out = os.path.join(run_dir, "data", "train.jsonl")
    if not dry_run:
        n = jsonl_to_openrlhf(os.path.abspath(path), out)
        print(f"[data] {n} rows {path} -> {out}", flush=True)
    return out


def subprocess_env(args) -> dict:
    env = dict(os.environ)
    env["RAY_ADDRESS"] = env.get("RAY_ADDRESS") or f"127.0.0.1:{args.ray_port}"
    env["HF_HUB_OFFLINE"] = "1"
    env["VLLM_USE_V1"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # DeepSpeed JIT-builds fused_adam by shelling out to `ninja`, and vLLM's FlashInfer backend does
    # the same for its kernels. Both need the binary on PATH, which on the venv route it is not.
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


# A console script next to the running interpreter. On the venv route it is not on PATH, and a
# bare name either fails with FileNotFoundError or silently picks up another environment's.
def venv_bin(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else name


def start_ray(port: int, n_gpus: int) -> bool:
    if os.environ.get("RAY_ADDRESS"):
        return False
    subprocess.run([venv_bin("ray"), "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Ray sizes its object store from /dev/shm. Where the shared dir is a large tmpfs, the default
    # claim collides with the weights living there and the raylet is OOM-killed mid-run;
    # GRPO_RAY_OBJECT_STORE_GB caps it (default 32 GB, ample for rollout batches).
    store_gb = float(os.environ.get("GRPO_RAY_OBJECT_STORE_GB", "32"))
    # Ray actors inherit the head's environment, not the launcher's. DeepSpeed JIT-builds fused_adam
    # inside an actor by shelling out to `ninja`, so the venv's bin/ must be on the head's PATH.
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    subprocess.run([venv_bin("ray"), "start", "--head", f"--port={port}", f"--num-gpus={n_gpus}", "--include-dashboard=false",
                    f"--object-store-memory={int(store_gb * 1024**3)}",
                    "--disable-usage-stats", "--temp-dir", "/tmp/ray_openrlhf"], check=True,
                   env=env, stdout=subprocess.DEVNULL)
    return True


def stop_ray():
    subprocess.run([venv_bin("ray"), "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_openrlhf(cmd: list, env: dict, run_dir: str) -> int:
    log_path = os.path.join(run_dir, "train.log")
    with open(os.path.join(run_dir, "t_start"), "w") as f:
        f.write(f"{int(time.time())}\n")
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
        log.write(f"EXIT_CODE={rc}\n")
    with open(os.path.join(run_dir, "t_end"), "w") as f:
        f.write(f"{int(time.time())}\n")
    return rc


def print_dry_run(cmd: list, env: dict, run_dir: str, ray_port: int, n_gpus: int, reward) -> None:
    keys = ("RAY_ADDRESS", "HF_HUB_OFFLINE", "VLLM_USE_V1", "GRPO_JUDGE_RUBRIC")
    print("env:", " ".join(f"{k}={shlex.quote(env[k])}" for k in keys if k in env))
    print(f"ray: {'RAY_ADDRESS set, reuse' if os.environ.get('RAY_ADDRESS') else f'ray start --head --port={ray_port} --num-gpus={n_gpus}'}")
    print(f"reward server: {reward.url} -> {reward.log_path}")
    print("cwd:", run_dir)
    print(shlex.join(cmd))


def record_command(cmd: list, run_dir: str):
    with open(os.path.join(run_dir, "command.txt"), "a") as f:
        f.write(shlex.join(cmd) + "\n")
