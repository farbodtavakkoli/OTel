import argparse
import collections
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def common_dir() -> str:
    for p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common"), "/workspace/grpo"):
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
    code = ("import torch, vllm, verl, transformers, peft, ray; print('versions torch', torch.__version__, "
            "'vllm', vllm.__version__, 'verl', verl.__version__, 'transformers', transformers.__version__, "
            "'peft', peft.__version__, 'ray', ray.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or ["versions unknown"])[-1]
    with open(path, "w") as f:
        f.write(line + "\n")
    return line


def jsonl_to_parquet(src: str, dst: str, split: str = "train") -> int:
    import pandas as pd

    rows = []
    with open(src, encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]
            gt = rec["ground_truth"]
            gt = json.loads(gt) if isinstance(gt, str) else dict(gt)
            reference = gt.get("reference", "")
            source = gt.get("source") or rec.get("source") or "unknown"
            if not messages or not reference:
                raise SystemExit(f"{src}: row {index}: empty messages or reference")
            rows.append({
                "data_source": source,
                "prompt": messages,
                "ability": "telecom",
                "reward_model": {"style": "judge", "ground_truth": reference},
                "extra_info": {"id": str(gt.get("id") or rec.get("id") or index), "source": source,
                               "index": index, "split": split, "question": gt.get("question")},
            })
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    return len(rows)


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


def _score_with_rubric(rubric, *args):
    import judge_reward

    saved = (judge_reward.RUBRIC, judge_reward.RUBRIC_VERSION)
    judge_reward.RUBRIC = rubric
    judge_reward.RUBRIC_VERSION = "telecom-judge-" + rubric
    try:
        return judge_reward.score_batch(*args)
    finally:
        judge_reward.RUBRIC, judge_reward.RUBRIC_VERSION = saved


try:
    import numpy as np
    import torch
    from verl import DataProto
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.ray_utils import auto_await

    class BatchedJudgeAgentLoopManager(AgentLoopManager):

        def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
            super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles=None)
            import judge_reward

            self.tokenizer = hf_tokenizer(copy_to_local(config.actor_rollout_ref.model.path), trust_remote_code=True)
            self.response_length = int(config.actor_rollout_ref.rollout.response_length)
            self.eval_rubric = os.environ.get("GRPO_EVAL_RUBRIC", "v1")
            run_dir = os.environ.get("GRPO_RUN_DIR", os.getcwd())
            os.makedirs(run_dir, exist_ok=True)
            self.step_log_path = os.path.join(run_dir, "judge_steps.jsonl")
            self._step = 0
            print(f"[BatchedJudgeAgentLoopManager] rubric={judge_reward.RUBRIC_VERSION} eval_rubric={self.eval_rubric} "
                  f"step_log={self.step_log_path}", flush=True)

        @auto_await
        async def generate_sequences(self, prompts: DataProto) -> DataProto:
            import judge_reward

            t0 = time.time()
            output = await super().generate_sequences(prompts)
            rollout_s = time.time() - t0

            validate = bool(prompts.meta_info.get("validate", False))
            batch = output.batch
            prompt_len = batch["prompts"].size(1)
            resp_lens = batch["attention_mask"][:, prompt_len:].sum(dim=1)
            responses = batch["responses"]
            n = len(output)

            completions = [self.tokenizer.decode(responses[i][: int(resp_lens[i])], skip_special_tokens=True) for i in range(n)]
            finish_reasons = ["length" if int(resp_lens[i]) >= self.response_length else "stop" for i in range(n)]
            nt = output.non_tensor_batch
            references = [rm["ground_truth"] for rm in nt["reward_model"]]
            sources = [str(s) for s in nt["data_source"]]
            extra = [dict(e or {}) for e in nt["extra_info"]] if "extra_info" in nt else [{} for _ in range(n)]
            # `raw_prompt` is present only because the launcher sets data.return_raw_chat=True
            if "raw_prompt" in nt:
                fallback = [list(p) for p in nt["raw_prompt"]]
            else:
                fallback = [self.tokenizer.decode(batch["prompts"][i], skip_special_tokens=True) for i in range(n)]
            questions = [extra[i].get("question") or fallback[i] for i in range(n)]

            t1 = time.time()
            if validate:
                scored = _score_with_rubric(self.eval_rubric, questions, completions, references, sources, finish_reasons)
            else:
                scored = judge_reward.score_batch(questions, completions, references, sources, finish_reasons)
            judge_wall_s = time.time() - t1
            rewards = scored["reward"]
            judge_raw = scored["judge"]

            rm_scores = torch.zeros_like(batch["response_mask"], dtype=torch.float32)
            last = (resp_lens - 1).clamp(min=0)
            rm_scores[torch.arange(n), last] = torch.tensor(rewards, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

            nt["acc"] = np.array(rewards, dtype=np.float32)
            nt["judge_score"] = np.array([(-1 if j is None else j) for j in judge_raw], dtype=np.float32)
            nt["truncated"] = np.array([1.0 if f == "length" else 0.0 for f in finish_reasons], dtype=np.float32)
            output.meta_info["reward_extra_keys"] = ["acc", "judge_score", "truncated"]

            timing = output.meta_info.setdefault("timing", {})
            timing["rollout_only"] = rollout_s
            timing["judge"] = float(scored["timing_s"])
            timing["judge_wall"] = judge_wall_s

            run_dir = os.path.dirname(self.step_log_path)
            if validate:  # labelled by the trainer's global step; does not advance _step
                step_label = int(prompts.meta_info.get("global_steps", -1))
                log_path = os.path.join(run_dir, "eval_steps.jsonl")
                dump_dir = os.path.join(run_dir, "eval")
            else:
                self._step += 1
                step_label = self._step
                log_path = self.step_log_path
                dump_dir = os.path.join(run_dir, "rollouts")
            num_resp_tokens = int(resp_lens.sum())
            rec = {
                "step": step_label,
                "global_step": int(prompts.meta_info.get("global_steps", -1)),
                "validate": validate,
                "n": n,
                "rollout_only_s": rollout_s,
                "judge_s": float(scored["timing_s"]),
                "judge_wall_s": judge_wall_s,
                "response_tokens": num_resp_tokens,
                "tokens_per_s": num_resp_tokens / rollout_s if rollout_s > 0 else None,
                "reward_mean": float(np.mean(rewards)),
                "judge_none": int(sum(1 for j in judge_raw if j is None)),
                "truncated_frac": float(np.mean(nt["truncated"])),
                "rubric": scored.get("rubric"),
                "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if validate:
                per_src = {}
                for s, r in zip(sources, rewards):
                    per_src.setdefault(s, []).append(float(r))
                rec["per_source"] = {s: float(np.mean(v)) for s, v in sorted(per_src.items())}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, f"step_{step_label:04d}.jsonl"), "w") as fh:
                for i in range(n):
                    fh.write(json.dumps({
                        "id": extra[i].get("id"), "source": sources[i],
                        "question": questions[i] if isinstance(questions[i], str) else None,
                        "prompt": questions[i] if not isinstance(questions[i], str) else None,
                        "completion": completions[i], "reference": references[i],
                        "finish_reason": finish_reasons[i], "response_tokens": int(resp_lens[i]),
                        "reward": float(rewards[i]), "judge": judge_raw[i],
                    }, default=str) + "\n")
            print(f"[BatchedJudgeAgentLoopManager] {json.dumps(rec)}", flush=True)
            return output
except ImportError:  # verl not importable outside the training container
    BatchedJudgeAgentLoopManager = None


def latest_actor_checkpoint(run_dir: str) -> str:
    steps = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "global_step_*")),
                   key=lambda p: int(re.search(r"global_step_(\d+)", p).group(1)))
    if not steps:
        raise FileNotFoundError(f"no checkpoints/global_step_* under {run_dir}")
    actor = os.path.join(steps[-1], "actor")
    return actor if os.path.isdir(actor) else steps[-1]


def export_lora_adapter(ckpt: str, out: str, base: str, r: int, alpha: int, target_regex: str) -> dict:
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file

    meta_path = os.path.join(ckpt, "lora_train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        r, alpha = int(meta.get("r", r)), int(meta.get("lora_alpha", alpha))

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
    nonzero_b = sum(1 for k, v in out_sd.items() if "lora_B" in k and float(v.abs().max()) > 0)
    print(f"[export] {len(out_sd)} tensors, {n_modules} LoRA modules, lora_B non-zero {nonzero_b}/{n_modules} -> {out}", flush=True)
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GRPO LoRA training with verl (colocated FSDP2 + vLLM)")
    p.add_argument("--model", required=True, help="HF checkpoint directory")
    p.add_argument("--data", required=True, help="train file: JSONL (`messages` + `ground_truth`) or a ready verl parquet")
    p.add_argument("--val-data", default=None, help="validation file (JSONL or parquet); default: the train file")
    p.add_argument("--output-dir", required=True, help="checkpoints, logs, data shim and the exported adapter go here")
    p.add_argument("--steps", type=int, default=22)
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step (data.train_batch_size)")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (rollout.n)")
    p.add_argument("--max-prompt-length", type=int, default=3072)
    p.add_argument("--max-tokens", type=int, default=1024, help="max response tokens")
    p.add_argument("--save-freq", type=int, default=None, help="checkpoint every N steps (default: only at --steps; -1 never)")
    p.add_argument("--test-freq", type=int, default=-1, help="validate every N steps (-1 never)")
    p.add_argument("--val-before-train", action="store_true")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-gpus", type=int, default=2)
    p.add_argument("--rollout-tp", type=int, default=1, help="vLLM tensor parallel size (n-gpus / tp replicas)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--ppo-micro-bs", type=int, default=2, help="sequences per GPU per forward/backward in the actor update")
    p.add_argument("--logprob-micro-bs", type=int, default=4, help="sequences per GPU per forward in the old-log-prob pass")
    p.add_argument("--weight-bucket-mb", type=int, default=8192, help="rollout.checkpoint_engine.update_weights_bucket_megabytes")
    p.add_argument("--remove-padding", action="store_true", help="packed sequences + flash_attention_2 instead of padded + sdpa")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-regex", default=None, help="verl target_modules regex (default: derived from the model class)")
    p.add_argument("--lora-adapter-path", default=None, help="start from this PEFT adapter (actor_rollout_ref.model.lora_adapter_path)")
    p.add_argument("--judge-rubric", default=None, help="GRPO_JUDGE_RUBRIC for the training reward (default: judge_reward's)")
    p.add_argument("--eval-rubric", default=None, help="GRPO_EVAL_RUBRIC for validation batches (default v1)")
    p.add_argument("--override", action="append", default=[], help="verl override key=value, replaces or appends (repeatable)")
    p.add_argument("--no-export", action="store_true", help="skip exporting the final LoRA adapter")
    p.add_argument("--export-only", action="store_true", help="skip training; export (and --verify) from the checkpoints already in --output-dir")
    p.add_argument("--verify", action="store_true", help="after export, load base+adapter on one GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the verl command and environment, then exit")
    return p.parse_args(argv)


def data_file(path: str, run_dir: str, split: str, dry_run: bool) -> str:
    path = os.path.abspath(path)
    if path.endswith(".parquet"):
        return path
    out = os.path.join(run_dir, "data", f"{split}.parquet")
    if not dry_run:
        n = jsonl_to_parquet(path, out, split)
        print(f"[data] {n} rows {path} -> {out}", flush=True)
    return out


def subprocess_env(args, run_dir: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(p for p in (HERE, common_dir(), os.environ.get("PYTHONPATH")) if p)
    env["GRPO_RUN_DIR"] = run_dir
    env["GRPO_COMMON_DIR"] = common_dir()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # Ray blanks the accelerator env var for num_gpus=0 actors
    if args.judge_rubric:
        env["GRPO_JUDGE_RUBRIC"] = args.judge_rubric
    if args.eval_rubric:
        env["GRPO_EVAL_RUBRIC"] = args.eval_rubric
    return env


def run_verl(cmd: list, env: dict, run_dir: str) -> int:
    log_path = os.path.join(run_dir, "train.log")
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
        log.write(f"=== TRAINING EXIT rc={rc}\n")
    return rc


def print_dry_run(cmd: list, env: dict, run_dir: str):
    keys = ("PYTHONPATH", "GRPO_RUN_DIR", "GRPO_COMMON_DIR", "GRPO_JUDGE_RUBRIC", "GRPO_EVAL_RUBRIC")
    print("env:", " ".join(f"{k}={shlex.quote(env[k])}" for k in keys if k in env))
    print("cwd:", run_dir)
    print(shlex.join(cmd))


def record_command(cmd: list, run_dir: str):
    with open(os.path.join(run_dir, "command.txt"), "a") as f:
        f.write(f"=== {os.path.basename(run_dir)} GRPO_RUN_DIR={run_dir}\n{shlex.join(cmd)}\n")


# SFT warm-up dataset. verl's MultiTurnSFTDataset must not be substituted here; see README.md.
# Selected by data.custom_cls.path=<this file> data.custom_cls.name=GrpoSFTDataset. pad_mode must
# be `no_padding`: verl's engine only supports nested tensors on this path. Config keys under
# `data.`: max_length, truncation (error|right), prompt_key, and an optional `sources` list.

END_OF_TURN = "<turn|>"


try:
    from torch.utils.data import Dataset as _Dataset
except Exception:  # torch absent: the GRPO path does not need it
    _Dataset = object


class GrpoSFTDataset(_Dataset):

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        import pandas as pd
        import torch
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.dataset_utils import DatasetPadMode
        from verl.utils.dataset.multiturn_sft_dataset import convert_nested_value_to_list_recursive
        from verl.utils.fs import copy_local_path_from_hdfs

        config = config or {}
        pad_mode = config.get("pad_mode", "no_padding")
        assert pad_mode == DatasetPadMode.NO_PADDING, f"GrpoSFTDataset supports pad_mode=no_padding only, got {pad_mode}"
        self.max_length = int(config.get("max_length", 4096))
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ("error", "right")
        self.prompt_key = config.get("prompt_key", "prompt")
        self.apply_chat_template_kwargs = dict(config.get("apply_chat_template_kwargs", {}) or {})
        sources = config.get("sources", None)
        self.sources = set(str(s) for s in sources) if sources else None

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
        assert isinstance(self.eot_id, int) and self.eot_id >= 0, f"{END_OF_TURN} not in vocab"

        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(copy_local_path_from_hdfs(p, verbose=True)) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        if self.sources is not None:
            df = df[df["data_source"].astype(str).isin(self.sources)].reset_index(drop=True)
        if max_samples is not None and max_samples > 0 and max_samples < len(df):
            df = df.iloc[:max_samples].reset_index(drop=True)
        self.df = df

        self.rows = []
        n_trunc = 0
        for _, rec in df.iterrows():
            messages = convert_nested_value_to_list_recursive(rec[self.prompt_key])
            reference = rec["reward_model"]["ground_truth"]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
            )
            if hasattr(prompt_ids, "input_ids"):  # BatchEncoding in some transformers versions
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = list(prompt_ids)
            resp_ids = list(tokenizer(reference, add_special_tokens=False)["input_ids"]) + [self.eot_id]
            ids = prompt_ids + resp_ids
            mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
            if len(ids) > self.max_length:
                if self.truncation == "error":
                    raise ValueError(f"row {rec.get('extra_info', {})}: {len(ids)} tokens > max_length {self.max_length}")
                ids, mask = ids[: self.max_length], mask[: self.max_length]
                n_trunc += 1
            self.rows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)))

        lens = [len(r[0]) for r in self.rows]
        loss_toks = [int(r[1].sum()) for r in self.rows]
        by_src = collections.Counter(str(s) for s in df["data_source"]) if "data_source" in df else {}
        print(
            f"[GrpoSFTDataset] {len(self.rows)} rows from {parquet_files} sources={dict(by_src)} "
            f"seq_len min/mean/max={min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)} "
            f"loss_tokens total={sum(loss_toks)} mean={sum(loss_toks) / len(loss_toks):.0f} truncated={n_trunc}",
            flush=True,
        )
        ids0, mask0 = self.rows[0]
        print("[GrpoSFTDataset] example prompt tail:", repr(tokenizer.decode(ids0[: int((mask0 == 0).sum())][-12:])), flush=True)
        print("[GrpoSFTDataset] example loss span  :", repr(tokenizer.decode(ids0[mask0.bool()])[:200]), flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        import torch
        ids, mask = self.rows[item]
        return {
            "input_ids": ids,
            "position_ids": torch.arange(ids.shape[0], dtype=torch.long),
            "loss_mask": mask,
        }
