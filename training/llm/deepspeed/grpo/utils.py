import argparse
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.request

# torch, datasets and transformers are imported inside the functions that need them, never at module
# level: in --server-mode docker the host process only orchestrates containers, and that interpreter
# is a plain venv with no ML stack. A top-level import would make `import utils` fail before the
# launcher could start anything.


def _message_content_to_text(message):
    content = message.get("content", "") if isinstance(message, dict) else ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _validate_messages_example(example):
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return "missing_messages"

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"message_{index}_not_object"
        role = str(message.get("role", "") or "").strip()
        if not role:
            return f"message_{index}_missing_role"

    final_message = messages[-1]
    if str(final_message.get("role", "") or "").strip() != "assistant":
        return "final_message_not_assistant"
    if not _message_content_to_text(final_message).strip():
        return "empty_final_assistant"
    if not any(_message_content_to_text(message).strip() for message in messages[:-1]):
        return "missing_prompt_messages"

    return None


def _sample_indices(total_count, sample_size):
    if total_count <= 0 or sample_size <= 0:
        return []
    if total_count <= sample_size:
        return list(range(total_count))
    if sample_size == 1:
        # A single sample: take the first row (avoids /0 in the step formula).
        return [0]

    step = (total_count - 1) / float(sample_size - 1)
    return sorted({min(total_count - 1, int(round(i * step))) for i in range(sample_size)})


def _run_messages_preflight(dataset, sample_size):
    sample_indices = _sample_indices(len(dataset), sample_size)
    invalid_examples = []
    for dataset_index in sample_indices:
        reason = _validate_messages_example(dataset[dataset_index])
        if reason is not None:
            invalid_examples.append((dataset_index, reason))
            if len(invalid_examples) >= 5:
                break

    if invalid_examples:
        formatted = ", ".join(f"index={index}: {reason}" for index, reason in invalid_examples)
        raise ValueError(
            "Messages preflight validation failed. Clean the dataset upstream before training. "
            f"Sample failures: {formatted}"
        )

    logging.info(
        "Messages preflight validation passed on %d sampled rows.",
        len(sample_indices),
    )


def _resolve_eval_sample_count(total_count, test_size, max_eval_samples=None, min_eval_samples=1):
    if total_count < 2:
        raise ValueError(f"Need at least 2 rows to create a train/eval split; got {total_count}.")

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError(f"Float test_size must be between 0 and 1; got {test_size}.")
        num_test = max(1, math.ceil(total_count * test_size))
    else:
        num_test = int(test_size)

    if max_eval_samples is not None:
        num_test = min(num_test, int(max_eval_samples))

    num_test = max(int(min_eval_samples), num_test)
    if num_test >= total_count:
        if total_count - 1 < min_eval_samples:
            raise ValueError(
                f"Dataset too small for the requested eval split: total={total_count}, "
                f"required_min_eval={min_eval_samples}."
            )
        logging.warning(
            "Requested eval split of %d rows exceeds dataset size %d; capping eval rows to %d.",
            num_test,
            total_count,
            total_count - 1,
        )
        num_test = total_count - 1

    return num_test


def _log_supervision_sanity(dataset, dataset_name, tokenizer, sample_size, decode_example=False):
    sample_indices = _sample_indices(len(dataset), sample_size)
    if not sample_indices:
        raise ValueError(f"{dataset_name} dataset is empty after preprocessing.")

    total_tokens = []
    masked_tokens = []
    trained_tokens = []
    first_decoded_index = sample_indices[0]

    for dataset_index in sample_indices:
        sample = dataset[dataset_index]
        sample_labels = sample["labels"]
        sample_total = len(sample_labels)
        sample_masked = sum(1 for label in sample_labels if label == -100)
        sample_trained = sample_total - sample_masked
        if sample_trained <= 0:
            raise ValueError(
                f"{dataset_name} sample at index {dataset_index} has zero supervised tokens after masking."
            )
        total_tokens.append(sample_total)
        masked_tokens.append(sample_masked)
        trained_tokens.append(sample_trained)

    logging.info(
        "%s supervision over %d sampled rows: total_tokens min=%d mean=%.0f max=%d | "
        "masked min=%d mean=%.0f max=%d | trained min=%d mean=%.0f max=%d",
        dataset_name,
        len(sample_indices),
        min(total_tokens),
        sum(total_tokens) / len(total_tokens),
        max(total_tokens),
        min(masked_tokens),
        sum(masked_tokens) / len(masked_tokens),
        max(masked_tokens),
        min(trained_tokens),
        sum(trained_tokens) / len(trained_tokens),
        max(trained_tokens),
    )

    if decode_example:
        sample = dataset[first_decoded_index]
        sample_ids = sample["input_ids"]
        sample_labels = sample["labels"]
        num_masked = sum(1 for label in sample_labels if label == -100)
        trained_ids = [token for token, label in zip(sample_ids, sample_labels) if label != -100]
        logging.info(
            "%s decoded trained region (first 300 chars): %s",
            dataset_name,
            tokenizer.decode(trained_ids)[:300],
        )
        logging.info(
            "%s decoded trained region (last 80 chars): ...%s",
            dataset_name,
            tokenizer.decode(trained_ids)[-80:],
        )
        logging.info(
            "%s decoded masked prompt tail (last 120 chars): ...%s",
            dataset_name,
            tokenizer.decode(sample_ids[:num_masked])[-120:],
        )


def _process_with_chat_template(messages, tokenizer, mask_user_prompt=True):
    # add_special_tokens=False: the template already injects BOS etc.
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if full_text.startswith(prompt_text):
        # Prompt/completion boundary is exact by construction (no token merge)
        completion_text = full_text[len(prompt_text):]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        labels = ([-100] * len(prompt_ids) + list(completion_ids)) if mask_user_prompt else list(input_ids)
    else:
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        n = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            n += 1
        input_ids = full_ids
        labels = ([-100] * n + list(full_ids[n:])) if mask_user_prompt else list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_deepspeed_config(zero_stage=3, offload_optimizer=False):
    offload_device = "cpu" if offload_optimizer else "none"

    zero_opt = {
        "stage": zero_stage,
        "offload_optimizer": {"device": offload_device, "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }

    if zero_stage == 3:
        zero_opt.update({
            "offload_param": {"device": "none", "pin_memory": True},
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    return {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": zero_opt,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


def _pref_to_conversational(example, tokenizer):
    msgs = []
    sys = (example.get("system") or "").strip()
    if sys:
        msgs.append({"role": "system", "content": sys})
    msgs.append({"role": "user", "content": example.get("prompt", "")})
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": prompt_text,
        "chosen": (example.get("chosen") or "").strip(),
        "rejected": (example.get("rejected") or "").strip(),
    }


HERE = os.path.dirname(os.path.abspath(__file__))

# A console script next to the running interpreter. On the venv route it is not on PATH, and a
# bare name either fails with FileNotFoundError or silently picks up another environment's.
def venv_bin(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else name

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LORA_PROJECTIONS = r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# The launcher re-execs itself (in the trainer container, and under `accelerate launch`). Name it
# explicitly rather than via __file__, which from this module would resolve to utils.py.
LAUNCHER = os.path.join(HERE, "train_llm_grpo_deepspeed.py")

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


def patch_trl_vllm_client():
    import trl.generation.vllm_client as vc
    vc._HAS_WEIGHT_UPDATE_LIFECYCLE = True


def grpo_versions(with_deepspeed):
    import torch
    import transformers
    import peft
    import trl
    import vllm
    return {"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__,
            "peft": peft.__version__, "vllm": vllm.__version__,
            "deepspeed": __import__("deepspeed").__version__ if with_deepspeed else None,
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


def make_judge_reward(max_completion_length, eos_ids, timers):
    import judge_reward

    def judge_reward_fn(prompts, completions, completion_ids=None, ground_truth=None, **kwargs):
        t = time.perf_counter()
        gts = [parse_ground_truth(g) for g in (ground_truth or [{}] * len(completions))]
        questions = [g.get("question") or judge_reward.question_text(p) for g, p in zip(gts, prompts)]
        references = [g.get("reference", "") for g in gts]
        sources = [g.get("source") for g in gts]
        finish = []
        for ids in completion_ids or [[]] * len(completions):
            truncated = len(ids) >= max_completion_length and (not ids or ids[-1] not in eos_ids)
            finish.append("length" if truncated else "stop")
        out = judge_reward.score_batch(questions, completions, references, sources, finish)
        timers["judge_s"].append(out["timing_s"])
        timers["reward_s"].append(time.perf_counter() - t)
        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            log_metric("judge_s", out["timing_s"])
            log_metric("judge_failed", float(sum(1 for j, s in zip(out["judge"], sources) if j is None and s != "3gpp_tsg")))
            log_metric("finish_length_frac", sum(f == "length" for f in finish) / max(1, len(finish)))
            ids_list = completion_ids or []
            log_metric("last_tok_eos_frac", sum(1 for ids in ids_list if ids and ids[-1] in eos_ids) / max(1, len(ids_list)))
        return out["reward"]

    judge_reward_fn.__name__ = "judge_reward"
    return judge_reward_fn


def eos_id_set(tokenizer, model):
    ids = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
    cfg_ids = model.config.eos_token_id if isinstance(model.config.eos_token_id, list) else [model.config.eos_token_id]
    return {i for i in list(ids) + list(cfg_ids) if i is not None}


class StepTimers:

    def __init__(self, steps_path):
        self.steps_path = steps_path
        self.step_rec = {}
        self.timers = {"judge_s": [], "reward_s": []}
        self.records = []
        open(steps_path, "w").close()

    def callbacks(self):
        import torch
        from transformers import TrainerCallback

        outer = self

        class StepTimer(TrainerCallback):
            def on_step_begin(self, a, state, control, **kw):
                outer.step_rec.clear()
                outer.step_rec["t_begin"] = time.perf_counter()
                outer.timers["judge_s"].clear()
                outer.timers["reward_s"].clear()

            def on_step_end(self, a, state, control, **kw):
                sr, tm = outer.step_rec, outer.timers
                rec = {
                    "step": state.global_step,
                    "cycle_s": time.perf_counter() - sr["t_begin"],
                    "rollout_s": sr.get("rollout_s"),
                    "sync_s": sr.get("sync_s"),
                    "gen_score_s": sr.get("gen_score_s"),
                    "judge_s": sum(tm["judge_s"]) if tm["judge_s"] else None,
                    "reward_s": sum(tm["reward_s"]) if tm["reward_s"] else None,
                    "completion_tokens": sr.get("completion_tokens"),
                    "num_completions": sr.get("num_completions"),
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                }
                rec["train_step_s"] = rec["cycle_s"] - rec["gen_score_s"] if rec["gen_score_s"] is not None else None
                rec["tokens_per_s"] = (rec["completion_tokens"] / rec["rollout_s"]
                                       if rec["rollout_s"] and rec["completion_tokens"] else None)
                sr["pending"] = rec

        class LogAttach(TrainerCallback):
            # on_log fires after on_step_end for the same step; attach TRL's metrics, then flush.
            def on_log(self, a, state, control, logs=None, **kw):
                rec = outer.step_rec.get("pending")
                if rec is None:
                    return
                rec["trl"] = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
                outer.records.append(rec)
                with open(outer.steps_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[grpo] step {rec['step']}: cycle={rec['cycle_s']:.1f}s rollout={rec['rollout_s']} "
                      f"judge={rec['judge_s']} sync={rec['sync_s']} train={rec['train_step_s']} "
                      f"tok/s={rec['tokens_per_s']} reward={rec['trl'].get('reward')} "
                      f"mean_len={rec['trl'].get('completions/mean_length')} peak={rec['peak_mem_gb']:.1f}GB", flush=True)
                outer.step_rec["pending"] = None

        return [StepTimer(), LogAttach()]

    def install(self, trainer):
        sr = self.step_rec
        vg = trainer.vllm_generation
        orig_sync, orig_gen, orig_gas = vg.sync_weights, vg.generate, trainer._generate_and_score_completions

        def timed_sync(*a, **kw):
            t = time.perf_counter()
            r = orig_sync(*a, **kw)
            sr["sync_s"] = sr.get("sync_s", 0.0) + (time.perf_counter() - t)
            return r

        def timed_generate(*a, **kw):
            t = time.perf_counter()
            r = orig_gen(*a, **kw)
            sr["rollout_s"] = sr.get("rollout_s", 0.0) + (time.perf_counter() - t)
            completion_ids = r[1]
            sr["completion_tokens"] = sr.get("completion_tokens", 0) + sum(len(c) for c in completion_ids)
            sr["num_completions"] = sr.get("num_completions", 0) + len(completion_ids)
            return r

        def timed_gas(*a, **kw):
            t = time.perf_counter()
            r = orig_gas(*a, **kw)
            sr["gen_score_s"] = sr.get("gen_score_s", 0.0) + (time.perf_counter() - t)
            return r

        vg.sync_weights, vg.generate = timed_sync, timed_generate
        trainer._generate_and_score_completions = timed_gas


def save_and_reload_adapter(trainer, adapter_dir):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    trainer.save_model(adapter_dir)
    print(f"[grpo] adapter saved to {adapter_dir}: {sorted(os.listdir(adapter_dir))}", flush=True)
    saved = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    live = {n: p.detach().float().cpu() for n, p in trainer.model.named_parameters() if "lora_" in n}
    base = trainer.model.unload()
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    n_match = n_lora = 0
    for n, p in reloaded.named_parameters():
        if "lora_" in n:
            n_lora += 1
            key = n.replace(".default", "")
            if key in saved and torch.allclose(saved[key].float(), p.detach().float().cpu()):
                n_match += 1
    if not (n_lora > 0 and n_match == n_lora):
        raise RuntimeError(f"adapter reload mismatch: {n_match}/{n_lora}")
    nonzero = sum(1 for n, p in live.items() if "lora_B" in n and p.abs().sum() > 0)
    print(f"[grpo] adapter reload OK: {n_match}/{n_lora} LoRA tensors match; lora_B non-zero tensors={nonzero}", flush=True)
    base = reloaded.unload()
    return base, n_match, n_lora, nonzero


def verify_adapter(base, adapter, prompts, max_new_tokens=160):
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
                 or not torch.equal(params[pk].detach().to("cpu", torch.bfloat16), v.to(torch.bfloat16))]
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[verify] LoRA modules attached: {n_lora}; exported tensors: {len(exported)}; unmatched: {len(unmatched)}", flush=True)

    lines = [f"base={base}\nadapter={adapter}\nlora_modules_attached={n_lora}\nexported_tensors={len(exported)}\nunmatched={len(unmatched)}\n"]
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
    del model
    torch.cuda.empty_cache()
    return proof


def write_grpo_summary(path, config, versions, timers, warmup_steps, steps_total, train_wall, adapter_dir,
                       trainable_params, rubric):
    records = timers.records
    measured = [r for r in records if r["step"] > warmup_steps]
    summary = {
        "label": config.get("label"), "config": config, "versions": versions,
        "steps_total": steps_total, "steps_measured": len(measured),
        "train_wall_s": train_wall, "peak_mem_gb": max((r["peak_mem_gb"] for r in records), default=None),
        "adapter_dir": adapter_dir, "adapter_reload_ok": True, "trainable_params": trainable_params,
        "rubric": rubric,
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


SERVER_NAME = "grpo-trl-vllm"
DOCKER_GPU_ARGS = ["--network", "host", "--device", "/dev/kfd", "--group-add", "video",
                   "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE", "--ipc", "host"]
DOCKER_GPU_ARGS_CUDA = ["--network", "host", "--security-opt", "seccomp=unconfined",
                        "--cap-add", "SYS_PTRACE", "--ipc", "host"]

PLATFORM = os.environ.get("GRPO_PLATFORM", "rocm")


def _is_cuda():
    return PLATFORM == "cuda"


def _visible_devices_var():
    return "CUDA_VISIBLE_DEVICES" if _is_cuda() else "ROCR_VISIBLE_DEVICES"


def _gpus_flag(gpus):
    return f'"device={gpus}"'


def _docker_gpu_args(gpus):
    if _is_cuda():
        return DOCKER_GPU_ARGS_CUDA + ["--gpus", _gpus_flag(gpus)]
    return DOCKER_GPU_ARGS


def parse_args():
    p = argparse.ArgumentParser(description="GRPO LoRA training with TRL GRPOTrainer + vLLM server")
    p.add_argument("--model", required=True, help="HF checkpoint directory (absolute path)")
    p.add_argument("--data", required=True, help="JSONL with `messages` and `ground_truth` per row")
    p.add_argument("--output-dir", required=True, help="steps.jsonl, summary.json, adapter/ and server log go here")
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
    p.add_argument("--zero-stage", type=int, default=2, help="DeepSpeed ZeRO stage via `accelerate launch`; 0 = plain Trainer")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="bench")
    p.add_argument("--server-mode", default="docker", choices=["docker", "process", "external"],
                   help="docker: server + trainer containers from the host; process: vllm-serve child process; "
                        "external: server already running at --server-host:--server-port")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8105)
    p.add_argument("--server-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--server-timeout", type=int, default=900, help="seconds to wait for the server to become healthy")
    p.add_argument("--server-gpu", default="0",
                   help="ROCR_VISIBLE_DEVICES (rocm) / CUDA_VISIBLE_DEVICES (cuda) for the server "
                        "(index into the visible GPUs)")
    p.add_argument("--train-gpu", default="1",
                   help="ROCR_VISIBLE_DEVICES (rocm) / CUDA_VISIBLE_DEVICES (cuda) for the trainer")
    p.add_argument("--dri-devices", default=os.environ.get("GRPO_DRI_DEVICES", "/dev/dri"),
                   help="docker, rocm: render nodes exposed to BOTH containers (space separated; both GPUs, for the RCCL push)")
    p.add_argument("--cuda-devices", default=os.environ.get("GRPO_CUDA_DEVICES", "0,1"),
                   help="docker, cuda: physical GPU indices exposed to BOTH containers (comma separated; "
                        "both GPUs, for the NCCL push). --server-gpu/--train-gpu index into this list.")
    p.add_argument("--image", default=os.environ.get("GRPO_IMAGE", "grpo-trl:mi355"))
    p.add_argument("--accelerate-port", type=int, default=29655, help="accelerate main_process_port")
    p.add_argument("--verify", action="store_true", help="after saving, load base+adapter fresh on the trainer GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--verify-only", action="store_true", help="skip the server and training; --verify the adapter already in --output-dir")
    p.add_argument("--dry-run", action="store_true", help="print the commands / config that would run, then exit")
    return p.parse_args()


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
    # Both containers get BOTH GPUs (the weight push is a cross-container NCCL/RCCL collective);
    # the visibility variable then selects which one each process trains or serves on.
    if _is_cuda():
        devices = []
        gpu_args = _docker_gpu_args(args.cuda_devices)
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
    for k in ("GRPO_SHARED_DIR", "GRPO_JUDGE_URL", "GRPO_JUDGE_MODEL", "GRPO_JUDGE_RUBRIC"):
        if os.environ.get(k):
            env += ["-e", f"{k}={os.environ[k]}"]
    train_name = "grpo-trl-train-" + os.path.basename(os.path.abspath(args.output_dir).rstrip("/"))
    trainer = (["docker", "run", "--rm", "--name", train_name] + gpu_args + devices +
               ["--shm-size", "64g"] + env + _bind_mounts(args) + ["-w", HERE, "--entrypoint", "python", args.image,
               LAUNCHER] + worker_argv())
    return server, trainer, train_name


def run_docker(args):
    server_cmd, trainer_cmd, train_name = docker_commands(args)
    if args.dry_run:
        print("[dry-run] server container:\n  " + shlex.join(server_cmd))
        print("[dry-run] trainer container:\n  " + shlex.join(trainer_cmd))
        print("[dry-run] inside the trainer container the script re-launches itself under "
              f"`accelerate launch --use_deepspeed` when --zero-stage > 0 (here {args.zero_stage}).")
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


def accelerate_command(args):
    return [venv_bin("accelerate"), "launch", "--num_processes=1", "--mixed_precision=bf16", "--use_deepspeed",
            "--main_process_port", str(args.accelerate_port), LAUNCHER] + worker_argv()


def run_local(args, train, grpo_config_kwargs):
    os.makedirs(args.output_dir, exist_ok=True)
    needs_relaunch = args.zero_stage > 0 and not args.verify_only and not os.environ.get("ACCELERATE_USE_DEEPSPEED")
    if args.dry_run:
        if args.server_mode == "process":
            print("[dry-run] server process: %s=%s python %s" % (_visible_devices_var(), args.server_gpu, shlex.join(server_command(args))))
        if needs_relaunch:
            print("[dry-run] trainer: " + shlex.join(accelerate_command(args)))
        print("[dry-run] GRPOConfig kwargs:\n" + json.dumps(grpo_config_kwargs(args, dry=True), indent=1, default=str))
        return

    server = None
    try:
        if args.server_mode == "process" and not args.verify_only:
            env = dict(os.environ, VLLM_USE_V1="1"); env[_visible_devices_var()] = args.server_gpu
            log = open(os.path.join(args.output_dir, "vllm_server.log"), "wb")
            server = subprocess.Popen([sys.executable] + server_command(args), env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[server] vllm-serve pid {server.pid} on GPU index {args.server_gpu}; waiting for health", flush=True)
            wait_for_server(args, lambda: server.poll() is None)
        if needs_relaunch:
            env = dict(os.environ)
            if args.server_mode == "process":
                env[_visible_devices_var()] = args.train_gpu
            cmd = accelerate_command(args)
            print("[trainer] " + shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)
        else:
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
