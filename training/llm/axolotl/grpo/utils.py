from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# A console script next to the running interpreter. On the venv route it is not on PATH, and a
# bare name either fails with FileNotFoundError or silently picks up another environment's.
def venv_bin(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else name

# Axolotl resolves these four YAML entries by module name, so PYTHONPATH must contain this
# directory and ../../common (the launcher sets both for the server and trainer):
#   trl.reward_funcs: [utils.judge_reward]      datasets[].type: utils.grpo_transform
#   plugins: [utils.GRPOBenchPlugin]            vllm.serve_module: utils   (-> utils.main)
for _p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common")):
    if _p and os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))


def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def is_multimodal(cfg: dict) -> bool:
    return any("ForConditionalGeneration" in a for a in cfg.get("architectures") or [])


def auto_model_class(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_multimodal(cfg) else AutoModelForCausalLM


def parse_ground_truth(gt) -> dict:
    try:
        return json.loads(gt) if isinstance(gt, str) else dict(gt)
    except Exception:
        return {"reference": str(gt)}


def grpo_transform(cfg, *args, **kwargs):

    def transform_fn(example, tokenizer=None):
        prompt = example.get("messages")
        if not (isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]):
            raise ValueError("each row needs `messages` = list of chat dicts (see the Data section of the top-level README.md)")
        if "ground_truth" not in example:
            raise ValueError("each row needs `ground_truth` (JSON string with reference/source/question)")
        return {"prompt": prompt, "ground_truth": example["ground_truth"]}

    return transform_fn, {"remove_columns": ["messages"]}


EOS_IDS = {int(x) for x in os.environ.get("GRPO_EOS_IDS", "").split(",") if x.strip()}  # set by the plugin per model
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_MAX_COMPLETION_LENGTH", "0")) or None  # set by the plugin
REWARD_LOG = os.environ.get("GRPO_REWARD_LOG")  # jsonl, one line per score_batch call (plugin default: <output_dir>/reward_calls.jsonl)
_CALLS = 0


def finish_reasons_from_ids(completion_ids, max_completion_length):
    out = []
    for ids in completion_ids:
        ids = list(ids) if ids is not None else []
        truncated = (max_completion_length is not None and len(ids) >= max_completion_length
                     and (not ids or ids[-1] not in EOS_IDS))
        out.append("length" if truncated else "stop")
    return out


def judge_reward(prompts, completions, completion_ids=None, ground_truth=None, **kwargs):
    global _CALLS
    import judge_reward as shared

    t_start = time.time()
    n = len(completions)
    if ground_truth is None:
        raise ValueError("dataset column 'ground_truth' did not reach the reward function")
    gts = [parse_ground_truth(g) for g in ground_truth]
    questions = [g.get("question") or shared.question_text(p) for g, p in zip(gts, prompts)]
    references = [g.get("reference", "") for g in gts]
    sources = [g.get("source") for g in gts]
    completion_ids = completion_ids if completion_ids is not None else [[] for _ in range(n)]
    finish = finish_reasons_from_ids(completion_ids, MAX_COMPLETION_LENGTH)

    out = shared.score_batch(questions, completions, references, sources, finish)
    t_end = time.time()
    _CALLS += 1

    n_len = sum(f == "length" for f in finish)
    failed = sum(1 for j, s in zip(out["judge"], sources) if j is None and s != "3gpp_tsg")
    rec = {
        "call": _CALLS, "n": n, "t_start": t_start, "t_end": t_end, "wall_s": t_end - t_start,
        "timing_s": out["timing_s"], "reward_mean": sum(out["reward"]) / max(1, n),
        "judge_failed": failed, "finish_length": n_len,
        "last_tok_eos_any": sum(1 for ids in completion_ids if ids and ids[-1] in EOS_IDS),
        "completion_tokens": sum(len(ids) for ids in completion_ids),
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "eos_ids": sorted(EOS_IDS),
        "trainer_step": getattr(kwargs.get("trainer_state"), "global_step", None),
    }
    if REWARD_LOG:
        with open(REWARD_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    log_metric = kwargs.get("log_metric")
    if log_metric is not None:
        log_metric("judge_s", out["timing_s"])
        log_metric("reward_wall_s", rec["wall_s"])
        log_metric("judge_failed", float(failed))
        log_metric("finish_length_frac", n_len / max(1, n))
    print(f"[reward] call {_CALLS}: n={n} judge_s={out['timing_s']:.2f} wall={rec['wall_s']:.2f} "
          f"reward_mean={rec['reward_mean']:.3f} truncated={n_len} judge_failed={failed}", flush=True)
    return out["reward"]


def main(script_args):
    import dataclasses

    import vllm

    orig = vllm.__version__
    relabelled = os.environ.get("GRPO_VLLM_VERSION_RELABEL", "0.28.0.dev0+g281955fc6")
    vllm.__version__ = relabelled
    try:
        import vllm.version as _vv
        _vv.__version__ = relabelled
    except Exception:  # noqa: BLE001
        pass
    from trl.scripts.vllm_serve import ScriptArguments
    from axolotl.scripts.vllm_serve_lora import LoRAScriptArguments, main as lora_main

    # axolotl sweeps the WHOLE HOST for `VLLM::EngineCore` on start-up and shutdown and SIGKILLs
    # every match. That is fatal here: it targets the shared judge and any other stack's rollout
    # engine. If those belong to another user the kill raises PermissionError and takes this server
    # down with it; if they belong to this user they die silently. Disabling it is safe: the OS
    # reaps this process's own children on exit.
    try:
        from axolotl.scripts import process_cleanup as _pc

        def _no_host_wide_kill(*_patterns):
            pass

        _pc.cleanup_orphan_processes = _no_host_wide_kill
    except Exception:  # noqa: BLE001 - upstream layout changed; leave the default behaviour
        pass

    base_fields = {f.name for f in dataclasses.fields(ScriptArguments)}
    kwargs = {k: getattr(script_args, k) for k in base_fields if hasattr(script_args, k)}
    lora_args = LoRAScriptArguments(**kwargs, enable_lora=True, max_lora_rank=int(os.environ.get("GRPO_LORA_R", "16")),
                                    max_loras=int(os.environ.get("GRPO_MAX_LORAS", "2")), lora_dtype="bfloat16")
    print(f"[vllm-serve] vllm.__version__ {orig!r} relabelled to {relabelled!r}; serving with {lora_args}", flush=True)
    return lora_main(lora_args)


try:
    from axolotl.integrations.base import BasePlugin
    from axolotl.utils.logging import get_logger

    LOG = get_logger(__name__)
    WARMUP_STEPS = int(os.environ.get("GRPO_WARMUP_STEPS", "2"))
    LABEL = os.environ.get("GRPO_LABEL", "bench")
    SAMPLES = int(os.environ.get("GRPO_SAMPLE_COMPLETIONS", "8"))

    class GRPOBenchPlugin(BasePlugin):

        def __init__(self):
            super().__init__()
            self.step_rec = {}
            self.records = []
            self.timers = {"reward_s": []}
            self._reward_lines_seen = 0
            self.trainer = None
            self.t_train = None

        def post_trainer_create(self, cfg, trainer):
            import torch
            from transformers import TrainerCallback

            me = sys.modules[__name__]
            self.trainer = trainer
            os.makedirs(cfg.output_dir, exist_ok=True)
            self.steps_path = os.path.join(cfg.output_dir, "steps.jsonl")
            open(self.steps_path, "w").close()
            self.samples_path = os.path.join(cfg.output_dir, "completions_samples.jsonl")
            open(self.samples_path, "w").close()
            me.MAX_COMPLETION_LENGTH = int(trainer.args.max_completion_length)
            tok0 = getattr(trainer, "_tokenizer", None) or trainer.processing_class
            if not me.EOS_IDS:  # tokenizer eos + generation_config eos ids (chat turns may end with a non-eos id)
                eos = set()
                for src in (getattr(tok0, "eos_token_id", None),
                            getattr(getattr(trainer.model, "generation_config", None), "eos_token_id", None)):
                    if isinstance(src, int):
                        eos.add(src)
                    elif isinstance(src, (list, tuple)):
                        eos.update(int(x) for x in src)
                me.EOS_IDS = eos
            self.eos_ids = sorted(me.EOS_IDS)
            self.probe_token_id = int(getattr(tok0, "bos_token_id", None) or 2)
            if not me.REWARD_LOG:
                me.REWARD_LOG = os.path.join(cfg.output_dir, "reward_calls.jsonl")
            open(me.REWARD_LOG, "w").close()
            step_rec, timers, records = self.step_rec, self.timers, self.records

            self.trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            self.total_params = sum(p.numel() for p in trainer.model.parameters())
            LOG.info(f"[bench] trainer={type(trainer).__name__} model={type(trainer.model).__name__} "
                     f"trainable={self.trainable_params/1e6:.1f}M of {self.total_params/1e9:.2f}B "
                     f"eos_ids={self.eos_ids} num_generations={trainer.num_generations} "
                     f"gas={trainer.args.gradient_accumulation_steps} max_completion_length={trainer.args.max_completion_length}")

            # reward wall time (TRL keeps the function objects in trainer.reward_funcs; swap in place)
            _orig_judge = me.judge_reward

            def timed_judge(*a, **kw):
                t = time.perf_counter()
                r = _orig_judge(*a, **kw)
                timers["reward_s"].append(time.perf_counter() - t)
                return r

            for i, rf in enumerate(trainer.reward_funcs):
                if rf is _orig_judge:
                    timed_judge.__name__ = rf.__name__
                    trainer.reward_funcs[i] = timed_judge

            # rollout: one HTTP /generate/ per generation batch
            vg = trainer.vllm_generation
            _orig_gen = vg.generate

            def timed_generate(*a, **kw):
                t = time.perf_counter()
                r = _orig_gen(*a, **kw)
                step_rec["rollout_s"] = step_rec.get("rollout_s", 0.0) + (time.perf_counter() - t)
                completion_ids = r[1]
                step_rec["completion_tokens"] = step_rec.get("completion_tokens", 0) + sum(len(c) for c in completion_ids)
                step_rec["num_completions"] = step_rec.get("num_completions", 0) + len(completion_ids)
                step_rec["rollout_calls"] = step_rec.get("rollout_calls", 0) + 1
                if "sample_ids" not in step_rec:
                    step_rec["sample_ids"] = [(list(p), list(c)) for p, c in zip(r[0][:SAMPLES], completion_ids[:SAMPLES])]
                return r

            vg.generate = timed_generate

            # sync: NCCL merge path (fallback) and the LoRA filesystem path (vllm_lora_sync: true)
            _orig_sync_nccl = vg.sync_weights

            def timed_sync_nccl(*a, **kw):
                t = time.perf_counter()
                r = _orig_sync_nccl(*a, **kw)
                step_rec["sync_s"] = step_rec.get("sync_s", 0.0) + (time.perf_counter() - t)
                step_rec["sync_calls"] = step_rec.get("sync_calls", 0) + 1
                step_rec["sync_mode"] = "nccl_merge"
                return r

            vg.sync_weights = timed_sync_nccl

            if hasattr(trainer, "_sync_lora_adapter"):
                _orig_sync_lora = trainer._sync_lora_adapter

                def timed_sync_lora(*a, **kw):
                    t = time.perf_counter()
                    r = _orig_sync_lora(*a, **kw)
                    t_save = time.perf_counter() - t
                    t2 = time.perf_counter()  # 1-token probe so vLLM loads the adapter now, not inside the next rollout
                    try:
                        if vg.accelerator.is_main_process:
                            vg.vllm_client.generate(prompts=[[self.probe_token_id]], n=1, max_tokens=1, temperature=1.0)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(f"[bench] adapter-load probe failed: {exc}")
                    t_load = time.perf_counter() - t2
                    step_rec["sync_s"] = step_rec.get("sync_s", 0.0) + t_save + t_load
                    step_rec["sync_save_register_s"] = step_rec.get("sync_save_register_s", 0.0) + t_save
                    step_rec["sync_probe_load_s"] = step_rec.get("sync_probe_load_s", 0.0) + t_load
                    step_rec["sync_calls"] = step_rec.get("sync_calls", 0) + 1
                    step_rec["sync_mode"] = "lora_filesystem"
                    return r

                trainer._sync_lora_adapter = timed_sync_lora

            _orig_gas = trainer._generate_and_score_completions

            def timed_gas(*a, **kw):
                t = time.perf_counter()
                r = _orig_gas(*a, **kw)
                step_rec["gen_score_s"] = step_rec.get("gen_score_s", 0.0) + (time.perf_counter() - t)
                return r

            trainer._generate_and_score_completions = timed_gas

            # Synchronous data-producer path (use_data_producer: true, async_prefetch: false) on axolotl 0.18.0 + TRL 1.8.0:
            #  * produce(skip_policy_logps=True) runs axolotl's own _generate_only + _compute_deferred_scores instead of
            #    TRL's stock _generate_and_score_completions, whose kwargs axolotl's logps override does not accept;
            #  * trainer._generate returns 9 values in TRL 1.8 (+images, tool_images); axolotl unpacks 7;
            #  * axolotl reads trainer.pad_token_id / eos_token_id, which TRL 1.8 no longer sets.
            dp = getattr(trainer, "data_producer", None)
            if dp is not None and os.environ.get("GRPO_SYNC_SKIP_POLICY_LOGPS", "1") == "1" \
                    and not getattr(trainer.args, "async_prefetch", False):
                _orig_produce = dp.produce

                def produce_inline(*a, **kw):
                    kw.setdefault("skip_policy_logps", True)
                    t = time.perf_counter()
                    r = _orig_produce(*a, **kw)
                    step_rec["produce_s"] = step_rec.get("produce_s", 0.0) + (time.perf_counter() - t)
                    return r

                dp.produce = produce_inline
                _orig_generate = trainer._generate

                def generate_7(*a, **kw):
                    r = _orig_generate(*a, **kw)
                    return tuple(r)[:7] if isinstance(r, (tuple, list)) and len(r) > 7 else r

                trainer._generate = generate_7
                if not hasattr(trainer, "pad_token_id"):
                    trainer.pad_token_id = tok0.pad_token_id
                if not hasattr(trainer, "eos_token_id"):
                    trainer.eos_token_id = tok0.eos_token_id
                LOG.info("[bench] data-producer shims active: produce(skip_policy_logps=True), _generate -> 7 values, "
                         f"pad_token_id={trainer.pad_token_id} eos_token_id={trainer.eos_token_id}")

            if hasattr(trainer, "_compute_deferred_scores"):
                _orig_cds = trainer._compute_deferred_scores

                def timed_cds(*a, **kw):
                    t = time.perf_counter()
                    r = _orig_cds(*a, **kw)
                    step_rec["score_s"] = step_rec.get("score_s", 0.0) + (time.perf_counter() - t)
                    return r

                trainer._compute_deferred_scores = timed_cds

            steps_path = self.steps_path
            plugin = self

            class StepTimer(TrainerCallback):
                def on_train_begin(self, a, state, control, **kw):
                    torch.cuda.reset_peak_memory_stats()
                    plugin.t_train = time.perf_counter()

                def on_step_begin(self, a, state, control, **kw):
                    step_rec.clear()
                    step_rec["t_begin"] = time.perf_counter()
                    timers["reward_s"].clear()

                def on_step_end(self, a, state, control, **kw):
                    now = time.perf_counter()
                    rec = {
                        "step": state.global_step,
                        "cycle_s": now - step_rec["t_begin"],
                        "rollout_s": step_rec.get("rollout_s"),
                        "rollout_calls": step_rec.get("rollout_calls"),
                        "sync_s": step_rec.get("sync_s"),
                        "sync_calls": step_rec.get("sync_calls"),
                        "sync_mode": step_rec.get("sync_mode"),
                        "sync_save_register_s": step_rec.get("sync_save_register_s"),
                        "sync_probe_load_s": step_rec.get("sync_probe_load_s"),
                        "produce_s": step_rec.get("produce_s"),
                        "score_s": step_rec.get("score_s"),
                        "reward_s": sum(timers["reward_s"]) if timers["reward_s"] else None,
                        "completion_tokens": step_rec.get("completion_tokens"),
                        "num_completions": step_rec.get("num_completions"),
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                    }
                    # gen_score_s: stock path = _generate_and_score_completions; data-producer path = produce + deferred scores
                    if step_rec.get("gen_score_s") is not None:
                        rec["gen_score_s"] = step_rec["gen_score_s"]
                    elif step_rec.get("produce_s") is not None:
                        rec["gen_score_s"] = step_rec["produce_s"] + (step_rec.get("score_s") or 0.0)
                    else:
                        rec["gen_score_s"] = None
                    rec["train_step_s"] = rec["cycle_s"] - rec["gen_score_s"] if rec["gen_score_s"] is not None else None
                    rec["tokens_per_s"] = (rec["completion_tokens"] / rec["rollout_s"]
                                           if rec["rollout_s"] and rec["completion_tokens"] else None)
                    step_rec["pending"] = rec

            class LogAttach(TrainerCallback):
                # on_log fires after on_step_end for the same step: attach TRL's metrics, then flush the record.
                def on_log(self, a, state, control, logs=None, **kw):
                    rec = step_rec.get("pending")
                    if rec is None:
                        return
                    rec["trl"] = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
                    calls = plugin._drain_reward_calls()  # judge_s from the reward jsonl (the deferred-score path drops log_metric)
                    n_all = max(1, sum(c["n"] for c in calls)) if calls else 1
                    rec["judge_s"] = sum(c["timing_s"] for c in calls) if calls else None
                    rec["judge_calls"] = len(calls)
                    rec["judge_failed"] = sum(c.get("judge_failed", 0) for c in calls) if calls else None
                    rec["finish_length_frac"] = sum(c.get("finish_length", 0) for c in calls) / n_all if calls else None
                    rec["last_tok_eos_any_frac"] = sum(c.get("last_tok_eos_any", 0) for c in calls) / n_all if calls else None
                    records.append(rec)
                    with open(steps_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    try:
                        with open(plugin.samples_path, "a") as f:
                            for i, (p_ids, c_ids) in enumerate(step_rec.get("sample_ids") or []):
                                f.write(json.dumps({"step": rec["step"], "i": i, "prompt_tokens": len(p_ids),
                                                    "completion_tokens": len(c_ids), "last_token_id": c_ids[-1] if c_ids else None,
                                                    "prompt_tail": tok0.decode(p_ids[-64:], skip_special_tokens=False)[-300:],
                                                    "completion": tok0.decode(c_ids, skip_special_tokens=False)}) + "\n")
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(f"[bench] completions sample dump failed: {exc}")
                    LOG.info(f"[bench] step {rec['step']}: cycle={rec['cycle_s']:.1f}s rollout={rec['rollout_s']} "
                             f"judge={rec['judge_s']} sync={rec['sync_s']} ({rec['sync_mode']}, {rec['sync_calls']} calls) "
                             f"train={rec['train_step_s']} tok/s={rec['tokens_per_s']} reward={rec['trl'].get('reward')} "
                             f"mean_len={rec['trl'].get('completions/mean_length')} peak={rec['peak_mem_gb']:.1f}GB")
                    step_rec["pending"] = None

            trainer.add_callback(StepTimer())
            trainer.add_callback(LogAttach())

        def _drain_reward_calls(self):
            path = sys.modules[__name__].REWARD_LOG
            if not path or not os.path.exists(path):
                return []
            with open(path) as f:
                lines = f.readlines()
            new = [json.loads(l) for l in lines[self._reward_lines_seen:] if l.strip()]
            self._reward_lines_seen = len(lines)
            return new

        def post_train(self, cfg, model):
            import torch
            import transformers

            trainer = self.trainer
            train_wall = time.perf_counter() - self.t_train if self.t_train else None
            records = self.records
            measured = [r for r in records if r["step"] > WARMUP_STEPS]
            check = adapter_file_check(cfg.output_dir, model)
            LOG.info(f"[bench] adapter file check: {check['lora_tensors_match']}/{check['lora_tensors']} LoRA tensors "
                     f"match the saved file; lora_B non-zero={check['lora_B_nonzero']}; err={check['error']}")

            def med(key):
                vals = [r.get(key) for r in measured if r.get(key) is not None]
                return statistics.median(vals) if vals else None

            summary = {
                "label": LABEL, "stack": "axolotl",
                "trainer_class": type(trainer).__name__, "model_class": type(model).__name__,
                "warmup_steps": WARMUP_STEPS, "steps_total": trainer.state.global_step, "steps_measured": len(measured),
                "train_wall_s": train_wall,
                "trainable_params": self.trainable_params, "total_params": self.total_params,
                "effective_args": {k: getattr(trainer.args, k, None) for k in (
                    "num_generations", "per_device_train_batch_size", "gradient_accumulation_steps",
                    "steps_per_generation", "generation_batch_size", "num_iterations", "max_completion_length",
                    "temperature", "top_p", "top_k", "beta", "learning_rate", "lr_scheduler_type", "warmup_steps",
                    "max_steps", "bf16", "gradient_checkpointing", "seed", "shuffle_dataset", "use_vllm", "vllm_mode",
                    "vllm_server_host", "vllm_server_port", "loss_type", "epsilon", "scale_rewards",
                    "mask_truncated_completions", "vllm_importance_sampling_correction", "vllm_importance_sampling_cap",
                    "vllm_lora_sync", "use_data_producer", "async_prefetch", "vllm_sync_interval",
                    "skip_zero_advantage_batches", "reward_num_workers", "streaming_partial_batch",
                    "replay_buffer_size", "optim")},
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "trl": __import__("trl").__version__, "peft": __import__("peft").__version__,
                             "vllm": __import__("vllm").__version__, "axolotl": __import__("axolotl").__version__,
                             "accelerate": __import__("accelerate").__version__, "python": sys.version.split()[0]},
                "peak_mem_gb": max((r["peak_mem_gb"] for r in records), default=None),
                "adapter_dir": cfg.output_dir, "adapter_file_check_ok": check["ok"],
                "adapter_lora_tensors": check["lora_tensors"], "adapter_lora_tensors_match": check["lora_tensors_match"],
                "lora_B_nonzero": check["lora_B_nonzero"], "adapter_check_error": check["error"],
                "rubric": __import__("judge_reward").RUBRIC_VERSION,
                "eos_ids": self.eos_ids,
                "base_model": str(getattr(cfg, "base_model", None)),
            }
            for k in ("rollout_s", "judge_s", "reward_s", "train_step_s", "sync_s", "cycle_s", "gen_score_s",
                      "produce_s", "score_s", "tokens_per_s", "completion_tokens", "sync_calls", "rollout_calls",
                      "sync_save_register_s", "sync_probe_load_s"):
                summary[k] = [r.get(k) for r in measured]
                summary[f"{k}_median"] = med(k)
            summary["sync_mode"] = sorted({r.get("sync_mode") for r in records if r.get("sync_mode")})
            summary["reward_per_step"] = [r["trl"].get("reward") for r in records]
            summary["mean_length_per_step"] = [r["trl"].get("completions/mean_length") for r in records]
            summary["finish_length_frac_per_step"] = [r.get("finish_length_frac") for r in records]
            summary["judge_failed_per_step"] = [r.get("judge_failed") for r in records]
            summary["reward_first"] = measured[0]["trl"].get("reward") if measured else None
            summary["reward_last"] = measured[-1]["trl"].get("reward") if measured else None
            with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1, default=str)
            LOG.info("[bench] DONE " + json.dumps({k: summary[k] for k in (
                "steps_total", "steps_measured", "peak_mem_gb", "reward_first", "reward_last", "adapter_file_check_ok",
                "rollout_s_median", "judge_s_median", "train_step_s_median", "sync_s_median", "cycle_s_median",
                "tokens_per_s_median")}, default=str))
except ImportError:  # axolotl not importable outside the training container
    GRPOBenchPlugin = None


def adapter_files(adapter_dir: str) -> dict:
    import struct

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    for p in (cfg_path, st_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found: training did not write the adapter")
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(st_path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    modules = {k.rsplit(".lora_", 1)[0] for k in keys if ".lora_" in k}
    return {"adapter_dir": adapter_dir, "tensors": len(keys), "modules": len(modules),
            "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"), "target_modules": cfg.get("target_modules")}


def adapter_file_check(adapter_dir: str, model) -> dict:
    import torch
    from safetensors.torch import load_file

    out = {"ok": False, "lora_tensors": 0, "lora_tensors_match": 0, "lora_B_nonzero": 0, "error": None}
    try:
        saved = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        for n, p in model.named_parameters():
            if "lora_" in n:
                out["lora_tensors"] += 1
                key = n.replace(".default", "")
                if key in saved and torch.allclose(saved[key].float().to(p.device), p.detach().float()):
                    out["lora_tensors_match"] += 1
                if "lora_B" in n and p.detach().abs().sum().item() > 0:
                    out["lora_B_nonzero"] += 1
        out["ok"] = out["lora_tensors"] > 0 and out["lora_tensors_match"] == out["lora_tensors"]
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def verify_adapter(base: str, adapter_dir: str, prompts: list, max_new_tokens: int = 128, meta: dict | None = None) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    t0 = time.perf_counter()
    model = auto_model_class(load_model_config(base)).from_pretrained(base, dtype=torch.bfloat16,
                                                                     attn_implementation="sdpa", device_map={"": 0})
    print(f"[verify] base {type(model).__name__} loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    check = adapter_file_check(adapter_dir, model)
    print(f"[verify] LoRA tensors: {check['lora_tensors_match']}/{check['lora_tensors']} match the file; "
          f"lora_B non-zero={check['lora_B_nonzero']}", flush=True)
    if not check["ok"]:
        raise RuntimeError(f"adapter tensors do not match the file: {check}")

    gens = {}
    for i, prompt in enumerate(prompts):
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
        for label in (("adapter", "base_only") if i == 0 else ("adapter",)):
            t = time.perf_counter()
            with torch.no_grad():
                if label == "adapter":
                    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
                else:
                    with model.disable_adapter():
                        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            key = label if i == 0 else f"{label}_{i}"
            gens[key] = {"text": text, "new_tokens": int(out.shape[1] - ids["input_ids"].shape[1]),
                         "gen_s": time.perf_counter() - t}
            print(f"[verify] prompt {i} {label} ({gens[key]['new_tokens']} tokens):\n{text}\n", flush=True)

    rec = {"adapter_dir": adapter_dir, "model_class": type(model.base_model.model).__name__,
           "lora_tensors": check["lora_tensors"], "lora_tensors_match": check["lora_tensors_match"],
           "lora_B_nonzero": check["lora_B_nonzero"], "generations": gens, **(meta or {})}
    proof = os.path.join(adapter_dir, "adapter_proof.json")
    with open(proof, "w") as f:
        json.dump(rec, f, indent=1)
    return proof


# The verify container re-execs the launcher. Name it explicitly rather than via __file__,
# which from this module would resolve to utils.py.
LAUNCHER = os.path.join(HERE, "train_llm_grpo_axolotl.py")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
COMMON = os.environ.get("GRPO_COMMON_DIR") or os.path.join(REPO_ROOT, "grpo_common")
PASS_ENV = ("GRPO_JUDGE_URL", "GRPO_JUDGE_MODEL", "GRPO_JUDGE_CONC", "GRPO_JUDGE_RUBRIC", "GRPO_EOS_IDS",
            "GRPO_SAMPLE_COMPLETIONS", "GRPO_VLLM_VERSION_RELABEL", "GRPO_SYNC_SKIP_POLICY_LOGPS", "GRPO_MAX_LORAS")
# rocm (default) pins a GPU with ROCR_VISIBLE_DEVICES and exposes /dev/kfd + the render nodes;
# cuda pins it with `--gpus "device=<n>"` and CUDA_VISIBLE_DEVICES, and has no kfd/render nodes.
PLATFORM = os.environ.get("GRPO_PLATFORM", "rocm")

def _is_cuda():
    return PLATFORM == "cuda"


def _visible_devices_var():
    return "CUDA_VISIBLE_DEVICES" if _is_cuda() else "ROCR_VISIBLE_DEVICES"


def _gpus_flag(gpus):
    return f'"device={gpus}"'


def parse_args():
    p = argparse.ArgumentParser(description="GRPO LoRA training with Axolotl (vLLM server + trainer)")
    p.add_argument("--config", required=True, help="Axolotl GRPO YAML (grpo_smoke*.yaml / grpo_bench*.yaml)")
    p.add_argument("--model", default=None, help="HF checkpoint directory (overrides base_model)")
    p.add_argument("--data", default=None, help="JSONL with `messages` and `ground_truth` per row (overrides datasets[0].path)")
    p.add_argument("--output-dir", default=None, help="adapter, logs, steps.jsonl, summary.json (overrides output_dir)")
    p.add_argument("--label", default=None, help="run label for container names and summary.json (default: output dir name)")
    p.add_argument("--warmup-steps", type=int, default=None, help="steps excluded from the summary medians (default: 0 if max_steps <= 2 else 2)")
    p.add_argument("--steps", type=int, default=None, help="max_steps override")
    p.add_argument("--prompt-batch-size", type=int, default=None, help="prompts per step (sets gradient_accumulation_steps)")
    p.add_argument("--group-size", type=int, default=None, help="completions per prompt (trl.num_generations)")
    p.add_argument("--max-tokens", type=int, default=None, help="trl.max_completion_length")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra YAML override, dotted key (repeatable)")
    p.add_argument("--server-mode", choices=("docker", "process"), default="docker",
                   help="docker: server, trainer and verify run in containers from --image; process: subprocesses in this environment")
    p.add_argument("--server-gpu", default="0", help="rollout server GPU: ROCR index, or /dev/dri/renderDNNN (docker mode, hard isolation)")
    p.add_argument("--train-gpu", default="1", help="trainer GPU, same forms as --server-gpu")
    p.add_argument("--image", default=os.environ.get("GRPO_IMAGE", "grpo-axolotl:mi355"))
    p.add_argument("--sync-dir", default=None, help="LoRA sync directory shared by server and trainer (docker: $GRPO_ENV_DIR/env_axolotl/lora_sync or <output-dir>/lora_sync; process: <tmp>/grpo_axolotl_lora_sync)")
    p.add_argument("--port", type=int, default=29656, help="accelerate main_process_port")
    p.add_argument("--server-timeout", type=int, default=1200, help="seconds to wait for the vLLM server /health/")
    p.add_argument("--no-export", action="store_true", help="skip the adapter check after training")
    p.add_argument("--export-only", action="store_true", help="skip training; check (and --verify) the adapter already in the output dir")
    p.add_argument("--verify", action="store_true", help="after training, load base+adapter on the trainer GPU and generate")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable; default: --verify-row of --data)")
    p.add_argument("--verify-row", type=int, default=0, help="row of the data file used as the --verify prompt")
    p.add_argument("--dry-run", action="store_true", help="print the effective config overrides and the commands, then exit")
    return p.parse_args()


class Launcher:

    def __init__(self, args, cfg, cfg_path):
        self.args, self.cfg, self.cfg_path = args, cfg, cfg_path
        self.out = cfg["output_dir"]
        self.label = args.label or os.path.basename(self.out.rstrip("/"))
        self.docker = args.server_mode == "docker"
        env_dir = os.environ.get("GRPO_ENV_DIR")
        if args.sync_dir:
            self.sync_dir = os.path.abspath(args.sync_dir)
        elif self.docker:  # mounted at /lora_sync in both containers
            self.sync_dir = os.path.join(env_dir, "env_axolotl", "lora_sync") if env_dir else os.path.join(self.out, "lora_sync")
        else:  # becomes the trainer's TMPDIR: keep it short (multiprocessing binds AF_UNIX sockets under it)
            self.sync_dir = os.path.join(tempfile.gettempdir(), "grpo_axolotl_lora_sync")
        self.server_name = f"grpo-axolotl-vllm-{self.label}"
        self.train_name = f"grpo-axolotl-train-{self.label}"
        self.host = cfg.get("vllm", {}).get("host", "127.0.0.1")
        self.port = int(cfg.get("vllm", {}).get("port", 8131))
        steps = int(cfg.get("max_steps") or 0)
        self.warmup = args.warmup_steps if args.warmup_steps is not None else (0 if steps <= 2 else 2)
        self.server_proc = None
        self.pythonpath = f"{HERE}:{COMMON}"

    def _gpu(self, spec):
        if _is_cuda():
            if spec.startswith("/dev/"):
                raise SystemExit("a /dev/dri render node cannot be pinned on CUDA; pass a GPU index")
            return [], "0"
        if spec.startswith("/dev/"):
            if not self.docker:
                raise SystemExit("a /dev/dri render node can only be pinned in --server-mode docker")
            return [spec], "0"
        return ["/dev/dri"], spec

    def _env(self, gpu, extra):
        # axolotl re-execs `accelerate` by bare name, so the interpreter's own bin/ must lead PATH:
        # otherwise it resolves against whatever environment the caller happened to have.
        path = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "AXOLOTL_DO_NOT_TRACK": "1", "PYTHONPATH": self.pythonpath, _visible_devices_var(): gpu,
               "PATH": path, "GRPO_LORA_R": str(self.cfg.get("lora_r", 16))}
        env.update({k: os.environ[k] for k in PASS_ENV if k in os.environ})
        env.update(extra)
        return env

    def _mounts(self):
        wanted = [(os.environ.get("GRPO_SHARED_DIR"), ""), (self.out, ""), (os.path.dirname(self.cfg["base_model"].rstrip("/")), ":ro"),
                  (os.path.dirname(self.cfg["datasets"][0]["path"]), ":ro"), (REPO_ROOT, ":ro")]
        mounts, seen = [], []
        for path, mode in wanted:
            if not path:
                continue
            path = os.path.abspath(path)
            if any(path == s or path.startswith(s + "/") for s in seen):
                continue
            seen.append(path)
            mounts += ["-v", f"{path}:{path}{mode}"]
        return mounts + ["-v", f"{self.sync_dir}:/lora_sync"]

    def _docker(self, name, gpu_spec, shm, env, entrypoint, argv, detach=False):
        devices, index = self._gpu(gpu_spec)
        cmd = ["docker", "run", "--name", name, "--network", "host"]
        if _is_cuda():
            # One GPU per container is enough: axolotl syncs the adapter through the shared
            # filesystem (trl.vllm_lora_sync), not a cross-GPU collective.
            cmd += ["--gpus", _gpus_flag(gpu_spec)]
        else:
            cmd += ["--device", "/dev/kfd"]
            for d in devices:
                cmd += ["--device", d]
            cmd += ["--group-add", "video"]
        cmd += ["--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE",
                "--ipc", "host", "--shm-size", shm, "-d" if detach else "--rm"]
        for k, v in self._env(index, env).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += self._mounts() + ["-w", HERE, "--entrypoint", entrypoint, self.args.image] + argv
        return cmd

    def server_cmd(self):
        if self.docker:
            return self._docker(self.server_name, self.args.server_gpu, "32g", {"VLLM_USE_V1": "1"},
                                "axolotl", ["vllm-serve", self.cfg_path], detach=True)
        return [venv_bin("axolotl"), "vllm-serve", self.cfg_path]

    def train_cmd(self):
        env = {"GRPO_LABEL": self.label, "GRPO_WARMUP_STEPS": str(self.warmup)}
        argv = ["train", self.cfg_path, "--num-processes", "1", "--main-process-port", str(self.args.port)]
        if self.docker:
            return self._docker(self.train_name, self.args.train_gpu, "64g", dict(env, TMPDIR="/lora_sync"), "axolotl", argv)
        return [venv_bin("axolotl")] + argv

    def verify_cmd(self):
        argv = [LAUNCHER, "--config", self.cfg_path, "--export-only", "--verify",
                "--server-mode", "process", "--verify-row", str(self.args.verify_row)]
        for p in self.args.verify_prompt or []:
            argv += ["--verify-prompt", p]
        return self._docker(f"grpo-axolotl-verify-{self.label}", self.args.train_gpu, "16g", {}, "python", argv)

    def start_server(self):
        os.makedirs(self.sync_dir, exist_ok=True)
        for d in os.listdir(self.sync_dir):  # stale adapters from an earlier run
            if d.startswith("lora_sync_"):
                shutil.rmtree(os.path.join(self.sync_dir, d), ignore_errors=True)
        log_path = os.path.join(self.out, "vllm_server.log")
        if self.docker:
            subprocess.run(["docker", "rm", "-f", self.server_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(self.server_cmd(), check=True, stdout=subprocess.DEVNULL)
        else:
            env = dict(os.environ, **self._env(self.args.server_gpu, {"VLLM_USE_V1": "1"}))
            self.server_proc = subprocess.Popen(self.server_cmd(), env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        print(f"[server] started ({self.args.server_mode}, GPU {self.args.server_gpu}); waiting for http://{self.host}:{self.port}/health/", flush=True)
        t0 = time.time()
        while time.time() - t0 < self.args.server_timeout:
            try:
                with urllib.request.urlopen(f"http://{self.host}:{self.port}/health/", timeout=3) as r:
                    if b"ok" in r.read().lower():
                        print(f"[server] healthy after {time.time()-t0:.0f}s", flush=True)
                        return
            except Exception:  # noqa: BLE001
                pass
            if not self._server_alive():
                self._dump_server_log(log_path, tail=True)
                raise SystemExit("vLLM server exited before becoming healthy")
            time.sleep(5)
        self._dump_server_log(log_path, tail=True)
        raise SystemExit(f"vLLM server not healthy after {self.args.server_timeout}s")

    def _server_alive(self):
        if self.docker:
            r = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{self.server_name}$"], capture_output=True, text=True)
            return bool(r.stdout.strip())
        return self.server_proc is not None and self.server_proc.poll() is None

    def _dump_server_log(self, log_path, tail=False):
        if self.docker:
            r = subprocess.run(["docker", "logs", self.server_name], capture_output=True, text=True, errors="replace")
            with open(log_path, "w") as f:
                f.write(r.stdout + r.stderr)
        if tail and os.path.exists(log_path):
            with open(log_path, errors="replace") as f:
                print("".join(f.readlines()[-40:]), flush=True)

    def stop_server(self):
        if self.docker:
            if subprocess.run(["docker", "inspect", self.server_name], capture_output=True).returncode == 0:
                self._dump_server_log(os.path.join(self.out, "vllm_server.log"))
                subprocess.run(["docker", "rm", "-f", self.server_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", "-f", self.train_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif self.server_proc is not None and self.server_proc.poll() is None:
            self.server_proc.terminate()
            try:
                self.server_proc.wait(30)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
        print("[server] stopped", flush=True)

    def run_logged(self, cmd, log_name, env=None) -> int:
        with open(os.path.join(self.out, log_name), "a") as log:
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            return p.wait()

    def train(self) -> int:
        env = None if self.docker else dict(os.environ, **self._env(self.args.train_gpu, {
            "GRPO_LABEL": self.label, "GRPO_WARMUP_STEPS": str(self.warmup), "TMPDIR": self.sync_dir}))
        t0 = time.time()
        rc = self.run_logged(self.train_cmd(), "train.log", env)
        print(f"[train] exit {rc} after {time.time()-t0:.0f}s", flush=True)
        return rc


def verify_prompts(args, cfg):
    if args.verify_prompt:
        return args.verify_prompt, {}
    with open(cfg["datasets"][0]["path"]) as f:
        for i, line in enumerate(f):
            if i == args.verify_row:
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"--verify-row {args.verify_row}: data file has fewer rows")
    gt = parse_ground_truth(row.get("ground_truth", "{}"))
    meta = {"row": args.verify_row, "id": gt.get("id"), "source": gt.get("source"),
            "question": (gt.get("question") or "")[:2000], "reference": (gt.get("reference") or "")[:2000]}
    return [row["messages"]], meta
