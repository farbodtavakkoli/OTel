import argparse
import functools
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
log = logging.getLogger("grpo")

for _p in (os.environ.get("GRPO_COMMON_DIR"), os.path.join(HERE, "..", "..", "grpo_common")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))


def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def is_multimodal(cfg: dict) -> bool:
    return any("ForConditionalGeneration" in a for a in cfg.get("architectures") or [])


def auto_model_class(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_multimodal(cfg) else AutoModelForCausalLM


def load_grpo_dataset(path: str, seed: int):
    from datasets import load_dataset
    from judge_reward import question_text

    def convert(row):
        try:
            gt = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else dict(row["ground_truth"])
        except Exception:
            gt = {"reference": str(row["ground_truth"])}
        return {"prompt": row["messages"],
                "question": gt.get("question") or question_text(row["messages"]),
                "reference": gt.get("reference", ""),
                "source": gt.get("source") or ""}

    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(convert, remove_columns=ds.column_names)
    return ds.shuffle(seed=seed)


BENCH = {"judge_s": [], "judge_scores": [], "judge_none": [], "reward_mean": [],
         "sync_s": [], "lora_add_s": [], "sync_step_s": [],
         "gen_wall_s": [], "gen_tokens": [], "gen_score_s": [],
         "cycle": [], "trl_logs": []}


def make_judge_reward(max_completion_length: int, dump_dir: str | None = None):
    from judge_reward import score_batch

    def judge_reward_fn(prompts, completions, completion_ids=None, question=None, reference=None,
                        source=None, **kwargs):
        n = len(completions)
        finish = None
        if completion_ids is not None:
            finish = ["length" if len(ids) >= max_completion_length else "stop" for ids in completion_ids]
        out = score_batch(question or prompts, completions, reference, source, finish)
        BENCH["judge_s"].append(out["timing_s"])
        scores = [s for s in out["judge"] if s is not None]
        BENCH["judge_scores"].append(statistics.mean(scores) if scores else None)
        # 3gpp_tsg rows are exact-match scored and never sent to the judge (judge=None by design).
        BENCH["judge_none"].append(sum(1 for s, src in zip(out["judge"], source or [None] * n)
                                       if s is None and src != "3gpp_tsg"))
        BENCH["reward_mean"].append(statistics.mean(out["reward"]))
        log.info(f"judge: n={n} judge_s={out['timing_s']:.2f} mean_reward={BENCH['reward_mean'][-1]:.4f} "
                 f"judge_failed={BENCH['judge_none'][-1]} truncated={sum(1 for f in (finish or []) if f == 'length')}")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, f"rollout_{len(BENCH['judge_s']):04d}.jsonl"), "w") as f:
                for i in range(n):
                    f.write(json.dumps({"source": (source or [None] * n)[i], "prompt": prompts[i],
                                        "completion": _text(completions[i]),
                                        "reference": (reference or [None] * n)[i],
                                        "finish_reason": (finish or [None] * n)[i],
                                        "judge": out["judge"][i], "reward": out["reward"][i]},
                                       ensure_ascii=False) + "\n")
        return out["reward"]

    return judge_reward_fn


def patch_vllm_fork_lora_manager():
    import torch
    try:
        import vllm.v1.worker.lora_model_runner_mixin as mixin
        from unsloth_zoo.vllm_lora_worker_manager import LRUCacheWorkerLoRAManager as UnslothLRU
    except Exception as e:  # noqa: BLE001
        log.warning(f"could not patch PTWorkerLoRAManager: {e}")
        return
    if hasattr(mixin, "PTWorkerLoRAManager"):
        mixin.PTWorkerLoRAManager = UnslothLRU
        log.info("patched vllm.v1.worker.lora_model_runner_mixin.PTWorkerLoRAManager -> unsloth_zoo LRUCacheWorkerLoRAManager")
    orig_add_adapter = UnslothLRU.add_adapter

    def timed_add_adapter(self, lora_request):
        try:
            already = lora_request.lora_int_id in self.list_adapters()
        except Exception:  # noqa: BLE001
            already = False
        if already:
            return orig_add_adapter(self, lora_request)
        t = time.perf_counter()
        r = orig_add_adapter(self, lora_request)
        torch.cuda.synchronize()
        BENCH["lora_add_s"].append(time.perf_counter() - t)
        return r
    UnslothLRU.add_adapter = timed_add_adapter


def patch_gemma4_fast_inference(max_num_seqs: int):
    import unsloth.models.vision as vision
    if "gemma4" not in vision.VLLM_SUPPORTED_VLM:
        vision.VLLM_SUPPORTED_VLM.append("gemma4")
    # unsloth_zoo wraps Gemma4TextConfig in a read-only proxy; vLLM writes tie_word_embeddings to it.
    from unsloth_zoo.temporary_patches.gemma4 import _Gemma4KVSharedSafeProxy

    def proxy_setattr(self, name, value):
        if name == "_real":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)
    _Gemma4KVSharedSafeProxy.__setattr__ = proxy_setattr

    # unsloth_zoo.vllm_utils.load_vllm caps max_num_seqs at 1 for any config with a vision_config;
    # prompts here are text-only, so restore the requested value (LLM is looked up by name at call time).
    import vllm
    orig_llm = vllm.LLM

    class BenchLLM(orig_llm):
        def __init__(self, *a, **kw):
            kw["max_num_seqs"] = max_num_seqs
            kw["max_num_batched_tokens"] = max(int(kw.get("max_num_batched_tokens") or 0), 8192)
            super().__init__(*a, **kw)
    vllm.LLM = BenchLLM
    log.info(f"gemma4 fast-inference patches applied (max_num_seqs={max_num_seqs})")


def time_load_lora(model):
    import torch
    if not hasattr(model, "load_lora"):
        log.warning("model has no load_lora attribute; sync_s will be null")
        return
    orig = model.load_lora

    @functools.wraps(orig)
    def timed(*a, **kw):
        t = time.perf_counter()
        r = orig(*a, **kw)
        torch.cuda.synchronize()
        BENCH["sync_s"].append(time.perf_counter() - t)
        return r
    model.load_lora = timed


def time_trainer(trainer, model):
    import torch
    vg = getattr(trainer, "vllm_generation", None)
    if vg is None:
        log.warning("trainer has no vllm_generation; rollout_s will be null")
    else:
        log.info(f"vllm_generation.llm is model.vllm_engine: {vg.llm is getattr(model, 'vllm_engine', None)}; "
                 f"shared_weights={getattr(vg.llm, 'shared_weights', None)}")
        orig_gen = vg.generate

        def timed_generate(*a, **kw):
            n_sync, n_add = len(BENCH["sync_s"]), len(BENCH["lora_add_s"])
            t = time.perf_counter()
            out = orig_gen(*a, **kw)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            completion_ids = out[1]
            BENCH["gen_wall_s"].append(wall)
            BENCH["gen_tokens"].append(int(sum(len(c) for c in completion_ids)))
            sync = sum(BENCH["sync_s"][n_sync:]) + sum(BENCH["lora_add_s"][n_add:])
            BENCH["sync_step_s"].append(sync)
            log.info(f"rollout: n={len(completion_ids)} tokens={BENCH['gen_tokens'][-1]} wall={wall:.2f}s "
                     f"sync={sync:.3f}s tokens/s={BENCH['gen_tokens'][-1] / max(wall - sync, 1e-6):.0f}")
            return out
        vg.generate = timed_generate

    orig_gas = trainer._generate_and_score_completions

    def timed_gas(*a, **kw):
        t = time.perf_counter()
        r = orig_gas(*a, **kw)
        torch.cuda.synchronize()
        BENCH["gen_score_s"].append(time.perf_counter() - t)
        return r
    trainer._generate_and_score_completions = timed_gas


def step_clock():
    import torch
    from transformers import TrainerCallback

    class StepClock(TrainerCallback):
        def on_step_begin(self, a, state, control, **kw):
            self._t = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, a, state, control, **kw):
            torch.cuda.synchronize()
            c = {"step": state.global_step, "cycle_s": time.perf_counter() - self._t,
                 "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                 "reserved_gb": torch.cuda.memory_reserved() / 1e9}
            BENCH["cycle"].append(c)
            log.info(f"step {c['step']}: cycle_s={c['cycle_s']:.2f} peak_alloc_gb={c['peak_alloc_gb']:.1f}")

        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                BENCH["trl_logs"].append({"step": state.global_step, **logs})

    return StepClock()


def assemble_steps() -> list:
    def at(key, i):
        return BENCH[key][i] if i < len(BENCH[key]) else None

    steps = []
    for i, c in enumerate(BENCH["cycle"]):
        gw, sy, gs, tk = at("gen_wall_s", i), at("sync_step_s", i), at("gen_score_s", i), at("gen_tokens", i)
        rollout = (gw - (sy or 0.0)) if gw is not None else None
        steps.append({
            "step": c["step"], "cycle_s": c["cycle_s"], "gen_score_s": gs,
            "rollout_s": rollout, "sync_s": sy, "judge_s": at("judge_s", i),
            "train_step_s": (c["cycle_s"] - gs) if gs is not None else None,
            "tokens": tk, "tokens_per_s": (tk / rollout) if (tk and rollout) else None,
            "reward_mean": at("reward_mean", i), "judge_mean_0_10": at("judge_scores", i),
            "judge_failed": at("judge_none", i),
            "peak_alloc_gb": c["peak_alloc_gb"], "reserved_gb": c["reserved_gb"],
        })
    return steps


def versions() -> dict:
    import peft, torch, transformers, trl, unsloth, vllm  # noqa: E401
    return {"torch": torch.__version__, "vllm": vllm.__version__, "trl": trl.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "unsloth": unsloth.__version__}


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
                 if (pk := re.sub(r"\.lora_(A|B)\.weight$", r".lora_\1.default.weight", k)) not in params
                 or not torch.equal(params[pk].detach().to("cpu", v.dtype), v)]
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    print(f"[verify] LoRA modules attached: {n_lora}; saved tensors: {len(exported)}; unmatched: {len(unmatched)}", flush=True)

    lines = [f"base={base}\nadapter={adapter}\nlora_modules_attached={n_lora}\nsaved_tensors={len(exported)}\nunmatched={len(unmatched)}\n"]
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


def _text(completion):
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return completion or ""


def _prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt or []:
        c = m.get("content", "")
        if isinstance(c, list):
            parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict)))
        else:
            parts.append(str(c))
    return "\n".join(parts).strip()


CORRECT_REWARD = 4.0
WRONG_REWARD = -0.5


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def reward_verifiable(prompts, completions, answer, **kwargs):
    scores = []
    for comp, gold in zip(completions, answer):
        try:
            ok = _normalize(_text(comp)) == _normalize(str(gold))
        except Exception:
            ok = False
        scores.append(CORRECT_REWARD if ok else WRONG_REWARD)
    return scores


_JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION and a MODEL ANSWER, rate how correct, "
    "relevant, and well-formed the answer is on a scale from 0 to 10. "
    "Reply with ONLY the integer score, nothing else."
)


def make_llm_judge_reward(model=None, base_url=None, api_key=None,
                          max_tokens=8, weight=1.0, timeout=30.0):
    model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    base_url = base_url or os.environ.get("JUDGE_BASE_URL")
    api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    def reward_llm_judge(prompts, completions, **kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        scores = []
        for prompt, comp in zip(prompts, completions):
            user_msg = f"QUESTION:\n{_prompt_text(prompt)}\n\nMODEL ANSWER:\n{_text(comp)}\n\nScore (0-10):"
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": _JUDGE_SYSTEM},
                              {"role": "user", "content": user_msg}],
                    max_tokens=max_tokens, temperature=0.0, stream=True, timeout=timeout,
                )
                text = ""
                for chunk in resp:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        text += delta
                m = re.search(r"\d+(?:\.\d+)?", text)
                raw = float(m.group(0)) if m else 0.0
                scores.append(weight * max(0.0, min(raw, 10.0)) / 10.0)
            except Exception:
                scores.append(0.0)  # never raise inside a reward: it aborts the GRPO step
        return scores

    reward_llm_judge.__name__ = "reward_llm_judge"
    return reward_llm_judge


def build_reward_funcs(reward_mode="rule", judge_model=None):
    if reward_mode == "rule":
        return [reward_verifiable]
    if reward_mode == "llm":
        return [make_llm_judge_reward(judge_model)]
    if reward_mode == "hybrid":
        return [reward_verifiable, make_llm_judge_reward(judge_model)]
    raise ValueError(f"unknown reward_mode {reward_mode!r}")


_SHARED = os.environ.get("GRPO_SHARED_DIR", "./shared")


def parse_args():
    p = argparse.ArgumentParser(description="GRPO LoRA training with Unsloth + TRL (one GPU, colocated vLLM)")
    p.add_argument("--model", default=_SHARED + "/models/unsloth--Meta-Llama-3.1-8B-Instruct", help="HF checkpoint directory")
    p.add_argument("--data", default=_SHARED + "/data/grpo_prompts.jsonl", help="JSONL with `messages` and `ground_truth` per row")
    p.add_argument("--output-dir", required=True, help="adapter (final_model/), rollouts/ and bench_steps.json go here")
    p.add_argument("--steps", type=int, default=22, help="total optimizer steps (warm-up + measured)")
    p.add_argument("--warmup-steps", type=int, default=2, help="leading steps the record writer drops")
    p.add_argument("--prompt-batch-size", type=int, default=32, help="prompts per step")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt")
    p.add_argument("--per-device-bs", type=int, default=2, help="micro-batch (completions per fwd/bwd)")
    p.add_argument("--max-tokens", type=int, default=1024, help="max completion length")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="share of the GPU given to the vLLM engine")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--logprob-chunks", type=int, default=None,
                   help="Unsloth `unsloth_grpo_mini_batch`: chunks for the old-logprob forward over the generation "
                        "batch (default: generation batch / 8, i.e. 8 rows per forward)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--allow-gemma4-vllm", action="store_true",
                   help="apply the gemma4 fast-inference patches (generation is corrupted for Gemma-4-31B; see README)")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--no-fast-inference", action="store_true", help="on-device HF generate instead of vLLM (debug only)")
    p.add_argument("--verify", action="store_true", help="after training, load base+adapter in a fresh process and generate")
    p.add_argument("--verify-only", action="store_true", help="skip training; run the load-and-generate check on --output-dir/final_model")
    p.add_argument("--verify-prompt", action="append", default=None, help="prompt for --verify (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the resolved run configuration, then exit")
    return p.parse_args()


def resolve(args) -> dict:
    gen_batch = args.prompt_batch_size * args.group_size
    if gen_batch % args.per_device_bs:
        sys.exit(f"generation batch {gen_batch} must be a multiple of --per-device-bs {args.per_device_bs}")
    if args.logprob_chunks is None:
        args.logprob_chunks = max(1, gen_batch // 8)
    return {"generation_batch_size": gen_batch, "gradient_accumulation_steps": gen_batch // args.per_device_bs,
            "steps_per_generation": gen_batch // args.per_device_bs, "lora_target_modules": LORA_TARGET_MODULES,
            "unsloth_grpo_mini_batch": args.logprob_chunks, "unsloth_logit_chunk_multiplier": 4,
            "env": {"UNSLOTH_GRPO_SEQ_PACKING": os.environ.get("UNSLOTH_GRPO_SEQ_PACKING")}}


def run_verify(args, launcher: str):
    cmd = [sys.executable, os.path.abspath(launcher), "--verify-only", "--model", args.model,
           "--output-dir", args.output_dir]
    for pr in args.verify_prompt or []:
        cmd += ["--verify-prompt", pr]
    subprocess.run(cmd, check=True, cwd=HERE)


def setup_logging():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="[grpo] %(asctime)s %(message)s", force=True)
