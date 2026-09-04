"""Unsloth LoRA/QLoRA SFT + GRPO trainer for Gemma-4-31B (DDP, one replica per GPU) — see readme_unsloth.md."""

# Unsloth must be imported before torch/transformers/trl — it patches them at import time.
from unsloth import FastModel

import os
import sys
import argparse
import logging

import torch
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("dev.env")

# Line-buffer stdio so log lines survive nohup redirection.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass


class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# force=True replaces the root handler `import unsloth` already installed.
logging.basicConfig(level=logging.INFO, format="[unsloth-sft] %(asctime)s %(levelname)s: %(message)s",
                    handlers=[_FlushStreamHandler(sys.stdout)], force=True)

def parse_args():
    p = argparse.ArgumentParser(description="Unsloth LoRA/QLoRA SFT or GRPO for Gemma-4-31B (DDP, no DeepSpeed)")
    # Training mode
    p.add_argument("--train_mode", type=str, default="sft", choices=["sft", "grpo"],
                   help="'sft' (default): supervised fine-tuning. 'grpo': Group Relative Policy "
                        "Optimization (RL) — regenerates completions per prompt each step and "
                        "scores them with reward functions; needs a prompt+answer+family dataset.")
    # Paths & model
    p.add_argument("--model_name", type=str, default="unsloth/gemma-4-31B-it")
    p.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Full output dir override. If unset, built as experiment_root/<RUN_ID>/output_subdir.")
    p.add_argument("--experiment_root", type=str, default="experiments",
                   help="Root under which run outputs are created.")
    p.add_argument("--output_subdir", type=str, default="gsma_sft_lora_unsloth",
                   help="Subdir name appended under experiment_root/<RUN_ID>.")
    # Model load / quantization
    p.add_argument("--max_seq_length", type=int, default=3100,
                   help="Over-length rows are dropped (not truncated).")
    p.add_argument("--load_in_4bit", action="store_true", help="On-the-fly 4-bit (QLoRA) or load a 4-bit ckpt.")
    p.add_argument("--load_in_8bit", action="store_true", help="On-the-fly 8-bit. Mutually exclusive with --load_in_4bit.")
    p.add_argument("--full_finetuning", action="store_true", help="Full FT instead of LoRA (heavy; usually off).")
    p.add_argument("--device_map", type=str, default=None,
                   help="'balanced' splits ONE replica across GPUs (launch as a single `python` process). "
                        "Leave unset for DDP (one replica per process).")
    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="all-linear",
                   help="'all-linear' (default) or a comma-separated list of module names.")
    # Trainer / hyperparameters
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_acc_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 = smoke test (overrides epochs).")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--optim", type=str, default="adamw_8bit")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])
    p.add_argument("--save_steps", type=int, default=250,
                   help="Checkpoint interval when --save_strategy steps (mid-run safety on a 1-epoch run).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_prompt", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable per-row masking policy (use --no-mask_prompt to disable). When ON "
                        "(default): rows with unmask=True get full-sequence loss; rows with unmask=False "
                        "(or no flag) get completion-only (prompt masked). When OFF: ALL rows get "
                        "full-sequence loss regardless of flag.")
    p.add_argument("--response_marker", type=str, default="<|turn>model\n",
                   help="Chat-template marker that opens the assistant turn. Completion-only rows "
                        "mask loss up to and including this marker.")
    p.add_argument("--loss_weight", type=float, default=1.0,
                   help="Per-token loss multiplier applied to COMPLETION-ONLY (unmask=False) rows. "
                        "Counteracts token-mass imbalance when short completion-only rows are mixed with long "
                        "full-sequence rows (the completion-only side contributes far fewer supervised tokens, "
                        "so its gradient signal is diluted). A weight >1 up-weights those rows. "
                        "Default 1.0 = no weighting (stock loss path).")
    p.add_argument("--report_to", type=str, default="none")
    # Data sizing
    p.add_argument("--eval_samples", type=int, default=1000, help="Held-out eval split size.")
    p.add_argument("--test_mode", action="store_true", help="Use only the first --test_mode_count rows.")
    p.add_argument("--test_mode_count", type=int, default=10000)
    # GRPO-only hyperparameters (ignored when --train_mode sft).
    p.add_argument("--num_generations", type=int, default=8,
                   help="[GRPO] Rollouts sampled per prompt. Larger => smoother advantages but "
                        "more memory/step time. The effective batch (per_device_batch * grad_acc * "
                        "world_size) MUST be divisible by this.")
    p.add_argument("--max_prompt_length", type=int, default=1024,
                   help="[GRPO] Prompts longer than this are dropped (kept off the completion budget).")
    p.add_argument("--max_completion_length", type=int, default=None,
                   help="[GRPO] Max tokens generated per rollout. Default: max_seq_length - max_prompt_length.")
    p.add_argument("--grpo_temperature", type=float, default=1.0,
                   help="[GRPO] Sampling temperature for rollouts (needs >0 for group diversity).")
    p.add_argument("--turn_end_token", type=str, default="<turn|>",
                   help="[GRPO] End-of-turn token set as EOS so rollouts stop at the end of the "
                        "model's reply (Gemma-4's processor reports a different EOS).")
    p.add_argument("--reward_mode", type=str, default="rule", choices=["rule", "llm", "hybrid"],
                   help="[GRPO] 'rule': verifiable rewards from --reward_module. "
                        "'llm': external-API LLM judge. 'hybrid': both.")
    p.add_argument("--reward_module", type=str, default="grpo_rewards",
                   help="[GRPO] Python module exposing build_reward_funcs(reward_mode, judge_model). "
                        "'grpo_rewards' (default) is the generic template; drop in your own module "
                        "to customize.")
    p.add_argument("--judge_model", type=str, default="claude-sonnet-5",
                   help="[GRPO] Model id for the LLM-judge reward (only used with llm/hybrid).")
    p.add_argument("--fast_inference", action="store_true",
                   help="[GRPO] Use vLLM (colocated in-process via Unsloth) for rollouts instead of "
                        "the on-device HF generate() path. Much faster rollouts, shares the same GPU/"
                        "VRAM as training. Requires vLLM installed AND vLLM support for the model arch "
                        "(gemma4). If unset (default), GRPO generates on-device.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                   help="[GRPO] vLLM KV-cache VRAM fraction (only used with --fast_inference).")
    # GRPO vLLM SERVER mode: rollouts run on a SEPARATE `trl vllm-serve` process (its own
    # GPU), sidestepping Unsloth's colocated allowlist that blocks gemma4. TRL syncs the
    # updated LoRA to the server each step; training uses the REMAINING GPUs.
    p.add_argument("--vllm_server", action="store_true",
                   help="[GRPO] Route rollouts to an external `trl vllm-serve` server "
                        "(vllm_mode=server). Start the server first on a dedicated GPU.")
    p.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    p.add_argument("--vllm_server_port", type=int, default=8000)
    return p.parse_args()


def run_grpo(args, model, tokenizer, output_dir, world_size, is_main):
    """GRPO (RL) training path — rollouts, rewards, group-relative advantages, LoRA update."""
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    import importlib
    build_reward_funcs = importlib.import_module(args.reward_module).build_reward_funcs

    _tok = getattr(tokenizer, "tokenizer", tokenizer)  # underlying fast tokenizer

    # EOS fix: point the tokenizer at the real end-of-turn token so rollouts stop.
    TURN_END = args.turn_end_token
    turn_end_id = _tok.convert_tokens_to_ids(TURN_END)
    if turn_end_id is not None and turn_end_id != _tok.unk_token_id:
        old_eos = _tok.eos_token
        _tok.eos_token = TURN_END
        if is_main:
            logging.info(f"[GRPO] EOS fix: eos_token {old_eos!r} -> {TURN_END!r} "
                         f"(id={turn_end_id}) so rollouts stop at end-of-turn.")
    elif is_main:
        logging.warning(f"[GRPO] Could not resolve {TURN_END!r} to a token id; leaving "
                        f"eos_token={_tok.eos_token!r}. Verify rollouts stop correctly.")

    # Data: expects prompt/answer/family rows.
    raw = load_dataset("json", data_files=args.train_file, split="train")
    required = {"prompt", "answer", "family"}
    missing = required - set(raw.column_names)
    if missing:
        raise SystemExit(
            f"[GRPO] --train_file {args.train_file} is missing columns {missing}. "
            f"GRPO needs a prompt/answer/family dataset "
            f"(found columns: {raw.column_names}).")
    if args.test_mode:
        raw = raw.select(range(min(args.test_mode_count, len(raw))))

    # Normalize prompt content to typed parts so both older and newer TRL accept it.
    def _typed_content(ex):
        new = []
        for m in ex["prompt"]:
            c = m["content"]
            if isinstance(c, str):
                new.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
            else:
                new.append(m)
        return {"prompt": new}
    raw = raw.map(_typed_content, desc="[GRPO] Normalize prompt content to typed parts")

    # Drop prompts longer than max_prompt_length so they don't eat the completion budget.
    def _prompt_len(ex):
        text = tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        return len(_tok(text=text, add_special_tokens=False)["input_ids"])
    n_before = len(raw)
    raw = raw.filter(lambda ex: _prompt_len(ex) <= args.max_prompt_length,
                     desc="[GRPO] Prompt-length filter")
    if is_main:
        from collections import Counter
        fam_counts = Counter(raw["family"])
        logging.info(f"[GRPO] Data ready: {len(raw)} prompts "
                     f"(dropped {n_before - len(raw)} over max_prompt_length={args.max_prompt_length}); "
                     f"family mix={dict(fam_counts)}")

    # Effective batch must be divisible by num_generations.
    eff_batch = args.batch_size * args.grad_acc_steps * world_size
    if eff_batch % args.num_generations != 0:
        raise SystemExit(
            f"[GRPO] effective batch (batch_size {args.batch_size} * grad_acc {args.grad_acc_steps} "
            f"* world_size {world_size} = {eff_batch}) must be divisible by --num_generations "
            f"{args.num_generations}. Adjust one of them.")

    max_completion_length = args.max_completion_length or (args.max_seq_length - args.max_prompt_length)

    reward_funcs = build_reward_funcs(args.reward_mode, args.judge_model)
    if is_main:
        logging.info(f"[GRPO] reward_mode={args.reward_mode} "
                     f"reward_funcs={[f.__name__ for f in reward_funcs]} "
                     f"num_generations={args.num_generations} "
                     f"max_prompt_length={args.max_prompt_length} "
                     f"max_completion_length={max_completion_length} "
                     f"lr={args.learning_rate} temperature={args.grpo_temperature}")

    # Only pass max_prompt_length if this TRL version still accepts it (removed by 0.29).
    import inspect as _inspect
    _grpo_params = set(_inspect.signature(GRPOConfig.__init__).parameters)
    _maybe_prompt_len = ({"max_prompt_length": args.max_prompt_length}
                         if "max_prompt_length" in _grpo_params else {})

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_generations=args.num_generations,
        **_maybe_prompt_len,
        max_completion_length=max_completion_length,
        temperature=args.grpo_temperature,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        report_to=args.report_to,
        # Truncated completions carry no clean reward signal; mask them out.
        mask_truncated_completions=True,
        use_vllm=args.fast_inference or args.vllm_server,
        **({"vllm_mode": "server",
            "vllm_server_host": args.vllm_server_host,
            "vllm_server_port": args.vllm_server_port} if args.vllm_server else {}),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=raw,
    )

    if is_main:
        logging.info(f"[GRPO] Starting training (max_steps={args.max_steps}, "
                     f"epochs={args.num_train_epochs}, effective_batch={eff_batch}). "
                     f"Watch the `reward` column rise and `frac_reward_zero_std` stay well "
                     f"below 1 (near 1 => groups agree, no learning: raise --num_generations).")
    trainer.train()

    if is_main:
        final_dir = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logging.info(f"[GRPO] Saved adapter + tokenizer to {final_dir}")


def main():
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Pass only ONE of --load_in_4bit / --load_in_8bit.")

    # GRPO wants a much smaller LR; drop to the RL default if the SFT default was left untouched.
    if args.train_mode == "grpo" and abs(args.learning_rate - 2e-4) < 1e-12:
        args.learning_rate = 5e-6

    # DDP rank (torchrun/accelerate set these).
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    # Pin each DDP process to its own GPU before loading.
    _ddp = world_size > 1 and args.device_map is None
    if _ddp:
        torch.cuda.set_device(local_rank)

    # RUN_ID comes from the env so all ranks share the same output dir.
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime, timezone
        run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.experiment_root, run_id, args.output_subdir)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {output_dir}")

    if is_main:
        logging.info(f"Loading via Unsloth FastModel: {args.model_name} "
                     f"(4bit={args.load_in_4bit} 8bit={args.load_in_8bit} "
                     f"device_map={args.device_map} world_size={world_size})")

    # Load model + tokenizer through Unsloth.
    load_kwargs = dict(
        model_name=args.model_name,
        dtype=None,                       # None => auto (bf16 on H100)
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
    )
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    elif _ddp:
        # One full replica per process, pinned to this rank's GPU.
        load_kwargs["device_map"] = {"": f"cuda:{local_rank}"}
    if args.train_mode == "grpo" and args.fast_inference:
        # Unsloth's colocated vLLM allowlist excludes gemma-4; fail fast with an
        # actionable message (full investigation in readme_unsloth.md).
        _mn = args.model_name.lower()
        if "gemma-4" in _mn or "gemma4" in _mn:
            raise SystemExit(
                "[GRPO] --fast_inference is NOT supported for gemma-4 with unsloth 2026.8.9: "
                "Unsloth's colocated vLLM allowlist (VLLM_SUPPORTED_VLM) excludes the multimodal "
                "gemma4 arch, and the text_only escape hatch crashes on this checkpoint. "
                "Re-run WITHOUT --fast_inference (on-device rollouts). See readme_unsloth.md "
                "§9 (GRPO limitations) for the full investigation.")
        load_kwargs["fast_inference"] = True
        load_kwargs["max_lora_rank"] = args.lora_r
        load_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        if is_main:
            logging.info(f"[GRPO] fast_inference=ON (vLLM colocated): "
                         f"max_lora_rank={args.lora_r} gpu_memory_utilization={args.gpu_memory_utilization}")
    try:
        model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    except Exception as e:
        logging.exception(f"Unsloth FastModel.from_pretrained FAILED for {args.model_name}: "
                          f"{type(e).__name__}: {e}")
        raise SystemExit(1)
    if is_main:
        logging.info("Unsloth load OK — fast path active.")

    # Attach LoRA adapters (skip if full FT).
    if not args.full_finetuning:
        tgt = args.lora_target_modules
        if tgt.strip() == "all-linear":
            tgt = None  # Unsloth's default covers all linear layers
        elif "," in tgt:
            tgt = [m.strip() for m in tgt.split(",") if m.strip()]
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
            target_modules=tgt,
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC (always on)
        )
        if is_main:
            logging.info(f"LoRA attached: r={args.lora_r} alpha={args.lora_alpha} "
                         f"dropout={args.lora_dropout} targets={args.lora_target_modules} "
                         f"grad_checkpointing=unsloth")

    # GRPO path: self-contained; the SFT path below is untouched for --train_mode sft.
    if args.train_mode == "grpo":
        run_grpo(args, model, tokenizer, output_dir, world_size, is_main)
        return

    # Data: pre-tokenize with per-row labels keyed by the `unmask` flag (mixed masking).
    raw = load_dataset("json", data_files=args.train_file, split="train")
    if args.test_mode:
        raw = raw.select(range(min(args.test_mode_count, len(raw))))
    if "messages" not in raw.column_names:
        raise SystemExit(f"Expected a `messages` column; found {raw.column_names}.")
    has_unmask = "unmask" in raw.column_names
    if is_main:
        logging.info(f"mask_prompt={args.mask_prompt}; per-row `unmask` column present={has_unmask}. "
                     f"Policy: unmask=True -> full-seq loss; unmask=False (or absent) -> completion-only.")

    _tok = getattr(tokenizer, "tokenizer", tokenizer)  # underlying fast tokenizer

    # Response-marker token ids; completion-only rows mask up to & including this marker.
    RESPONSE_MARKER = args.response_marker
    marker_ids = _tok(text=RESPONSE_MARKER, add_special_tokens=False)["input_ids"]
    mlen = len(marker_ids)

    def _find_marker_end(ids):
        """Index just past the LAST occurrence of the response marker, or None."""
        for i in range(len(ids) - mlen, -1, -1):
            if ids[i:i + mlen] == marker_ids:
                return i + mlen
        return None

    # Counters for a sanity summary.
    stats = {"full": 0, "completion": 0, "over_len": 0}

    def _tokenize(examples):
        n = len(examples["messages"])
        unmask_flags = examples["unmask"] if has_unmask else [False] * n
        out_ids, out_labels, out_weights = [], [], []
        for i in range(n):
            msgs = examples["messages"][i]
            # add_generation_prompt=False: full conversation incl. the assistant target.
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = _tok(text=text, add_special_tokens=False)["input_ids"]
            unmask = bool(unmask_flags[i])
            if unmask or not args.mask_prompt:
                # Full-sequence loss.
                labels = list(ids)
                weight = 1.0
                stats["full"] += 1
            else:
                # Completion-only: mask up to & including the response marker.
                end = _find_marker_end(ids)
                if end is None:
                    # A completion-only row must contain the marker.
                    raise SystemExit(
                        f"[MASKING] Completion-only row is missing response marker "
                        f"{RESPONSE_MARKER!r} (ids={marker_ids}). Rendered head: {text[:200]!r}"
                    )
                labels = [-100] * end + list(ids[end:])
                weight = args.loss_weight
                stats["completion"] += 1
            out_ids.append(ids)
            out_labels.append(labels)
            out_weights.append(weight)
        return {"input_ids": out_ids, "labels": out_labels, "sample_weight": out_weights}

    tokenized = raw.map(_tokenize, batched=True, remove_columns=raw.column_names,
                        desc="Tokenize + build per-row labels")

    # Drop rows longer than max_seq_length rather than silently truncating them.
    def _len_ok(ex):
        return len(ex["input_ids"]) <= args.max_seq_length
    n_before = len(tokenized)
    tokenized = tokenized.filter(_len_ok, desc="Length filter")
    # attention_mask is all-ones (no padding yet; the collator pads per batch).
    tokenized = tokenized.map(lambda ex: {"attention_mask": [1] * len(ex["input_ids"])},
                              desc="Attention mask")

    split = tokenized.train_test_split(test_size=min(args.eval_samples, max(1, len(tokenized) // 20)),
                                       seed=args.seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    if is_main:
        logging.info(f"Data ready: train={len(train_dataset)} eval={len(eval_dataset)} "
                     f"(dropped {n_before-len(tokenized)} over-length @ max_seq_length={args.max_seq_length}) "
                     f"masking_counts(full={stats['full']}, completion_only={stats['completion']}) "
                     f"cols={train_dataset.column_names}")

    # Trainer (Unsloth-patched TRL); pre-tokenized rows skip TRL's text prep.
    from trl import SFTTrainer, SFTConfig
    from transformers import DataCollatorForSeq2Seq
    import torch.nn.functional as F

    weighting_on = args.mask_prompt and abs(args.loss_weight - 1.0) > 1e-9

    # Carries the scalar per-row sample_weight alongside the padded token tensors.
    class WeightAwareCollator:
        def __init__(self, base):
            self.base = base
        def __call__(self, features):
            weights = [f.pop("sample_weight", 1.0) for f in features]
            batch = self.base(features)
            batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
            return batch

    # Applies per-example loss weights; upcasts one example's logits at a time to avoid OOM.
    class WeightedLossTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weights = inputs.pop("sample_weight")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (B, T, V)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].to(shift_logits.device)
            weights = weights.to(shift_logits.device)
            B = shift_logits.shape[0]
            total_loss = shift_logits.new_zeros(())
            total_denom = shift_logits.new_zeros(())
            for b in range(B):
                lb = shift_labels[b]
                mask = lb != -100
                if not mask.any():
                    continue
                lg = shift_logits[b].float()               # (T-1, V) — one example only
                tl = F.cross_entropy(lg, lb.clamp_min(0), reduction="none")
                tl = tl * mask.float()
                w = weights[b]
                total_loss = total_loss + tl.sum() * w
                total_denom = total_denom + mask.float().sum() * w
            loss = total_loss / total_denom.clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss
    
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        report_to=args.report_to,
        max_length=args.max_seq_length,
        packing=False,
        # Rows are already tokenized; tell TRL not to re-render a text field.
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    # Pad input_ids with pad_token and labels with -100 (so pad positions carry no loss).
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=_tok,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    # Only use the custom weighted-loss path when a non-trivial weight is set; otherwise
    # keep the fast default path (Unsloth's fused CE) and just drop the weight column.
    if weighting_on:
        train_dataset = train_dataset  # sample_weight stays; collator forwards it
        eval_dataset = eval_dataset
        collator = WeightAwareCollator(base_collator)
        TrainerCls = WeightedLossTrainer
        if is_main:
            logging.info(f"Per-row loss weighting ON: completion-only rows weighted "
                         f"×{args.loss_weight} vs full-sequence rows ×1.0 (custom compute_loss).")
    else:
        # Strip the weight column so the stock collator/loss path is unaffected.
        train_dataset = train_dataset.remove_columns(
            [c for c in ["sample_weight"] if c in train_dataset.column_names])
        eval_dataset = eval_dataset.remove_columns(
            [c for c in ["sample_weight"] if c in eval_dataset.column_names])
        collator = base_collator
        TrainerCls = SFTTrainer
        if is_main:
            logging.info("Per-row loss weighting OFF (loss_weight=1.0): stock loss path.")
    trainer = TrainerCls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        data_collator=collator,
    )

    if is_main:
        logging.info(f"Starting training (max_steps={args.max_steps}, "
                     f"epochs={args.num_train_epochs}, effective_batch="
                     f"{args.batch_size * args.grad_acc_steps * world_size})")
    trainer.train()

    # Save the LoRA adapter (rank 0 only).
    if is_main:
        final_dir = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logging.info(f"Saved adapter + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
