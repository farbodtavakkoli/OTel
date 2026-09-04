"""Data/eval/loss helpers and DPO/GRPO dataset builders for train_llm_deepspeed.py; design notes in readme_deepspeed.md."""

import logging
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainerCallback


def _message_content_to_text(message):
    """Coerce a chat message's `content` to a string ("" for None/non-dict)."""
    content = message.get("content", "") if isinstance(message, dict) else ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _validate_messages_example(example):
    """Check one record against the `messages` contract; return None if valid, else a short reason string."""
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
    """Pick up to `sample_size` evenly-spaced indices over [0, total_count)."""
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
    """Validate evenly-spaced rows against the `messages` contract; raise ValueError on malformed rows."""
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
    """Resolve the eval split to an absolute, distributed-safe row count."""
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
    """Log token supervision stats over a sample; raise ValueError if any sampled row has zero supervised tokens."""
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


def _stable_causal_lm_loss(outputs, labels, num_items_in_batch=None):
    """Completion-only causal-LM loss with a clamped denominator so zero-supervised micro-batches yield 0.0, not NaN."""
    shift_logits = outputs.logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    valid_tokens = (shift_labels != -100).sum()
    denominator = valid_tokens if num_items_in_batch is None else num_items_in_batch
    if not torch.is_tensor(denominator):
        denominator = torch.tensor(denominator, device=shift_logits.device)
    else:
        denominator = denominator.to(shift_logits.device)

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss / denominator.clamp_min(1)


def _process_with_chat_template(messages, tokenizer, mask_user_prompt=True):
    """Tokenize a conversation via the tokenizer's chat template with completion-only loss masking."""
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
        # Defensive fallback: mask the shared leading token prefix
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


def get_datasets(
    path,
    tokenizer,
    seed,
    max_token_length,
    test_mode_count,
    preflight_sample_size,
    max_eval_samples=None,
    test_size=0.01,
    sample_fraction=1.0,
    max_samples=None,
    test_mode=False,
    mask_user_prompt=True,
    num_proc=8,
    min_eval_samples=1,
):
    """Load, validate, tokenize, length-filter and split a `messages` dataset; returns (train, eval)."""
    full_ds = load_dataset("json", data_files=path, split="train")
    initial_count = len(full_ds)

    # Messages-only by contract; any other shape is a hard error (clean upstream)
    if "messages" not in full_ds.column_names:
        raise ValueError(
            "Dataset must be in canonical `messages` format "
            "({\"messages\": [{\"role\", \"content\"}, ...]}). "
            f"Found columns {full_ds.column_names}. Clean/normalize the data upstream "
            "before training."
        )

    _run_messages_preflight(full_ds, preflight_sample_size)

    if test_mode:
        logging.info("TEST MODE: Limiting dataset for verification")
        full_ds = full_ds.select(range(min(test_mode_count, len(full_ds))))

    if sample_fraction < 1.0:
        sample_size = int(len(full_ds) * sample_fraction)
        logging.info(f"Subsampling dataset to {sample_fraction:.2%} ({sample_size} samples)")
        full_ds = full_ds.shuffle(seed=seed).select(range(sample_size))

    if max_samples is not None and max_samples < len(full_ds):
        logging.info(f"Limiting dataset to {max_samples} samples")
        full_ds = full_ds.select(range(max_samples))

    full_ds = full_ds.map(
        lambda x: _process_with_chat_template(x["messages"], tokenizer, mask_user_prompt=mask_user_prompt),
        remove_columns=full_ds.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # Drop over-length examples rather than truncating
    pre_filter_count = len(full_ds)
    full_ds = full_ds.filter(
        lambda x: len(x["input_ids"]) <= max_token_length,
        num_proc=num_proc,
        desc="Dropping over-length examples",
    )
    dropped = pre_filter_count - len(full_ds)
    pct = (100.0 * dropped / pre_filter_count) if pre_filter_count else 0.0
    logging.info(
        f"Dropped {dropped}/{pre_filter_count} ({pct:.2f}%) examples exceeding "
        f"{max_token_length} tokens; {len(full_ds)} remain."
    )

    logging.info("Loaded %d training data from %s (initial %d rows)", len(full_ds), path, initial_count)

    num_test = _resolve_eval_sample_count(
        len(full_ds),
        test_size=test_size,
        max_eval_samples=max_eval_samples,
        min_eval_samples=min_eval_samples,
    )
    logging.info(f"Using {num_test} samples for evaluation")

    split_ds = full_ds.train_test_split(test_size=num_test, seed=seed)
    return split_ds["train"], split_ds["test"]


def run_generation_sanity_check(model_dir=None, eval_dataset=None, tokenizer=None, num_samples=2, max_new_tokens=256, device=None, model=None):
    """Decode a few eval prompts and log prompt / generation / reference side by side (rank-0 only)."""
    if num_samples <= 0:
        return
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        logging.info("--- Generation Sanity Check (loading %s) ---", model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
    else:
        logging.info("--- Generation Sanity Check (using preloaded model) ---")
        model = model.to(device)
    model.eval()

    sample_indices = _sample_indices(len(eval_dataset), num_samples)
    for order, dataset_index in enumerate(sample_indices):
        sample = eval_dataset[dataset_index]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        num_masked = sum(1 for label in labels if label == -100)
        prompt_ids = input_ids[:num_masked]
        reference_ids = [token for token, label in zip(input_ids, labels) if label != -100]
        if not prompt_ids:
            # Fully-unmasked row (mask_user_prompt=False) — nothing to prompt with
            continue

        input_tensor = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = generated[0][len(prompt_ids):].tolist()

        logging.info("[gen %d/%d] eval index=%d", order + 1, len(sample_indices), dataset_index)
        logging.info("  prompt tail (last 200 chars): ...%s", tokenizer.decode(prompt_ids)[-200:])
        logging.info("  generated (first 400 chars): %s", tokenizer.decode(generated_ids, skip_special_tokens=False)[:400])
        logging.info("  reference (first 400 chars): %s", tokenizer.decode(reference_ids, skip_special_tokens=False)[:400])

    logging.info("--- End Generation Sanity Check ---")


def build_deepspeed_config(zero_stage=3, offload_optimizer=False):
    """Build an in-memory DeepSpeed config dict for TrainingArguments(deepspeed=...)."""
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
    """Map a {system?, prompt, chosen, rejected} row to DPO format with a chat-templated prompt."""
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


def build_pref_dataset(pref_file, tokenizer, seed, eval_size=200, num_proc=8):
    """Load a DPO preference JSONL and return (train, eval) datasets, dropping degenerate pairs."""
    ds = load_dataset("json", data_files=pref_file, split="train")
    ds = ds.map(lambda x: _pref_to_conversational(x, tokenizer),
                remove_columns=ds.column_names, num_proc=num_proc, desc="Building DPO pairs")
    ds = ds.filter(lambda x: x["chosen"] and x["rejected"] and x["chosen"] != x["rejected"],
                   num_proc=num_proc, desc="Dropping degenerate pairs")
    if len(ds) < 2:
        raise ValueError(f"Too few usable preference pairs in {pref_file}: {len(ds)}")
    n_eval = min(eval_size, max(1, len(ds) // 10))
    split = ds.train_test_split(test_size=n_eval, seed=seed)
    return split["train"], split["test"]


def build_grpo_prompts(train_file, tokenizer, seed, max_samples=None, num_proc=8):
    """Build a GRPO prompt dataset (chat-templated `prompt` + gold `reference`) from a messages JSONL."""
    ds = load_dataset("json", data_files=train_file, split="train")

    def _to_prompt(ex):
        m = ex.get("messages") or []
        prompt_msgs = [x for x in m if x.get("role") != "assistant"]
        ref = next((x.get("content", "") for x in reversed(m) if x.get("role") == "assistant"), "")
        return {
            "prompt": tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True),
            "reference": ref,
        }

    ds = ds.map(_to_prompt, remove_columns=ds.column_names, num_proc=num_proc, desc="Building GRPO prompts")
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds


def build_qlora_bnb_config():
    """Return a BitsAndBytesConfig for 4-bit (QLoRA) base-weight quantization."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def run_custom_eval(
    model,
    tokenizer,
    test_dir,
    scorer_module="step8_score_eval",
    max_new_tokens=768,
    batch_size=16,
    device=None,
):
    """Generate on each test/<name>_eval.jsonl and score with a pluggable scorer module (rank-0 only)."""
    import importlib
    import sys

    # The scorer module may live beside the test files; retry with test_dir's parent on sys.path
    try:
        scorer = importlib.import_module(scorer_module)
    except ModuleNotFoundError:
        extra = os.path.dirname(os.path.abspath(test_dir))
        if extra not in sys.path:
            sys.path.insert(0, extra)
        scorer = importlib.import_module(scorer_module)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    was_training = model.training
    model.eval()
    # Batched decoder-only generation requires LEFT padding; restore afterwards
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = {}
    try:
        for name in scorer.EVAL_DATASETS:
            path = os.path.join(test_dir, f"{name}_eval.jsonl")
            if not os.path.exists(path):
                logging.warning("custom eval: missing %s, skipping %s", path, name)
                continue
            rows = scorer.load_eval(path)
            prompts = [
                tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                for r in rows
            ]
            completions = []
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                enc = tokenizer(
                    chunk, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Slice off the uniform left-padded prompt width per row
                gen = out[:, enc["input_ids"].shape[1] :]
                completions.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
            results[name] = scorer.score_dataset(name, rows, completions)
    finally:
        tokenizer.padding_side = saved_padding_side
        if was_training:
            model.train()
    macro = scorer.macro_average(results) if results else 0.0
    return {"per_dataset": results, "macro_avg": macro}


class EmptyCacheCallback(TrainerCallback):
    """Free the CUDA allocator cache every N optimizer steps on all ranks (0 disables)."""

    def __init__(self, every_n_steps=1):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            torch.cuda.empty_cache()


class CustomEvalCallback(TrainerCallback):
    """Run run_custom_eval() at each epoch end and log per-dataset + macro accuracy (rank-0 only, crash-safe)."""

    def __init__(self, tokenizer, test_dir, scorer_module="step8_score_eval",
                 max_new_tokens=768, batch_size=16, rank=0):
        self.tokenizer = tokenizer
        self.test_dir = test_dir
        self.scorer_module = scorer_module
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.rank = rank

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.rank != 0 or model is None:
            return
        try:
            res = run_custom_eval(
                model, self.tokenizer, self.test_dir,
                scorer_module=self.scorer_module,
                max_new_tokens=self.max_new_tokens, batch_size=self.batch_size,
            )
            logging.info("=== custom eval @ epoch %.2f ===", state.epoch or 0.0)
            for name, r in res["per_dataset"].items():
                logging.info("  %-14s acc=%.4f (%d/%d)", name, r["accuracy"], r["correct"], r["n"])
            logging.info("  MACRO AVG acc=%.4f", res["macro_avg"])
        except Exception as e:  # never let eval crash training
            logging.warning("custom eval failed @ epoch %s: %s", state.epoch, e)


def grpo_reward_funcs():
    """Return the reward functions for GRPOTrainer — a PLACEHOLDER that rewards non-empty completions; replace before real GRPO."""
    def _length_reward(prompts, completions, **kwargs):
        rewards = []
        for c in completions:
            text = c if isinstance(c, str) else (c[-1]["content"] if c else "")
            rewards.append(1.0 if text.strip() else 0.0)
        return rewards

    return [_length_reward]
