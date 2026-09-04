"""RapidFire AI hyperparallel post-training — run N TRL SFT/DPO/GRPO configs concurrently on the same GPUs (see readme_rapidfire.md)."""

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv("dev.env")

from datasets import load_dataset
from rapidfireai import Experiment
from rapidfireai.automl import (
    List,
    RFDPOConfig,
    RFGRPOConfig,
    RFGridSearch,
    RFLoraConfig,
    RFModelConfig,
    RFRandomSearch,
    RFSFTConfig,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("rapidfire-posttrain")

GRPO_SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def parse_args():
    p = argparse.ArgumentParser(description="RapidFire AI multi-config SFT/DPO/GRPO over TRL")
    p.add_argument("--trainer_type", choices=["sft", "dpo", "grpo"], default="sft")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train_file", default="data/OTel_LLM_sample_10.jsonl",
                   help="JSONL; schema depends on --trainer_type. Default is the shipped chat sample.")
    p.add_argument("--eval_file", default=None, help="Optional JSONL eval split.")
    p.add_argument("--experiment_name", default="rf-posttrain", help="Must be unique per experiment.")
    # Knobs swept across configs (comma-separated -> cross-product).
    p.add_argument("--learning_rates", default="2e-4,5e-5")
    p.add_argument("--lora_r", default="8,32")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Knobs shared by every config.
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 = smoke test (overrides epochs).")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--logging_steps", type=int, default=2)
    p.add_argument("--num_generations", type=int, default=8, help="GRPO only.")
    p.add_argument("--beta", type=float, default=0.1, help="DPO/GRPO KL coefficient.")
    # Scheduling / search.
    p.add_argument("--num_chunks", type=int, default=4, help="Swap granularity; higher = finer comparison.")
    p.add_argument("--search", choices=["grid", "random"], default="grid")
    p.add_argument("--num_samples", type=int, default=4, help="Configs sampled when --search random.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# The formatting, reward, and model-creation helpers below run inside RapidFire
# worker processes, so they must stay module-level and not close over argparse state.

def sft_formatting_func(row):
    """{"messages": [...]} -> TRL prompt/completion pair (last turn is the target)."""
    messages = row["messages"]
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def grpo_formatting_func(row):
    """Chat `messages` or flat {prompt, answer} -> GRPO prompt + gold answer."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt = [{"role": "system", "content": GRPO_SYSTEM_PROMPT}]
        gold = ""
        if messages[-1].get("role") == "assistant":
            gold = messages[-1].get("content", "")
            prompt.extend(messages[:-1])
        else:
            prompt.extend(messages)
        return {"prompt": prompt, "answer": gold}
    return {
        "prompt": [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ],
        "answer": row["answer"],
    }


def _extract_xml_answer(text):
    return text.split("<answer>")[-1].split("</answer>")[0].strip()


def correctness_reward_func(prompts, completions, answer, **kwargs):
    """2.0 when the extracted answer matches the ground truth. Replace for real tasks."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


def format_reward_func(completions, **kwargs):
    """0.5 when the completion respects the <reasoning>/<answer> envelope."""
    import re

    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def create_model(model_config):
    """Must return (model, tokenizer). Called once per config by each worker."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], **model_config["model_kwargs"]
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"], **model_config["tokenizer_kwargs"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer)


def build_training_args(args, learning_rate):
    """One RF*Config (a drop-in TRL config) per swept learning rate."""
    common = dict(
        learning_rate=learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        bf16=True,
    )
    if args.max_steps > 0:
        common["max_steps"] = args.max_steps

    if args.trainer_type == "sft":
        return RFSFTConfig(max_length=args.max_length, **common)
    if args.trainer_type == "dpo":
        return RFDPOConfig(
            beta=args.beta,
            loss_type="sigmoid",
            max_length=args.max_length,
            max_prompt_length=args.max_length // 2,
            **common,
        )
    return RFGRPOConfig(
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=args.max_length // 2,
        max_completion_length=args.max_length // 2,
        **common,
    )


def main():
    args = parse_args()

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN not set (dev.env missing?); gated models will fail to download.")

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = (
        load_dataset("json", data_files=args.eval_file, split="train") if args.eval_file else None
    )
    # This repo's LLM sample carries extra columns (unmask, flow, source_*) and
    # two all-null fields. Keep only what the selected trainer actually reads.
    keep = {
        "sft": ["messages"],
        "dpo": ["prompt", "chosen", "rejected", "messages"],
        "grpo": ["prompt", "answer", "messages"],
    }[args.trainer_type]
    drop = [c for c in train_dataset.column_names if c not in keep]
    if drop:
        train_dataset = train_dataset.remove_columns(drop)
        if eval_dataset is not None:
            eval_drop = [c for c in eval_dataset.column_names if c not in keep]
            if eval_drop:
                eval_dataset = eval_dataset.remove_columns(eval_drop)
    logger.info("Loaded %d train rows for %s", len(train_dataset), args.trainer_type)

    learning_rates = [float(x) for x in args.learning_rates.split(",")]
    lora_ranks = [int(x) for x in args.lora_r.split(",")]

    # A List() of LoRA configs is a swept knob: it multiplies with the config list below.
    peft_configs = List(
        [
            RFLoraConfig(
                r=r,
                lora_alpha=2 * r,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            for r in lora_ranks
        ]
    )

    formatting_func = {"sft": sft_formatting_func, "grpo": grpo_formatting_func}.get(
        args.trainer_type
    )
    extra = {"reward_funcs": [correctness_reward_func, format_reward_func]} if args.trainer_type == "grpo" else {}
    if formatting_func is not None:
        extra["formatting_func"] = formatting_func

    config_set = List(
        [
            RFModelConfig(
                model_name=args.model_name,
                peft_config=peft_configs,
                training_args=build_training_args(args, lr),
                model_type="causal_lm",
                model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
                tokenizer_kwargs={
                    "model_max_length": args.max_length,
                    "padding_side": "right",
                    "truncation": True,
                },
                **extra,
            )
            for lr in learning_rates
        ]
    )

    trainer_type = args.trainer_type.upper()
    if args.search == "grid":
        config_group = RFGridSearch(configs=config_set, trainer_type=trainer_type)
        logger.info("Grid search: %d configs", len(learning_rates) * len(lora_ranks))
    else:
        config_group = RFRandomSearch(
            configs=config_set, trainer_type=trainer_type, num_samples=args.num_samples
        )
        logger.info("Random search: %d configs sampled", args.num_samples)

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")
    try:
        experiment.run_fit(
            config_group,
            create_model,
            train_dataset,
            eval_dataset,
            num_chunks=args.num_chunks,
            seed=args.seed,
        )
    finally:
        # Always release workers/GPU state, otherwise the next experiment name collides.
        experiment.end()
    logger.info("Experiment %s complete. Dashboard: http://localhost:8853", args.experiment_name)


if __name__ == "__main__":
    main()
