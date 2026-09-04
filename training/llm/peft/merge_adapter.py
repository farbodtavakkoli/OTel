"""Merge a trained PEFT adapter into its base model and save an HF-ready checkpoint -- see readme_peft.md."""

import os
import json
import logging
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser(description="Merge a PEFT adapter into base weights")
    p.add_argument("--adapter", type=str, required=True, help="Trained adapter dir (e.g. <output_dir>/final_adapter)")
    p.add_argument("--base_model", type=str, default=None, help="Base model id/path; defaults to base_model_name_or_path from the adapter config")
    p.add_argument("--output_dir", type=str, required=True, help="Where the merged, HF-ready model is written")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPES), help="Precision to merge and save in")
    p.add_argument("--device_map", type=str, default="cpu", help="Device map for the merge ('cpu' is safest; 'auto' is faster if the model fits in HBM)")
    p.add_argument("--max_shard_size", type=str, default="5GB", help="Shard size for the saved safetensors files")
    return p.parse_args()


def resolve_base_model(adapter_dir, override):
    """Take --base_model if given, else read it out of the adapter's config."""
    if override:
        return override
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}; pass --base_model explicitly.")
    with open(config_path, "r", encoding="utf-8") as f:
        base = json.load(f).get("base_model_name_or_path")
    if not base:
        raise ValueError(f"{config_path} has no base_model_name_or_path; pass --base_model explicitly.")
    return base


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    base_model = resolve_base_model(args.adapter, args.base_model)
    logger.info("Merging adapter %s into base model %s (%s)", args.adapter, base_model, args.dtype)

    # Full precision on purpose: LoRA deltas cannot merge into 4-bit/8-bit weights.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=DTYPES[args.dtype],
        device_map=args.device_map,
        token=hf_token,
    )

    model = PeftModel.from_pretrained(model, args.adapter, token=hf_token)
    # merge_and_unload is not in place -- the result must be reassigned.
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    # Prefer the tokenizer saved next to the adapter; fall back to the base model's.
    tokenizer_src = args.adapter if os.path.isfile(os.path.join(args.adapter, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Merged model written to %s (tokenizer from %s)", args.output_dir, tokenizer_src)
    logger.info("Load it with: AutoModelForCausalLM.from_pretrained('%s')", args.output_dir)


if __name__ == "__main__":
    main()
