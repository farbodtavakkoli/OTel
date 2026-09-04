"""Thin launcher for Tevatron dense-retriever training: resolve the ROCm/CUDA knobs, then shell out to tevatron.retriever.driver.train (see readme_embedding_tevatron.md)."""

import argparse
import logging
import sys

from dotenv import load_dotenv

from utils import (build_env, build_train_command, prepare_output_dir, resolve_attn, run,
                   summarize, visible_device_count, warn_on_environment)

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("training/embedding/tevatron")


def parse_args():
    """Parse every tunable this launcher forwards to the Tevatron training driver."""
    parser = argparse.ArgumentParser(description="Train a Tevatron dense retriever")

    parser.add_argument("--model_name_or_path", default="BAAI/bge-small-en-v1.5",
                        help="Backbone model id or path")
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean", "eos", "last"],
                        help="Representation pooling; cls for BERT encoders, eos for decoder LMs")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="L2-normalize query and passage vectors (cosine similarity)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false",
                        help="Disable normalization (raw dot product)")
    parser.add_argument("--append_eos_token", action="store_true",
                        help="Append EOS to query and passage — required by the RepLLaMA/eos-pooling recipe")
    parser.add_argument("--temperature", type=float, default=0.02,
                        help="Softmax temperature for the contrastive loss")
    parser.add_argument("--attn_implementation", default="auto",
                        choices=["auto", "sdpa", "eager", "flash_attention_2"],
                        help="Attention kernel; auto picks sdpa on ROCm/CPU and flash_attention_2 on CUDA")

    parser.add_argument("--dataset_name", default="json",
                        help="HF dataset name, or 'json' to read --dataset_path")
    parser.add_argument("--dataset_path", default="OTel_tevatron_sample_100.jsonl",
                        help="Local training JSONL in Tevatron format (see convert_data.py)")
    parser.add_argument("--dataset_config", default=None, help="HF dataset config name")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to train on")
    parser.add_argument("--corpus_name", default=None,
                        help="HF corpus name, for the docid-referencing data format")
    parser.add_argument("--corpus_path", default=None,
                        help="Local corpus JSONL, for the docid-referencing data format")
    parser.add_argument("--query_prefix", default=None, help="Instruction prefix prepended to queries")
    parser.add_argument("--passage_prefix", default=None, help="Instruction prefix prepended to passages")
    parser.add_argument("--query_max_len", type=int, default=64, help="Query truncation length")
    parser.add_argument("--passage_max_len", type=int, default=192, help="Passage truncation length")
    parser.add_argument("--train_group_size", type=int, default=6,
                        help="Passages per query: 1 positive + (n-1) hard negatives")

    parser.add_argument("--output_dir", default="outputs/tevatron_run",
                        help="Where checkpoints and the final encoder are written")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete a non-empty output_dir before training")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size (queries)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=float, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Hard cap on optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (use ~1e-4 with LoRA)")
    parser.add_argument("--lr_scheduler_type", default=None, help="LR schedule, e.g. linear or cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup fraction of total steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Train in bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Train in float16 instead of bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations to trade compute for memory")
    parser.add_argument("--dataloader_num_workers", type=int, default=None, help="Dataloader worker processes")
    parser.add_argument("--logging_steps", type=int, default=1, help="Steps between loss log lines")
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"],
                        help="Checkpoint cadence")
    parser.add_argument("--save_steps", type=int, default=None, help="Steps between checkpoints")
    parser.add_argument("--report_to", default="none", help="Trainer reporting integration, e.g. tensorboard")
    parser.add_argument("--deepspeed", default=None, help="Path to a DeepSpeed config JSON")

    parser.add_argument("--grad_cache", action="store_true",
                        help="Enable GradCache — chunked re-forward, so batch size stops being memory-bound")
    parser.add_argument("--gc_q_chunk_size", type=int, default=8, help="GradCache query chunk size")
    parser.add_argument("--gc_p_chunk_size", type=int, default=16, help="GradCache passage chunk size")

    parser.add_argument("--lora", action="store_true", help="Parameter-efficient fine-tuning with LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules",
                        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
                        help="Comma-separated LoRA target module names")

    parser.add_argument("--devices", default=None,
                        help="GPU ids for HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES, e.g. '2,3'")
    parser.add_argument("--master_port", type=int, default=29820,
                        help="torchrun rendezvous port; 29500 collides on shared hosts")
    parser.add_argument("--dry_run", action="store_true", help="Print the command and exit")
    return parser.parse_args()


def main():
    """Resolve the launch configuration, print a summary, and run the Tevatron driver."""
    args = parse_args()

    num_gpus = visible_device_count(args.devices)
    attn = resolve_attn(args.attn_implementation)
    warn_on_environment(args)

    cmd = build_train_command(args, num_gpus, attn)
    summarize(args, cmd, num_gpus, attn)

    if args.dry_run:
        logger.info("--dry_run set; not launching")
        return

    prepare_output_dir(args.output_dir, args.overwrite)

    code = run(cmd, build_env(args.devices))
    if code != 0:
        logger.error("tevatron.retriever.driver.train exited with code %s", code)
        sys.exit(code)
    logger.info("training finished; encoder written to %s", args.output_dir)


if __name__ == "__main__":
    main()
