"""Reranker fine-tuner driving FlagEmbedding's own trainer, incl. LLM rerankers — see readme_reranker_flagembedding.md."""
import argparse
import os
import sys
import torch
from dotenv import load_dotenv
from utils import build_argv, gpu_banner, llm_argv, patch_transformers5_compat, resolve_path

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Reranker fine-tuner on FlagEmbedding")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-reranker-base",
                        help="Base reranker model")
    parser.add_argument("--reranker_type", type=str, default="encoder",
                        choices=["encoder", "llm", "llm_layerwise"],
                        help="encoder = cross-encoder head; llm = decoder-only LLM reranker; "
                             "llm_layerwise = LLM reranker with per-layer heads")
    parser.add_argument("--train_data", type=str, default="OTel_reranker_flagembedding_100.jsonl",
                        help="Training JSONL in FlagEmbedding format (query/pos/neg)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output dir for checkpoints and final model")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size (queries per step)")
    parser.add_argument("--train_group_size", type=int, default=4,
                        help="Passages per query: 1 positive + (n-1) sampled negatives")
    parser.add_argument("--query_max_len", type=int, default=128, help="Max query token length")
    parser.add_argument("--passage_max_len", type=int, default=256, help="Max passage token length")
    parser.add_argument("--max_len", type=int, default=512, help="Max combined length (LLM rerankers)")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="LR warmup steps (transformers 5.x dropped warmup_ratio)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--knowledge_distillation", type=str, default="False",
                        help="Distill from pos_scores/neg_scores columns in the training data")
    parser.add_argument("--query_instruction", type=str, default=None, help="Query instruction prefix")
    parser.add_argument("--passage_instruction", type=str, default=None, help="Passage instruction prefix")
    parser.add_argument("--use_lora", type=str, default="True", help="LoRA training (LLM rerankers)")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (LLM rerankers)")
    parser.add_argument("--lora_alpha", type=float, default=64, help="LoRA alpha (LLM rerankers)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (LLM rerankers)")
    parser.add_argument("--target_modules", type=str, default="q_proj k_proj v_proj o_proj",
                        help="LoRA target modules (LLM rerankers)")
    parser.add_argument("--save_merged_lora_model", type=str, default="False",
                        help="Merge the adapter into the base model when saving")
    parser.add_argument("--start_layer", type=int, default=None, help="First scored layer (llm_layerwise)")
    parser.add_argument("--head_multi", type=str, default=None, help="Use one head per layer (llm_layerwise)")
    parser.add_argument("--head_type", type=str, default=None, help="Head type (llm_layerwise)")
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        help="auto (sdpa on ROCm), sdpa, eager, or flash_attention_2")
    parser.add_argument("--bf16", action="store_true", default=True, help="Train in bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--gc_use_reentrant", type=str, default="False",
                        help="Reentrant checkpointing; must stay False under DDP or backward raises")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"],
                        help="Checkpoint strategy")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Max checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps")
    parser.add_argument("--dataloader_drop_last", type=str, default="True", help="Drop the last partial batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Trainer reporting backend")
    parser.add_argument("--deepspeed", type=str, default=None, help="Optional DeepSpeed config JSON")
    parser.add_argument("--cache_dir", type=str, default=None, help="Model cache dir override")
    parser.add_argument("--cache_path", type=str, default=None, help="Tokenized-dataset cache dir")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading the model")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional HF_HOME model-cache override")
    parser.add_argument("--extra_args", type=str, default=None,
                        help="Raw extra flags forwarded verbatim to FlagEmbedding")
    return parser.parse_args()


def load_runner(reranker_type):
    """Return the (parser dataclasses, runner) pair for the selected FlagEmbedding reranker family."""
    from FlagEmbedding.abc.finetune.reranker import (AbsRerankerDataArguments, AbsRerankerModelArguments,
                                                     AbsRerankerTrainingArguments)
    if reranker_type == "encoder":
        from FlagEmbedding.finetune.reranker.encoder_only.base import EncoderOnlyRerankerRunner
        return ((AbsRerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments),
                EncoderOnlyRerankerRunner)
    if reranker_type == "llm":
        from FlagEmbedding.finetune.reranker.decoder_only.base import (DecoderOnlyRerankerRunner,
                                                                       RerankerModelArguments)
        return ((RerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments),
                DecoderOnlyRerankerRunner)
    from FlagEmbedding.finetune.reranker.decoder_only.layerwise import (
        DecoderOnlyRerankerRunner as LayerwiseRunner, RerankerModelArguments as LayerwiseModelArguments)
    return ((LayerwiseModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments), LayerwiseRunner)


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    patch_transformers5_compat()
    from transformers import HfArgumentParser

    argv = build_argv(args, BASE_DIR)
    if args.reranker_type != "encoder":
        argv += llm_argv(args)

    dataclasses, runner_cls = load_runner(args.reranker_type)
    model_args, data_args, training_args = HfArgumentParser(dataclasses).parse_args_into_dataclasses(args=argv)

    # tf32 is a CUDA-only knob; setting it on ROCm raises inside TrainingArguments.
    if torch.version.cuda is None:
        training_args.tf32 = None

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[flagembedding] {gpu_banner()}", file=sys.stderr)
        print(f"[flagembedding] type={args.reranker_type} model={args.model_name_or_path}", file=sys.stderr)

    runner_cls(model_args=model_args, data_args=data_args, training_args=training_args).run()

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Training complete. Model saved to {resolve_path(args.output_dir, BASE_DIR)}")


if __name__ == "__main__":
    main()
