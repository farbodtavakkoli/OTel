"""ColBERT late-interaction (multi-vector) fine-tuner built on PyLate — see readme_embedding_pylate.md."""

import argparse
import os
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv
from pylate import utils as pylate_utils
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import utils

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")

# Registry-backed defaults; any field is overridable by the matching CLI flag.
DEFAULT_CFG = {
    "embedding_dim": 128,
    "query_length": 32,
    "document_length": 180,
    "batch_size": 16,
    "epochs": 1,
    "lr": 3e-6,
    "n_negatives": 1,
}

MODELS = {
    # BERT-style bi-encoder backbones — PyLate appends a fresh Dense projection to embedding_dim.
    "BAAI/bge-small-en-v1.5": {"batch_size": 16, "lr": 3e-6},
    "bert-base-uncased": {"batch_size": 8, "lr": 3e-6},
    "sentence-transformers/all-MiniLM-L6-v2": {"batch_size": 32, "lr": 3e-6},

    # Existing ColBERT checkpoints — the projection ships with the checkpoint.
    "lightonai/GTE-ModernColBERT-v1": {"batch_size": 8, "lr": 1e-6, "document_length": 300},
    "colbert-ir/colbertv2.0": {"batch_size": 16, "lr": 1e-6},
    "answerdotai/answerai-colbert-small-v1": {"batch_size": 16, "lr": 1e-6},
}

def parse_args():
    """Parse CLI arguments; registry-backed flags default to None, meaning "use the registry value"."""
    parser = argparse.ArgumentParser(description="ColBERT late-interaction fine-tuner (PyLate)")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-small-en-v1.5",
                        help="Base encoder or ColBERT checkpoint; selects the registry entry")
    parser.add_argument("--train_file", type=str, default="OTel_embedding_pylate_100.jsonl",
                        help="Training JSONL with query/positive/negative columns")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir; defaults to <experiment_root>/<RUN_ID>/colbert_<model>")
    parser.add_argument("--experiment_root", type=str, default="experiments",
                        help="Root for the default output dir")
    parser.add_argument("--run_id", type=str, default=None, help="Run id used in the default output dir")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Per-token output dimension of the ColBERT projection (overrides registry)")
    parser.add_argument("--query_length", type=int, default=None,
                        help="Query token budget, padded with expansion tokens (overrides registry)")
    parser.add_argument("--document_length", type=int, default=None,
                        help="Document token budget (overrides registry)")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size (overrides registry)")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Per-device eval/encode batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (overrides registry)")
    parser.add_argument("--max_steps", type=int, default=-1, help="Hard cap on optimizer steps; -1 disables")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides registry)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of steps used for LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--loss", type=str, default="contrastive",
                        choices=["contrastive", "cached_contrastive", "distillation"],
                        help="Late-interaction objective")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for the contrastive loss")
    parser.add_argument("--gather_across_devices", action="store_true",
                        help="Gather document embeddings across ranks to enlarge the in-batch negative pool")
    parser.add_argument("--score_mini_batch_size", type=int, default=None,
                        help="Chunk queries during loss scoring to cut transient memory")
    parser.add_argument("--n_negatives", type=int, default=None,
                        help="Hard negative columns to feed the loss (overrides registry)")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of the dataset to use")
    parser.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction held out for the triplet evaluator")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data splits and init")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Model compute dtype")
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        choices=["auto", "sdpa", "eager", "flash_attention_2"],
                        help="Attention kernel; auto selects sdpa on ROCm, flash_attention_2 on CUDA")
    parser.add_argument("--scores_backend", type=str, default="auto", choices=["auto", "torch", "flash", "lik"],
                        help="MaxSim kernel; auto forces torch on ROCm (flash/lik are CUDA-only)")
    parser.add_argument("--tf32", action="store_true", help="Enable tf32 matmuls; ignored on ROCm builds")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade compute for activation memory")
    parser.add_argument("--logging_steps", type=int, default=1, help="Steps between loss log lines")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"],
                        help="Triplet-evaluator cadence")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"],
                        help="Checkpoint cadence")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps between checkpoints when save_strategy=steps")
    parser.add_argument("--dataloader_drop_last", action="store_true",
                        help="Drop the trailing partial batch (off by default so tiny samples still yield steps)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Dataloader worker processes")
    parser.add_argument("--skip_late_interaction_check", action="store_true",
                        help="Skip the post-training multi-vector / MaxSim proof")
    parser.add_argument("--index_backend", type=str, default="none", choices=["none", "plaid"],
                        help="Build an index and run end-to-end retrieval after training")
    return parser.parse_args()


def resolve_output_dir(args, cfg):
    """Default the output dir to <experiment_root>/<run_id>/colbert_<model>."""
    if args.output_dir:
        return args.output_dir
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = cfg["model_name"].replace("/", "_")
    return os.path.join(args.experiment_root, run_id, "colbert_%s" % slug)


def build_training_args(args, cfg, output_dir):
    """Assemble the sentence-transformers training arguments PyLate trains through."""
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=cfg["lr"],
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        tf32=utils.resolve_tf32(args.tf32),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,
        dataloader_drop_last=args.dataloader_drop_last,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to=[],
        run_name="pylate_colbert",
    )


def main():
    args = parse_args()
    cfg = utils.resolve_cfg(DEFAULT_CFG, MODELS, args.model_name, args)
    cfg["model_name"] = args.model_name

    output_dir = resolve_output_dir(args, cfg)
    is_main = os.environ.get("RANK", "0") == "0"
    logger = utils.setup_logging(output_dir, is_main)
    utils.set_seed(args.seed)

    os.environ["PYLATE_SCORES_BACKEND"] = utils.resolve_scores_backend(args.scores_backend)
    attn = utils.resolve_attn_implementation(args.attn_implementation)
    logger.info("torch %s (hip=%s cuda=%s) devices=%d attn=%s scores_backend=%s",
                torch.__version__, torch.version.hip, torch.version.cuda,
                torch.cuda.device_count(), attn, os.environ["PYLATE_SCORES_BACKEND"])
    logger.info("Resolved config: %s", cfg)

    train_dataset, eval_dataset, columns = utils.load_triplets(
        args.train_file, args.sample_fraction, args.eval_fraction, cfg["n_negatives"], args.seed
    )

    model = utils.build_model(cfg, attn, args.dtype)
    loss = utils.build_loss(args.loss, model, args.temperature, args.gather_across_devices,
                            args.score_mini_batch_size)
    evaluator = build_evaluator_safe(eval_dataset, columns, args.eval_batch_size, logger)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=build_training_args(args, cfg, output_dir),
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=pylate_utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()

    final_dir = os.path.join(output_dir, "final_model")
    if is_main:
        model.save_pretrained(final_dir)
        logger.info("Saved final ColBERT model to %s", final_dir)

    if is_main and not args.skip_late_interaction_check:
        run_late_interaction_check(model, train_dataset, columns, args, output_dir, logger)


def build_evaluator_safe(eval_dataset, columns, batch_size, logger):
    """Build the triplet evaluator, logging and skipping it when the split has no negatives."""
    evaluator = utils.build_evaluator(eval_dataset, columns, batch_size)
    if evaluator is None:
        logger.info("No eval split or no negative column — skipping the triplet evaluator")
    return evaluator


def run_late_interaction_check(model, train_dataset, columns, args, output_dir, logger):
    """Post-training proof that the model emits per-token vectors and ranks by MaxSim."""
    positive_column = [key for key in columns if key.startswith("positive")][0]
    negative_columns = [key for key in columns if key.startswith("negative")]
    query = train_dataset[0]["query"]
    documents = [train_dataset[0][positive_column]]
    if negative_columns:
        documents.append(train_dataset[0][negative_columns[0]])
    documents.append(train_dataset[1][positive_column])

    logger.info("Late-interaction check — query: %s", query[:90].replace("\n", " "))
    maxsim, ranking = utils.late_interaction_report(model, query, documents, args.eval_batch_size, logger)
    logger.info("Relevant document is rank 1: %s", ranking[0] == 0)
    utils.rerank_report(model, query, documents, args.eval_batch_size, logger)

    if args.index_backend == "plaid":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        utils.index_report(model, query, documents, os.path.join(output_dir, "indexes"),
                           "colbert", args.eval_batch_size, device, logger)
    return maxsim


if __name__ == "__main__":
    main()
