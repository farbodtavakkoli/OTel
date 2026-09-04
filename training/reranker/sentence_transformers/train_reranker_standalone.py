"""Cross-encoder reranker fine-tuner (sentence-transformers CrossEncoder) — see readme_reranker.md."""
import argparse
import json, os, random
import torch
import torch.distributed as dist
from datasets import Dataset
from dotenv import load_dotenv
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Cross-encoder reranker fine-tuner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="Cross-encoder base model")
    parser.add_argument("--data", type=str, default="OTel_reranker_sample_100.jsonl",
                        help="Training JSONL with anchor/positive/negative_1..negative_N columns")
    parser.add_argument("--out", type=str, default="output", help="Output dir for checkpoints, logs, final model")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length (query + document)")
    parser.add_argument("--batch", type=int, default=64, help="Per-GPU train/eval batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--n_neg", type=int, default=5, help="Hard negatives per query (negative_1..negative_N)")
    parser.add_argument("--eval_frac", type=float, default=0.003, help="Fraction of rows held out for evaluation")
    parser.add_argument("--test_mode", action="store_true", help="Cap the dataset at 200 rows for a quick test")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional HF_HOME model-cache override")
    return parser.parse_args()


def load_and_split(path: str, eval_frac: float, n_neg: int, test_mode: bool = False):
    """Expand each JSONL row into labeled (query, doc) pairs and a held-out eval set."""
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(raw)

    if test_mode:
        raw = raw[:200]

    k = int(len(raw) * (1 - eval_frac))

    pairs = {"sentence_0": [], "sentence_1": [], "label": []}
    for e in raw[:k]:
        q = e["anchor"]
        for doc, lbl in [(e["positive"], 1.0)] + [(e[f"negative_{i}"], 0.0) for i in range(1, n_neg + 1)]:
            pairs["sentence_0"].append(q)
            pairs["sentence_1"].append(doc)
            pairs["label"].append(lbl)

    eval_samples = [{"query": e["anchor"], "positive": [e["positive"]],
                     "negative": [e[f"negative_{i}"] for i in range(1, n_neg + 1)]} for e in raw[k:]]
    return Dataset.from_dict(pairs), eval_samples


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        # Set device before init_process_group to avoid warning
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    
    is_main = local_rank == 0

    # Load the cross-encoder with flash-attention 2 in bf16.
    # flash-attn is a CUDA-only build; on ROCm (AMD) fall back to PyTorch SDPA.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_len,
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_impl
        }
    )

    # Qwen has no dedicated pad token; use eos and sync it to the model config
    # (otherwise the trainer raises a "pad_token_id not set" ValueError).
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.pad_token_id

    train_ds, eval_samples = load_and_split(args.data, args.eval_frac, args.n_neg, test_mode=args.test_mode)

    evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_samples, name="reranking", at_k=10,
        show_progress_bar=is_main, batch_size=32
    )

    # Baseline eval on rank 0 before training (barriers keep ranks in sync).
    dist.barrier()
    if is_main:
        print("Running baseline evaluation...")
        baseline = evaluator(model, output_path=args.out)
        print("Baseline Results:", baseline)
    dist.barrier()

    training_args = CrossEncoderTrainingArguments(
            output_dir=args.out,
            logging_dir=f"{args.out}/logs",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=2,
            dataloader_drop_last=True,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            bf16=True,
            gradient_checkpointing=True,
            report_to="tensorboard",
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="reranking_ndcg@10",
            greater_is_better=True,
            ddp_find_unused_parameters=False
        )

    if not hasattr(training_args, "save_safetensors"):
        training_args.save_safetensors = True

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=BinaryCrossEntropyLoss(model),
        evaluator=evaluator,
    )

    trainer.train()

    # Save the best model (loaded at end).
    if is_main:
        final_path = os.path.join(args.out, "final")
        model.save_pretrained(final_path)
        print(f"Training complete. Best model loaded and saved to {final_path}")

if __name__ == "__main__":
    main()