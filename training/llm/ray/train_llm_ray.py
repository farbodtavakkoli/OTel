"""Ray Train (TorchTrainer) orchestration of an HF/TRL SFT fine-tune -- see readme_ray.md."""

import os
import json
import uuid
import logging
import argparse
import tempfile

import ray
import ray.train
import ray.train.torch
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from dotenv import load_dotenv

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

HF_TOKEN = os.getenv("HF_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray Train (TorchTrainer) orchestration of an HF/TRL SFT fine-tune"
    )

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="HF repo id or local path to fine-tune")
    parser.add_argument("--train_file", type=str, default="data/OTel_LLM_sample_10.jsonl",
                        help="Chat JSONL, one {'messages': [...]} per line. Extra columns are dropped after rendering.")
    parser.add_argument("--storage_path", type=str, default="./ray_results",
                        help="Persistent storage for checkpoints. MUST be s3:// or a shared "
                             "NFS mount for multi-node runs; a local path errors there.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Unique run name. Reuse the SAME (storage_path, run_name) to resume "
                             "an interrupted run; a fresh name starts from scratch.")

    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of training workers (one per GPU, across all nodes)")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="Reserve 1 GPU per worker")
    parser.add_argument("--max_failures", type=int, default=3,
                        help="Worker/node failure retries; -1 for unlimited. 0 disables recovery.")
    parser.add_argument("--num_to_keep", type=int, default=2,
                        help="Max checkpoints to retain in persistent storage")
    parser.add_argument("--ray_address", type=str, default=None,
                        help="Ray cluster address, e.g. 'auto' to attach to a running cluster. "
                             "Omit to start a local single-node Ray.")

    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max tokens per example")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-worker (per-GPU) batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Report a checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for activation memory")

    parser.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        help="PEFT target modules: 'all-linear' or a comma-separated list")

    return parser.parse_args()


def train_func(config):
    """Per-worker training loop; builds everything locally so nothing large is serialized from the driver."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTTrainer, SFTConfig

    from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer

    set_seed(config["seed"])

    if config["hf_token"]:
        os.environ["HF_TOKEN"] = config["hf_token"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    rank = ray.train.get_context().get_world_rank()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {config['model_name']} has no chat_template; this trainer "
            "renders `messages` with apply_chat_template and requires one."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Render `messages` with the model's own chat template; TRL tokenizes the `text` column.
    dataset = load_dataset("json", data_files=config["train_file"], split="train")

    def render(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = dataset.map(render, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # device_map stays None: Ray assigns one device per worker.
        device_map=None,
    )
    model.config.use_cache = False

    peft_config = None
    if config["use_lora"]:
        from peft import LoraConfig
        targets = config["lora_target_modules"]
        if targets != "all-linear" and "," in targets:
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )

    args = SFTConfig(
        output_dir=tempfile.mkdtemp(),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_acc_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        max_length=config["max_seq_length"],
        packing=False,
        report_to="none",  # Ray Train collects metrics via the report callback
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Report metrics + checkpoints to Ray Train; this is what makes recovery resume, not restart.
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    # Resume from the latest Ray checkpoint if this is a recovery attempt.
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as ckpt_dir:
            if rank == 0:
                logging.info("Resuming from Ray checkpoint: %s", ckpt_dir)
            trainer.train(resume_from_checkpoint=ckpt_dir)
    else:
        trainer.train()


def main():
    args = parse_args()

    if not HF_TOKEN:
        logging.warning("HF_TOKEN is not set (dev.env missing?). Gated models will fail to download.")

    if not os.path.exists(args.train_file):
        raise SystemExit(f"--train_file not found: {args.train_file}")

    # Fail fast on the data contract before reserving a cluster's worth of GPUs.
    with open(args.train_file) as f:
        for i, line in enumerate(f):
            if i >= 64:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise SystemExit(f"{args.train_file} line {i + 1}: expected a non-empty 'messages' list.")

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    logging.info("Ray cluster resources: %s", ray.cluster_resources())

    # A run is identified by (storage_path, name); reusing the pair resumes it.
    run_name = args.run_name or f"llm_sft-{uuid.uuid4().hex[:8]}"
    if not args.storage_path.startswith(("s3://", "gs://", "/mnt")):
        logging.warning(
            "storage_path %s looks local. Multi-node runs REQUIRE s3:// or a shared "
            "NFS mount; Ray errors on checkpoint if workers span nodes.", args.storage_path
        )

    train_loop_config = {
        "model_name": args.model_name,
        "train_file": os.path.abspath(args.train_file),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_acc_steps": args.grad_acc_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "hf_token": HF_TOKEN,
    }

    trainer = TorchTrainer(
        train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
        run_config=RunConfig(
            storage_path=args.storage_path,
            name=run_name,
            failure_config=FailureConfig(max_failures=args.max_failures),
            checkpoint_config=CheckpointConfig(num_to_keep=args.num_to_keep),
        ),
    )

    logging.info("Launching run '%s' | workers=%d | max_failures=%d | storage=%s",
                 run_name, args.num_workers, args.max_failures, args.storage_path)

    result = trainer.fit()

    logging.info("Training complete. Metrics: %s", result.metrics)
    logging.info("Result path: %s", result.path)
    if result.checkpoint:
        logging.info("Latest checkpoint: %s", result.checkpoint)
    logging.info("To resume this exact run: --storage_path %s --run_name %s",
                 args.storage_path, run_name)


if __name__ == "__main__":
    main()
