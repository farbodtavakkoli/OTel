"""torchtitan run configs (CPT + SFT) selected via --module/--config -- see readme_torchtitan.md."""

import os
from pathlib import Path

from datasets import load_dataset

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import (
    ChatDataLoader,
    DatasetConfig,
    DATASETS,
    HuggingFaceTextDataLoader,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

_HERE = Path(__file__).resolve().parent


def _resolve(path: str) -> str:
    """Anchor a relative path to this folder; torchtitan runs from its own clone."""
    return path if os.path.isabs(path) else str(_HERE / path)


HF_ASSETS = _resolve(os.environ.get("TITAN_HF_ASSETS", "assets/hf/Qwen3-8B"))
CPT_JSONL = _resolve(os.environ.get("TITAN_CPT_JSONL", "data/OTel_LLM_sample_10.jsonl"))
SFT_JSON = _resolve(os.environ.get("TITAN_SFT_JSON", "data/OTel_LLM_sample_10.jsonl"))
OUT_DIR = _resolve(os.environ.get("TITAN_OUT", "outputs"))

# Local dataset name registered below and referenced by the CPT recipes.
LOCAL_DATASET = "domain_corpus"


def _load_local_jsonl(dataset_path: str, **kwargs):
    """Loader for a plain local JSONL corpus, streamed like the built-in c4 loader."""
    return load_dataset("json", data_files=dataset_path, split="train", streaming=True)


def _text_from_sample(sample) -> str:
    """Pull a document string from one row: `text` if present, else concatenated `messages`."""
    if sample.get("text"):
        return sample["text"]
    messages = sample.get("messages") or []
    parts = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else ""
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)


def _process_local_text(sample) -> str:
    """Pre-training sample processor: return the raw document text."""
    return _text_from_sample(sample)


# Register the local corpus in torchtitan's dataset registry (docs/datasets.md pattern).
DATASETS[LOCAL_DATASET] = DatasetConfig(
    path=CPT_JSONL,
    loader=_load_local_jsonl,
    sample_processor=_process_local_text,
)


def _chat_pair(sample):
    """SFT sample processor: `messages` preferred, else flat prompt/response pairs."""
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
            if isinstance(m, dict)
        ]
    prompt = sample.get("prompt") or sample.get("question") or ""
    response = sample.get("response") or sample.get("answer") or ""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _base_config(model_spec, *, steps, lr, warmup_steps, seq_len, local_batch_size):
    """Shared skeleton: bf16 mixed precision, FSDP2 over every rank, full AC."""
    return Trainer.Config(
        model_spec=model_spec,
        hf_assets_path=HF_ASSETS,
        dump_folder=OUT_DIR,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        optimizer=default_adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
        ),
        parallelism=ParallelismConfig(
            # -1 = use every leftover rank for FSDP2 sharding. On a single
            # 8xH100 node with TP/PP/CP all 1 this means FSDP2 across 8 GPUs.
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=True,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            keep_latest_k=3,
            # Cold start: read the HF safetensors sitting in hf_assets_path.
            # initial_load_in_hf requires initial_load_model_only.
            initial_load_path=HF_ASSETS,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            # Write the final checkpoint back out as HF safetensors so the result
            # can be served / further post-trained with the rest of this repo.
            last_save_in_hf=True,
            last_save_model_only=True,
            export_dtype="bfloat16",
        ),
        activation_checkpoint=FullAC.Config(),
    )


def cpt_qwen3_8b() -> Trainer.Config:
    """Continued pre-training of Qwen3-8B on a local domain corpus (8x H100)."""
    from torchtitan.models.qwen3 import model_registry

    config = _base_config(
        model_registry("8B"),
        steps=2000,
        lr=1e-5,
        warmup_steps=100,
        seq_len=4096,
        local_batch_size=1,
    )
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset=LOCAL_DATASET)
    return config


def cpt_llama3_8b() -> Trainer.Config:
    """Continued pre-training of Llama 3.1 8B on a local domain corpus (8x H100)."""
    from torchtitan.models.llama3 import model_registry

    config = _base_config(
        model_registry("8B"),
        steps=2000,
        lr=1e-5,
        warmup_steps=100,
        seq_len=4096,
        local_batch_size=1,
    )
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset=LOCAL_DATASET)
    return config


def sft_qwen3_8b() -> Trainer.Config:
    """SFT of Qwen3-8B on a local chat dataset, prompt tokens masked."""
    from torchtitan.models.qwen3 import model_registry

    config = _base_config(
        model_registry("8B"),
        steps=500,
        lr=5e-6,
        warmup_steps=25,
        seq_len=4096,
        local_batch_size=1,
    )
    config.dataloader = ChatDataLoader.Config(
        dataset_path="json",
        load_dataset_kwargs={"data_files": SFT_JSON, "split": "train"},
        sample_processor=_chat_pair,
    )
    return config


def cpt_qwen3_8b_smoke() -> Trainer.Config:
    """20-step smoke test: same wiring, tiny step count, frequent logging."""
    config = cpt_qwen3_8b()
    config.training.steps = 20
    config.lr_scheduler.warmup_steps = 2
    config.metrics.log_freq = 1
    config.checkpoint.interval = 20
    return config


# --- MI355X / ROCm bring-up config (see readme_torchtitan.md section 7) ----------------
# Same wiring as the CPT recipes above (local-JSONL dataset registry, FSDP2 with
# data_parallel_shard_degree=-1, ChunkedLoss, FullAC) but on torchtitan's `debugmodel`
# flavor (dim 256, 8 layers, vocab 2048) with random init and checkpointing OFF, so it
# needs no HF weight download and writes essentially nothing to disk. This is the config
# used to validate the folder on 2x AMD Instinct MI355X (gfx950); it is hardware-agnostic
# and works unchanged on NVIDIA.
def cpt_debugmodel_smoke() -> Trainer.Config:
    """Tiny random-init FSDP2 smoke (no HF download, no checkpoints) for bring-up."""
    from torchtitan.models.qwen3 import model_registry

    config = _base_config(
        model_registry("debugmodel"),
        steps=8,
        lr=1e-4,
        warmup_steps=2,
        seq_len=512,
        local_batch_size=2,
    )
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset=LOCAL_DATASET)
    # The debugmodel needs only a tokenizer, not model weights. Default points at the
    # test tokenizer inside the torchtitan clone (TITAN_TOKENIZER overrides it).
    config.hf_assets_path = os.environ.get("TITAN_TOKENIZER", "./tests/assets/tokenizer")
    config.metrics.log_freq = 1
    # Random init + no checkpoint I/O at all: nothing is loaded, nothing is written.
    config.checkpoint.enable = False
    config.checkpoint.initial_load_path = None
    config.checkpoint.initial_load_in_hf = False
    config.checkpoint.last_save_in_hf = False
    return config
