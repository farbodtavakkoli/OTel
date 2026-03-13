from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)
 
from tokenformer.llama_tokenformer_model import create_llama_tokenformer_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config
from ml.get_local_job_config import load_local_training_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification

import torch
import torch.nn as nn

import logging
import time
import warnings

from ml.cray_megatron.collectives.main_rank_only import log_if_main_rank

logger = logging.getLogger(__name__)


def load_tokenformer_model(): 
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_tokenformer_adapter(model_info)

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    model_info = load_checkpoint_weights_if_exist(model_info)

    total_time = time.time() - start_time
    log_if_main_rank(f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    return model_info


def load_model_config():
    job_config = get_job_config()

    model_name = job_config["llm_name"]

    model_config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_info = {
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer": tokenizer,
    }

    return model_info


def apply_tokenformer_adapter(model_info):
    return model_info


def _apply_freeze_keywords(model, job_config):
    """Selectively freeze parameters whose names match any of the configured keywords."""
    freeze_keywords = job_config.get("freeze_keywords", [])
    if not freeze_keywords:
        return
    log_if_main_rank(f"Selective freezing enabled for keywords: {freeze_keywords}")
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(key in name for key in freeze_keywords):
            param.requires_grad = False
            frozen_count += 1
    log_if_main_rank(f"Froze {frozen_count} parameters matching keywords")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_if_main_rank(f"After selective freezing: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")


def _apply_gradient_checkpointing(model, job_config):
    """Enable gradient checkpointing when configured."""
    if job_config.get("gradient_checkpointing", False):
        # Disable use_cache before enabling gradient checkpointing to suppress warning
        model.config.use_cache = False
        # Suppress backward hook warning from gradient checkpointing
        warnings.filterwarnings("ignore", message=".*Full backward hook.*")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        log_if_main_rank("Gradient checkpointing enabled")


def _convert_dtype(model):
    """Convert model to the dtype specified in the global config."""
    start_time = time.time()
    config = get_config()
    config_dtype = config["dtype"]
    dtype = (
        torch.float16
        if config_dtype == "float16"
        else torch.float32 if config_dtype == "float32" else torch.bfloat16
    )
    log_if_main_rank(f"Converting model to {dtype}...")
    model = model.to(dtype=dtype)
    total_time = time.time() - start_time
    log_if_main_rank(f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    return model


def _apply_distribution_and_move_to_device(model, model_info):
    """Wrap the model with the distribution strategy (if any) and move to device."""
    if (
        "distribution_strategy" in model_info
        and "strategy" in model_info["distribution_strategy"]
    ):
        model = model_info["distribution_strategy"]["strategy"](model)
    log_if_main_rank(f"Model: {model}")
    model.to(model_info["distribution_strategy"]["device"])
    return model


def _materialize_embedding(model_info, job_config):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses.CoSENTLoss import CoSENTLoss

    start_time = time.time()
    log_if_main_rank("Loading model for embedding training with SentenceTransformer...")
    model_info["model"] = SentenceTransformer(
        model_info["model_name"],
        device=model_info["distribution_strategy"]["device"],
    )
    total_time = time.time() - start_time
    log_if_main_rank(f"create embedding model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    model_info["model"] = _convert_dtype(model_info["model"])
    model_info["model"] = _apply_distribution_and_move_to_device(model_info["model"], model_info)
    model_info["loss"] = CoSENTLoss(model_info["model"])
    return model_info


def _materialize_classification(model_info, job_config):
    attn_impl = job_config.get("attn_implementation", "flash_attention_2")
    num_labels = job_config.get("num_labels")
    classification_dropout = job_config.get("classification_dropout", 0.1)

    if num_labels is None:
        raise ValueError("num_labels must be specified for classification training")

    log_if_main_rank(f"Loading model for classification training (AutoModelForSequenceClassification) with attn_implementation={attn_impl}...")
    log_if_main_rank(f"Number of labels: {num_labels}, Classification dropout: {classification_dropout}")

    start_time = time.time()
    model_info["model"] = AutoModelForSequenceClassification.from_pretrained(
        model_info["model_name"],
        num_labels=num_labels,
        dtype="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    total_time = time.time() - start_time
    log_if_main_rank(f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    # Replace classification head with dropout version
    log_if_main_rank(f"Replacing classification head with Dropout({classification_dropout}) + Linear...")
    hidden_size = model_info["model"].config.hidden_size
    model_info["model"].score = nn.Sequential(
        nn.Dropout(classification_dropout),
        nn.Linear(hidden_size, num_labels, bias=False),
    )
    torch.nn.init.normal_(model_info["model"].score[1].weight, mean=0.0, std=0.01)
    log_if_main_rank(f"Head: Dropout({classification_dropout}) → Linear({hidden_size}, {num_labels}, bias=False)")

    # Pad token / cache config
    model_info["model"].config.pad_token_id = model_info["tokenizer"].pad_token_id
    model_info["model"].config.use_cache = False  # Required for gradient checkpointing

    _apply_gradient_checkpointing(model_info["model"], job_config)
    _apply_freeze_keywords(model_info["model"], job_config)
    model_info["model"] = _convert_dtype(model_info["model"])
    model_info["model"] = _apply_distribution_and_move_to_device(model_info["model"], model_info)
    return model_info


def _materialize_language_model(model_info, job_config):
    attn_impl = job_config.get("attn_implementation", "flash_attention_2")
    log_if_main_rank(f"Loading model for language model training (AutoModelForCausalLM) with attn_implementation={attn_impl}...")

    start_time = time.time()
    model_info["model"] = AutoModelForCausalLM.from_pretrained(
        model_info["model_name"],
        dtype="auto",
        attn_implementation=attn_impl,
    )
    total_time = time.time() - start_time
    log_if_main_rank(f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    # Apply adapter (lora / tokenformer) if configured
    start_time = time.time()
    adapter_type = job_config.get("adapter_type", "lora")
    if adapter_type == "none":
        log_if_main_rank("No adapter inserted - full parameter training enabled")
    elif adapter_type in ["lora", "tokenformer"]:
        model_info["model"] = create_llama_tokenformer_model(
            model_info["model"],
            model_info["distribution_strategy"]["device"],
            adapter_type=adapter_type,
        )
        total_time = time.time() - start_time
        log_if_main_rank(f"Adding adapter latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

    _apply_gradient_checkpointing(model_info["model"], job_config)
    _apply_freeze_keywords(model_info["model"], job_config)
    model_info["model"] = _convert_dtype(model_info["model"])
    model_info["model"] = _apply_distribution_and_move_to_device(model_info["model"], model_info)
    return model_info


_TRAINING_MODE_MATERIALIZERS = {
    "embedding": _materialize_embedding,
    "classification": _materialize_classification,
    "language_model": _materialize_language_model,
}


def materialize_model(model_info):
    download_model(model_info["model_name"])

    job_config = load_local_training_config()
    training_mode = job_config.get("training_mode", "language_model")

    materializer = _TRAINING_MODE_MATERIALIZERS.get(training_mode)
    if materializer is None:
        raise ValueError(f"Unknown training_mode: {training_mode}")

    model_info = materializer(model_info, job_config)
    return model_info

def load_checkpoint_weights_if_exist(model_info):
    return model_info
