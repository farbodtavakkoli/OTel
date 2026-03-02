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
from transformers import AutoModel

import torch
import yaml
import os

import logging
import time
from gpu_aware_mpi import get_rank

logger = logging.getLogger(__name__)


def is_main_rank():
    """Check if current process is the main rank (rank 0)."""
    return get_rank() == 0


def load_tokenformer_model(): 
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_tokenformer_adapter(model_info)

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    model_info = load_checkpoint_weights_if_exist(model_info)

    total_time = time.time() - start_time
    if is_main_rank():
        logger.info(f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
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


def materialize_model(model_info):
    download_model(model_info["model_name"])

    job_config = load_local_training_config()
    if is_main_rank():
        logger.info(f"Loading local training config from: {job_config}")
    training_mode = job_config.get("training_mode", "language_model")

    start_time = time.time()
     
    if training_mode == "embedding":

        from sentence_transformers import SentenceTransformer
        from sentence_transformers.losses.CoSENTLoss import CoSENTLoss

        if is_main_rank():
            logger.info("Loading model for embedding training with SentenceTransformer...")
        model_info["model"] = SentenceTransformer(
            model_info["model_name"], 
            device=model_info["distribution_strategy"]["device"]
        )

        total_time = time.time() - start_time

        if is_main_rank():
            logger.info(f"create embedding model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        start_time = time.time()
        config = get_config()
        config_dtype = config["dtype"]
        dtype = (
            torch.float16
            if config_dtype == "float16"
            else torch.float32 if config_dtype == "float32" else torch.bfloat16
        )
        if is_main_rank():
            logger.info(f"Converting model to {dtype}...")

        model_info["model"] = model_info["model"].to(dtype=dtype)

        total_time = time.time() - start_time
        if is_main_rank():
            logger.info(f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

        if (
            "distribution_strategy" in model_info 
            and "strategy" in model_info["distribution_strategy"]
        ):
            model_info["model"] = model_info["distribution_strategy"]["strategy"](
                model_info["model"]
            )

        if is_main_rank():
            logger.info(f"Model: {model_info['model']}")

        model_info["model"].to(model_info["distribution_strategy"]["device"])

        model_info["loss"] = CoSENTLoss(model_info["model"])

    else:
        attn_impl = job_config.get("attn_implementation", "flash_attention_2")
        if is_main_rank():
            logger.info(f"Loading model for language model training (AutoModelForCausalLM) with attn_implementation={attn_impl}...")
        model_info["model"] = AutoModelForCausalLM.from_pretrained(
            model_info["model_name"],
            dtype="auto",           # Use model's native dtype
            attn_implementation=attn_impl,
            #device_map="auto",            # Enable Big Model Inference
            #low_cpu_mem_usage=True,       # Reduce CPU memory usage
            #_fast_init=True               # Skip weight initialization (default True)
        )

        total_time = time.time() - start_time
        if is_main_rank():
            logger.info(f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

        
        start_time = time.time()
        if not job_config.get("full_parameter_training", False):
            model_info["model"] = create_llama_tokenformer_model(
                model_info["model"], 
                model_info["distribution_strategy"]["device"],
                use_lora=job_config.get("use_lora", False),
                lora_config=job_config
            )
            total_time = time.time() - start_time
            if is_main_rank():
                logger.info(f"create_llama_tokenformer_model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        else:
            if is_main_rank():
                logger.info("Full parameter training enabled - skipping adapter application")
        
        if job_config.get("gradient_checkpointing", False):
            model_info["model"].gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if is_main_rank():
                logger.info("Gradient checkpointing enabled")
        
        freeze_keywords = job_config.get("freeze_keywords", [])
        if freeze_keywords:
            if is_main_rank():
                logger.info(f"Selective freezing enabled for keywords: {freeze_keywords}")
            frozen_count = 0
            for name, param in model_info["model"].named_parameters():
                if any(key in name for key in freeze_keywords):
                    param.requires_grad = False
                    frozen_count += 1
            if is_main_rank():
                logger.info(f"Froze {frozen_count} parameters matching keywords")
            
            trainable = sum(p.numel() for p in model_info["model"].parameters() if p.requires_grad)
            total = sum(p.numel() for p in model_info["model"].parameters())
            if is_main_rank():
                logger.info(f"After selective freezing: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
        
        start_time = time.time()
        config = get_config()
        config_dtype = config["dtype"]
        dtype = (
            torch.float16
            if config_dtype == "float16"
            else torch.float32 if config_dtype == "float32" else torch.bfloat16
        )
        if is_main_rank():
            logger.info(f"Converting model to {dtype}...")

        model_info["model"] = model_info["model"].to(dtype=dtype)

        total_time = time.time() - start_time
        if is_main_rank():
            logger.info(f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)")

        if (
            "distribution_strategy" in model_info
            and "strategy" in model_info["distribution_strategy"]
        ):
            model_info["model"] = model_info["distribution_strategy"]["strategy"](
                model_info["model"]
            )

        if is_main_rank():
            logger.info(f"Model: {model_info['model']}")

        model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info

def load_checkpoint_weights_if_exist(model_info):
    return model_info
