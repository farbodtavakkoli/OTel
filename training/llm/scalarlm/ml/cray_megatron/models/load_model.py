from cray_megatron.huggingface.download_model import download_model
from cray_megatron.megatron.distribution.apply_distribution_strategy import (
    apply_distribution_strategy,
)
from cray_megatron.collectives.main_rank_only import is_main_rank, log_if_main_rank

from cray_infra.training.distributed import get_size, get_rank, allgather, allreduce

from adapters.add_adapters_to_model import add_adapters_to_model

from cray_infra.util.get_job_config import get_job_config
from cray_infra.util.get_config import get_config

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoModelForSequenceClassification

from cray_megatron.megatron.doc_mask import is_multimodal, is_diffusion

import torch

import logging
import time

logger = logging.getLogger(__name__)


def load_model():
    start_time = time.time()
    model_info = load_model_config()

    model_info = apply_distribution_strategy(model_info)

    model_info = materialize_model(model_info)

    total_time = time.time() - start_time
    logger.info(
        f"Total model loading time: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )
    return model_info


def load_model_config():
    job_config = get_job_config()

    model_name = job_config["llm_name"]

    # Opt-in execution of a repo's custom modeling code; defaults False.
    trust_remote_code = bool(job_config.get("trust_remote_code", False))

    model_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    model_info = {
        "model_name": model_name,
        "model_config": model_config,
        "tokenizer": tokenizer,
    }

    return model_info


def sync_freshly_initialized_heads(model):
    """Make randomly-initialized task heads identical on every rank."""
    if get_size() <= 1:
        return

    head_prefixes = ("score.", "classifier.")
    synced = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.startswith(head_prefixes):
                continue
            if get_rank() != 0:
                param.zero_()
            allreduce(param.data)
            synced.append(name)

    if synced:
        logger.info(
            "Broadcast %d freshly-initialized head tensor(s) from rank 0 so all "
            "%d ranks share one head: %s",
            len(synced),
            get_size(),
            ", ".join(synced),
        )


def apply_freeze_layer_keywords(model, job_config):
    """Freeze every parameter whose name contains one of freeze_layer_keywords."""
    keywords = job_config.get("freeze_layer_keywords")
    if not keywords:
        return

    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if not keywords:
        return

    logger.info(f"Selective freezing enabled for keywords: {keywords}")
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(key in name for key in keywords):
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100 * trainable / total) if total else 0.0
    logger.info(
        f"Froze {frozen_count} parameter tensors matching keywords; "
        f"{trainable:,} / {total:,} trainable ({pct:.2f}%)"
    )


def _resolve_attn_impl(job_config):
    """Decide the HF attention backend for training."""
    override = job_config.get("attn_implementation", "auto")
    if override in (None, "auto", "flash_attention_2"):
        return "sdpa"
    return override


def _resolve_dtype(job_config):
    """Per-job dtype (train_args["dtype"]) wins over global cray-config.yaml."""
    job_dtype = job_config.get("dtype", "auto")
    config_dtype = job_dtype if job_dtype != "auto" else get_config()["dtype"]
    if config_dtype == "auto":
        return None
    return (
        torch.float16
        if config_dtype == "float16"
        else torch.float32 if config_dtype == "float32" else torch.bfloat16
    )


def _materialize_embedding(model_info):
    """Embedding training: SentenceTransformer body + CoSENT pairwise loss."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses.CoSENTLoss import CoSENTLoss

    job_config = get_job_config()
    device = model_info["distribution_strategy"]["device"]

    log_if_main_rank("Loading SentenceTransformer for embedding training...")
    download_model(model_info["model_name"])

    start_time = time.time()
    model_info["model"] = SentenceTransformer(model_info["model_name"], device=device)
    log_if_main_rank(
        f"SentenceTransformer load latency: {time.time() - start_time:.2f}s"
    )

    dtype = _resolve_dtype(job_config)
    if dtype is not None:
        model_info["model"] = model_info["model"].to(dtype=dtype)

    model_info["model"] = model_info["distribution_strategy"]["strategy"](
        model_info["model"]
    )
    model_info["model"].to(device)

    # CoSENTLoss takes the wrapped model so its forward runs through the distribution strategy.
    model_info["loss"] = CoSENTLoss(model_info["model"])
    return model_info


def materialize_model(model_info):
    job_config = get_job_config()

    # Embedding training swaps in a SentenceTransformer pair and shares none of the load path below.
    if job_config.get("training_mode") == "embedding":
        return _materialize_embedding(model_info)

    download_model(model_info["model_name"])

    attn_impl = _resolve_attn_impl(job_config)

    # Routing order matters: diffusion must be checked before multimodal.
    is_classification = job_config["training_mode"] == "classification"

    if is_classification:
        model_cls = AutoModelForSequenceClassification
    elif is_diffusion(model_info["model_config"]):
        # No Auto* class accepts DiffusionGemmaConfig; imported lazily so other modes still run.
        from transformers.models.diffusion_gemma import (
            DiffusionGemmaForBlockDiffusion,
        )

        model_cls = DiffusionGemmaForBlockDiffusion
    elif is_multimodal(model_info["model_config"]):
        model_cls = AutoModelForImageTextToText
    else:
        model_cls = AutoModelForCausalLM
    logger.info(
        "Loading model with %s, attn_implementation=%s",
        model_cls.__name__,
        attn_impl,
    )

    # Load weights straight onto the target GPU to avoid a transient ~2x memory peak.
    device = model_info["distribution_strategy"]["device"]
    on_gpu = isinstance(device, int) or (
        isinstance(device, torch.device) and device.type == "cuda"
    )
    # Resolved before load so an fp32 checkpoint is not materialized full-width under device_map.
    load_kwargs = {"torch_dtype": _resolve_dtype(job_config) or "auto"}
    # Shared by both from_pretrained calls, so the flag reaches SDPA and the eager fallback alike.
    load_kwargs["trust_remote_code"] = bool(job_config.get("trust_remote_code", False))
    if on_gpu:
        load_kwargs["device_map"] = {"": device}
        load_kwargs["low_cpu_mem_usage"] = True

    if is_classification:
        classification = job_config.get("classification") or {}
        num_labels = classification.get("num_labels")
        if not num_labels:
            raise ValueError(
                "training_mode 'classification' requires classification.num_labels "
                "in train_args (the classifier head width)."
            )
        load_kwargs["num_labels"] = num_labels
        if classification.get("id2label"):
            load_kwargs["id2label"] = classification["id2label"]
        if classification.get("label2id"):
            load_kwargs["label2id"] = classification["label2id"]

        # Head dropout goes on the config and its attribute name is architecture-specific.
        dropout = classification.get("dropout")
        if dropout is not None:
            config_obj = model_info["model_config"]
            applied = [
                attr
                for attr in (
                    "classifier_dropout",
                    "classifier_dropout_prob",
                    "seq_classif_dropout",
                )
                if hasattr(config_obj, attr)
            ]
            for attr in applied:
                setattr(config_obj, attr, dropout)
            if applied:
                load_kwargs["config"] = config_obj
                logger.info("Classification head dropout: %s -> %s", dropout, applied)
            else:
                logger.info(
                    "Classification head dropout %s ignored: %s exposes no "
                    "classifier-dropout attribute (score-head architecture).",
                    dropout,
                    type(config_obj).__name__,
                )
        logger.info("Classification head: num_labels=%s", num_labels)

    start_time = time.time()
    try:
        model_info["model"] = model_cls.from_pretrained(
            model_info["model_name"],
            attn_implementation=attn_impl,
            **load_kwargs,
        )
    except (ValueError, ImportError, RuntimeError) as e:
        # Some configs reject SDPA at load time; eager is the universal fallback.
        if attn_impl != "eager":
            logger.warning(
                "from_pretrained refused attn_implementation=%s (%s); "
                "falling back to eager.",
                attn_impl,
                e,
            )
            attn_impl = "eager"
            model_info["model"] = model_cls.from_pretrained(
                model_info["model_name"],
                attn_implementation="eager",
                **load_kwargs,
            )
        else:
            raise

    total_time = time.time() - start_time
    logger.info(
        f"from_pretrained latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    if is_classification:
        # HF pools the last non-pad token via config.pad_token_id; None raises at batch_size > 1.
        tokenizer = model_info["tokenizer"]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        config_obj = model_info["model"].config
        config_obj.pad_token_id = tokenizer.pad_token_id
        text_config = (
            config_obj.get_text_config()
            if hasattr(config_obj, "get_text_config")
            else config_obj
        )
        if text_config is not config_obj:
            text_config.pad_token_id = tokenizer.pad_token_id

    # use_cache allocates an unused DynamicCache on every training forward; memory-only.
    model_info["model"].config.use_cache = False

    if job_config.get("gradient_checkpointing", False):
        # Must run on the bare HF model before PEFT wraps it, or LoRA gets zero grad.
        logger.info("Enabling gradient checkpointing")
        model_info["model"].gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model_info["model"], "enable_input_require_grads"):
            model_info["model"].enable_input_require_grads()

    start_time = time.time()
    model_info["model"] = add_adapters_to_model(
        model=model_info["model"], device=model_info["distribution_strategy"]["device"]
    )
    total_time = time.time() - start_time

    logger.info(
        f"create_tokenformer_model latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    # Applied AFTER the adapter so it also freezes adapter params on excluded towers.
    apply_freeze_layer_keywords(model_info["model"], job_config)

    start_time = time.time()
    # Per-job dtype wins over global cray-config.yaml; both default to "auto".
    job_dtype = job_config.get("dtype", "auto")
    config_dtype = job_dtype if job_dtype != "auto" else get_config()["dtype"]

    if config_dtype != "auto":
        dtype = (
            torch.float16
            if config_dtype == "float16"
            else torch.float32 if config_dtype == "float32" else torch.bfloat16
        )
        logger.info(f"Converting model to {dtype}...")

        model_info["model"] = model_info["model"].to(dtype=dtype)
    else:
        logger.info("Using model's native dtype, no conversion needed.")

    total_time = time.time() - start_time
    logger.info(
        f"model dtype conversion latency: {total_time:.2f}s ({total_time/60:.1f} minutes)"
    )

    # Must happen BEFORE the distribution strategy wraps/shards the model.
    sync_freshly_initialized_heads(model_info["model"])

    model_info["model"] = model_info["distribution_strategy"]["strategy"](
        model_info["model"]
    )

    if is_main_rank():
        logger.info(f"Model: {model_info['model']}")

    if on_gpu:
        # device_map already materialized the model on-device; a further .to(device) would re-peak.
        logger.info(
            f"Model already on device via device_map: "
            f"{model_info['distribution_strategy']['device']}"
        )
    else:
        logger.info(
            f"Moving model to device: {model_info['distribution_strategy']['device']}..."
        )
        model_info["model"].to(model_info["distribution_strategy"]["device"])

    return model_info
