"""Helpers for train_reranker_flagembedding.py — transformers 5.x compat and argv building."""
import os
import torch


def patch_transformers5_compat():
    """Re-add the Trainer.tokenizer name FlagEmbedding still uses; transformers 5.x renamed it."""
    from transformers.trainer import Trainer

    if getattr(Trainer, "_flagembedding_compat", False):
        return

    original_init = Trainer.__init__

    def init(self, *args, **kwargs):
        """Accept the removed tokenizer= kwarg that FlagEmbedding's reranker runners still pass."""
        if "tokenizer" in kwargs:
            kwargs.setdefault("processing_class", kwargs.pop("tokenizer"))
        original_init(self, *args, **kwargs)

    Trainer.__init__ = init
    if not hasattr(Trainer, "tokenizer"):
        Trainer.tokenizer = property(lambda self: getattr(self, "processing_class", None))
    Trainer._flagembedding_compat = True


def resolve_attn(requested):
    """Pick the attention implementation; flash_attention_2 is CUDA-only, so ROCm falls back to sdpa."""
    if requested != "auto":
        return requested
    if torch.version.hip is not None:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def resolve_path(path, base):
    """Make a relative path absolute against the folder that holds the script."""
    return path if os.path.isabs(path) else os.path.join(base, path)


def flag(name, value):
    """Render one HfArgumentParser flag, dropping it when the value is None."""
    return [] if value is None else [f"--{name}", str(value)]


def build_argv(args, base_dir):
    """Translate the parsed CLI namespace into the argv FlagEmbedding's HfArgumentParser expects."""
    argv = [
        "--model_name_or_path", args.model_name_or_path,
        "--train_data", resolve_path(args.train_data, base_dir),
        "--output_dir", resolve_path(args.output_dir, base_dir),
        "--per_device_train_batch_size", str(args.batch_size),
        "--train_group_size", str(args.train_group_size),
        "--query_max_len", str(args.query_max_len),
        "--passage_max_len", str(args.passage_max_len),
        "--max_len", str(args.max_len),
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--warmup_steps", str(args.warmup_steps),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--knowledge_distillation", str(args.knowledge_distillation),
        "--logging_steps", str(args.logging_steps),
        "--save_strategy", args.save_strategy,
        "--dataloader_drop_last", str(args.dataloader_drop_last),
        "--seed", str(args.seed),
        "--report_to", args.report_to,
    ]
    argv += flag("query_instruction_for_rerank", args.query_instruction)
    argv += flag("passage_instruction_for_rerank", args.passage_instruction)
    argv += flag("cache_dir", args.cache_dir)
    argv += flag("cache_path", args.cache_path)
    argv += flag("save_total_limit", args.save_total_limit)
    argv += flag("deepspeed", args.deepspeed)

    if args.bf16:
        argv += ["--bf16"]
    if args.gradient_checkpointing:
        reentrant = str(args.gc_use_reentrant).lower()
        argv += ["--gradient_checkpointing",
                 "--gradient_checkpointing_kwargs", f'{{"use_reentrant":{reentrant}}}']
    if args.trust_remote_code:
        argv += ["--trust_remote_code", "True"]
    if args.extra_args:
        argv += args.extra_args.split()
    return argv


def llm_argv(args):
    """Extra flags for the decoder-only (LLM) reranker families — LoRA and attention."""
    argv = ["--model_type", "decoder", "--use_lora", str(args.use_lora),
            "--lora_rank", str(args.lora_rank), "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
            "--save_merged_lora_model", str(args.save_merged_lora_model),
            "--target_modules"] + args.target_modules.split()
    # flash-attn is a CUDA-only build here; FlagEmbedding falls back to the model default (sdpa).
    argv += ["--use_flash_attn", str(resolve_attn(args.attn_implementation) == "flash_attention_2")]
    if args.start_layer is not None:
        argv += ["--start_layer", str(args.start_layer)]
    if args.head_multi is not None:
        argv += ["--head_multi", str(args.head_multi)]
    if args.head_type is not None:
        argv += ["--head_type", args.head_type]
    return argv


def gpu_banner():
    """One line describing the visible accelerators — handy in multi-rank logs."""
    if not torch.cuda.is_available():
        return "no GPU visible"
    name = torch.cuda.get_device_name(0)
    build = f"hip {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}"
    return f"{torch.cuda.device_count()}x {name} (torch {torch.__version__}, {build})"
