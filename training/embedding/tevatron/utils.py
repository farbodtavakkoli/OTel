"""Helpers for the Tevatron launcher: ROCm detection, launch-command assembly, and output-dir hygiene."""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger("training/embedding/tevatron")

TRAIN_MODULE = "tevatron.retriever.driver.train"
ENCODE_MODULE = "tevatron.retriever.driver.encode"
SEARCH_MODULE = "tevatron.retriever.driver.search"


def torch_build():
    """Return ('rocm'|'cuda'|'cpu', version string) for the installed torch."""
    import torch

    if getattr(torch.version, "hip", None):
        return "rocm", torch.__version__
    if getattr(torch.version, "cuda", None):
        return "cuda", torch.__version__
    return "cpu", torch.__version__


def resolve_attn(requested):
    """Pick the attention implementation; flash_attention_2 is a CUDA-only build."""
    build, _ = torch_build()
    if requested != "auto":
        return requested
    return "sdpa" if build != "cuda" else "flash_attention_2"


def visible_device_count(devices):
    """Number of GPUs the run will use, from --devices or the ambient visibility vars."""
    if devices:
        return len([d for d in devices.split(",") if d.strip()])
    ambient = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if ambient:
        return len([d for d in ambient.split(",") if d.strip()])
    import torch

    return max(torch.cuda.device_count(), 1)


def build_env(devices):
    """Child environment: pin the GPUs on both ROCm and CUDA variable names."""
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if devices:
        env["HIP_VISIBLE_DEVICES"] = devices
        env["CUDA_VISIBLE_DEVICES"] = devices
    return env


def prepare_output_dir(output_dir, overwrite):
    """Clear a stale output dir.

    Upstream guards this with `--overwrite_output_dir`, which transformers 5.x removed;
    the driver would raise AttributeError on a non-empty dir instead.
    """
    if not os.path.isdir(output_dir) or not os.listdir(output_dir):
        return
    if not overwrite:
        raise SystemExit(f"output_dir is not empty: {output_dir} (pass --overwrite to clear it)")
    logger.info("clearing non-empty output_dir %s", output_dir)
    shutil.rmtree(output_dir)


def flag(name, value):
    """Emit ['--name', str(value)] for a set value, or [] for None."""
    return [] if value is None else [f"--{name}", str(value)]


def switch(name, value):
    """Emit ['--name'] for a truthy store_true flag."""
    return [f"--{name}"] if value else []


def build_train_command(args, num_gpus, attn):
    """Assemble the python/torchrun command line for tevatron.retriever.driver.train."""
    if num_gpus > 1:
        launcher = ["torchrun", "--nproc_per_node", str(num_gpus),
                    "--master_port", str(args.master_port), "-m", TRAIN_MODULE]
    else:
        launcher = ["python", "-m", TRAIN_MODULE]

    cmd = launcher + [
        "--output_dir", args.output_dir,
        "--model_name_or_path", args.model_name_or_path,
        "--dataset_name", args.dataset_name,
        "--attn_implementation", attn,
        "--pooling", args.pooling,
        "--temperature", str(args.temperature),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--train_group_size", str(args.train_group_size),
        "--query_max_len", str(args.query_max_len),
        "--passage_max_len", str(args.passage_max_len),
        "--num_train_epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--warmup_ratio", str(args.warmup_ratio),
        "--logging_steps", str(args.logging_steps),
        "--save_strategy", args.save_strategy,
        "--seed", str(args.seed),
        "--report_to", args.report_to,
    ]
    cmd += flag("dataset_path", args.dataset_path)
    cmd += flag("dataset_config", args.dataset_config)
    cmd += flag("dataset_split", args.dataset_split)
    cmd += flag("corpus_name", args.corpus_name)
    cmd += flag("corpus_path", args.corpus_path)
    cmd += flag("query_prefix", args.query_prefix)
    cmd += flag("passage_prefix", args.passage_prefix)
    cmd += flag("save_steps", args.save_steps)
    cmd += flag("max_steps", args.max_steps)
    cmd += flag("lr_scheduler_type", args.lr_scheduler_type)
    cmd += flag("dataloader_num_workers", args.dataloader_num_workers)
    cmd += flag("deepspeed", args.deepspeed)
    cmd += switch("normalize", args.normalize)
    cmd += switch("append_eos_token", args.append_eos_token)
    cmd += switch("bf16", args.bf16)
    cmd += switch("fp16", args.fp16)
    cmd += switch("gradient_checkpointing", args.gradient_checkpointing)

    if args.grad_cache:
        cmd += ["--grad_cache",
                "--gc_q_chunk_size", str(args.gc_q_chunk_size),
                "--gc_p_chunk_size", str(args.gc_p_chunk_size)]
    if args.lora:
        cmd += ["--lora",
                "--lora_r", str(args.lora_r),
                "--lora_alpha", str(args.lora_alpha),
                "--lora_dropout", str(args.lora_dropout),
                "--lora_target_modules", args.lora_target_modules]
    return cmd


def summarize(args, cmd, num_gpus, attn):
    """One compact block so the log says exactly what is about to run."""
    build, version = torch_build()
    effective = args.batch_size * args.gradient_accumulation_steps * num_gpus
    logger.info("torch           : %s (%s)", version, build)
    logger.info("model           : %s (pooling=%s, lora=%s)", args.model_name_or_path, args.pooling, args.lora)
    logger.info("attention       : %s", attn)
    logger.info("data            : %s (%s)", args.dataset_path or args.dataset_name, args.dataset_split)
    logger.info("batch           : %s per device x accum %s x %s GPU(s) = %s queries/update",
                args.batch_size, args.gradient_accumulation_steps, num_gpus, effective)
    logger.info("in-batch pool   : %s passages/update (queries x train_group_size %s)",
                effective * args.train_group_size, args.train_group_size)
    logger.info("grad_cache      : %s (q_chunk=%s, p_chunk=%s)",
                args.grad_cache, args.gc_q_chunk_size, args.gc_p_chunk_size)
    logger.info("output_dir      : %s", args.output_dir)
    logger.info("command         : %s", " ".join(cmd))


def warn_on_environment(args):
    """Surface the failure modes that otherwise show up minutes into a run."""
    build, _ = torch_build()
    if build == "rocm" and args.attn_implementation == "flash_attention_2":
        logger.warning("flash_attention_2 requested on a ROCm build - flash-attn is CUDA-only here; expect a load failure")
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set - gated models on the Hub will fail to download")
    if not os.environ.get("HF_HOME"):
        logger.warning("HF_HOME is not set - models will download into the home directory cache")


def run(cmd, env):
    """Run the child process and return its exit code."""
    return subprocess.run(cmd, env=env, check=False).returncode
