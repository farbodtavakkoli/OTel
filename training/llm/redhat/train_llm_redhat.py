"""OSFT (Orthogonal Subspace Fine-Tuning) trainer via training_hub — see readme_redhat.md."""
import os
import sys
import time
import json
import argparse
import glob
import contextlib
from datetime import datetime

from dotenv import load_dotenv

# Loaded before training_hub imports transformers/datasets/hf_hub, which resolve
# HF_TOKEN and cache locations at import time.
load_dotenv("dev.env")

from training_hub import osft
from speed_monitor import speed_monitor


def parse_args():
    parser = argparse.ArgumentParser(description='OSFT Training Example')

    # Required parameters
    parser.add_argument('--model-path', required=True,
    help='Model path or HuggingFace name (any causal LM supported by '
    'training_hub\'s dependencies)')
    parser.add_argument('--data-path', default='data/OTel_LLM_sample_10.jsonl',
    help='Path to training data (JSONL format; default: the bundled sample)')
    parser.add_argument('--ckpt-output-dir', required=True,
    help='Directory to save checkpoints')

    # Optional overrides
    parser.add_argument('--num-epochs', type=int, default=4,
    help='Number of epochs (default: 4)')
    parser.add_argument('--unfreeze-rank-ratio', type=float, default=0.3,
    help='Unfreeze rank ratio for OSFT (0.0-1.0, default: 0.3)')
    parser.add_argument('--effective-batch-size', type=int, default=128,
    help='Effective batch size (default: 128, good for >10k sample datasets)')
    parser.add_argument('--learning-rate', type=float, default=5e-6,
    help='Learning rate for training (default: 5e-6)')
    parser.add_argument('--max-seq-len', type=int, default=4096,
    help='Max sequence length in tokens (default: 4096)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=8192,
    help='Max tokens per GPU (default: 8192; a conservative starting '
    'point. Lower it for larger models or large-vocabulary models '
    'if you hit OOM, raise it on H100 if you have memory headroom)')
    parser.add_argument('--nproc-per-node', type=int, default=8,
    help='Number of GPUs (default: 8 for a single 8x H100 node)')
    parser.add_argument('--data-output-dir', default='data_output',
    help='Directory for processed data (default: data_output; point it at '
    'a RAM disk such as /dev/shm for speed)')
    parser.add_argument('--unmask-messages', action=argparse.BooleanOptionalAction,
    default=True,
    help='Control which conversation turns contribute to the training '
    'loss/gradient. When enabled (default), ALL turns (system, user, '
    'and assistant) are unmasked and contribute to the gradient. Pass '
    '--no-unmask-messages to mask non-assistant turns so that ONLY '
    'assistant turns contribute to the gradient (standard SFT behavior).')
    parser.add_argument('--eos-token', default=None,
    help='Override the tokenizer\'s EOS token used for data '
    'processing and saved with the checkpoint. Needed when the '
    'chat template ends the assistant turn with a token other '
    'than the tokenizer\'s default eos_token. For example, Gemma 4 '
    'closes turns with "<turn|>" (not <eos>), so instructlab\'s '
    'unmasking never unmasks the terminator and the model never '
    'learns to stop; pass --eos-token "<turn|>" to fix it. When '
    'unset (default), the tokenizer is used unchanged.')
    parser.add_argument('--seed', type=int, default=42,
    help='Random seed for training (default: 42)')
    parser.add_argument('--speed-steps', type=int, default=0,
    help='If > 0, print a live speed/ETA report every N completed '
    'training steps while training runs. Reads '
    'training_metrics_0.jsonl from --ckpt-output-dir and reports '
    'estimated total remaining time, time to finish the current '
    'epoch, max peak memory, max peak tokens/sec, and last '
    'validation loss. 0 disables (default).')

    # Validation options (off by default; matches prior behavior when unset)
    parser.add_argument('--validation-split', type=float, default=0.0,
    help='Fraction of data held out for validation, [0.0, 1.0). '
    '0.0 disables validation (default). e.g. 0.01 = 1%%.')
    parser.add_argument('--validation-frequency', type=int, default=None,
    help='Run validation every N steps. Required when '
    '--validation-split > 0.')
    parser.add_argument('--save-best-val-loss', action='store_true', default=False,
    help='Save a checkpoint whenever validation loss improves '
    '(saved as hf_format/samples_*_best_val_loss).')

    args = parser.parse_args()

    if args.validation_split > 0.0 and not (args.validation_frequency and args.validation_frequency > 0):
        parser.error('--validation-frequency must be a positive integer when --validation-split > 0')

    return args


def find_most_recent_checkpoint(output_dir):
    """Return the most recently created hf_format/samples_* checkpoint under output_dir."""
    checkpoint_pattern = os.path.join(output_dir, "hf_format", "samples_*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {os.path.join(output_dir, 'hf_format')}")

    most_recent_checkpoint = max(checkpoint_dirs, key=os.path.getctime)

    return most_recent_checkpoint


def save_run_args(args, log_root="logs"):
    """Write the run's arguments to {log_root}/<timestamp>/run_args.json and return that dir."""
    run_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        # default=str keeps the dump robust to non-JSON-serializable values.
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    return run_dir


def prepare_model_with_eos(model_path, eos_token, staging_dir):
    """Stage a symlinked copy of model_path with the tokenizer's EOS overridden to eos_token."""
    import shutil
    import tempfile
    from transformers import AutoTokenizer, AutoConfig

    # Resolve the model to a local directory (uses the HF cache for hub ids).
    if os.path.isdir(model_path):
        src = model_path
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(model_path)

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    # Symlink every source file so weights are available without copying GBs.
    for name in os.listdir(src):
        os.symlink(os.path.join(src, name), os.path.join(staging_dir, name))

    # Validate the requested EOS token exists, then override it on the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(src)
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_id is None or eos_id == tokenizer.unk_token_id:
        raise ValueError(
        f"--eos-token {eos_token!r} is not a known token in {model_path}'s vocabulary"
        )
    tokenizer.eos_token = eos_token

    # Regenerate tokenizer files in a temp dir, then replace the staging symlinks
    # with real files (so save_pretrained never writes through into the base model).
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        for name in os.listdir(tmp):
            dst = os.path.join(staging_dir, name)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(os.path.join(tmp, name), dst)

    # Keep config.json's eos_token_id in sync (also propagates to the checkpoint,
    # so the served model stops on the terminator too).
    config = AutoConfig.from_pretrained(src)
    config.eos_token_id = eos_id
    cfg_path = os.path.join(staging_dir, "config.json")
    if os.path.islink(cfg_path) or os.path.exists(cfg_path):
        os.remove(cfg_path)
    config.save_pretrained(staging_dir)

    # Ensure the chat template survives (save_pretrained may embed it in
    # tokenizer_config.json instead of writing chat_template.jinja).
    chat_tmpl = os.path.join(staging_dir, "chat_template.jinja")
    src_tmpl = os.path.join(src, "chat_template.jinja")
    if not os.path.exists(chat_tmpl) and os.path.exists(src_tmpl):
        shutil.copy2(src_tmpl, chat_tmpl)

    return staging_dir, eos_id


def main():
    args = parse_args()

    # Record the exact arguments this run was launched with, before training starts.
    log_dir = save_run_args(args)

    # If an EOS override was requested, prepare a model directory whose tokenizer
    # uses it, and train from that instead. Leaves args.model_path untouched for
    # display/logging.
    model_path = args.model_path
    if args.eos_token:
        staging_dir = os.path.join(args.data_output_dir, 'eos_override_model')
        model_path, eos_id = prepare_model_with_eos(
            args.model_path, args.eos_token, staging_dir
        )

    # OSFT configuration
    print("🚀 OSFT Training")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    if args.eos_token:
        print(f"EOS override: {args.eos_token!r} (id {eos_id}) — training from {model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"Run args logged to: {log_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Unfreeze Rank Ratio: {args.unfreeze_rank_ratio}")
    print(f"Effective batch size: {args.effective_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_len:,}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    if args.speed_steps > 0:
        print(f"Speed/ETA reporting: every {args.speed_steps} step(s)")
    if args.validation_split > 0.0:
        print(f"Validation split: {args.validation_split:.1%} "
    f"(every {args.validation_frequency} steps, "
    f"save_best_val_loss={args.save_best_val_loss})")
    print()
    print(f"📝 OSFT Benefits for {args.model_path}:")
    print(" • Preserve the base model's general capabilities")
    print(" • Add domain-specific knowledge efficiently")
    print(" • No need for complex data mixing or replay buffers")
    print(" • Continually train from the instruction-tuned base")
    print()

    # Training configuration for OSFT on 8x H100
    start_time = time.time()

    try:
        osft_params = {
            # Model and data
            'model_path': model_path,
            'data_path': args.data_path,
            'ckpt_output_dir': args.ckpt_output_dir,

            # OSFT-specific parameters
            'unfreeze_rank_ratio': args.unfreeze_rank_ratio, # 0.3 balances adaptation vs preservation

            # Training parameters for OSFT on 8x H100
            'num_epochs': args.num_epochs,
            'effective_batch_size': args.effective_batch_size,
            'learning_rate': args.learning_rate,
            'max_seq_len': args.max_seq_len,
            'max_tokens_per_gpu': args.max_tokens_per_gpu,

            # Data processing
            'data_output_dir': args.data_output_dir, # RAM disk for speed by default
            'warmup_steps': 0,
            'unmask_messages': args.unmask_messages,

            # Optimization
            'use_liger': True, # Liger kernels reduce memory, especially for large-vocab models
            'osft_memory_efficient_init': True, # Recommended for OOMs at model load time
            'seed': args.seed,
            'lr_scheduler': 'cosine', # Cosine scheduler works well with OSFT

            # Validation (forwarded to mini-trainer via kwargs; None values are ignored)
            'validation_split': args.validation_split,
            'validation_frequency': args.validation_frequency,
            'save_best_val_loss': args.save_best_val_loss,

            # Checkpointing
            'checkpoint_at_epoch': True,
            'save_final_checkpoint': True,

            # Single-node multi-GPU setup (8x H100)
            'nproc_per_node': args.nproc_per_node,
            'nnodes': 1,
            'node_rank': 0,
            'rdzv_id': 105,
            # Default single-node rendezvous endpoint. Override on a shared box
            # (concurrent torchrun collides on a fixed port) via the
            # RDZV_ENDPOINT env var, e.g. RDZV_ENDPOINT=127.0.0.1:29647.
            'rdzv_endpoint': os.environ.get("RDZV_ENDPOINT", "127.0.0.1:29500"),
        }

        # Optionally monitor training speed / ETA in the background. When disabled
        # (--speed-steps 0), this is a no-op context so training runs unchanged.
        if args.speed_steps > 0:
            monitor_ctx = speed_monitor(
            ckpt_output_dir=args.ckpt_output_dir,
            data_path=args.data_path,
            effective_batch_size=args.effective_batch_size,
            num_epochs=args.num_epochs,
            report_every_steps=args.speed_steps,
        )
        else:
            monitor_ctx = contextlib.nullcontext()

        with monitor_ctx:
            osft(**osft_params)

        end_time = time.time()
        duration = end_time - start_time

        most_recent_checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)

        print("=" * 50)
        print("✅ OSFT Training completed successfully!")
        print(f"⏱️ Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format")
        print(f" Most recent checkpoint: {most_recent_checkpoint}")
        print()
        print(f"🎯 Your {args.model_path} model has been successfully adapted!")
        print(" The model now incorporates your domain-specific knowledge")
        print(" while maintaining its original high-quality capabilities.")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print("=" * 50)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("💡 Troubleshooting tips:")
        print(" - Reduce --max-tokens-per-gpu if you see OOM errors (large / large-vocab models are memory-hungry)")
        print(" - For domain adaptation, try --unfreeze-rank-ratio between 0.2-0.3")
        print(" - Reduce --effective-batch-size further for memory constraints")
        sys.exit(1)


if __name__ == "__main__":
    main()