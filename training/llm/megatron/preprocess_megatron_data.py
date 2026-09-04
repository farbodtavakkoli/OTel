"""Build Megatron's indexed .bin/.idx dataset from a JSONL corpus -- see readme_megatron.md."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils import _maybe_flatten_chat_jsonl

load_dotenv("dev.env")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("preprocess_megatron_data")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tokenize a JSONL corpus into Megatron's indexed .bin/.idx format.",
    )
    p.add_argument("--megatron-repo", required=True,
                   help="Path to a checked-out Megatron-LM clone (or /workspace/megatron "
                        "inside the NGC container).")
    p.add_argument("--input", default="data/OTel_LLM_sample_10.jsonl",
                   help="Input JSONL (one JSON object per line; default: the shipped sample). "
                        "With --partitions > 1 this is a glob, per upstream "
                        "tools/preprocess_data.py.")
    p.add_argument("--output-prefix", default="data/otel",
                   help="Output path prefix, without a suffix (default: data/otel).")
    p.add_argument("--tokenizer-model", default="meta-llama/Llama-3.1-8B",
                   help="Tokenizer to use — must match the training tokenizer. For "
                        "--tokenizer-type HuggingFaceTokenizer this is an HF repo id or a "
                        "local tokenizer directory (default: meta-llama/Llama-3.1-8B).")
    p.add_argument("--tokenizer-type", default="HuggingFaceTokenizer",
                   help="Megatron tokenizer type (default: HuggingFaceTokenizer).")
    p.add_argument("--json-keys", nargs="+", default=["text"],
                   help="JSON field(s) holding the document text (default: text). "
                        "Chat JSONL with a `messages` column is flattened to `text` first.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 16,
                   help="Worker processes (default: all cores). Upstream's rule of thumb is "
                        "workers * partitions = available cores, and workers must be "
                        "divisible by partitions.")
    p.add_argument("--partitions", type=int, default=1,
                   help="Split the input into N partitions processed in parallel, then merge "
                        "(default: 1).")
    p.add_argument("--no-append-eod", action="store_true",
                   help="Do not append the end-of-document token after each document. "
                        "Appending it is the right default for pre-training.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command and exit without running it.")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(args.megatron_repo).expanduser().resolve()
    script = repo / "tools" / "preprocess_data.py"
    if not script.is_file():
        raise SystemExit(
            f"{script} not found. Point --megatron-repo at a Megatron-LM clone: "
            "git clone https://github.com/NVIDIA/Megatron-LM.git"
        )

    # Absolutize paths: the upstream tool runs with the Megatron clone as cwd.
    args.input = str(Path(args.input).expanduser().resolve())
    args.output_prefix = str(Path(args.output_prefix).expanduser().resolve())

    input_path = Path(args.input)
    if not input_path.exists() and args.partitions == 1:
        raise SystemExit(f"--input not found: {args.input}")

    # This repo's LLM sample is chat JSONL (`messages`), not `{text: ...}`.
    # Flatten it so Megatron's --json-keys text still works.
    flattened = None
    if input_path.is_file() and args.partitions == 1:
        flattened = _maybe_flatten_chat_jsonl(input_path)
        if flattened is not None:
            log.info("flattened chat `messages` -> temporary {text} JSONL for Megatron")
            args.input = flattened

    if args.workers % args.partitions != 0:
        raise SystemExit(
            f"--workers ({args.workers}) must be divisible by --partitions "
            f"({args.partitions}); upstream drops non-divisible worker counts."
        )

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "tools/preprocess_data.py",
        "--input", args.input,
        "--output-prefix", args.output_prefix,
        "--tokenizer-type", args.tokenizer_type,
        "--tokenizer-model", args.tokenizer_model,
        "--json-keys", *args.json_keys,
        "--workers", str(args.workers),
        "--partitions", str(args.partitions),
    ]
    if not args.no_append_eod:
        cmd.append("--append-eod")

    log.info("cwd: %s", repo)
    log.info("command: %s", " ".join(cmd))
    if args.dry_run:
        log.info("--dry-run set; not executing.")
        return 0

    proc = subprocess.run(cmd, cwd=str(repo), env=os.environ.copy())
    if proc.returncode != 0:
        log.error("preprocessing failed with code %s", proc.returncode)
        return proc.returncode

    for key in args.json_keys:
        prefix = f"{args.output_prefix}_{key}_document"
        ok = all(Path(prefix + suffix).is_file() for suffix in (".bin", ".idx"))
        log.info("%s -> %s.bin / %s.idx", "wrote" if ok else "EXPECTED but missing:",
                 prefix, prefix)
        log.info("use this as data_path in megatron_cpt_config.toml: %s", prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
