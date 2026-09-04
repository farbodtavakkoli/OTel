"""
Fix a published HuggingFace model repo that is missing the multimodal processor
files (preprocessor_config.json / processor_config.json).

This patches the gap left by merge_lora_and_push.py before the AutoProcessor
save was added.  The fix:
  1. Reads the published model's config.json to find the base model name
     (via _name_or_path or a --base-model override).
  2. Loads AutoProcessor from the base model.
  3. Saves it to a temp dir.
  4. Uploads only the new files to the target repo (non-destructive).

Usage:
    python fix_missing_processor.py --repo-id gdiamos/relm-3-e2b-it [--token $HF_TOKEN]
    python fix_missing_processor.py --repo-id gdiamos/relm-3-e2b-it \
        --base-model google/gemma-4-2b-it --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSOR_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
}


def _resolve_token(cli_token: str | None) -> str | None:
    return (
        cli_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _fetch_published_config(repo_id: str, token: str | None) -> dict:
    from huggingface_hub import hf_hub_download

    logger.info("Fetching config.json from %s", repo_id)
    path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        token=token,
        repo_type="model",
    )
    with open(path) as f:
        return json.load(f)


def _existing_files(repo_id: str, token: str | None) -> set[str]:
    from huggingface_hub import list_repo_files

    return set(list_repo_files(repo_id, repo_type="model", token=token))


def _resolve_base_model(
    repo_id: str, token: str | None, override: str | None
) -> str:
    if override:
        logger.info("Using explicit base model: %s", override)
        return override

    cfg = _fetch_published_config(repo_id, token)
    candidates = [
        cfg.get("_name_or_path"),
        cfg.get("base_model"),
        cfg.get("base_model_name_or_path"),
    ]
    for candidate in candidates:
        if candidate and "/" in candidate:
            logger.info("Inferred base model from config.json: %s", candidate)
            return candidate

    model_type = cfg.get("model_type", "")
    archs = cfg.get("architectures", [])
    logger.error(
        "Cannot infer base model from config.json of %s "
        "(model_type=%r, architectures=%r, present fields: %s). "
        "Pass --base-model explicitly.",
        repo_id,
        model_type,
        archs,
        sorted(cfg.keys()),
    )
    raise ValueError(
        f"Cannot infer base model from config.json of {repo_id}. "
        "Pass --base-model explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="HuggingFace repo to fix (e.g. myorg/mymodel)")
    p.add_argument("--base-model", default=None, help="Base model to pull the processor from. Auto-detected from config.json if omitted.")
    p.add_argument("--token", default=None, help="HuggingFace token (falls back to HF_TOKEN env var)")
    p.add_argument("--dry-run", action="store_true", help="Save processor locally but do not upload")
    p.add_argument("--output-dir", type=Path, default=None, help="Where to save processor files (default: temp dir)")
    args = p.parse_args(argv)

    token = _resolve_token(args.token)

    base_model = _resolve_base_model(args.repo_id, token, args.base_model)

    logger.info("Checking which processor files already exist in %s", args.repo_id)
    try:
        existing = _existing_files(args.repo_id, token)
    except Exception as e:
        logger.warning("Could not list existing repo files: %s", e)
        existing = set()

    already_present = {f for f in PROCESSOR_FILES if f in existing}
    if already_present:
        logger.info("Already present: %s", sorted(already_present))

    from transformers import AutoProcessor

    logger.info("Loading processor from base model: %s", base_model)
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load processor from %s: %s", base_model, e)
        return 1

    tmp_dir, cleanup = (
        (Path(tempfile.mkdtemp(prefix="fix-processor-")), True)
        if args.output_dir is None
        else (args.output_dir, False)
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Saving processor to %s", tmp_dir)
        processor.save_pretrained(tmp_dir)

        saved = list(tmp_dir.iterdir())
        logger.info("Processor files saved: %s", [f.name for f in saved])

        to_upload = [f for f in saved if f.name not in already_present]
        skipped = [f for f in saved if f.name in already_present]
        if skipped:
            logger.info("Skipping (already in repo): %s", [f.name for f in skipped])
        if not to_upload:
            logger.info("Nothing new to upload — repo already has all processor files.")
            return 0

        logger.info("Files to upload: %s", [f.name for f in to_upload])

        if args.dry_run:
            logger.info("Dry run — skipping upload. Files are at %s", tmp_dir)
            cleanup = False  # keep for inspection
            return 0

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        for f in to_upload:
            logger.info("Uploading %s -> %s/%s", f.name, args.repo_id, f.name)
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"fix: add missing processor file {f.name}",
            )

        logger.info(
            "Done. Uploaded %d file(s) to https://huggingface.co/%s",
            len(to_upload),
            args.repo_id,
        )
        return 0

    finally:
        if cleanup:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
