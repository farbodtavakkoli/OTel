"""Fold a trained LoRA adapter back into its base model and push it to HuggingFace Hub."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


# Pure helpers — kept free of transformers / peft / huggingface_hub imports.


def find_latest_checkpoint(job_dir: Path) -> Path:
    """Return the `checkpoint_<step>.pt` with the highest step in `job_dir`."""
    candidates = []
    for p in job_dir.iterdir():
        name = p.name
        if not (name.startswith("checkpoint_") and name.endswith(".pt")):
            continue
        try:
            step = int(name[len("checkpoint_") : -len(".pt")])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_<step>.pt files found in {job_dir}"
        )
    candidates.sort()
    return candidates[-1][1]


def load_job_config(job_dir: Path) -> dict:
    """Load the saved job config (model name, lora_config, etc.)."""
    config_path = job_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f) or {}


def classify_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Split a saved adapter state_dict into `lora`, `tokenformer` and `base` buckets."""
    lora: dict[str, torch.Tensor] = {}
    tokenformer: dict[str, torch.Tensor] = {}
    base: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        leaf = k.rsplit(".", 1)[-1]
        if leaf in ("tokenformer_k", "tokenformer_v", "tokenformer_p"):
            tokenformer[k] = v
        elif (
            ".lora_A." in k
            or ".lora_B." in k
            or leaf in ("lora_embedding_A", "lora_embedding_B")
        ):
            lora[k] = v
        else:
            base[k] = v
    return {"lora": lora, "tokenformer": tokenformer, "base": base}


def infer_lora_rank(lora_keys: dict[str, torch.Tensor]) -> int:
    """Read the leading dim of any `lora_A` tensor — that's `r`."""
    for k, v in lora_keys.items():
        # Either `...lora_A.weight` or `...lora_A.<adapter_name>.weight`.
        if ".lora_A." in k:
            return int(v.shape[0])
    raise ValueError(
        "Cannot infer LoRA rank: no lora_A.* tensor in the state dict"
    )


def resolve_lora_config_args(
    job_config: dict,
    metadata: dict,
    lora_keys: dict[str, torch.Tensor],
    lora_alpha_override: int | None = None,
) -> dict:
    """Build peft.LoraConfig kwargs; rank comes from the saved tensors, not the job config."""
    rank = infer_lora_rank(lora_keys)
    user_lora = (job_config.get("lora_config") or {})
    if lora_alpha_override is not None:
        alpha = int(lora_alpha_override)
        alpha_source = "cli"
    elif metadata.get("lora_alpha") is not None:
        alpha = int(metadata["lora_alpha"])
        alpha_source = "metadata"
    elif user_lora.get("lora_alpha") is not None:
        alpha = int(user_lora["lora_alpha"])
        alpha_source = "job_config"
    else:
        alpha = 2 * rank
        alpha_source = "default"

    target_modules = user_lora.get("target_modules") or "all-linear"

    use_rslora = bool(metadata.get("use_rslora", user_lora.get("use_rslora", False)))

    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": float(user_lora.get("lora_dropout", 0.0)),
        "target_modules": target_modules,
        "bias": "none",
        "use_rslora": use_rslora,
        "_alpha_source": alpha_source,  # popped out before LoraConfig sees it
    }


PEFT_OUTER_PREFIX = "base_model.model."


def _prefix_for_peft_load(
    lora_keys: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-prefix raw checkpoint keys to match `peft.PeftModel.state_dict()`. Idempotent."""
    out: dict[str, torch.Tensor] = {}
    for k, v in lora_keys.items():
        if k.startswith(PEFT_OUTER_PREFIX):
            out[k] = v
        else:
            out[f"{PEFT_OUTER_PREFIX}{k}"] = v
    return out


# CLI / main flow. transformers, peft and huggingface_hub are imported lazily.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="merge_lora_and_push",
        description=(
            "Fold a LoRA adapter back into its base model and push the "
            "merged weights to HuggingFace Hub."
        ),
    )
    p.add_argument(
        "--job-dir",
        required=True,
        type=Path,
        help="Path to the training-job directory (contains config.yaml and checkpoint_*.pt).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo to push the merged model to (e.g. myorg/mymodel).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint .pt to load. Defaults to the latest in --job-dir.",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Override the lora_alpha used when reconstructing the LoRA. Defaults to checkpoint metadata, then job config, then 2*rank.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it doesn't exist.",
    )
    p.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the upload. Defaults to a generated one.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Where to materialize the merge math. CPU is safe for any size; CUDA is faster but needs free VRAM.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to save the merged model before upload. Defaults to a temp dir that's cleaned up.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Merge and save locally; skip the HF upload.",
    )
    p.add_argument(
        "--dtype",
        default=None,
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Override the dtype used for the merged model. Defaults to the base model's native dtype.",
    )
    p.add_argument(
        "--mode",
        default="merged",
        choices=("merged", "adapter"),
        help=(
            "What to publish. `merged` (default) folds the LoRA into the "
            "base model and uploads a self-contained model repo. "
            "`adapter` exports a PEFT-format adapter repo "
            "(adapter_config.json + adapter_model.safetensors) — small, "
            "loadable via `PeftModel.from_pretrained(base, repo)`. "
            "(Adapter mode is wired in a follow-up commit.)"
        ),
    )
    p.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file the script writes phase markers into "
            "while running. Used by the publish-SLURM-job orchestrator "
            "(see launch_publish_job.py) to drive the UI's progress view."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    status_writer = _StatusWriter(args.status_file, mode=args.mode)
    status_writer.update(phase="validating", started_at=_now())

    job_dir = args.job_dir.resolve()
    if not job_dir.is_dir():
        msg = f"Job directory does not exist: {job_dir}"
        status_writer.update(phase="error", error=msg, completed_at=_now())
        logger.error(msg)
        return 2

    job_config = load_job_config(job_dir)
    base_model_name = job_config.get("llm_name") or job_config.get("model")
    if not base_model_name:
        msg = f"Job config at {job_dir}/config.yaml has no llm_name/model field."
        status_writer.update(phase="error", error=msg, completed_at=_now())
        logger.error(msg)
        return 2

    checkpoint_path = args.checkpoint or find_latest_checkpoint(job_dir)
    logger.info("Loading checkpoint %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    metadata = checkpoint.get("metadata", {}) or {}

    buckets = classify_state_dict(state_dict)
    lora_count = len(buckets["lora"])
    tk_count = len(buckets["tokenformer"])
    base_count = len(buckets["base"])
    logger.info(
        "Classified state_dict: lora=%d, tokenformer=%d, base_overrides=%d",
        lora_count,
        tk_count,
        base_count,
    )
    if tk_count > 0:
        logger.warning(
            "Checkpoint contains %d tokenformer parameter(s); they will be "
            "DROPPED — folding tokenformer adapters into a vanilla model "
            "isn't supported. The export will only carry the LoRA delta"
            "%s.",
            tk_count,
            (
                " plus any base-weight overrides"
                if args.mode == "merged"
                else ""
            ),
        )
    if lora_count == 0:
        logger.error(
            "No LoRA tensors in checkpoint. Tokenformer-only and "
            "base-overrides-only checkpoints are not supported here."
        )
        return 2

    from peft import LoraConfig

    lora_kwargs = resolve_lora_config_args(
        job_config,
        metadata,
        buckets["lora"],
        lora_alpha_override=args.lora_alpha,
    )
    alpha_source = lora_kwargs.pop("_alpha_source")
    logger.info(
        "LoRA: r=%d, alpha=%d (from %s), use_rslora=%s, target_modules=%s",
        lora_kwargs["r"],
        lora_kwargs["lora_alpha"],
        alpha_source,
        lora_kwargs["use_rslora"],
        lora_kwargs["target_modules"],
    )
    lora_config = LoraConfig(**lora_kwargs)

    output_dir, cleanup_tmp = _resolve_output_dir(args.output_dir, args.mode)

    if args.mode == "adapter":
        status_writer.update(phase="saving")
        if base_count > 0:
            logger.warning(
                "Adapter-mode export drops %d base-weight override(s); "
                "those tensors only make sense paired with a merged base. "
                "Re-run with --mode merged if you need them.",
                base_count,
            )
        _export_adapter_repo(
            output_dir=output_dir,
            lora_tensors=buckets["lora"],
            lora_config=lora_config,
            base_model_name=base_model_name,
        )
    else:
        # mode == "merged" — fold LoRA into base, save full model.
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = _resolve_dtype(args.dtype)
        status_writer.update(phase="loading_base", base_model=base_model_name)
        logger.info("Materializing base model: %s", base_model_name)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype if torch_dtype is not None else "auto",
            low_cpu_mem_usage=True,
        )
        if args.device == "cuda":
            base = base.to("cuda")

        if base_count > 0:
            _apply_base_overrides(base, buckets["base"])

        status_writer.update(phase="merging")
        peft_model = get_peft_model(base, lora_config)

        # On-disk keys lack the outer prefix; without it `merge_and_unload` folds zero deltas.
        prefixed_lora = _prefix_for_peft_load(buckets["lora"])
        load_result = peft_model.load_state_dict(prefixed_lora, strict=False)
        loaded_count = len(prefixed_lora) - len(load_result.unexpected_keys)
        logger.info(
            "LoRA load: matched %d / %d tensors against the wrapped model.",
            loaded_count,
            len(prefixed_lora),
        )
        if load_result.unexpected_keys:
            # Usually vision/audio towers AutoModelForCausalLM never materialises. Not fatal.
            logger.warning(
                "LoRA load: %d tensors unmatched after prefix fix (e.g. %s). "
                "These were saved by the trainer but the merge target's "
                "module tree doesn't expose them — usually vision/audio "
                "tower modules absent from AutoModelForCausalLM.",
                len(load_result.unexpected_keys),
                load_result.unexpected_keys[:3],
            )
        if loaded_count == 0:
            raise RuntimeError(
                "merge_lora_and_push: 0 LoRA tensors matched the rebuilt "
                "PEFT model. Aborting before producing a no-op merge that "
                "looks successful but uploads the unchanged base model."
            )
        # "Missing" keys are non-LoRA params absent from the adapter dict; expected.
        logger.info("Merging LoRA into base...")
        merged = peft_model.merge_and_unload()

        status_writer.update(phase="saving")
        logger.info("Saving merged model to %s", output_dir)
        merged.save_pretrained(output_dir, safe_serialization=True)
        from .backfill_weights import backfill_missing_weights
        backfill_missing_weights(output_dir, base_model_name)
        AutoTokenizer.from_pretrained(base_model_name).save_pretrained(output_dir)
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(
                base_model_name, trust_remote_code=True
            ).save_pretrained(output_dir)
            logger.info("Saved multimodal processor to %s", output_dir)
        except Exception as proc_err:
            logger.debug(
                "No multimodal processor for %s (text-only model): %s",
                base_model_name,
                proc_err,
            )

    if args.dry_run:
        logger.info(
            "Dry run — skipping upload. Output is at %s", output_dir
        )
        status_writer.update(phase="done", completed_at=_now(), repo_url=None)
        if cleanup_tmp:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        return 0

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    commit_message = args.commit_message or _default_commit_message(
        job_dir, checkpoint_path, lora_kwargs
    )
    status_writer.update(phase="uploading", repo_id=args.repo_id)
    try:
        _push(output_dir, args.repo_id, token, args.private, commit_message)
    except Exception as e:
        status_writer.update(
            phase="error",
            error=f"upload failed: {e}",
            completed_at=_now(),
        )
        if cleanup_tmp:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        raise

    status_writer.update(
        phase="done",
        completed_at=_now(),
        repo_url=f"https://huggingface.co/{args.repo_id}",
    )

    if cleanup_tmp:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    return 0


def _resolve_dtype(name: str | None):
    if name is None or name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolve_output_dir(
    requested: Path | None, mode: str
) -> tuple[Path, bool]:
    """Return (output_dir, cleanup_tmp), creating a temp dir when none was requested."""
    if requested is None:
        prefix = "scalarlm-merged-" if mode == "merged" else "scalarlm-adapter-"
        return Path(tempfile.mkdtemp(prefix=prefix)), True
    requested.mkdir(parents=True, exist_ok=True)
    return requested, False


# The trainer's raw PEFT state_dict keeps the `.default.` segment save_pretrained strips.
def strip_default_adapter_segment(key: str) -> str:
    """Drop the adapter-name segment between `lora_A`/`lora_B` and `weight`, if present."""
    parts = key.split(".")
    # Only drop the segment after lora_* when it isn't "weight"/"bias" itself.
    for i in range(len(parts) - 2):
        if parts[i] in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            tail = parts[i + 1]
            terminal = parts[i + 2] if i + 2 < len(parts) else ""
            if tail not in ("weight", "bias") and terminal in ("weight", "bias"):
                return ".".join(parts[: i + 1] + parts[i + 2 :])
    return key


def _export_adapter_repo(
    output_dir: Path,
    lora_tensors: "dict[str, torch.Tensor]",
    lora_config: "object",
    base_model_name: str,
) -> None:
    """Write a HF-standard PEFT adapter repo: config JSON plus adapter safetensors."""
    import json
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    renamed: dict[str, "torch.Tensor"] = {}
    for k, v in lora_tensors.items():
        renamed[strip_default_adapter_segment(k)] = v.contiguous()

    safetensors_path = output_dir / "adapter_model.safetensors"
    save_file(renamed, str(safetensors_path))
    logger.info(
        "Wrote %d adapter tensor(s) to %s", len(renamed), safetensors_path
    )

    # base_model_name_or_path lets `PeftModel.from_pretrained(repo)` find the base itself.
    cfg_dict = lora_config.to_dict()
    cfg_dict["base_model_name_or_path"] = base_model_name
    cfg_dict.setdefault("peft_type", "LORA")
    cfg_dict.setdefault("task_type", "CAUSAL_LM")
    config_path = output_dir / "adapter_config.json"
    with config_path.open("w") as f:
        json.dump(cfg_dict, f, indent=2)
    logger.info("Wrote adapter config to %s", config_path)


def _now() -> float:
    import time
    return time.time()


class _StatusWriter:
    """Phase tracker for the publish flow; a `path` of None makes every update a no-op."""

    def __init__(self, path: Path | None, mode: str):
        self.path = path
        self._state = {"mode": mode, "phase": "queued"}

    def update(self, **fields) -> None:
        if self.path is None:
            return
        self._state.update(fields)
        # Atomic write: dump to a sibling tempfile, then rename.
        import json
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w") as f:
                json.dump(self._state, f)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning("Failed to write status file %s: %s", self.path, e)


def _apply_base_overrides(base, overrides: dict[str, torch.Tensor]) -> None:
    sd = base.state_dict()
    applied = 0
    skipped = 0
    for name, tensor in overrides.items():
        if name not in sd:
            logger.debug("Base override key not present in base model: %s", name)
            skipped += 1
            continue
        target = sd[name]
        if target.shape != tensor.shape:
            logger.warning(
                "Skipping base override %s: shape mismatch %s vs %s",
                name,
                tuple(target.shape),
                tuple(tensor.shape),
            )
            skipped += 1
            continue
        target.copy_(tensor.to(target.dtype).to(target.device))
        applied += 1
    logger.info("Applied %d base-weight overrides (%d skipped)", applied, skipped)


def _default_commit_message(
    job_dir: Path, checkpoint_path: Path, lora_kwargs: dict
) -> str:
    return (
        f"merge_lora_and_push: job={job_dir.name} "
        f"checkpoint={checkpoint_path.name} "
        f"r={lora_kwargs['r']} alpha={lora_kwargs['lora_alpha']}"
    )


def _push(output_dir: Path, repo_id: str, token: str | None, private: bool, message: str) -> None:
    from huggingface_hub import HfApi, create_repo

    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    logger.info("Creating repo %s (exist_ok=True, private=%s)", repo_id, private)
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")

    api = HfApi(token=token)
    logger.info("Uploading %s -> %s", output_dir, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    logger.info("Push complete: https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    sys.exit(main())
