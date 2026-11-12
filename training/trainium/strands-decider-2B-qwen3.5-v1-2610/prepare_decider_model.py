"""Download a pinned released model and verify architecture-preserving LoRA merge."""

import hashlib
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from decider_recipe_utils import (
    MODEL_ID,
    MODEL_REVISION,
    StrictTokenizer,
    digest,
    load_rows,
    write_json,
    require,
)


def architecture(model):
    decider = json.loads(model.config.to_json())
    for key in ("temperature", "temperature_by_kind"):
        decider.pop(
            key, None
        )  # Intentional training/calibration state, not architecture.
    # Compare the same JSON representation used by the persisted snapshot.
    # For example, HF id2label has integer keys in memory and string keys on disk.
    torso = json.loads(json.dumps(model.torso.config.to_dict()))
    for key in (
        "_name_or_path",
        "transformers_version",
        "dtype",
        "torch_dtype",
        "architectures",
    ):
        torso.pop(key, None)
    vocab_hash = hashlib.sha256(
        json.dumps(model.tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    return dict(
        decider_config=decider,
        tokenizer_vocab_sha256=vocab_hash,
        special_tokens=model.tokenizer.special_tokens_map,
        dropout={
            n: m.p
            for n, m in model.named_modules()
            if hasattr(m, "p") and "Dropout" in type(m).__name__
        },
        parameters={n: list(p.shape) for n, p in model.named_parameters()},
        modules={
            n: type(m).__module__ + "." + type(m).__name__
            for n, m in model.named_modules()
        },
        torso_config=torso,
        head_type=type(model.head).__name__,
        parameter_count=sum(p.numel() for p in model.parameters()),
    )


def download_snapshot_for_preparation(output, cache):
    """Leave output untouched until the pinned Hub snapshot is available."""
    import httpx
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        LocalTokenNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if output.exists() or output.is_symlink():
        raise SystemExit(
            f"Preparation output already exists: {output}. Choose a fresh output directory."
        )
    try:
        released = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache,
        )
    except RevisionNotFoundError:
        reason = (
            "The pinned revision was not found. Verify the recipe's model revision."
        )
    except (GatedRepoError, LocalTokenNotFoundError):
        reason = "Authenticate with `hf auth login` and confirm access to this model."
    except RepositoryNotFoundError:
        reason = "The repository was not found or is private. Verify the model ID and your access."
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            reason = "Authentication or access was denied. Run `hf auth login` and confirm model access."
        else:
            reason = f"The Hub returned HTTP {status or 'error'}. Check Hub availability and retry."
    except (httpx.TransportError, LocalEntryNotFoundError, OfflineModeIsEnabled):
        # IncompleteSnapshotError is a LocalEntryNotFoundError subclass in 1.33.
        reason = (
            "The Hub is unreachable or the cached snapshot is incomplete. "
            "Check connectivity, proxy/TLS settings and HF_HUB_OFFLINE, then retry."
        )
    except (OSError, RuntimeError):
        reason = (
            "The snapshot transfer or cache operation failed. "
            "Check connectivity, free disk space and cache write permissions, then retry."
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
        return released
    raise SystemExit(
        f"Failed to download {MODEL_ID} at revision {MODEL_REVISION}. {reason}"
    ) from None


def main(args):
    import torch
    from strands_decider.modeling import StrandsDeciderModel
    from decider_model_io import load_full, save_full

    sys.path.insert(0, str(args.upstream))
    from training.data.collate import CollatorConfig, SystemOneCollator
    from training.data.format import Example

    os.umask(0o077)
    torch.set_num_threads(6)
    rows = [Example.from_dict(r) for r in load_rows(args.data)]
    released = download_snapshot_for_preparation(args.output, args.cache)
    model = (
        StrandsDeciderModel.load(released, trainable=True, attn_implementation="eager")
        .float()
        .eval()
    )
    collator = SystemOneCollator(
        StrictTokenizer(model.tokenizer, args.pad_to),
        CollatorConfig(max_length=args.pad_to),
        train=False,
    )
    # Check every row for truncation; bound retained batches used for numerical parity.
    lengths = [collator([r])["input_ids"].shape[1] for r in rows]
    batches = [collator([r]) for r in rows[:10]]
    with torch.no_grad():
        original = [model(**b)["logits"].cpu() for b in batches]
    model.torso = model.torso.merge_and_unload(safe_merge=True)
    model.config.use_lora = False
    model.requires_grad_(True)
    require(
        not any("lora_" in n for n, _ in model.named_parameters()),
        "Unmerged adapter remains",
    )
    with torch.no_grad():
        merged = [model(**b)["logits"].cpu() for b in batches]
    for left, right in zip(original, merged):
        torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)
    shape = architecture(model)
    save_full(model, args.output / "merged")
    del model
    gc.collect()
    restored = load_full(args.output / "merged").eval()
    require(
        architecture(restored)["parameters"] == shape["parameters"],
        "Reload parameter layout changed",
    )
    with torch.no_grad():
        for expected, batch in zip(merged, batches):
            torch.testing.assert_close(
                expected, restored(**batch)["logits"], atol=1e-6, rtol=1e-6
            )
    write_json(args.output / "architecture.json", shape)
    write_json(
        args.output / "preparation.json",
        dict(
            status="passed",
            rows=len(rows),
            token_lengths=lengths,
            parity_rows=len(batches),
            merge_parity=True,
            reload_parity=True,
            truncated_rows=0,
            parameter_count=shape["parameter_count"],
            data_sha256=digest(args.data),
        ),
    )
    print("Prepared and verified full-weight initialization:", args.output / "merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("upstream", "data", "output", "cache"):
        parser.add_argument("--" + n, type=Path, required=True)
    parser.add_argument("--pad-to", type=int, default=768)
    main(parser.parse_args())
