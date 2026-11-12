"""Input, provenance and no-truncation controls shared by recipe commands."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "13faf5214f7cbea07d005278f1aeaf3973611eba"
MODEL_ID = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
MODEL_REVISION = "216d1ccb46a9e9b036eb9187b248db98480f77d4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def load_rows(path):
    path = Path(path)
    require(
        path.stat().st_size <= 64 * 1024 * 1024,
        "Dataset exceeds the 64 MiB in-memory recipe limit",
    )
    rows, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(
            isinstance(row, dict) and row.get("kind") == "choice",
            f"Row {n}: this recipe supports choice tasks",
        )
        require(
            isinstance(row.get("state"), str) and row["state"].strip(),
            f"Row {n}: state must be nonempty text",
        )
        require(
            isinstance(row.get("instructions"), str) and row["instructions"].strip(),
            f"Row {n}: instructions missing",
        )
        options = row.get("options")
        require(
            isinstance(options, list) and 2 <= len(options) <= 32,
            f"Row {n}: expected 2–32 options",
        )
        require(
            all(
                isinstance(x, list)
                and len(x) == 2
                and all(isinstance(v, str) and v.strip() for v in x)
                for x in options
            ),
            f"Row {n}: options must be [name, description] pairs",
        )
        require(
            len({x[0] for x in options}) == len(options),
            f"Row {n}: duplicate option name",
        )
        require(
            type(row.get("label")) is int and 0 <= row["label"] < len(options),
            f"Row {n}: invalid label index",
        )
        require(
            row.get("weight", 1) == 1,
            f"Row {n}: weighted examples are not supported in this recipe",
        )
        require(
            not row.get("instruction_variants"),
            f"Row {n}: use one instruction per example for static-shape validation",
        )
        key = " ".join(row["state"].split())
        require(key not in seen, f"Row {n}: duplicate normalized state")
        seen.add(key)
        rows.append(row)
    require(rows, "Dataset is empty")
    return rows


class StrictTokenizer:
    """Use the upstream renderer/collator, but reject rather than truncate input."""

    def __init__(self, tokenizer, limit):
        self.tokenizer, self.limit = tokenizer, limit

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        kwargs["truncation"] = False
        kwargs.pop("max_length", None)
        result = self.tokenizer(*args, **kwargs)
        require(
            result["input_ids"].shape[1] <= self.limit,
            f"Prompt exceeds {self.limit} tokens; shorten examples or explicitly raise --pad-to",
        )
        return result


def training_signature(args, world):
    root = Path(__file__).resolve().parent
    rev = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    require(rev == UPSTREAM_REVISION, "Unexpected upstream revision")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.upstream),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    require(not dirty, "Upstream tracked source is modified")
    return dict(
        format=1,
        world=world,
        upstream_revision=rev,
        pad_to=args.pad_to,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        optimizer="AdamW(.9,.95),weight_decay=0,clip=1",
        source={
            name: digest(root / name)
            for name in (
                "train_decider_full.py",
                "decider_training_step.py",
                "decider_checkpoints.py",
                "decider_model_io.py",
                "decider_recipe_utils.py",
            )
        },
        initialization=digest(args.merged / "manifest.json"),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torch-neuronx",
                "transformers",
                "peft",
                "safetensors",
            )
        },
    )


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Missing model file manifest")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    require(actual == set(files), "Model file set differs from manifest")
    for relative, expected in files.items():
        path = root / relative
        require(
            not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Invalid manifest path",
        )
        require(
            not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "Model manifest must contain local regular files",
        )
        require(digest(path) == expected, "Model checksum mismatch: " + relative)
