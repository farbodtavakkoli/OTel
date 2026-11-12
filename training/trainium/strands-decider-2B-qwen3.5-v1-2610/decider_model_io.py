"""Full-weight serialization for the released decoder torso and pointer head."""

import json

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from decider_recipe_utils import digest, verify_manifest, write_json


def save_full(model, path):
    path.mkdir(parents=True, exist_ok=False)
    model.torso.save_pretrained(path / "backbone", safe_serialization=True)
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        path / "head.safetensors",
    )
    model.tokenizer.save_pretrained(path)
    (path / "strands_decider_config.json").write_text(model.config.to_json())
    (path / "format.json").write_text(
        json.dumps({"type": "strands_full", "version": 1})
    )
    manifest_files = {
        str(p.relative_to(path)): digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }
    write_json(path / "manifest.json", {"format": 1, "files": manifest_files})


def load_full(path):
    verify_manifest(path)
    if json.loads((path / "format.json").read_text()).get("type") != "strands_full":
        raise ValueError("Unsupported full-model format")
    config = StrandsDeciderConfig.from_json(str(path / "strands_decider_config.json"))
    torso = AutoModel.from_pretrained(
        path / "backbone",
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = StrandsDeciderModel(
        config,
        torso,
        AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        ),
    )
    model.head.load_state_dict(load_file(path / "head.safetensors"), strict=True)
    return model
