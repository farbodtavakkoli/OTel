"""Run a trained checkpoint through HuggingFace instead of vLLM.

"Trained model answers with garbage" has two causes that look identical
from outside: a bad checkpoint, or a serving path that loses weights.
Same weights through a second inference stack tells them apart in a
minute, with no rebuild:

  - HF correct, vLLM garbage -> serving is losing weights
  - both garbage             -> the checkpoint is bad, serving is innocent

That distinction is what ended a long debugging session where training,
checkpointing and adapter loading all reported success.

Mirrors the trainer: same model class, same Tokenformer surgery, same
prompt construction (input + output concatenated, no chat template).

    docker compose exec cray-nvidia \
        python /app/cray/test/deployment/verify_checkpoint_hf.py \
            [--job-dir /app/cray/jobs/<hash>] [--prompt "What is 3 + 3?"]
"""

import argparse
import glob
import os

import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from tokenformer.tokenformer_surgeon import TokenformerSurgeon


def latest_job_dir(jobs_root):
    candidates = [
        d for d in glob.glob(os.path.join(jobs_root, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.pt"))
    ]
    if not candidates:
        raise SystemExit(f"No job directory with a .pt checkpoint under {jobs_root}")
    return max(candidates, key=os.path.getmtime)


def latest_checkpoint(job_dir):
    # Match the serving side: sorted() so a directory with several
    # checkpoints picks deterministically, then prefer the highest step.
    checkpoints = sorted(glob.glob(os.path.join(job_dir, "*.pt")))
    if not checkpoints:
        raise SystemExit(f"No .pt checkpoint in {job_dir}")

    def step_of(path):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(checkpoints, key=step_of)


def build_model(model_name):
    """Load the base model the way the trainer does, then apply the surgeon.

    The surgery matters: the checkpoint contains tokenformer_{k,v,p}
    tensors that only exist as parameters after the MLP layers are
    wrapped. Loading into an unsurgered model would report all of them
    as unexpected keys.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    multimodal = getattr(config, "vision_config", None) is not None
    loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    print(f"Base model: {model_name} (multimodal={multimodal}, via {loader.__name__})")

    model = loader.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return TokenformerSurgeon(model, torch.device(device)).insert_adapter_modules(), device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--jobs-root", default="/app/cray/jobs")
    parser.add_argument("--prompt", default="What is 3 + 3?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    job_dir = args.job_dir or latest_job_dir(args.jobs_root)
    checkpoint_path = latest_checkpoint(job_dir)
    print(f"Job:        {job_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    with open(os.path.join(job_dir, "config.yaml")) as f:
        model_name = yaml.safe_load(f)["llm_name"]

    model, device = build_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    print(f"\nCheckpoint holds {len(state_dict)} tensors (step {checkpoint.get('step')})")

    # strict=False plus an explicit report: a silent mismatch here would
    # invalidate the whole comparison, so name what didn't land.
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    print(f"Unexpected keys (in checkpoint, not in model): {len(unexpected)}")
    for key in unexpected[:10]:
        print(f"    {key}")
    if unexpected:
        print("    ^ these did NOT load; the comparison below is only")
        print("      meaningful if this list is empty.")

    loaded = len(state_dict) - len(unexpected)
    print(f"Applied {loaded}/{len(state_dict)} checkpoint tensors\n")

    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"Prompt:     {args.prompt!r}")
    print(f"HF output:  {completion!r}")
    print(
        "\nCompare against what vLLM returned for the same prompt and model.\n"
        "Different -> the serving path is losing weights.\n"
        "Same garbage -> the checkpoint is the problem."
    )


if __name__ == "__main__":
    main()
