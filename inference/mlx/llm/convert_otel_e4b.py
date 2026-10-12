"""Convert farbodtavakkoli/OTel-LLM-E4B-IT to an MLX 4-bit checkpoint.

The published checkpoint is six PyTorch .bin shards (31 GB; the vision and audio towers of
Gemma4ForConditionalGeneration are included and fp32 in places). mlx_lm.convert reads
safetensors only, so this script rewrites the shards to bf16 safetensors one shard at a time
(peak RAM ~ the largest shard, 11 GB) and then quantizes. Only the text model is kept.

    python convert_otel_e4b.py --mlx_path ./OTel-LLM-E4B-IT-4bit

Known fix -- the checkpoint stores lm_head.weight as a separate tensor although
tie_word_embeddings is true and it is bit-identical to embed_tokens.weight; mlx_lm.convert
rejects the extra parameter ("Received 1 parameters not in model: lm_head.weight"), so it is
dropped after the identity is verified.
"""
import argparse, glob, json, os, shutil, tempfile

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hf_path", type=str, default="farbodtavakkoli/OTel-LLM-E4B-IT", help="Source repo id or local dir")
    p.add_argument("--mlx_path", type=str, default="./OTel-LLM-E4B-IT-4bit", help="Output MLX model dir")
    p.add_argument("--q_bits", type=int, default=4)
    p.add_argument("--q_group_size", type=int, default=64)
    p.add_argument("--no_quantize", action="store_true", help="Write bf16 MLX weights instead of quantizing")
    p.add_argument("--keep_bf16", type=str, default=None, help="Keep the intermediate bf16 safetensors dir here")
    return p.parse_args()


def to_safetensors(src: str, dst: str):
    idx = json.load(open(f"{src}/pytorch_model.bin.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    weight_map = {}
    for i, sh in enumerate(shards):
        sd = torch.load(f"{src}/{sh}", map_location="cpu", weights_only=True)
        out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, f"{dst}/{name}", metadata={"format": "pt"})
        weight_map.update({k: name for k in out})
        print(f"{sh} -> {name} ({len(out)} tensors)")
        del sd, out
    # Known fix: drop the redundant tied lm_head (see module docstring).
    lm, emb = "lm_head.weight", "model.language_model.embed_tokens.weight"
    if lm in weight_map:
        a = load_file(f"{dst}/{weight_map[lm]}")
        b = load_file(f"{dst}/{weight_map[emb]}")[emb]
        assert torch.equal(a[lm], b), "lm_head.weight is not tied to embed_tokens; refusing to drop it"
        a.pop(lm)
        save_file(a, f"{dst}/{weight_map[lm]}", metadata={"format": "pt"})
        weight_map.pop(lm)
        print("dropped lm_head.weight (identical to embed_tokens.weight, tie_word_embeddings=true)")
    json.dump({"metadata": {}, "weight_map": weight_map}, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
    for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        shutil.copy(f"{src}/{f}", dst)


def main():
    args = parse_args()
    src = args.hf_path if os.path.isdir(args.hf_path) else snapshot_download(args.hf_path, max_workers=2)
    bf16 = args.keep_bf16 or tempfile.mkdtemp(prefix="otel-e4b-bf16-")
    os.makedirs(bf16, exist_ok=True)
    if not glob.glob(f"{bf16}/*.safetensors"):
        to_safetensors(src, bf16)
    from mlx_lm import convert
    kwargs = {} if args.no_quantize else {"quantize": True, "q_bits": args.q_bits, "q_group_size": args.q_group_size}
    convert(bf16, mlx_path=args.mlx_path, **kwargs)
    if not args.keep_bf16:
        shutil.rmtree(bf16)
    print(f"wrote {args.mlx_path}")


if __name__ == "__main__":
    main()
