"""EmbeddingGemma reference embedding inference (sentence-transformers) — see README.md."""
import argparse
import json, os, time

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# dev.env supplies HF_TOKEN for gated-model downloads (embeddinggemma is gated).
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_QUERIES = [
    "What GPU runtimes support ROCm?",
    "Which database is embedded and serverless?",
]

DEFAULT_DOCUMENTS = [
    "vLLM supports NVIDIA CUDA and AMD ROCm.",
    "SGLang provides a ROCm build for AMD Instinct accelerators.",
    "SQLite is an embedded database.",
    "The Eiffel Tower is located in Paris, France.",
]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="EmbeddingGemma reference embedding inference")
    parser.add_argument("--model", type=str, default="google/embeddinggemma-300m", help="Embedding model id or path")
    parser.add_argument("--queries_file", type=str, default=None, help="JSON/JSONL file of query strings; omit for the built-in set")
    parser.add_argument("--documents_file", type=str, default=None, help="JSON/JSONL file of document strings; omit for the built-in set")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32", "float16"],
                        help="Model dtype; the model card recommends bfloat16 or float32, not float16")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated devices. One device = single-GPU; two or more = multi-process encoding")
    parser.add_argument("--batch_size", type=int, default=32, help="Encode batch size")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Override the model's max sequence length")
    parser.add_argument("--truncate_dim", type=int, default=None, help="Matryoshka output dim (768/512/256/128); omit for full")
    parser.add_argument("--normalize", action="store_true", default=True, help="L2-normalize embeddings (default on)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    parser.add_argument("--attn_impl", type=str, default=None, help="Attention implementation; default sdpa on ROCm, eager elsewhere")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed for reproducible outputs")
    parser.add_argument("--output", type=str, default=None, help="Path for the reference-output JSON artifact")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional HF_HOME model-cache override")
    return parser.parse_args()


def read_texts(path: str, fallback: list[str]) -> list[str]:
    """Read a JSON list or a JSONL/plain-text file of strings; fall back to the built-in set."""
    if path is None:
        return list(fallback)
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l)["text"] if l.lstrip().startswith("{") else l for l in raw.splitlines() if l.strip()]


def resolve_devices(spec: str) -> list[str]:
    """Turn the --devices string into a device list, validating against the visible GPU count."""
    devices = [d.strip() for d in spec.split(",") if d.strip()]
    n = torch.cuda.device_count()
    for d in devices:
        if d.startswith("cuda:") and int(d.split(":")[1]) >= n:
            raise SystemExit(f"{d} requested but only {n} GPU(s) visible — check HIP_VISIBLE_DEVICES")
    return devices


def load_model(args):
    """Load the SentenceTransformer, selecting sdpa on ROCm since flash-attn is a CUDA-only build."""
    attn = args.attn_impl or ("sdpa" if torch.version.hip else "eager")
    model = SentenceTransformer(
        args.model,
        device="cuda:0",
        model_kwargs={"dtype": getattr(torch, args.dtype), "attn_implementation": attn},
    )
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    return model, attn


def encode_all(model, queries, documents, args, devices):
    """Encode queries and documents with the model's prompt templates; multi-device uses a process pool."""
    kwargs = {"batch_size": args.batch_size, "normalize_embeddings": args.normalize,
              "truncate_dim": args.truncate_dim, "convert_to_numpy": True}
    if len(devices) > 1:
        kwargs["device"] = devices
    q = model.encode_query(queries, **kwargs)
    d = model.encode_document(documents, **kwargs)
    return np.asarray(q, dtype=np.float32), np.asarray(d, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine-similarity matrix between two batches of row vectors."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def vram_mib() -> list[float]:
    """Per-visible-GPU allocated VRAM in MiB, as torch sees it."""
    return [round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())]


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)

    devices = resolve_devices(args.devices)
    queries = read_texts(args.queries_file, DEFAULT_QUERIES)
    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    t0 = time.time()
    model, attn = load_model(args)
    load_s = time.time() - t0
    print(f"model={args.model} dtype={args.dtype} attn={attn} devices={devices}")
    print(f"load: {load_s:.1f}s | max_seq_length={model.max_seq_length} | VRAM(MiB)={vram_mib()}")

    t0 = time.time()
    q_emb, d_emb = encode_all(model, queries, documents, args, devices)
    encode_s = time.time() - t0

    sims = cosine(q_emb, d_emb)
    print(f"encode: {encode_s:.2f}s | query_shape={q_emb.shape} doc_shape={d_emb.shape} "
          f"| dim={q_emb.shape[1]} | VRAM(MiB)={vram_mib()}")
    print(f"query[0] norm={np.linalg.norm(q_emb[0]):.6f} first8={np.round(q_emb[0][:8], 6).tolist()}")

    ok = True
    for i, query in enumerate(queries):
        order = np.argsort(-sims[i])
        print(f"\nQ{i}: {query}")
        for rank, j in enumerate(order):
            print(f"  #{rank + 1} cos={sims[i][j]:+.6f}  {documents[j]}")
        # Sanity: the intended relevant doc for each built-in query must outrank the irrelevant ones.
        if args.queries_file is None and args.documents_file is None:
            relevant, irrelevant = ([0, 1], [2, 3]) if i == 0 else ([2], [0, 1, 3])
            best_rel, worst_irr = sims[i][relevant].min(), sims[i][irrelevant].max()
            passed = best_rel > worst_irr
            ok &= passed
            print(f"  sanity: min(relevant)={best_rel:+.6f} > max(irrelevant)={worst_irr:+.6f} -> {'PASS' if passed else 'FAIL'}")

    artifact = {
        "model": args.model, "dtype": args.dtype, "attn_implementation": attn,
        "devices": devices, "seed": args.seed, "truncate_dim": args.truncate_dim,
        "normalized": args.normalize, "embedding_dim": int(q_emb.shape[1]),
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_properties(0).gcnArchName,
        "load_seconds": round(load_s, 2), "encode_seconds": round(encode_s, 3),
        "queries": queries, "documents": documents,
        "cosine_similarities": np.round(sims, 6).tolist(),
        "query_embeddings_first16": np.round(q_emb[:, :16], 6).tolist(),
        "query_embedding_norms": np.round(np.linalg.norm(q_emb, axis=1), 6).tolist(),
        "sanity_check_passed": bool(ok),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nwrote reference artifact: {args.output}")

    print(f"\nSANITY: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
