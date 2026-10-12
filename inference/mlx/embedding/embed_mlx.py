import argparse
import json, os, subprocess, time

import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from mlx_embeddings import load

# dev.env supplies HF_TOKEN. The default model is an ungated mlx-community conversion, so
# the token is only needed if you point --model at the gated google/embeddinggemma-300m.
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Same query/document sets as inference/transformers/embedding -- the reference artifact
# this leaf is diffed against.
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

# EmbeddingGemma's own prompt templates, as applied by sentence-transformers'
# encode_query / encode_document in the transformers baseline.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "


def parse_args():
    parser = argparse.ArgumentParser(description="EmbeddingGemma embedding inference on MLX (Apple silicon)")
    parser.add_argument("--model", type=str, default="mlx-community/embeddinggemma-300m-bf16",
                        help="MLX model id or path (an mlx-community conversion of google/embeddinggemma-300m)")
    parser.add_argument("--queries_file", type=str, default=None, help="JSON/JSONL file of query strings; omit for the built-in set")
    parser.add_argument("--documents_file", type=str, default=None, help="JSON/JSONL file of document strings; omit for the built-in set")
    parser.add_argument("--batch_size", type=int, default=32, help="Encode batch size")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Tokenizer truncation length (model max is 2048)")
    parser.add_argument("--truncate_dim", type=int, default=None, help="Matryoshka output dim (768/512/256/128); omit for full")
    parser.add_argument("--normalize", action="store_true", default=True, help="L2-normalize embeddings (default on)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    parser.add_argument("--output", type=str, default=None, help="Path for the reference-output JSON artifact")
    parser.add_argument("--hf_home", type=str, default=None, help="Optional HF_HOME model-cache override")
    return parser.parse_args()


def read_texts(path: str, fallback: list[str]) -> list[str]:
    if path is None:
        return list(fallback)
    with open(path) as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [json.loads(l)["text"] if l.lstrip().startswith("{") else l for l in raw.splitlines() if l.strip()]


# Known fix -- mlx-embeddings 0.1.0 generate() calls the model as model(**inputs), i.e.
# with the tokenizer's `input_ids=` keyword, but the Gemma3 text model's __call__ takes
# `inputs` positionally (models/gemma3_text.py), so generate() raises
# "unexpected keyword argument 'input_ids'". The encoder below tokenizes with the wrapped
# HF tokenizer and calls the model directly, which is all generate() does after that point.
def encode(model, tokenizer, texts: list[str], prompt: str, args) -> np.ndarray:
    out = []
    for i in range(0, len(texts), args.batch_size):
        batch = [prompt + t for t in texts[i:i + args.batch_size]]
        enc = tokenizer._tokenizer(batch, padding=True, truncation=True,
                                   max_length=args.max_seq_length, return_tensors="mlx")
        emb = model(enc["input_ids"], attention_mask=enc["attention_mask"]).text_embeds
        mx.eval(emb)
        out.append(np.asarray(emb.astype(mx.float32)))
    emb = np.concatenate(out, axis=0)
    if args.truncate_dim:
        emb = emb[:, :args.truncate_dim]
    if args.normalize:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"


def peak_mib() -> float:
    # Unified memory: peak bytes the MLX allocator has held in this process. There is no
    # separate VRAM counter on Apple silicon; this is the number to compare with VRAM(MiB).
    return round(mx.get_peak_memory() / 2**20, 1)


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    queries = read_texts(args.queries_file, DEFAULT_QUERIES)
    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    t0 = time.time()
    model, tokenizer = load(args.model)
    load_s = time.time() - t0
    print(f"model={args.model} device=gpu (Metal, unified memory) chip={chip()}")
    print(f"load: {load_s:.1f}s | max_seq_length={args.max_seq_length} | peak(MiB)={peak_mib()}")

    # One warm-up call so encode_seconds measures the steady state, not Metal kernel compilation.
    encode(model, tokenizer, ["warm-up"], QUERY_PROMPT, args)

    t0 = time.time()
    q_emb = encode(model, tokenizer, queries, QUERY_PROMPT, args)
    d_emb = encode(model, tokenizer, documents, DOCUMENT_PROMPT, args)
    encode_s = time.time() - t0

    sims = cosine(q_emb, d_emb)
    print(f"encode: {encode_s:.3f}s | query_shape={q_emb.shape} doc_shape={d_emb.shape} "
          f"| dim={q_emb.shape[1]} | peak(MiB)={peak_mib()}")
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

    import mlx_embeddings, mlx_lm
    artifact = {
        "model": args.model, "dtype": "bfloat16", "attn_implementation": "mlx",
        "devices": ["gpu"], "seed": None, "truncate_dim": args.truncate_dim,
        "normalized": args.normalize, "embedding_dim": int(q_emb.shape[1]),
        "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "mlx_embeddings": mlx_embeddings.__version__,
        "gpu": chip(), "peak_memory_mib": peak_mib(),
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
