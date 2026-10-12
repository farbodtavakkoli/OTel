import argparse
import json, os, subprocess, time

import mlx.core as mx
import numpy as np
from dotenv import load_dotenv
from mlx_lm import load

# dev.env supplies HF_TOKEN. The default model is an ungated mlx-community conversion.
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Same query/document set as inference/transformers/reranker -- the reference artifact
# this leaf is diffed against.
DEFAULT_QUERY = "Which inference engines support AMD ROCm?"

DEFAULT_DOCUMENTS = [
    "vLLM supports AMD ROCm.",
    "SGLang ships a ROCm build for AMD Instinct accelerators.",
    "PostgreSQL is a relational database.",
    "The Eiffel Tower is located in Paris, France.",
]

# Index of the documents above that are genuinely relevant to DEFAULT_QUERY.
DEFAULT_RELEVANT = [0, 1]

# Qwen3-Reranker is a decoder-style yes/no reranker: the (Instruct, Query, Document) triple is
# rendered into this prompt and the score is logit(yes) - logit(no) at the last position (the
# LogitScore the transformers baseline records; --activation sigmoid turns it into the P(yes)
# that llama.cpp's /v1/rerank reports). Same prompt as the transformers leaf's pair template.
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query."
PREFIX = ('<|im_start|>system\n'
          'Judge whether the Document meets the requirements based on the Query and the Instruct '
          'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 reranker inference on MLX (Apple silicon)")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-Reranker-0.6B-4bit",
                        help="MLX model id or path (an mlx-community conversion of Qwen/Qwen3-Reranker-0.6B)")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to rerank documents against")
    parser.add_argument("--documents_file", type=str, default=None, help="JSON/JSONL file of document strings; omit for the built-in set")
    parser.add_argument("--instruct", type=str, default=INSTRUCT, help="Task instruction rendered into the reranker prompt")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length (prompt + query + document), truncated from the document end")
    parser.add_argument("--activation", type=str, default="default", choices=["default", "sigmoid", "none"],
                        help="'default'/'none' = raw logit(yes) - logit(no), the same LogitScore the transformers "
                             "baseline records; 'sigmoid' = P(yes) in [0, 1] as llama.cpp's /v1/rerank reports")
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


def chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"


def peak_mib() -> float:
    # Unified memory: peak bytes the MLX allocator has held in this process (no separate VRAM).
    return round(mx.get_peak_memory() / 2**20, 1)


def score(model, tokenizer, query: str, documents: list[str], args) -> np.ndarray:
    yes = tokenizer.encode("yes", add_special_tokens=False)[0]
    no = tokenizer.encode("no", add_special_tokens=False)[0]
    head = tokenizer.encode(f"{PREFIX}<Instruct>: {args.instruct}\n<Query>: {query}\n<Document>: ", add_special_tokens=False)
    tail = tokenizer.encode(SUFFIX, add_special_tokens=False)
    out = []
    for doc in documents:
        body = tokenizer.encode(doc, add_special_tokens=False)[: max(0, args.max_len - len(head) - len(tail))]
        logits = model(mx.array([head + body + tail]))[0, -1].astype(mx.float32)
        pair = logits[mx.array([yes, no])]
        diff = pair[0] - pair[1]
        s = mx.sigmoid(diff) if args.activation == "sigmoid" else diff
        out.append(float(s))
    return np.asarray(out, dtype=np.float32)


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    t0 = time.time()
    model, tokenizer = load(args.model)
    load_s = time.time() - t0
    print(f"model={args.model} device=gpu (Metal, unified memory) chip={chip()}")
    print(f"load: {load_s:.1f}s | max_length={args.max_len} | peak(MiB)={peak_mib()}")

    score(model, tokenizer, args.query, documents[:1], args)  # warm-up: Metal kernel compilation

    t0 = time.time()
    scores = score(model, tokenizer, args.query, documents, args)
    score_s = time.time() - t0
    print(f"score: {score_s:.3f}s | {len(documents)} pairs | peak(MiB)={peak_mib()}")

    order = np.argsort(-scores)
    print(f"\nQ: {args.query}")
    for rank, j in enumerate(order):
        print(f"  #{rank + 1} score={scores[j]:+.6f}  {documents[j]}")

    ok = True
    if args.documents_file is None:
        irrelevant = [i for i in range(len(documents)) if i not in DEFAULT_RELEVANT]
        best_rel, worst_irr = scores[DEFAULT_RELEVANT].min(), scores[irrelevant].max()
        ok = bool(best_rel > worst_irr)
        print(f"\nsanity: min(relevant)={best_rel:+.6f} > max(irrelevant)={worst_irr:+.6f} -> {'PASS' if ok else 'FAIL'}")
        print(f"sanity: top-{len(DEFAULT_RELEVANT)} == relevant set -> "
              f"{'PASS' if sorted(order[:len(DEFAULT_RELEVANT)]) == DEFAULT_RELEVANT else 'FAIL'}")

    import mlx_lm
    artifact = {
        "model": args.model, "dtype": "4-bit (group size 64), bfloat16 activations", "attn_implementation": "mlx",
        "devices": ["gpu"], "seed": None, "max_len": args.max_len, "activation": args.activation,
        "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
        "gpu": chip(), "peak_memory_mib": peak_mib(),
        "load_seconds": round(load_s, 2), "score_seconds": round(score_s, 3),
        "query": args.query, "documents": documents,
        "scores": np.round(scores, 6).tolist(),
        "ranking": order.tolist(),
        "sanity_check_passed": ok,
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
