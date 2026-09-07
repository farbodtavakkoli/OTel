"""Qwen3 reranker reference inference (sentence-transformers CrossEncoder) — see README.md."""
import argparse
import json, os, time

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_QUERY = "Which inference engines support AMD ROCm?"

DEFAULT_DOCUMENTS = [
    "vLLM supports AMD ROCm.",
    "SGLang ships a ROCm build for AMD Instinct accelerators.",
    "PostgreSQL is a relational database.",
    "The Eiffel Tower is located in Paris, France.",
]

# Index of the documents above that are genuinely relevant to DEFAULT_QUERY.
DEFAULT_RELEVANT = [0, 1]


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Qwen3 reranker reference inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Reranker-0.6B", help="Cross-encoder model id or path")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query to rerank documents against")
    parser.add_argument("--documents_file", type=str, default=None, help="JSON/JSONL file of document strings; omit for the built-in set")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32", "float16"],
                        help="Model dtype; float32 is the reproducible choice for reference scoring")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated devices. One device = single-GPU; two or more = multi-process scoring")
    parser.add_argument("--batch_size", type=int, default=32, help="Scoring batch size")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length (query + document)")
    parser.add_argument("--activation", type=str, default="default", choices=["default", "sigmoid", "none"],
                        help="Score activation; 'default' keeps the model's configured LogitScore head")
    parser.add_argument("--attn_impl", type=str, default=None, help="Attention implementation; default sdpa on ROCm, eager elsewhere")
    parser.add_argument("--fix_pair_template", action="store_true", default=True,
                        help="Install a Query/Document pair chat template if the checkpoint's own can't carry both roles (default on)")
    parser.add_argument("--no_fix_pair_template", dest="fix_pair_template", action="store_false",
                        help="Leave the loaded chat template untouched (fails on a bare checkpoint under sentence-transformers 5.7+)")
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


# Qwen3-Reranker judges a (Query, Document) pair with a yes/no instruction prompt. A bare
# checkpoint (no packaged chat_template.jinja) carries only the generic Qwen3 *generation*
# template, which branches on system/user/assistant and never renders 'query'/'document'.
# sentence-transformers 5.x maps a pair to one message per role and, since 5.7.0, refuses a
# template that drops both (_verify_pair_roles_supported), so predict() raises. This template
# renders the two roles into the reranker's yes/no prompt. Roles are selected by name because
# ST passes role-tagged messages, not positional ones.
QWEN3_RERANKER_PAIR_TEMPLATE = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
    '<Instruct>: Given a web search query, retrieve relevant passages that answer the query.\n'
    '<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}\n'
    '<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\n'
)


def install_pair_chat_template(model) -> bool:
    """Install a query/document pair chat template if the loaded one can't carry both roles.

    Returns True when the template was replaced. A packaged reranker that already renders both
    roles is left untouched; this only fires for a bare checkpoint carrying the plain generation
    template (the common case when loading from a minimal local cache).
    """
    try:
        formatter = model[0].input_formatter
        if formatter.pair_roles_failure({}) is None:
            return False  # existing template already handles query/document pairs
    except Exception:
        pass  # older sentence-transformers without the probe: install unconditionally
    try:
        model.processor.chat_template = QWEN3_RERANKER_PAIR_TEMPLATE
    except Exception:
        model.tokenizer.chat_template = QWEN3_RERANKER_PAIR_TEMPLATE
    return True


def load_model(args):
    """Load the CrossEncoder, selecting sdpa on ROCm since flash-attn is a CUDA-only build."""
    attn = args.attn_impl or ("sdpa" if torch.version.hip else "eager")
    model = CrossEncoder(
        args.model,
        max_length=args.max_len,
        trust_remote_code=True,
        model_kwargs={"dtype": getattr(torch, args.dtype), "attn_implementation": attn},
    )
    # Qwen has no dedicated pad token; use eos and sync it to the model config.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.pad_token_id
    if args.fix_pair_template and install_pair_chat_template(model):
        print("reranker fix: installed Query/Document pair chat template (bare checkpoint)")
    return model, attn


def score(model, query, documents, args, devices):
    """Score every (query, document) pair; multi-device uses a process pool."""
    kwargs = {"batch_size": args.batch_size, "convert_to_numpy": True}
    if args.activation != "default":
        kwargs["activation_fn"] = torch.nn.Sigmoid() if args.activation == "sigmoid" else torch.nn.Identity()
    if len(devices) > 1:
        kwargs["device"] = devices
    pairs = [(query, d) for d in documents]
    return np.asarray(model.predict(pairs, **kwargs), dtype=np.float32).reshape(-1)


def vram_mib() -> list[float]:
    """Per-visible-GPU allocated VRAM in MiB, as torch sees it."""
    return [round(torch.cuda.memory_allocated(i) / 2**20, 1) for i in range(torch.cuda.device_count())]


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    torch.manual_seed(args.seed)

    devices = resolve_devices(args.devices)
    documents = read_texts(args.documents_file, DEFAULT_DOCUMENTS)

    t0 = time.time()
    model, attn = load_model(args)
    load_s = time.time() - t0
    print(f"model={args.model} dtype={args.dtype} attn={attn} devices={devices}")
    print(f"load: {load_s:.1f}s | max_length={args.max_len} | VRAM(MiB)={vram_mib()}")

    t0 = time.time()
    scores = score(model, args.query, documents, args, devices)
    score_s = time.time() - t0
    print(f"score: {score_s:.2f}s | {len(documents)} pairs | VRAM(MiB)={vram_mib()}")

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

    artifact = {
        "model": args.model, "dtype": args.dtype, "attn_implementation": attn,
        "devices": devices, "seed": args.seed, "max_len": args.max_len, "activation": args.activation,
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_properties(0).gcnArchName,
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
