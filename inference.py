"""
OTel inference + scoring script — single entry point for all three benchmarked model families

Usage:
    python inference.py --mode llm       --model_name <scalarlm_model_id> --model_type rnj-1
    python inference.py --mode embedding --model_name farbodtavakkoli/OTel-Embedding-300M
    python inference.py --mode reranker  --model_name farbodtavakkoli/OTel-Reranker-0.6B

Default splits (configurable via --eval_ratio and --seed):
    LLM, embedding: 90% train / 10% eval, seed 42
    Reranker:       95% train /  5% eval, seed 42

Metrics:
    LLM:       LLM-as-judge correctness (%) — judge runs via OpenAI API
    Embedding: NDCG@10
    Reranker:  MRR@10

Tokens:
    HF_TOKEN and OPENAI_API_KEY are read from `dev.env` by default.
    Override with --hf_token / --openai_api_key.
"""

import argparse
import datetime
import json
import os
import random
import re
import time
import traceback

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("dev.env")

def get_args():

    parser = argparse.ArgumentParser(description="OTel inference + scoring script.")

    # Default OTel HF dataset for each mode. Used when --dataset is not
    # provided; the resolution happens after parse_args() below so the help
    # text and resolved value share a single source of truth.
    DATASETS = {
        "llm":       "farbodtavakkoli/OTel-LLM",
        "embedding": "farbodtavakkoli/OTel-Embedding",
        "reranker":  "farbodtavakkoli/OTel-Reranker",
    }

    # Default eval split fraction for each mode. Used when --eval_ratio is
    # not provided. The OTel paper uses 90/10 for LLM and embedding, and
    # 95/5 for reranker (rerankers train on more data because cross-encoder
    # scoring is comparatively cheap to evaluate).
    EVAL_RATIO = {
        "llm":       0.10,
        "embedding": 0.10,
        "reranker":  0.05,
    }

    # Model family to run. Determines which dataset is loaded, which inference
    # backend is used (ScalarLM for LLM; sentence-transformers for embedding /
    # reranker), and which scoring metric is computed
    # (correctness % / NDCG@10 / MRR@10).
    parser.add_argument('--mode', required=True, choices=["llm", "embedding", "reranker"])

    # ScalarLM model_id (LLM mode) OR Hugging Face repo ID
    # (embedding / reranker mode). In LLM mode this is the content hash
    # returned by ScalarLM after deployment, e.g.
    # 9f0122ddc2e55a14f1ae195c871c736b0673d98d2f0cfcf77b85a5863b98b052.
    # In embedding / reranker mode use the released HF repo, e.g.
    # farbodtavakkoli/OTel-Embedding-300M.
    parser.add_argument('--model_name', required=True)

    # Override the default OTel dataset for this mode. Leave unset to use the
    # mode-appropriate default from DATASETS above:
    #   llm       -> farbodtavakkoli/OTel-LLM
    #   embedding -> farbodtavakkoli/OTel-Embedding
    #   reranker  -> farbodtavakkoli/OTel-Reranker
    # Pass any HF dataset id (e.g. --dataset my-org/my-eval-set) to override.
    parser.add_argument('--dataset', default=None)

    # Eval split fraction (e.g. 0.10 for a 90 / 10 train / eval split). Leave
    # unset to use the mode-appropriate default from EVAL_RATIO above:
    #   llm       -> 0.10
    #   embedding -> 0.10
    #   reranker  -> 0.05
    # The split is deterministic given --seed, so the same fraction + seed
    # always yields the same held-out rows.
    parser.add_argument('--eval_ratio', type=float, default=None)

    # Random seed used by the train / test split. Default 42 matches the OTel
    # paper. Change only if you intentionally want a different held-out split.
    parser.add_argument('--seed', type=int, default=42)

    # OpenAI judge model used for LLM-mode correctness scoring (LLM-as-judge).
    # Default gpt-4.1 matches the paper. Override with --judge gpt-4o-mini
    # for a faster / cheaper run.
    parser.add_argument('--judge', default="gpt-4.1")

    # LLM family used to apply the correct chat template to prompts in LLM
    # mode. Supported families:
    # gemma3, qwen3, llama3, rnj-1, olmo3, mistral, lfm, phi4,
    # gpt-oss_reasoning, gpt_oss_it. Pick the family of the BASE model your
    # fine-tuned ScalarLM model was post-trained from. Ignored in embedding
    # and reranker modes.
    parser.add_argument('--model_type', default="rnj-1")

    # Max generated tokens per prompt in LLM mode. Total context
    # = input tokens + this value must fit within the deployed model's
    # context window. ScalarLM default deployment context is 4096:
    # 4096 (deployed context length) = 3000 (input tokens) + 1096 (max output).
    parser.add_argument('--max_tokens', type=int, default=500)

    # Number of prompts per ScalarLM batch in LLM mode. Reduce if you hit
    # timeouts on long-context generation; raise for short prompts. Ignored
    # in embedding and reranker modes (those use sentence-transformers'
    # internal batching).
    parser.add_argument('--batch_size', type=int, default=2)

    # Hugging Face access token used to load datasets and, in embedding /
    # reranker mode, to load the base model. Defaults to HF_TOKEN from
    # dev.env. Pass --hf_token explicitly to override or to run without
    # dev.env.
    parser.add_argument('--hf_token', default=os.getenv("HF_TOKEN"))

    # OpenAI API key used by the LLM-as-judge in LLM mode. Required only
    # in LLM mode. Defaults to OPENAI_API_KEY from dev.env. Pass
    # --openai_api_key explicitly to override or to run without dev.env.
    parser.add_argument('--openai_api_key', default=os.getenv("OPENAI_API_KEY"))

    args = parser.parse_args()

    # Resolve mode-keyed defaults so downstream code can read args.dataset /
    # args.eval_ratio directly without re-checking for None.
    if args.dataset is None:
        args.dataset = DATASETS[args.mode]
    if args.eval_ratio is None:
        args.eval_ratio = EVAL_RATIO[args.mode]

    return args


def format_conversation_only(prompt, model_type):

    prompt = str(prompt or "")
    mt = model_type.lower()

    if mt == "qwen3":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if mt == "llama3":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    if mt == "gemma3":
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n\n\n<start_of_turn>model"

    if mt == "rnj-1":
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    if mt == "olmo3":
        return f"<|endoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if mt == "mistral":
        return f"<s>[SYSTEM_PROMPT][/SYSTEM_PROMPT][INST]{prompt}[/INST]"

    if mt == "lfm":
        return f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    if mt == "phi4":
        return f"<|user|>{prompt}<|end|><|assistant|>"

    if mt == "gpt-oss_reasoning":
        return (
            f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Knowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n"
            f"# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>developer<|message|># Instructions\n\nreasoning_language: English\n\n"
            f"You are an Open Source Telecom Question answering model that uses provided contexts to answer telecom questions.\n\n"
            f"<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>analysis<|message|>\n"
        )

    if mt == "gpt_oss_it":
        return (
            f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Knowledge cutoff: 2024-06\nCurrent date: 2026-02-11\n\nReasoning: low\n\n"
            f"# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
            f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>"
        )

    return f"User: {prompt}\nAssistant:"


def split(ds, eval_ratio, seed):
    return ds.train_test_split(test_size=eval_ratio, seed=seed)

CORRECTNESS_SYSTEM = (
    "You are an expert evaluator comparing an AI response against a ground truth answer."
)
CORRECTNESS_USER_TEMPLATE = """Compare the model response to the expected answer and determine if it's correct.
Consider the response correct if it captures the key information and is factually accurate.

Question: {question}
Expected Answer: {expected}
Model Response: {response}

Respond with a JSON object structured exactly as below:
{{"score": "pass" or "fail"}}"""


def extract_score(text):
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end]).get("score")
    except Exception:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1)).get("score")
            except Exception:
                pass
    return None


def _judge_one(client, judge_model, question, expected, response):
    try:
        r = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": CORRECTNESS_SYSTEM},
                {"role": "user", "content": CORRECTNESS_USER_TEMPLATE.format(
                    question=question, expected=expected, response=response,
                )},
            ],
            max_tokens=50,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return extract_score(r.choices[0].message.content.strip())
    except Exception as e:
        print(f"  judge error: {e}")
        return None


def infer_llm(config, dataset_id, eval_ratio):
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY missing (set in dev.env or pass --openai_api_key).")

    import scalarlm
    from openai import OpenAI

    llm = scalarlm.SupermassiveIntelligence()
    judge_client = OpenAI(api_key=config.openai_api_key)

    # Load HF dataset (data_files="*.jsonl" excludes croissant.json from being treated as data).
    ds = load_dataset(dataset_id, data_files="*.jsonl", split="train", token=config.hf_token)
    eval_ds = split(ds, eval_ratio, config.seed)["test"]

    prompts = [format_conversation_only(row["prompt"], config.model_type) for row in eval_ds]
    expected = [row["completion"] for row in eval_ds]
    questions = [row.get("anchor") or row["prompt"] for row in eval_ds]

    # Batched generation via ScalarLM, with per-batch error tolerance.
    results = []
    processed_indices = []
    for i in range(0, len(prompts), config.batch_size):
        batch = prompts[i: i + config.batch_size]
        try:
            t0 = time.time()
            out = llm.generate(prompts=batch, model_name=config.model_name, max_tokens=config.max_tokens)
            results += out
            processed_indices += list(range(i, min(i + config.batch_size, len(prompts))))
            t1 = time.time()
            print(f"  generated batch {i//config.batch_size + 1}/{(len(prompts) + config.batch_size - 1)//config.batch_size} in {t1-t0:.2f}s")
        except Exception:
            traceback.print_exc()
            print(f"  batch starting at index {i} failed; skipping")
            continue

    # Persist raw generations before scoring so re-scoring does not require re-generation.
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = os.path.join("scalarlm_inference", "runs", now)
    os.makedirs(save_dir, exist_ok=True)

    output = []
    for k, row_idx in enumerate(processed_indices):
        output.append({
            "ground_truth_response": expected[row_idx],
            "input_prompt": prompts[row_idx],
            "generated_response": results[k],
        })

    out_path = os.path.join(save_dir, "inference_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"  saved raw generations to {out_path}")

    # Score with OpenAI judge using anchor as the question and completion as the reference.
    correct = scored = 0
    for k, row_idx in enumerate(processed_indices):
        verdict = _judge_one(
            judge_client, config.judge,
            questions[row_idx], expected[row_idx], results[k],
        )
        if verdict is None:
            continue
        scored += 1
        if verdict == "pass":
            correct += 1

    pct = 100.0 * correct / max(scored, 1)
    print(f"\nCorrectness: {correct}/{scored} = {pct:.2f}% (judge={config.judge})")
    return pct


def infer_embedding(config, dataset_id, eval_ratio):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import (
        InformationRetrievalEvaluator,
        SimilarityFunction,
    )

    model = SentenceTransformer(config.model_name, trust_remote_code=True, token=config.hf_token)
    # Instruction-tuned embedding models (Gemma3-Embedding, Qwen3-Embedding) use
    # query/document prefixes; classic ST models (MiniLM, mpnet, BGE) ignore them.
    model.prompts = {
        "Retrieval-query":    "search_query: ",
        "Retrieval-document": "search_document: ",
    }

    splits = split(load_dataset(dataset_id, split="train", token=config.hf_token), eval_ratio, config.seed)
    train_ds, eval_ds = splits["train"], splits["test"]

    all_positives = list(set(train_ds["positive"]) | set(eval_ds["positive"]))
    corpus = {f"doc_{i}": t for i, t in enumerate(all_positives)}
    text_to_id = {t: i for i, t in corpus.items()}

    evaluator = InformationRetrievalEvaluator(
        queries={f"q_{i}": row["anchor"] for i, row in enumerate(eval_ds)},
        corpus=corpus,
        relevant_docs={f"q_{i}": {text_to_id[row["positive"]]} for i, row in enumerate(eval_ds)},
        name="otel_embedding",
        main_score_function=SimilarityFunction.COSINE,
        query_prompt_name="Retrieval-query",
        corpus_prompt_name="Retrieval-document",
    )
    metrics = evaluator(model)
    ndcg = metrics["otel_embedding_cosine_ndcg@10"]
    print(f"\nNDCG@10: {ndcg:.4f}")
    return ndcg


def infer_reranker(config, dataset_id, eval_ratio):
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

    model = CrossEncoder(config.model_name, trust_remote_code=True, token=config.hf_token)
    # Qwen3-based cross-encoders ship without a pad token; required for batched eval.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.model.config.pad_token_id = model.tokenizer.pad_token_id

    eval_ds = split(load_dataset(dataset_id, split="train", token=config.hf_token), eval_ratio, config.seed)["test"]

    # Reranker rows are flat (sentence_0, sentence_1, label); regroup by query.
    grouped = {}
    for row in eval_ds:
        g = grouped.setdefault(row["sentence_0"], {"positive": [], "negative": []})
        (g["positive"] if row["label"] >= 0.5 else g["negative"]).append(row["sentence_1"])
    samples = [
        {"query": q, "positive": v["positive"], "negative": v["negative"]}
        for q, v in grouped.items() if v["positive"] and v["negative"]
    ]

    evaluator = CrossEncoderRerankingEvaluator(samples=samples, name="otel_reranker", at_k=10)
    metrics = evaluator(model)
    mrr = metrics["otel_reranker_mrr@10"]
    print(f"\nMRR@10: {mrr:.4f}")
    return mrr


if __name__ == "__main__":
    config = get_args()

    random.seed(config.seed)

    print(f"Mode:       {config.mode}")
    print(f"Model:      {config.model_name}")
    print(f"Dataset:    {config.dataset}")
    print(f"Eval split: {config.eval_ratio:.0%} (seed={config.seed})")

    if config.mode == "llm":
        infer_llm(config, config.dataset, config.eval_ratio)
    elif config.mode == "embedding":
        infer_embedding(config, config.dataset, config.eval_ratio)
    elif config.mode == "reranker":
        infer_reranker(config, config.dataset, config.eval_ratio)
