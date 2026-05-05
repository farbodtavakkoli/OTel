"""
OTel evaluation script — single entry point for all three benchmarked model families.

Usage:
    python eval.py --mode llm       --model farbodtavakkoli/OTel-LLM-1.2B-IT
    python eval.py --mode embedding --model farbodtavakkoli/OTel-Embedding-300M
    python eval.py --mode reranker  --model farbodtavakkoli/OTel-Reranker-0.6B

Splits (seed 42):
    LLM, embedding: 90% train / 10% eval
    Reranker:       95% train /  5% eval

Metrics:
    LLM:       LLM-as-judge correctness (%) — judge runs via OpenAI API (default: gpt-4.1)
    Embedding: NDCG@10
    Reranker:  MRR@10

Tokens:
    Reads HF_TOKEN and OPENAI_API_KEY from `dev.env` in the working directory.
"""

import argparse
import json
import os
import random
import re

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("dev.env")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEED = 42

DATASETS = {
    "llm":       "farbodtavakkoli/OTel-LLM",
    "embedding": "farbodtavakkoli/OTel-Embedding",
    "reranker":  "farbodtavakkoli/OTel-Reranker",
}

EVAL_RATIO = {
    "llm":       0.10,
    "embedding": 0.10,
    "reranker":  0.05,
}


def format_conversation_only(tokenizer, user_msg, system_msg=None):

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def split(ds, eval_ratio):
    return ds.train_test_split(test_size=eval_ratio, seed=SEED)


# -------------------- LLM: OpenAI-API LLM-as-judge correctness --------------------

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


def eval_llm(model_id, dataset_id, eval_ratio, judge_model):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing from dev.env (required for LLM mode).")

    import torch
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    judge_client = OpenAI(api_key=OPENAI_API_KEY)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    ).eval()

    def generate(prompt, max_new_tokens=512):
        text = format_conversation_only(tok, prompt)
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def judge(question, expected, response):
        try:
            r = judge_client.chat.completions.create(
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

    # data_files="*.jsonl" excludes croissant.json from being treated as a data file.
    ds = load_dataset(dataset_id, data_files="*.jsonl", split="train", token=HF_TOKEN)
    eval_ds = split(ds, eval_ratio)["test"]

    correct = scored = 0
    for row in eval_ds:
        answer = generate(row["prompt"])
        verdict = judge(row.get("anchor", row["prompt"]), row["completion"], answer)
        if verdict is not None:
            scored += 1
            if verdict == "pass":
                correct += 1

    pct = 100.0 * correct / max(scored, 1)
    print(f"\nCorrectness: {correct}/{scored} = {pct:.2f}% (judge={judge_model})")
    return pct


# -------------------- Embedding: NDCG@10 --------------------

def eval_embedding(model_id, dataset_id, eval_ratio):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import (
        InformationRetrievalEvaluator,
        SimilarityFunction,
    )

    model = SentenceTransformer(model_id, trust_remote_code=True, token=HF_TOKEN)
    splits = split(load_dataset(dataset_id, split="train", token=HF_TOKEN), eval_ratio)
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
    )
    metrics = evaluator(model)
    ndcg = metrics["otel_embedding_cosine_ndcg@10"]
    print(f"\nNDCG@10: {ndcg:.4f}")
    return ndcg


# -------------------- Reranker: MRR@10 --------------------

def eval_reranker(model_id, dataset_id, eval_ratio):
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

    model = CrossEncoder(model_id, trust_remote_code=True, token=HF_TOKEN)
    eval_ds = split(load_dataset(dataset_id, split="train", token=HF_TOKEN), eval_ratio)["test"]

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


# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="OTel evaluation script.")
    p.add_argument("--mode", required=True, choices=list(DATASETS))
    p.add_argument("--model", required=True, help="HF model ID or local path.")
    p.add_argument("--dataset", default=None, help="Override the default OTel dataset.")
    p.add_argument(
        "--judge",
        default="gpt-4.1",
        help="OpenAI judge model for LLM mode (default: gpt-4.1).",
    )
    args = p.parse_args()

    random.seed(SEED)
    dataset_id = args.dataset or DATASETS[args.mode]
    eval_ratio = EVAL_RATIO[args.mode]

    print(f"Mode:       {args.mode}")
    print(f"Model:      {args.model}")
    print(f"Dataset:    {dataset_id}")
    print(f"Eval split: {eval_ratio:.0%} (seed={SEED})")

    if args.mode == "llm":
        eval_llm(args.model, dataset_id, eval_ratio, args.judge)
    elif args.mode == "embedding":
        eval_embedding(args.model, dataset_id, eval_ratio)
    elif args.mode == "reranker":
        eval_reranker(args.model, dataset_id, eval_ratio)


if __name__ == "__main__":
    main()
 
