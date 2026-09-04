"""Helpers for train_embedding_pylate.py — config resolution, data, model/loss build, late-interaction proof."""

import json
import logging
import os
import random

import numpy as np
import torch
from datasets import Dataset
from pylate import indexes, losses, models, rank, retrieve, scores

LOGGER = logging.getLogger("training/embedding/pylate")


def setup_logging(output_dir, is_main):
    """Log to stdout and to run.log under the output dir; non-main ranks stay quiet."""
    handlers = [logging.StreamHandler()]
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(output_dir, "run.log")))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[rank=%s] %%(levelname)s: %%(message)s" % os.environ.get("RANK", "0"),
        handlers=handlers,
        force=True,
    )
    return LOGGER


def set_seed(seed):
    """Seed python/numpy/torch so every rank splits the data identically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_rocm():
    """True when the installed torch is a ROCm/HIP build."""
    return torch.version.hip is not None


def resolve_attn_implementation(requested):
    """Pick the attention kernel; flash_attention_2 is CUDA-only, so ROCm falls back to sdpa."""
    if requested != "auto":
        return requested
    return "sdpa" if is_rocm() else "flash_attention_2"


def resolve_tf32(requested):
    """tf32 is an NVIDIA Ampere+ feature; enabling it on a ROCm build raises."""
    return bool(requested) and torch.version.cuda is not None


def resolve_scores_backend(requested):
    """PyLate's flash/lik MaxSim kernels are CUDA-only; force the torch backend on ROCm."""
    if requested != "auto":
        return requested
    return "torch" if is_rocm() else "auto"


def resolve_cfg(defaults, registry, model_name, args_cli):
    """Merge defaults <- registry entry <- non-None CLI overrides."""
    cfg = dict(defaults)
    cfg.update(registry.get(model_name, {}))
    for key, value in vars(args_cli).items():
        if value is not None and key in cfg:
            cfg[key] = value
    return cfg


def read_jsonl(path):
    """Read a JSONL file into a list of dicts."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def document_columns(rows):
    """Return the document column names (positive first, then any negatives) in a stable order."""
    keys = [key for key in rows[0] if key != "query"]
    positives = [key for key in keys if key.startswith("positive")]
    negatives = sorted(key for key in keys if key.startswith("negative"))
    return positives + negatives


def load_triplets(train_file, sample_fraction, eval_fraction, n_negatives, seed):
    """Load the PyLate triplet JSONL and split it into train/eval Datasets."""
    rows = read_jsonl(train_file)
    columns = document_columns(rows)
    negatives = [key for key in columns if key.startswith("negative")][:n_negatives]
    keep = ["query"] + [key for key in columns if key.startswith("positive")] + negatives

    rows = [{key: row[key] for key in keep} for row in rows]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if sample_fraction < 1.0:
        rows = rows[: max(1, int(len(rows) * sample_fraction))]

    n_eval = max(1, int(len(rows) * eval_fraction)) if eval_fraction > 0 else 0
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    train = Dataset.from_list(train_rows)
    evaluation = Dataset.from_list(eval_rows) if eval_rows else None
    LOGGER.info("Loaded %d train / %d eval rows; columns %s", len(train_rows), len(eval_rows), keep)
    return train, evaluation, keep


def build_model(cfg, attn_implementation, dtype):
    """Build a PyLate ColBERT model from a base encoder or an existing ColBERT checkpoint."""
    model_kwargs = {"attn_implementation": attn_implementation}
    if dtype != "float32":
        model_kwargs["dtype"] = getattr(torch, dtype)
    model = models.ColBERT(
        model_name_or_path=cfg["model_name"],
        embedding_size=cfg["embedding_dim"],
        query_length=cfg["query_length"],
        document_length=cfg["document_length"],
        model_kwargs=model_kwargs,
    )
    LOGGER.info(
        "Loaded %s — %d modules, per-token dim %d, query_length %d, document_length %d",
        cfg["model_name"], len(model), model.get_sentence_embedding_dimension(),
        model.query_length, model.document_length,
    )
    return model


def build_loss(loss_name, model, temperature, gather_across_devices, score_mini_batch_size):
    """Build the late-interaction training objective."""
    if loss_name == "contrastive":
        return losses.Contrastive(
            model=model, temperature=temperature,
            gather_across_devices=gather_across_devices,
            score_mini_batch_size=score_mini_batch_size,
        )
    if loss_name == "cached_contrastive":
        return losses.CachedContrastive(
            model=model, temperature=temperature,
            gather_across_devices=gather_across_devices,
        )
    if loss_name == "distillation":
        return losses.Distillation(model=model)
    raise ValueError("Unknown loss %s" % loss_name)


def build_evaluator(eval_dataset, columns, batch_size):
    """ColBERT triplet accuracy evaluator over the held-out split."""
    from pylate.evaluation import ColBERTTripletEvaluator

    negatives = [key for key in columns if key.startswith("negative")]
    if eval_dataset is None or not negatives:
        return None
    positive = [key for key in columns if key.startswith("positive")][0]
    return ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset[positive],
        negatives=eval_dataset[negatives[0]],
        name="otel_triplet",
        batch_size=batch_size,
    )


def pad_multi_vector(embeddings):
    """Pad a ragged list of [n_tokens, dim] arrays into a [n, max_tokens, dim] tensor plus its mask."""
    tensors = [torch.as_tensor(np.asarray(e), dtype=torch.float32) for e in embeddings]
    max_tokens = max(tensor.shape[0] for tensor in tensors)
    padded = torch.zeros(len(tensors), max_tokens, tensors[0].shape[1])
    mask = torch.zeros(len(tensors), max_tokens)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.shape[0]] = tensor
        mask[i, : tensor.shape[0]] = 1.0
    return padded, mask


def encode_multi_vector(model, texts, is_query, batch_size):
    """Encode texts and return one variable-length [n_tokens, dim] array per text."""
    embeddings = model.encode(sentences=texts, batch_size=batch_size, is_query=is_query, show_progress_bar=False)
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    return embeddings


def late_interaction_report(model, query, documents, batch_size, logger):
    """Prove the model is genuinely multi-vector: per-token shapes plus a MaxSim ranking."""
    query_embeddings = encode_multi_vector(model, [query], is_query=True, batch_size=batch_size)
    document_embeddings = encode_multi_vector(model, documents, is_query=False, batch_size=batch_size)

    logger.info("LATE-INTERACTION query embedding shape: %s (ndim=%d)",
                tuple(np.asarray(query_embeddings[0]).shape), np.asarray(query_embeddings[0]).ndim)
    for i, embedding in enumerate(document_embeddings):
        logger.info("LATE-INTERACTION document[%d] embedding shape: %s (ndim=%d)",
                    i, tuple(np.asarray(embedding).shape), np.asarray(embedding).ndim)

    assert np.asarray(query_embeddings[0]).ndim == 2, "query embedding is not multi-vector"
    assert all(np.asarray(e).ndim == 2 for e in document_embeddings), "document embeddings are not multi-vector"

    queries_padded, queries_mask = pad_multi_vector(query_embeddings)
    documents_padded, documents_mask = pad_multi_vector(document_embeddings)
    maxsim = scores.colbert_scores(
        queries_embeddings=queries_padded,
        documents_embeddings=documents_padded,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
    )
    ranking = torch.argsort(maxsim[0], descending=True).tolist()
    for position, index in enumerate(ranking):
        logger.info("MAXSIM rank %d: score %.4f | doc[%d] %s",
                    position + 1, float(maxsim[0][index]), index, documents[index][:70].replace("\n", " "))
    return maxsim[0].tolist(), ranking


def rerank_report(model, query, documents, batch_size, logger):
    """Run PyLate's rerank() over the same documents — the reranking use case, no index needed."""
    query_embeddings = encode_multi_vector(model, [query], is_query=True, batch_size=batch_size)
    document_embeddings = encode_multi_vector(model, documents, is_query=False, batch_size=batch_size)
    reranked = rank.rerank(
        documents_ids=[[str(i) for i in range(len(documents))]],
        queries_embeddings=query_embeddings,
        documents_embeddings=[document_embeddings],
    )
    for position, hit in enumerate(reranked[0]):
        logger.info("RERANK rank %d: id %s score %.4f", position + 1, hit["id"], float(hit["score"]))
    return reranked[0]


def index_report(model, query, documents, index_folder, index_name, batch_size, device, logger):
    """Build a PLAID index over the documents and retrieve — end-to-end late-interaction retrieval."""
    document_ids = [str(i) for i in range(len(documents))]
    document_embeddings = encode_multi_vector(model, documents, is_query=False, batch_size=batch_size)
    index = indexes.PLAID(index_folder=index_folder, index_name=index_name, override=True)
    index.add_documents(documents_ids=document_ids, documents_embeddings=document_embeddings)
    retriever = retrieve.ColBERT(index=index)
    query_embeddings = encode_multi_vector(model, [query], is_query=True, batch_size=batch_size)
    results = retriever.retrieve(queries_embeddings=query_embeddings, k=min(5, len(documents)), device=device)
    for position, hit in enumerate(results[0]):
        logger.info("PLAID rank %d: id %s score %.4f", position + 1, hit["id"], float(hit["score"]))
    return results[0]
