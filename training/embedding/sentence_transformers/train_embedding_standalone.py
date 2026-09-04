"""Unified embedding fine-tuner (sentence-transformers) with hard negatives — see readme_embedding.md."""

import mteb
import os
import logging
import random
import shutil
import sys
import argparse
import torch
import torch.distributed as dist
import warnings
from mteb.cache import ResultCache
from mteb.models import EncoderProtocol
from datetime import datetime, timezone
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator, SimilarityFunction
from dotenv import load_dotenv

# dev.env supplies HF_TOKEN for gated-model downloads.
load_dotenv("dev.env")

# Per-model defaults; any field can be overridden by a CLI flag. Field meanings: see readme_embedding.md.
DEFAULT_CFG = {
    "loader": "transformer",
    "query_prefix": "",
    "doc_prefix": "",
    "train_batch": 64,
    "eval_batch": 128,
    "epochs": 2,
    "learning_rate": 1e-5,
    "use_checkpointing": True,
    "max_seq_length": None,
    "matryoshka_dims": None,
    "eval_corpus_size": None,
    "eval_on_start": False,
}

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODELS = {
    # sentence-transformers family
    "sentence-transformers/all-mpnet-base-v2": {
        "train_batch": 128, "eval_batch": 256, "use_checkpointing": False, "eval_on_start": True,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "train_batch": 512, "eval_batch": 1024, "use_checkpointing": False, "eval_on_start": True,
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "train_batch": 384, "eval_batch": 768, "use_checkpointing": False, "eval_on_start": True,
    },

    # BGE family
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 256, "eval_batch": 512, "epochs": 4, "use_checkpointing": True,
    },
    "BAAI/bge-large-en-v1.5": {
        "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 32, "eval_batch": 64, "epochs": 4, "use_checkpointing": True,
    },
    "BAAI/bge-m3": {
        "query_prefix": "",  # Not required for M3 dense retrieval
        "train_batch": 16, "eval_batch": 32, "epochs": 4, "use_checkpointing": True,
    },

    # Gemma
    "google/embeddinggemma-300m": {
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "train_batch": 96,   # Downscaled to fix OOM (7x inputs: 1 anchor + 1 pos + 5 negs)
        "eval_batch": 192,
        "epochs": 2,
        "use_checkpointing": True,
        "max_seq_length": 1024,             # Covers >99.9% of Tele-Eval data
        "matryoshka_dims": [768, 512, 256, 128],
        "eval_corpus_size": 607000,         # Large distractor pool
    },

    # Static model2vec family; most potion models are distilled from BGE v1.5, hence the BGE-style query prefix.
    "minishlab/potion-base-2M": {
        "loader": "static", "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 1024, "eval_batch": 1024, "epochs": 3,
        "learning_rate": 2e-3, "use_checkpointing": False,
    },
    "minishlab/potion-base-32M": {
        "loader": "static", "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 1024, "eval_batch": 1024, "epochs": 3,
        "learning_rate": 2e-3, "use_checkpointing": False,
    },
    "minishlab/potion-retrieval-32M": {
        "loader": "static", "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 1024, "eval_batch": 1024, "epochs": 3,
        "learning_rate": 2e-3, "use_checkpointing": False,
    },
    "minishlab/potion-base-128M": {
        "loader": "static", "query_prefix": _BGE_QUERY_PREFIX,
        "train_batch": 512, "eval_batch": 512, "epochs": 3,
        "learning_rate": 2e-3, "use_checkpointing": False,
    },
}

def parse_args():
    """Parse CLI arguments; registry-backed flags default to None, meaning "use the registry value"."""
    parser = argparse.ArgumentParser(description="Unified embedding fine-tuner (ST / BGE / Gemma / model2vec)")
    parser.add_argument("--model_name", type=str, default="google/embeddinggemma-300m", help="Model name or path")
    parser.add_argument("--train_file", type=str, default="OTel_embedding_sample_100.jsonl",
                        help="Training JSONL with anchor/positive/negative_1..negative_5 columns")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--experiment_root", type=str, default="experiments",
                        help="Root directory used for HF_HOME and the default output dir")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of the dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data splits and eval sampling")
    parser.add_argument("--loader", type=str, default=None, choices=["transformer", "static"],
                        help="Model loader type (overrides registry)")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size (overrides registry)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides registry)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides registry)")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length (overrides registry)")
    parser.add_argument("--eval_corpus_size", type=int, default=None,
                        help="Distractor corpus size for eval (overrides registry; omit for all-positives corpus)")
    return parser.parse_args()


def resolve_cfg(model_name, args_cli):
    """Merge DEFAULT_CFG <- registry entry <- CLI overrides."""
    cfg = dict(DEFAULT_CFG)
    cfg.update(MODELS.get(model_name, {}))

    if args_cli.loader is not None:
        cfg["loader"] = args_cli.loader
    if args_cli.batch_size is not None:
        cfg["train_batch"] = args_cli.batch_size
        cfg["eval_batch"] = args_cli.batch_size * 2
    if args_cli.epochs is not None:
        cfg["epochs"] = args_cli.epochs
    if args_cli.lr is not None:
        cfg["learning_rate"] = args_cli.lr
    if args_cli.max_seq_length is not None:
        cfg["max_seq_length"] = args_cli.max_seq_length
    if args_cli.eval_corpus_size is not None:
        cfg["eval_corpus_size"] = args_cli.eval_corpus_size
    return cfg


def safe_get_model_meta(model_name):
    """mteb.get_model_meta with a fallback for models not on the MTEB hub (static models, local paths)."""
    try:
        meta = mteb.get_model_meta(model_name)
    except Exception:
        from mteb.model_meta import ModelMeta
        meta = ModelMeta(name=model_name, revision="main", release_date=None, languages=["en"])
    meta.revision = "main"
    return meta


def get_datasets(path, max_eval_samples=3000, test_size=0.1, sample_fraction=1.0, seed=42):
    """Load the training JSONL and produce a deterministic train/eval split shared by all ranks."""
    full_ds = load_dataset("json", data_files=path, split="train")
    logging.info("Loaded %d training data from %s", len(full_ds), path)

    if sample_fraction < 1.0:
        sample_size = int(len(full_ds) * sample_fraction)
        logging.info(f"Subsampling dataset to {sample_fraction:.2%} ({sample_size} samples)")
        # Deterministic shuffle + select so all ranks agree
        full_ds = full_ds.shuffle(seed=seed).select(range(sample_size))

    num_test = min(max_eval_samples, int(len(full_ds) * test_size))
    logging.info(f"Using {num_test} samples for evaluation (capped at {max_eval_samples})")

    # A fixed seed ensures all GPU ranks split the data identically
    split_ds = full_ds.train_test_split(test_size=num_test, seed=seed)
    return split_ds["train"], split_ds["test"]


class MTEBWrapper:
    def __init__(self, tasks, model_name):
        self.task_names = tasks
        self.model_name = model_name
        self.name = "mteb"
        self.primary_metric = "mteb_average_score"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        is_main_process = int(os.environ.get("RANK", 0)) == 0
        rank = int(os.environ.get("RANK", 0))

        results = {self.primary_metric: 0.0}
        for t_name in self.task_names:
            results[f"mteb_{t_name}"] = 0.0

        if rank == 0:
            logging.info(f"[rank=0] MTEB Evaluation: Epoch {epoch}, Step {steps}")
            run_id = f"mteb_epoch_{epoch}_step_{steps}"
            target_dir = os.path.join(output_path, run_id) if output_path else run_id

            try:
                fresh_tasks = mteb.get_tasks(tasks=self.task_names)
                custom_cache = ResultCache(cache_path=target_dir)

                # WORKAROUND: MTEB 2.7.x crashes if 'model' is a SentenceTransformer
                # but doesn't have a strict 'organization/model' name format.
                model_name_outer = self.model_name

                class SafeSTWrapper(EncoderProtocol):
                    def __init__(self, st_model, m_name):
                        self.model = st_model
                        self._mteb_model_meta = safe_get_model_meta(m_name)

                    @property
                    def mteb_model_meta(self):
                        return self._mteb_model_meta

                    def encode(self, sentences, **kwargs):
                        # MTEB passes extra args SentenceTransformer.encode doesn't accept
                        for ignore_arg in ['hf_split', 'task_metadata', 'hf_subset', 'prompt_type']:
                            kwargs.pop(ignore_arg, None)

                        # Robust input coercion: handle ndarrays, dicts, and
                        # nested/columnar lists that would otherwise crash encode().
                        if not isinstance(sentences, list):
                            if hasattr(sentences, "tolist"):
                                sentences = sentences.tolist()
                            else:
                                sentences = list(sentences)

                        if len(sentences) > 0:
                            first_item = sentences[0]
                            if isinstance(first_item, dict):
                                found_key = None
                                if "text" in first_item:
                                    found_key = "text"
                                elif "sentence" in first_item:
                                    found_key = "sentence"

                                if found_key:
                                    extracted = [s[found_key] for s in sentences]
                                else:
                                    logging.warning(
                                        f"[rank=0] encode received dicts without known keys: {first_item.keys()}"
                                    )
                                    extracted = [list(s.values())[0] for s in sentences]

                                # Flatten columnar format e.g. [{'text': ['a', 'b']}]
                                if len(extracted) > 0 and isinstance(extracted[0], list):
                                    flattened = []
                                    for sub in extracted:
                                        flattened.extend(sub)
                                    sentences = flattened
                                else:
                                    sentences = extracted

                            # Handle lists like [['text']] or other non-string iterables
                            elif hasattr(first_item, "__iter__") and not isinstance(first_item, (str, bytes, dict)):
                                new_sentences = []
                                for s in sentences:
                                    if hasattr(s, "__iter__") and not isinstance(s, (str, bytes, dict)):
                                        parts = [str(x) for x in s]
                                        new_sentences.append(parts[0] if len(parts) == 1 else " ".join(parts))
                                    else:
                                        new_sentences.append(str(s))
                                sentences = new_sentences

                        return self.model.encode(sentences, **kwargs)

                    def similarity(self, embeddings1, embeddings2):
                        return self.model.similarity(embeddings1, embeddings2)

                    def similarity_pairwise(self, embeddings1, embeddings2):
                        return self.model.similarity_pairwise(embeddings1, embeddings2)

                safe_model = SafeSTWrapper(model, model_name_outer)

                mteb_output = mteb.evaluate(
                    model=safe_model,
                    tasks=fresh_tasks,
                    cache=custom_cache,
                    encode_kwargs={
                        "batch_size": 256,
                        "normalize_embeddings": True,
                        "show_progress_bar": is_main_process,
                        "convert_to_tensor": False,
                    },
                )

                if mteb_output:
                    task_scores = []
                    for task_result in mteb_output:
                        t_name = task_result.task.metadata.name
                        score = task_result.get_score()
                        if score is None:
                            main_met = task_result.task.metadata.main_score
                            score = task_result.get_score(metric=main_met)

                        if score is not None:
                            val = float(score)
                            task_scores.append(val)
                            results[f"mteb_{t_name}"] = val

                    if task_scores:
                        avg_score = sum(task_scores) / len(task_scores)
                        results[self.primary_metric] = float(avg_score)
                        logging.info(f"[rank=0] MTEB Success. Avg Score: {avg_score:.4f}")

            except Exception as e:
                import traceback
                logging.error(f"[rank=0] MTEB Execution Error: {e}")
                logging.error(traceback.format_exc())

        if dist.is_initialized():
            dist.barrier()

        return results


def create_telco_evaluator(train_ds, eval_ds, model_name, corpus_size=None, seed=42):
    """Build the SequentialEvaluator: unseen/seen IR evaluators plus an MTEB check."""
    random.seed(seed)

    n_eval = min(10000, len(eval_ds))
    n_seen = min(10000, len(train_ds))

    eval_sample = eval_ds.select(random.sample(range(len(eval_ds)), n_eval))
    seen_sample = train_ds.select(random.sample(range(len(train_ds)), n_seen))

    global_positives_pool = list(set(train_ds["positive"]) | set(eval_ds["positive"]))

    if corpus_size is None:
        # Simple mode (st/baai default): corpus = all known positives.
        all_chunks = global_positives_pool
    else:
        # Distractor mode (gemma/minishlab): required answers + sampled distractors.
        required_answers = list(set(eval_sample["positive"]) | set(seen_sample["positive"]))
        required_answers_set = set(required_answers)
        true_distractor_candidates = [t for t in global_positives_pool if t not in required_answers_set]

        target_corpus_size = min(corpus_size, len(global_positives_pool))
        num_distractors_needed = max(0, target_corpus_size - len(required_answers))
        distractors = random.sample(
            true_distractor_candidates,
            min(num_distractors_needed, len(true_distractor_candidates)),
        )
        all_chunks = required_answers + distractors

    corpus = {f"doc_{i}": text for i, text in enumerate(all_chunks)}
    text_to_id = {text: doc_id for doc_id, text in corpus.items()}

    is_main_process = int(os.environ.get("RANK", 0)) == 0

    unseen_eval = InformationRetrievalEvaluator(
        queries={f"u_{i}": row["anchor"] for i, row in enumerate(eval_sample)},
        corpus=corpus,
        relevant_docs={f"u_{i}": {text_to_id[row["positive"]]} for i, row in enumerate(eval_sample)},
        name="telco_unseen",
        main_score_function=SimilarityFunction.DOT_PRODUCT,
        batch_size=512,
        corpus_chunk_size=150000,
        show_progress_bar=is_main_process,
        query_prompt_name="Retrieval-query",
        corpus_prompt_name="Retrieval-document",
    )

    seen_eval = InformationRetrievalEvaluator(
        queries={f"s_{i}": row["anchor"] for i, row in enumerate(seen_sample)},
        corpus=corpus,
        relevant_docs={f"s_{i}": {text_to_id[row["positive"]]} for i, row in enumerate(seen_sample)},
        name="telco_seen",
        main_score_function=SimilarityFunction.DOT_PRODUCT,
        batch_size=512,
        corpus_chunk_size=150000,
        show_progress_bar=is_main_process,
        query_prompt_name="Retrieval-query",
        corpus_prompt_name="Retrieval-document",
    )

    mteb_eval = MTEBWrapper(tasks=["SciFact", "NFCorpus"], model_name=model_name)

    return SequentialEvaluator([unseen_eval, seen_eval, mteb_eval], main_score_function=lambda scores: scores[0])


def load_model(model_name, cfg, local_rank):
    """Load an embedding model according to its loader type ("transformer" or "static")."""
    if cfg["loader"] == "static":
        # Static model2vec: a lookup table, so no dtype/attention/tokenizer kwargs.
        static_embedding = StaticEmbedding.from_model2vec(model_name)
        model = SentenceTransformer(modules=[static_embedding], device=f"cuda:{local_rank}")
        return model

    # Transformer-backed models.
    # flash-attn as pinned is a CUDA-only build; on ROCm (torch.version.hip) fall back to
    # PyTorch SDPA — verified on AMD MI355X / ROCm 7.2 (see readme_embedding.md).
    # On CUDA, prefer flash_attention_2 only when the flash-attn package is actually
    # importable; otherwise fall back to SDPA (verified on NVIDIA H100 / CUDA 13.0 where
    # no prebuilt flash-attn wheel exists for torch 2.11+cu130 — see readme_embedding.md).
    if torch.version.hip:
        attn_impl = "sdpa"
    else:
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    model = SentenceTransformer(
        model_name,
        model_kwargs={"dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": attn_impl},
        tokenizer_kwargs={"padding_side": "right"},
        device=f"cuda:{local_rank}",
    )

    # Remove a problematic tokenizer flag before it gets serialized (from gemma).
    # Harmless no-op for models (e.g. BGE / ST) that never had it.
    if hasattr(model.tokenizer, "init_kwargs"):
        model.tokenizer.init_kwargs.pop("fix_mistral_regex", None)

    # Gradient checkpointing is incompatible with KV caching; disable use_cache.
    if hasattr(model[0].auto_model, "config"):
        model[0].auto_model.config.use_cache = False

    return model


def main():
    args_cli = parse_args()

    warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*n_jobs.*")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = args_cli.experiment_root
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    cfg = resolve_cfg(args_cli.model_name, args_cli)

    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        run_id = os.environ.get("RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        output_dir = os.path.join(args_cli.experiment_root, run_id, f"trained_{args_cli.model_name.replace('/', '_')}")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[rank=0] %(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"))
        file_handler.setFormatter(logging.Formatter("[rank=0] %(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
        logging.info("Resolved config for %s: %s", args_cli.model_name, cfg)

    dist.barrier()

    train_dataset, eval_dataset = get_datasets(
        args_cli.train_file, sample_fraction=args_cli.sample_fraction, seed=args_cli.seed
    )

    if rank == 0:
        logging.info("Loading Model: %s (loader=%s)", args_cli.model_name, cfg["loader"])

    model = load_model(args_cli.model_name, cfg, local_rank)

    # Metadata injection so MTEB applies the correct instruction templates
    # (falls back gracefully for static / off-hub models).
    model.mteb_model_meta = safe_get_model_meta(args_cli.model_name)

    if cfg["max_seq_length"] is not None:
        model.max_seq_length = cfg["max_seq_length"]

    # Prompt configuration
    model.prompts = {
        "Retrieval-query": cfg["query_prefix"],
        "Retrieval-document": cfg["doc_prefix"],
        "query": cfg["query_prefix"],
        "document": cfg["doc_prefix"],
    }
    model.default_prompt_name = "Retrieval-query"
    model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
    model.encode_kwargs = {'normalize_embeddings': True, 'batch_size_sorting': False}

    # Matryoshka dims: registry override or dynamic from hidden size
    hidden_dim = model.get_sentence_embedding_dimension()
    if cfg["matryoshka_dims"] is not None:
        matryoshka_dims = cfg["matryoshka_dims"]
    else:
        possible_dims = [1024, 768, 512, 256, 128, 64, 32]
        matryoshka_dims = [d for d in possible_dims if d <= hidden_dim]
        if hidden_dim not in matryoshka_dims:
            matryoshka_dims = [hidden_dim] + matryoshka_dims

    if rank == 0:
        logging.info("Model dimension: %d | Matryoshka: %s", hidden_dim, matryoshka_dims)

    evaluator = create_telco_evaluator(
        train_dataset, eval_dataset, args_cli.model_name, corpus_size=cfg["eval_corpus_size"], seed=args_cli.seed
    )

    base_loss = losses.MultipleNegativesRankingLoss(
        model, scale=20.0, similarity_fct=util.dot_score, gather_across_devices=True
    )
    train_loss = losses.MatryoshkaLoss(
        model, base_loss, matryoshka_dims=matryoshka_dims, matryoshka_weights=[1.0] * len(matryoshka_dims)
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["train_batch"],
        per_device_eval_batch_size=cfg["eval_batch"],
        gradient_checkpointing=cfg["use_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["use_checkpointing"] else None,
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        optim="adamw_torch",
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        dataloader_num_workers=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_telco_unseen_dot_ndcg@10",
        greater_is_better=True,
        dataloader_drop_last=True,
        eval_on_start=cfg["eval_on_start"],
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=1,
        ddp_timeout=3600,
        ddp_find_unused_parameters=False,
    )

    # WORKAROUND: transformers 5.0.0.dev0 misses save_safetensors
    if not hasattr(args, "save_safetensors"):
        args.save_safetensors = True

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
        loss=train_loss, evaluator=evaluator,
    )

    if rank == 0:
        shutil.copyfile(sys.argv[0], os.path.join(output_dir, "train_script_backup.py"))
        logging.info("Beginning training with model %s", args_cli.model_name)

    trainer.train()

    if rank == 0:
        logging.info("Saving final model to %s", output_dir)
        model.save(os.path.join(output_dir, "final_model"))


if __name__ == "__main__":
    main()
