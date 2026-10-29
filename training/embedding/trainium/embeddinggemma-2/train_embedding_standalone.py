#!/usr/bin/env python3
"""Standalone native TorchNeuron text embedding training; no local module imports.

MNRL + MatryoshkaLoss, grouped data splits, baseline/reload validation and
weight-change verification. Only EmbeddingGemma-2 has been validated on Trainium.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time

DEFAULT_MODEL = "google/embeddinggemma-2"
DEFAULT_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return " ".join(value.split())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def prepare_data(args, data_dir):
    data_dir.mkdir(parents=True, exist_ok=False)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    rows, documents, queries = [], {}, {}
    positives = defaultdict(set)
    audit = Counter()
    pairs = set()
    fuzzy = {}
    for index, line in enumerate(args.train_file.open()):
        row = json.loads(line)
        audit["source_rows"] += 1
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("anchor", "positive")):
            audit["invalid_pairs"] += 1
            continue
        q, p = digest(normalized(row["anchor"])), digest(normalized(row["positive"]))
        positives[q].add(p)
        queries.setdefault(q, row["anchor"])
        documents.setdefault(p, row["positive"])
        union("q" + q, "d" + p)
        # Conservatively group punctuation/whitespace variants of positives.
        signature = digest(" ".join(re.findall(r"\w+", row["positive"])))
        if signature in fuzzy:
            union("d" + p, "d" + fuzzy[signature])
        fuzzy[signature] = p
        negatives = []
        for key in ("negative_1", "negative_2", "negative_3", "negative_4", "negative_5"):
            text = row.get(key)
            if not isinstance(text, str) or not text.strip():
                audit["invalid_negatives"] += 1
                continue
            n = digest(normalized(text))
            documents.setdefault(n, text)
            if n not in negatives:
                negatives.append(n)
            else:
                audit["duplicate_negatives"] += 1
        if (q, p) in pairs:
            audit["duplicate_pairs"] += 1
            continue
        pairs.add((q, p))
        rows.append((index, q, p, negatives))

    query_split = {}
    for q in queries:
        bucket = int(digest(f"{args.seed}:" + find("q" + q))[:8], 16) % 1000
        query_split[q] = "train" if bucket < 900 else "validation" if bucket < 950 else "test"
    heldout_positive = set().union(*(
        positives[q] for q in queries if query_split[q] != "train"
    ))
    split_rows = defaultdict(list)
    for index, q, p, negatives in rows:
        clean = []
        for n in negatives:
            if n in positives[q]:
                audit["known_positive_negative_conflicts"] += 1
            elif query_split[q] == "train" and n in heldout_positive:
                audit["heldout_positives_removed_from_training_negatives"] += 1
            else:
                clean.append(n)
        if len(clean) < args.num_negatives:
            audit["pairs_without_clean_negative"] += 1
            continue
        split_rows[query_split[q]].append({
            "row": index, "qid": q, "pid": p, "nid": clean[0],
            "anchor": queries[q], "positive": documents[p],
            **{f"negative_{i+1}": documents[n] for i, n in enumerate(clean[:args.num_negatives])},
        })
    split_hashes = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in split_rows[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit[f"{split}_pairs"] = len(split_rows[split])
        audit[f"{split}_queries"] = len({r["qid"] for r in split_rows[split]})
    # The whole source has >500k passages. Freeze a sampled benchmark explicitly.
    selected = {}
    required = set()
    for split in ("validation", "test"):
        qs = sorted({r["qid"] for r in split_rows[split]}, key=lambda q: digest(f"{args.seed}:" + q))[:512]
        selected[split] = qs
        for q in qs:
            required.update(positives[q])
        selected_queries = set(qs)
        required.update(r["nid"] for r in split_rows[split] if r["qid"] in selected_queries)
    distractors = sorted(set(documents) - required, key=lambda d: digest(f"{args.seed}:" + d))[:args.eval_corpus_size]
    corpus_ids = sorted(required | set(distractors))
    with (data_dir / "corpus.jsonl").open("w") as stream:
        for d in corpus_ids:
            stream.write(json.dumps({"id": d, "text": documents[d]}, ensure_ascii=False) + "\n")
    for split, qs in selected.items():
        (data_dir / f"{split}-qrels.json").write_text(json.dumps({
            "queries": {q: queries[q] for q in qs},
            "relevant_docs": {q: sorted(positives[q]) for q in qs},
        }, ensure_ascii=False))
    spec = {
        "dataset_id": str(args.train_file),
        "dataset_revision": "local-file-sha256",
        "source_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_hashes": split_hashes, "seed": args.seed,
        "query_ids": selected, "corpus_ids": corpus_ids,
        "corpus_sampling": f"Required positives/negatives plus {args.eval_corpus_size} hash-ranked distractors",
        "maximum_length": args.eval_max_seq_length, "dimensions": args.matryoshka_dims, "normalize": True,
        "similarity": "cosine", "precision": "fp32 weights, bf16 autocast",
        "query_prompt": args.query_prompt,
        "document_prompt": args.document_prompt,
        "objective": "ndcg@10", "secondary": ["recall@10", "mrr@10"],
        "minimum_delta": 0.005,
        "truncation": "Right truncate after native prefix and special tokens",
        "limitations": [
            "Source document IDs are absent; document-level isolation cannot be certified.",
            "Repeated query/positive components and punctuation variants are grouped; semantic near-duplicates may remain.",
            "Negatives are source-ranked, not reranker-verified; semantic false negatives may remain.",
            "Metrics describe a fixed sampled corpus, not retrieval over all source documents.",
        ],
    }
    spec["evaluation_id"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (data_dir / "evaluation-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    audit.update(unique_queries=len(queries), unique_documents=len(documents),
                 evaluation_documents=len(corpus_ids))
    (data_dir / "audit.json").write_text(json.dumps(dict(audit), indent=2) + "\n")
    return split_rows, spec, dict(audit)



def configure_model(model):
    import torch
    original = model.preprocess

    def preprocess(inputs, **kwargs):
        length = model.max_seq_length
        kwargs.update(max_length=length, truncation=True)
        features = original(inputs, **kwargs)
        for key in ('input_ids', 'attention_mask', 'token_type_ids'):
            if key in features:
                value = model.tokenizer.pad_token_id if key == 'input_ids' else 0
                tensor = features[key]
                if tensor.shape[1] > length:
                    raise ValueError('Preprocessing exceeded the fixed sequence length.')
                features[key] = torch.nn.functional.pad(
                    tensor, (0, length - tensor.shape[1]), value=value)
        return features
    model.preprocess = preprocess


def build_loss(model, dimensions):
    from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.util import dot_score
    width = model.get_embedding_dimension()
    if (not dimensions or len(set(dimensions)) != len(dimensions)
            or width not in dimensions or any(d <= 0 or d > width for d in dimensions)):
        raise ValueError(f'Use unique positive dimensions <= {width}, including {width}.')
    base = MultipleNegativesRankingLoss(model, scale=20.0, similarity_fct=dot_score,
                                       gather_across_devices=False)
    return MatryoshkaLoss(model, base, matryoshka_dims=dimensions,
                         matryoshka_weights=[1.0] * len(dimensions), n_dims_per_step=-1)


def cpu_cosine(queries, documents):
    import torch
    import torch.nn.functional as F
    with torch.autocast('cpu', enabled=False):
        q = queries.detach().to(device='cpu', dtype=torch.float32)
        d = documents.detach().to(device='cpu', dtype=torch.float32)
        return F.normalize(q, dim=1) @ F.normalize(d, dim=1).T


def metrics(raw):
    result = {name: float(next(v for k, v in raw.items() if k.endswith('cosine_' + name)))
              for name in ('ndcg@10', 'recall@10', 'mrr@10')}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError('Invalid retrieval metrics.')
    return result


def state_hashes(model):
    import torch
    result = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite model tensor: {name}')
        result[name] = hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def self_test():
    """CPU loss/gradient test; does not establish Neuron hardware compatibility."""
    import copy
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Dense
    torch.manual_seed(42)
    model = SentenceTransformer(modules=[Dense(8, 768, activation_function=torch.nn.Identity())], device='cpu')
    reference = copy.deepcopy(model)
    model.forward = torch.compile(model.forward, backend='eager', dynamic=False)
    original = model.forward
    xs = [torch.randn(3, 8) for _ in range(3)]
    dims = [768, 512, 256, 128]
    loss = build_loss(model, dims)
    assert not loss.uses_gradient_cache
    actual = loss([{'sentence_embedding': x.clone()} for x in xs], None)
    actual.backward()
    reps = [reference({'sentence_embedding': x.clone()})['sentence_embedding'] for x in xs]
    expected = 0
    for dim in dims:
        q, p, n = [F.normalize(x[:, :dim], dim=-1) for x in reps]
        expected += F.cross_entropy(20 * q @ torch.cat([p, n]).T, torch.arange(3))
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for p, r in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-4, atol=1e-5)
        assert torch.isfinite(p.grad).all()
    assert model.forward is original
    before = state_hashes(model)
    torch.optim.AdamW(model.parameters(), lr=1e-5).step()
    assert before != state_hashes(model)
    for bad in ([769, 128], [128], [768, 0], [768, 768]):
        try:
            build_loss(model, bad)
        except ValueError:
            continue
        raise AssertionError(f'Accepted invalid dimensions: {bad}')
    print(json.dumps({'passed': True, 'device': 'cpu', 'checks':
                     ['loss', 'gradients', 'finite_gradients', 'forward_restoration', 'weight_update', 'invalid_dimensions']}))


def train(args, output, rows, spec):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from sentence_transformers import (SentenceTransformer, SentenceTransformerModelCardData,
        SentenceTransformerTrainer, SentenceTransformerTrainingArguments)
    from sentence_transformers.base.data_collator import BaseDataCollator
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    if not hasattr(torch, 'neuron') or not torch.neuron.is_available():
        raise RuntimeError('Native torch.neuron is unavailable. Install the documented native backend and runtime.')
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 32
    set_seed(args.seed)
    versions = {}
    for name in ('torch', 'torch-neuronx', 'neuronx-cc', 'nki', 'transformers',
                 'sentence-transformers', 'datasets', 'accelerate'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(output / 'environment.json', {'versions': versions, 'python': sys.version,
               'python_executable': sys.executable, 'sys_path': sys.path, 'device': args.device})

    class NativeTrainer(SentenceTransformerTrainer):
        def _save(self, output_dir=None, state_dict=None):
            dest = Path(output_dir or self.args.output_dir)
            dest.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(dest), create_model_card=False)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(str(dest))
            torch.save(self.args, dest / 'training_args.bin')

    class Timing(TrainerCallback):
        def __init__(self):
            self.seconds, self.logged = [], []
        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
        def on_step_end(self, args, state, control, **kwargs):
            self.seconds.append(time.monotonic() - self.started)
            logging.info('Optimizer step %s completed in %.3f seconds', state.global_step, self.seconds[-1])
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                record = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                if not all(math.isfinite(v) for v in record.values()):
                    raise RuntimeError('Non-finite training loss/gradient log.')
                self.logged.append(record)
                write_json(output / 'training_steps.json', self.logged)

    class FixedEvaluator(InformationRetrievalEvaluator):
        def __call__(self, model, *pos, **kw):
            previous = model.max_seq_length
            model.max_seq_length = args.eval_max_seq_length
            try:
                with torch.autocast('neuron', dtype=torch.bfloat16):
                    return super().__call__(model, *pos, **kw)
            finally:
                model.max_seq_length = previous

    def load(path, revision=None):
        model = SentenceTransformer(str(path), revision=revision, device='cpu',
            cache_folder=str(args.model_cache_dir),
            config_kwargs={'vision_config': None, 'audio_config': None},
            model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'},
            model_card_data=SentenceTransformerModelCardData(language='en', license='apache-2.0',
                model_name='EmbeddingGemma 2 Matryoshka text fine-tune'))
        model.max_seq_length = args.max_seq_length
        return model

    def activate(model):
        model.to(args.device)
        configure_model(model)
        model.forward = torch.compile(model.forward, backend='neuron', dynamic=False)

    model = load(args.model_name, args.model_revision)
    before = state_hashes(model)
    write_json(output / 'base_tensor_hashes.json', before)
    loss = build_loss(model, args.matryoshka_dims)
    activate(model)
    columns = ['anchor', 'positive'] + [f'negative_{i+1}' for i in range(args.num_negatives)]
    train_data = Dataset.from_list(rows['train']).select_columns(columns)
    valid_data = Dataset.from_list(rows['validation']).select_columns(columns)
    data = output / 'data'
    qrels = json.loads((data / 'validation-qrels.json').read_text())
    corpus = {r['id']: r['text'] for r in map(json.loads, (data / 'corpus.jsonl').open())}
    evaluator = FixedEvaluator(queries=qrels['queries'], corpus=corpus,
        relevant_docs={q: set(v) for q, v in qrels['relevant_docs'].items()},
        name='validation', batch_size=args.retrieval_batch_size,
        query_prompt=args.query_prompt, corpus_prompt=args.document_prompt,
        score_functions={'cosine': cpu_cosine}, main_score_function='cosine',
        mrr_at_k=[10], ndcg_at_k=[10], precision_recall_at_k=[10], show_progress_bar=False)

    def evaluate_dimensions(model):
        result = {}
        try:
            for dim in sorted(args.matryoshka_dims, reverse=True):
                evaluator.truncate_dim = dim
                result[str(dim)] = metrics(evaluator(model))
        finally:
            evaluator.truncate_dim = None
        return result

    baseline = evaluate_dimensions(model)
    write_json(output / 'baseline.json', baseline)
    metric_key = f'eval_{evaluator.primary_metric}'
    prompts = {c: args.query_prompt if c == 'anchor' else args.document_prompt for c in columns}
    # Store prompts in the export, not just in the training collator.
    model.prompts.update({'query': args.query_prompt, 'document': args.document_prompt})
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output / 'checkpoints'), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr, weight_decay=0.01, warmup_steps=0.05,
        optim='adamw_torch', lr_scheduler_type='linear', max_grad_norm=1.0,
        bf16=True, fp16=False, gradient_checkpointing=False,
        gradient_accumulation_steps=1, batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True, dataloader_pin_memory=False,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        logging_steps=1, logging_first_step=True, load_best_model_at_end=True,
        metric_for_best_model=metric_key, greater_is_better=True, report_to='none',
        seed=args.seed, prompts=prompts, push_to_hub=False)
    timing = Timing()
    trainer = NativeTrainer(model=model, args=training_args, train_dataset=train_data,
        eval_dataset=valid_data, loss=loss, evaluator=evaluator,
        data_collator=BaseDataCollator(preprocess_fn=model.preprocess, prompts=prompts), callbacks=[timing])
    if trainer.args.device.type != 'neuron':
        raise RuntimeError(f'Trainer selected {trainer.args.device}, not Neuron.')
    if len(trainer.get_train_dataloader()) == 0:
        raise ValueError('No complete duplicate-free batches; reduce --batch_size.')
    trainer.train()
    steps, selected = trainer.state.global_step, trainer.state.best_model_checkpoint
    del model.forward
    model.to('cpu')
    after = state_hashes(model)
    if before.keys() != after.keys():
        raise RuntimeError('Model state keys changed unexpectedly.')
    changed = sum(after[k] != before[k] for k in before)
    if not changed or not steps or not any(e.get('learning_rate', 0) > 0 for e in timing.logged):
        raise RuntimeError('No verified weight update; use enough training steps beyond warmup.')
    final = output / 'final_model'
    model.save_pretrained(str(final), create_model_card=False)
    del trainer, loss, model
    gc.collect()
    exported = load(final)
    if state_hashes(exported) != after:
        raise RuntimeError('Exported weights do not match the selected checkpoint.')
    activate(exported)
    scores = evaluate_dimensions(exported)
    artifact_hash = hashlib.sha256()
    for path in sorted(final.rglob('*')):
        if path.is_file():
            artifact_hash.update(str(path.relative_to(final)).encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    artifact_hash.update(block)
    width = str(exported.get_embedding_dimension())
    delta = scores[width]['ndcg@10'] - baseline[width]['ndcg@10']
    return {'status': 'completed', 'model': args.model_name, 'model_revision': args.model_revision,
        'loss': 'MatryoshkaLoss(MultipleNegativesRankingLoss)', 'steps': steps,
        'baseline_by_dimension': baseline, 'metrics_by_dimension': scores,
        'full_width_ndcg_delta': delta, 'improved': delta > 0,
        'selected_checkpoint': selected, 'artifact': str(final), 'artifact_sha256': artifact_hash.hexdigest(),
        'artifact_verified': True, 'reload_encoding_device': args.device,
        'tensors_compared': len(before), 'tensors_changed': changed, 'all_weights_finite': True,
        'training_logs_finite': True, 'optimizer_step_seconds': timing.seconds,
        'test_set_used': False}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_name', default=DEFAULT_MODEL)
    p.add_argument('--model_revision', default=DEFAULT_REVISION)
    p.add_argument('--train_file', type=Path, default=Path(__file__).resolve().with_name('OTel_embedding_sample_100.jsonl'))
    p.add_argument('--output_dir', type=Path, default=Path('outputs/embeddinggemma2'))
    p.add_argument('--model_cache_dir', type=Path, default=Path.home() / '.cache/huggingface/hub')
    p.add_argument('--compile_cache_dir', type=Path, default=None)
    p.add_argument('--device', default='neuron:0', choices=['neuron:0'])
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--eval_batch_size', type=int, default=1)
    p.add_argument('--retrieval_batch_size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max_steps', type=int, default=-1)
    p.add_argument('--max_seq_length', type=int, default=512)
    p.add_argument('--eval_max_seq_length', type=int, default=1024)
    p.add_argument('--matryoshka_dims', type=int, nargs='+', default=[768, 512, 256, 128])
    p.add_argument('--num_negatives', type=int, choices=range(1, 6), default=1)
    p.add_argument('--eval_corpus_size', type=int, default=20000, help='Additional hash-ranked distractors.')
    p.add_argument('--query_prompt', default='task: search result | query: ')
    p.add_argument('--document_prompt', default='title: none | text: ')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prepare_only', action='store_true')
    p.add_argument('--self_test', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    if args.self_test:
        self_test()
        return
    if args.model_name != DEFAULT_MODEL:
        raise ValueError('This release validates only google/embeddinggemma-2; other architectures need a tested profile.')
    if (min(args.batch_size, args.eval_batch_size, args.retrieval_batch_size, args.epochs,
            args.max_seq_length, args.eval_max_seq_length) < 1 or args.eval_corpus_size < 0
            or not math.isfinite(args.lr) or args.lr <= 0
            or args.max_steps == 0 or args.max_steps < -1):
        raise ValueError('Invalid batch, length, epoch, learning-rate or step setting.')
    args.train_file = args.train_file.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir = output
    args.model_cache_dir = args.model_cache_dir.resolve()
    cache = (args.compile_cache_dir or output / 'compiled').resolve()
    for variable, name in [('TORCH_NEURONX_NEFF_CACHE_DIR', 'neff'),
            ('TORCH_NEURONX_HLO_CACHE_DIR', 'hlo'), ('TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR', 'local'),
            ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        (cache / name).mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache / name)
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'run.log')])
    for name in ('httpx', 'httpcore', 'huggingface_hub', 'urllib3', 'filelock', 'fsspec'):
        logging.getLogger(name).setLevel(logging.WARNING)
    write_json(output / 'config.json', vars(args))
    write_json(output / 'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).resolve().parent.iterdir() if p.is_file()})
    started = time.monotonic()
    try:
        rows, spec, audit = prepare_data(args, output / 'data')
        if len(rows['train']) < args.batch_size or not rows['validation']:
            raise ValueError('Insufficient train/validation rows after grouping; use more data or another seed/smaller batch.')
        if args.prepare_only:
            write_json(output / 'result.json', {'status': 'prepared', 'audit': audit})
            print(json.dumps(audit))
            return
        result = train(args, output, rows, spec)
        result.update(elapsed_seconds=time.monotonic() - started, audit=audit,
                      evaluation_id=spec['evaluation_id'])
        write_json(output / 'result.json', result)
        logging.info('Completed: %s', output / 'result.json')
    except Exception as exc:
        write_json(output / 'result.json', {'status': 'failed', 'error': str(exc),
                                          'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
