"""Retrieval-data parsing, grouping, splitting, and evaluation artifacts."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from embedding_training_config import (
    MAX_EVALUATION_QUERIES_PER_SPLIT,
    SPLIT_BUCKET_COUNT,
    SPLIT_HASH_PREFIX_LENGTH,
    TRAIN_SPLIT_CUTOFF,
    VALIDATION_SPLIT_CUTOFF,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value):
    return " ".join(value.split())


class ConnectedComponents:
    """Small deterministic union-find used to keep related rows in one split."""

    def __init__(self):
        self.parent = {}

    def find(self, component_id):
        self.parent.setdefault(component_id, component_id)
        while self.parent[component_id] != component_id:
            self.parent[component_id] = self.parent[self.parent[component_id]]
            component_id = self.parent[component_id]
        return component_id

    def union(self, first_component_id, second_component_id):
        first_root = self.find(first_component_id)
        second_root = self.find(second_component_id)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(
                first_root, second_root
            )


def load_and_group_rows(train_file):
    components = ConnectedComponents()
    rows = []
    documents = {}
    queries = {}
    positives = defaultdict(set)
    audit = Counter()
    pairs = set()
    fuzzy_positive_ids = {}

    with train_file.open() as stream:
        for row_index, line in enumerate(stream):
            row = json.loads(line)
            audit["source_rows"] += 1
            if any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("anchor", "positive")
            ):
                audit["invalid_pairs"] += 1
                continue

            query_id = digest(normalized(row["anchor"]))
            positive_id = digest(normalized(row["positive"]))
            positives[query_id].add(positive_id)
            queries.setdefault(query_id, row["anchor"])
            documents.setdefault(positive_id, row["positive"])
            components.union("q" + query_id, "d" + positive_id)

            signature = digest(" ".join(re.findall(r"\w+", row["positive"])))
            if signature in fuzzy_positive_ids:
                components.union(
                    "d" + positive_id,
                    "d" + fuzzy_positive_ids[signature],
                )
            fuzzy_positive_ids[signature] = positive_id

            negative_ids = []
            for key in (
                "negative_1",
                "negative_2",
                "negative_3",
                "negative_4",
                "negative_5",
            ):
                text = row.get(key)
                if not isinstance(text, str) or not text.strip():
                    audit["invalid_negatives"] += 1
                    continue
                negative_id = digest(normalized(text))
                documents.setdefault(negative_id, text)
                if negative_id not in negative_ids:
                    negative_ids.append(negative_id)
                else:
                    audit["duplicate_negatives"] += 1

            pair = (query_id, positive_id)
            if pair in pairs:
                audit["duplicate_pairs"] += 1
                continue
            pairs.add(pair)
            rows.append((row_index, query_id, positive_id, negative_ids))

    return rows, documents, queries, positives, components, audit


def assign_query_splits(queries, components, seed):
    query_split = {}
    for query_id in queries:
        split_hash = digest(f"{seed}:" + components.find("q" + query_id))
        bucket = (
            int(split_hash[:SPLIT_HASH_PREFIX_LENGTH], 16)
            % SPLIT_BUCKET_COUNT
        )
        if bucket < TRAIN_SPLIT_CUTOFF:
            query_split[query_id] = "train"
        elif bucket < VALIDATION_SPLIT_CUTOFF:
            query_split[query_id] = "validation"
        else:
            query_split[query_id] = "test"
    return query_split


def filter_and_split_rows(
    rows,
    documents,
    queries,
    positives,
    query_split,
    num_negatives,
    audit,
):
    heldout_positive_ids = set().union(
        *(
            positives[query_id]
            for query_id in queries
            if query_split[query_id] != "train"
        )
    )
    split_rows = defaultdict(list)
    for row_index, query_id, positive_id, negative_ids in rows:
        clean_negative_ids = []
        for negative_id in negative_ids:
            if negative_id in positives[query_id]:
                audit["known_positive_negative_conflicts"] += 1
            elif (
                query_split[query_id] == "train"
                and negative_id in heldout_positive_ids
            ):
                audit[
                    "heldout_positives_removed_from_training_negatives"
                ] += 1
            else:
                clean_negative_ids.append(negative_id)

        if len(clean_negative_ids) < num_negatives:
            audit["pairs_without_clean_negative"] += 1
            continue
        split_rows[query_split[query_id]].append(
            {
                "row": row_index,
                "qid": query_id,
                "pid": positive_id,
                "nid": clean_negative_ids[0],
                "anchor": queries[query_id],
                "positive": documents[positive_id],
                **{
                    f"negative_{negative_index + 1}": documents[negative_id]
                    for negative_index, negative_id in enumerate(
                        clean_negative_ids[:num_negatives]
                    )
                },
            }
        )
    return split_rows


def write_split_files(data_dir, split_rows, audit):
    split_hashes = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in split_rows[split]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit[f"{split}_pairs"] = len(split_rows[split])
        audit[f"{split}_queries"] = len(
            {row["qid"] for row in split_rows[split]}
        )
    return split_hashes


def select_evaluation_corpus(
    split_rows,
    documents,
    positives,
    seed,
    eval_corpus_size,
):
    selected_query_ids_by_split = {}
    required_document_ids = set()
    for split in ("validation", "test"):
        selected_query_ids = sorted(
            {row["qid"] for row in split_rows[split]},
            key=lambda query_id: digest(f"{seed}:" + query_id),
        )[:MAX_EVALUATION_QUERIES_PER_SPLIT]
        selected_query_ids_by_split[split] = selected_query_ids
        for query_id in selected_query_ids:
            required_document_ids.update(positives[query_id])
        selected_query_id_set = set(selected_query_ids)
        required_document_ids.update(
            row["nid"]
            for row in split_rows[split]
            if row["qid"] in selected_query_id_set
        )

    distractor_ids = sorted(
        set(documents) - required_document_ids,
        key=lambda document_id: digest(f"{seed}:" + document_id),
    )[:eval_corpus_size]
    corpus_ids = sorted(required_document_ids | set(distractor_ids))
    return selected_query_ids_by_split, corpus_ids


def write_evaluation_files(
    data_dir,
    selected_query_ids_by_split,
    corpus_ids,
    documents,
    queries,
    positives,
):
    with (data_dir / "corpus.jsonl").open("w") as stream:
        for document_id in corpus_ids:
            stream.write(
                json.dumps(
                    {"id": document_id, "text": documents[document_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    for split, selected_query_ids in selected_query_ids_by_split.items():
        (data_dir / f"{split}-qrels.json").write_text(
            json.dumps(
                {
                    "queries": {
                        query_id: queries[query_id]
                        for query_id in selected_query_ids
                    },
                    "relevant_docs": {
                        query_id: sorted(positives[query_id])
                        for query_id in selected_query_ids
                    },
                },
                ensure_ascii=False,
            )
        )


def build_evaluation_spec(
    args,
    selected_query_ids_by_split,
    corpus_ids,
    split_hashes,
):
    preprocessing_source = Path(__file__)
    spec = {
        "dataset_id": str(args.train_file),
        "dataset_revision": "local-file-sha256",
        "source_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "preprocessing_sha256": hashlib.sha256(
            preprocessing_source.read_bytes()
        ).hexdigest(),
        "split_hashes": split_hashes,
        "seed": args.seed,
        "query_ids": selected_query_ids_by_split,
        "corpus_ids": corpus_ids,
        "corpus_sampling": (
            "Required positives/negatives plus "
            f"{args.eval_corpus_size} hash-ranked distractors"
        ),
        "maximum_length": args.eval_max_seq_length,
        "dimensions": args.matryoshka_dims,
        "normalize": True,
        "similarity": "cosine",
        "precision": "fp32 weights, bf16 autocast",
        "query_prompt": args.query_prompt,
        "document_prompt": args.document_prompt,
        "objective": "ndcg@10",
        "secondary": ["recall@10", "mrr@10"],
        "minimum_delta": 0.005,
        "truncation": "Right truncate after native prefix and special tokens",
        "limitations": [
            "Source document IDs are absent; document-level isolation cannot be certified.",
            "Repeated query/positive components and punctuation variants are grouped; "
            "semantic near-duplicates may remain.",
            "Negatives are source-ranked, not reranker-verified; semantic false "
            "negatives may remain.",
            "Metrics describe a fixed sampled corpus, not retrieval over all "
            "source documents.",
        ],
    }
    spec["evaluation_id"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return spec


def prepare_data(args, data_dir):
    data_dir.mkdir(parents=True, exist_ok=False)
    (
        rows,
        documents,
        queries,
        positives,
        components,
        audit,
    ) = load_and_group_rows(args.train_file)
    query_split = assign_query_splits(queries, components, args.seed)
    split_rows = filter_and_split_rows(
        rows,
        documents,
        queries,
        positives,
        query_split,
        args.num_negatives,
        audit,
    )
    split_hashes = write_split_files(data_dir, split_rows, audit)
    selected_query_ids_by_split, corpus_ids = select_evaluation_corpus(
        split_rows,
        documents,
        positives,
        args.seed,
        args.eval_corpus_size,
    )
    write_evaluation_files(
        data_dir,
        selected_query_ids_by_split,
        corpus_ids,
        documents,
        queries,
        positives,
    )
    spec = build_evaluation_spec(
        args,
        selected_query_ids_by_split,
        corpus_ids,
        split_hashes,
    )
    (data_dir / "evaluation-spec.json").write_text(
        json.dumps(spec, indent=2) + "\n"
    )
    audit.update(
        unique_queries=len(queries),
        unique_documents=len(documents),
        evaluation_documents=len(corpus_ids),
    )
    audit_result = dict(audit)
    (data_dir / "audit.json").write_text(
        json.dumps(audit_result, indent=2) + "\n"
    )
    return split_rows, spec, audit_result
