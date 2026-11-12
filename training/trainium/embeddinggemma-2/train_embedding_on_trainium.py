"""Trainium training, evaluation, checkpoint selection, and export verification."""

import gc
import json
import logging
import math
from pathlib import Path
import time

from datasets import Dataset
import torch
from transformers import TrainerCallback, set_seed
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.data_collator import BaseDataCollator
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from embedding_model import (
    activate_on_neuron,
    build_loss,
    cpu_cosine,
    device_smoke,
    extract_retrieval_metrics,
    load_embedding_model,
    state_hashes,
)
from embedding_run_artifacts import collect_environment, hash_artifact, write_json


class NativeTrainer(SentenceTransformerTrainer):
    """Save Sentence Transformers checkpoints without framework-only state."""

    def _save(self, output_dir=None, state_dict=None):
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(
            str(destination), create_model_card=False
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(destination))
        torch.save(self.args, destination / "training_args.bin")


class TimingCallback(TrainerCallback):
    """Persist finite logs and optimizer-step durations."""

    def __init__(self, output):
        self.output = output
        self.seconds = []
        self.logged = []
        self.started = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.monotonic() - self.started
        self.seconds.append(elapsed)
        logging.info(
            "Optimizer step %s completed in %.3f seconds",
            state.global_step,
            elapsed,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        record = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float))
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise RuntimeError("Non-finite training loss/gradient log.")
        self.logged.append(record)
        write_json(self.output / "training_steps.json", self.logged)


class FixedLengthEvaluator(InformationRetrievalEvaluator):
    """Evaluate at a fixed length under Neuron BF16 autocast."""

    def __init__(self, *args, eval_max_seq_length, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_max_seq_length = eval_max_seq_length

    def __call__(self, model, *positional, **keyword):
        previous_length = model.max_seq_length
        model.max_seq_length = self.eval_max_seq_length
        try:
            with torch.autocast("neuron", dtype=torch.bfloat16):
                return super().__call__(model, *positional, **keyword)
        finally:
            model.max_seq_length = previous_length


def run_model_preflight(args, model):
    features = model.preprocess([args.query_prompt + "Test query"])
    if not features:
        raise RuntimeError("Model processor returned no features.")
    device_smoke()
    return {
        "status": "preflight_passed",
        "model": args.model_name,
        "device": args.device,
        "model_processor_loaded": True,
        "compiled_linear_update": True,
        "full_model_training_validated": False,
    }


def build_datasets(args, rows):
    columns = ["anchor", "positive"] + [
        f"negative_{index + 1}" for index in range(args.num_negatives)
    ]
    train_data = Dataset.from_list(rows["train"]).select_columns(columns)
    validation_data = Dataset.from_list(rows["validation"]).select_columns(
        columns
    )
    return columns, train_data, validation_data


def build_evaluator(args, output):
    data_directory = output / "data"
    qrels = json.loads(
        (data_directory / "validation-qrels.json").read_text()
    )
    with (data_directory / "corpus.jsonl").open() as stream:
        corpus = {
            row["id"]: row["text"]
            for row in map(json.loads, stream)
        }
    return FixedLengthEvaluator(
        queries=qrels["queries"],
        corpus=corpus,
        relevant_docs={
            query_id: set(document_ids)
            for query_id, document_ids in qrels["relevant_docs"].items()
        },
        name="validation",
        batch_size=args.retrieval_batch_size,
        query_prompt=args.query_prompt,
        corpus_prompt=args.document_prompt,
        score_functions={"cosine": cpu_cosine},
        main_score_function="cosine",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        precision_recall_at_k=[10],
        show_progress_bar=False,
        eval_max_seq_length=args.eval_max_seq_length,
    )


def evaluate_dimensions(model, evaluator, dimensions):
    result = {}
    try:
        for dimension in sorted(dimensions, reverse=True):
            evaluator.truncate_dim = dimension
            result[str(dimension)] = extract_retrieval_metrics(
                evaluator(model)
            )
    finally:
        evaluator.truncate_dim = None
    return result


def build_training_arguments(args, output, evaluator, prompts):
    return SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.05,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_drop_last=True,
        dataloader_pin_memory=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{evaluator.primary_metric}",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        prompts=prompts,
        push_to_hub=False,
    )


def fit_model(
    args,
    output,
    model,
    loss,
    evaluator,
    columns,
    train_data,
    validation_data,
):
    prompts = {
        column: args.query_prompt
        if column == "anchor"
        else args.document_prompt
        for column in columns
    }
    model.prompts.update(
        {"query": args.query_prompt, "document": args.document_prompt}
    )
    timing = TimingCallback(output)
    trainer = NativeTrainer(
        model=model,
        args=build_training_arguments(
            args, output, evaluator, prompts
        ),
        train_dataset=train_data,
        eval_dataset=validation_data,
        loss=loss,
        evaluator=evaluator,
        data_collator=BaseDataCollator(
            preprocess_fn=model.preprocess, prompts=prompts
        ),
        callbacks=[timing],
    )
    if trainer.args.device.type != "neuron":
        raise RuntimeError(
            f"Trainer selected {trainer.args.device}, not Neuron."
        )
    if len(trainer.get_train_dataloader()) == 0:
        raise ValueError(
            "No complete duplicate-free batches; reduce --batch_size."
        )
    trainer.train()
    return trainer, timing


def verify_weight_updates(before, after, steps, training_logs):
    if before.keys() != after.keys():
        raise RuntimeError("Model state keys changed unexpectedly.")
    changed = sum(after[name] != before[name] for name in before)
    has_nonzero_learning_rate = any(
        record.get("learning_rate", 0) > 0 for record in training_logs
    )
    if not changed or not steps or not has_nonzero_learning_rate:
        raise RuntimeError(
            "No verified weight update; use enough training steps beyond warmup."
        )
    return changed


def save_selected_model(output, model):
    final_directory = output / "final_model"
    model.save_pretrained(
        str(final_directory), create_model_card=False
    )
    return final_directory


def reload_and_verify_model(args, final_directory, expected_hashes):
    exported_model = load_embedding_model(args, final_directory)
    if state_hashes(exported_model) != expected_hashes:
        raise RuntimeError(
            "Exported weights do not match the selected checkpoint."
        )
    activate_on_neuron(exported_model, args.device)
    return exported_model


def build_training_result(
    args,
    before,
    baseline,
    scores,
    steps,
    selected_checkpoint,
    final_directory,
    changed,
    timing,
    exported_model,
):
    width = str(exported_model.get_embedding_dimension())
    delta = scores[width]["ndcg@10"] - baseline[width]["ndcg@10"]
    return {
        "status": "completed",
        "model": args.model_name,
        "model_revision": args.model_revision,
        "loss": "MatryoshkaLoss(MultipleNegativesRankingLoss)",
        "steps": steps,
        "baseline_by_dimension": baseline,
        "metrics_by_dimension": scores,
        "full_width_ndcg_delta": delta,
        "improved": delta > 0,
        "selected_checkpoint": selected_checkpoint,
        "artifact": str(final_directory),
        "artifact_sha256": hash_artifact(final_directory),
        "artifact_verified": True,
        "reload_encoding_device": args.device,
        "tensors_compared": len(before),
        "tensors_changed": changed,
        "all_weights_finite": True,
        "training_logs_finite": True,
        "optimizer_step_seconds": timing.seconds,
        "test_set_used": False,
    }


def train(args, output, rows):
    """Sequence preflight or full training through verified export."""
    if not hasattr(torch, "neuron") or not torch.neuron.is_available():
        raise RuntimeError(
            "Native torch.neuron is unavailable. Install the documented "
            "native backend and runtime."
        )
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 32
    set_seed(args.seed)
    write_json(
        output / "environment.json", collect_environment(args.device)
    )

    model = load_embedding_model(
        args, args.model_name, args.model_revision
    )
    if args.preflight:
        return run_model_preflight(args, model)

    before = state_hashes(model)
    write_json(output / "base_tensor_hashes.json", before)
    loss = build_loss(model, args.matryoshka_dims)
    activate_on_neuron(model, args.device)
    columns, train_data, validation_data = build_datasets(args, rows)
    evaluator = build_evaluator(args, output)
    baseline = evaluate_dimensions(
        model, evaluator, args.matryoshka_dims
    )
    write_json(output / "baseline.json", baseline)

    trainer, timing = fit_model(
        args,
        output,
        model,
        loss,
        evaluator,
        columns,
        train_data,
        validation_data,
    )
    steps = trainer.state.global_step
    selected_checkpoint = trainer.state.best_model_checkpoint
    del model.forward
    model.to("cpu")
    after = state_hashes(model)
    changed = verify_weight_updates(
        before, after, steps, timing.logged
    )

    final_directory = save_selected_model(output, model)
    del trainer, loss, model
    gc.collect()
    exported_model = reload_and_verify_model(
        args, final_directory, after
    )
    scores = evaluate_dimensions(
        exported_model, evaluator, args.matryoshka_dims
    )
    return build_training_result(
        args,
        before,
        baseline,
        scores,
        steps,
        selected_checkpoint,
        final_directory,
        changed,
        timing,
        exported_model,
    )
