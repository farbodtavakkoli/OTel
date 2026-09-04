"""Inference / evaluation for the trained sequence classifier — see readme_classification.md."""
import argparse
import torch
import os
import sys
import json
import numpy as np
import evaluate
import logging
import warnings
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed
)
from transformers.utils import logging as hf_logging
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_info()


def parse_args():
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained sequence classifier on a labeled CSV")
    parser.add_argument("--model_path", type=str, default="best_model_hf_ready",
                        help="HF-ready checkpoint dir (from prepare_model_for_hf.py)")
    parser.add_argument("--test_file", type=str, default="data/classification_sample.csv",
                        help="Test CSV with a text column (text/question) and a label column (label/answer)")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save results")
    parser.add_argument("--max_length", type=int, default=5000, help="Max input tokens")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--sample_fraction", type=float, default=0.2,
                        help="Fraction of the test set to evaluate (0 < f <= 1)")
    return parser.parse_args()


# Metrics are loaded individually because they take different arguments
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def load_test_dataset(test_file, label2id):
    """Load only the test dataset and map labels using the provided label2id."""
    test_file = Path(test_file)

    if not test_file.exists():
        raise FileNotFoundError(f"Could not find test file at {test_file}")

    logger.info(f"Loading data from: {test_file}")
    dataset = load_dataset('csv', data_files={'test': str(test_file)})

    cols = dataset["test"].column_names
    logger.info(f"Columns: {cols}")

    if "answer" in cols:
        target_col = "answer"
    elif "label" in cols:
        target_col = "label"
    else:
        raise ValueError(f"No 'answer' or 'label' column found in {cols}.")

    def map_labels(examples):
        return {"labels": [label2id[l] for l in examples[target_col]]}
        
    dataset = dataset.map(map_labels, batched=True)
    
    # Rename text column if needed for uniformity
    if "question" in cols and "text" not in cols:
        dataset = dataset.rename_column("question", "text")
        
    return dataset["test"]


def compute_metrics(predictions, labels, num_classes):
    """Compute metrics from predictions and labels."""
    # Debug logging — show ALL classes including zeros
    dist_dict = {i: int(np.sum(predictions == i)) for i in range(num_classes)}
    logger.info(f"\nPred Distribution (all {num_classes} classes): {dist_dict}")
    
    # Compute accuracy (no average arg)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Compute others (with macro average)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        prec = precision_metric.compute(predictions=predictions, references=labels, average="macro")
        rec = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        **acc,
        **f1,
        **prec,
        **rec
    }


def inference(args):
    set_seed(42)

    logger.info(f"Loading model from: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Left padding is required for decoder-based sequence classification.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # HF-ready checkpoint has a standard score.weight, so it loads directly.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # Get label mappings from model config
    label2id = model.config.label2id
    id2label = model.config.id2label
    num_labels = len(label2id)
    logger.info(f"Classes: {num_labels} => {label2id}")
    
    logger.info(f"Loading test data...")
    test_dataset = load_test_dataset(args.test_file, label2id)
    original_size = len(test_dataset)

    if args.sample_fraction < 1.0:
        num_samples = max(1, int(len(test_dataset) * args.sample_fraction))
        test_dataset = test_dataset.shuffle(seed=42).select(range(num_samples))
        logger.info(f"Sampled {num_samples} examples from {original_size} (fraction={args.sample_fraction})")
    
    logger.info(f"Test size: {len(test_dataset)}")

    # Store original texts for per-record logging
    original_texts = test_dataset["text"]
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    logger.info("Tokenizing...")
    cols_to_drop = [c for c in test_dataset.column_names if c not in ("labels",)]
    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)
    
    # Set format for PyTorch
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized_test, batch_size=args.batch_size, collate_fn=data_collator)
    
    # Inference
    logger.info("Running inference...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, num_labels)
    
    logger.info("\n" + "="*50)
    logger.info("INFERENCE RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"F1 (macro): {metrics['f1']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall']:.4f}")
    logger.info("="*50)
    
    # Save results locally
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.output_dir) / f"inference_results_{timestamp}.json"
    
    # Build per-record predictions
    per_record_results = []
    for i in range(len(all_labels)):
        pred_id = int(all_predictions[i])
        true_id = int(all_labels[i])
        per_record_results.append({
            "index": i,
            "text": original_texts[i][:500] + "..." if len(original_texts[i]) > 500 else original_texts[i],
            "prediction_id": pred_id,
            "prediction_label": id2label[pred_id],
            "ground_truth_id": true_id,
            "ground_truth_label": id2label[true_id],
            "correct": pred_id == true_id
        })
    
    results = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "test_size": len(all_labels),
        "sample_fraction": args.sample_fraction,
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "f1_macro": float(metrics['f1']),
            "precision_macro": float(metrics['precision']),
            "recall_macro": float(metrics['recall'])
        },
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "predictions": per_record_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return metrics


def main():
    args = parse_args()
    logger.info("Starting inference script")
    inference(args)


if __name__ == "__main__":
    main()
