# Open Telco (OTel) AI — Datasets, Benchmarks, and Models

<p align="center">
  <a href="https://huggingface.co/farbodtavakkoli">🤗 HuggingFace</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-llm">LLM Collection</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-embedding">Embedding Collection</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-reranker">Reranker Collection</a> •
  <a href="https://huggingface.co/farbodtavakkoli/datasets">Datasets</a> •
  <a href="https://github.com/farbodtavakkoli/OTel/blob/main/docs/media_coverage.md">Media Coverage</a>
</p>

This repository contains the training and inference code for the **Open Telco (OTel) AI** project — an open-source initiative delivering curated telecom datasets, evaluation partitions, and a family of telecom-specialized embedding, reranker, and language models. The repository is built on top of **[ScalarLM](https://www.scalarlm.com)**, a GPU-agnostic open-source framework for training, inference, and deployment across both AMD and NVIDIA hardware.

## 🎯 Project Overview

The OTel models are full-parameter post-trained embedding models, rerankers, and language models optimized for retrieval-augmented generation (RAG) and agentic applications in telecommunications. They are trained on high-quality data curated by 100+ telecom domain experts and contributed by specific institutional partners:

| Source | Contributor | Raw samples |
|---|---|---:|
| arXiv telecom papers, 3GPP standards, telecom Wikipedia, telecom Common Crawl pages | Yale University | 681,172 |
| GSMA Permanent Reference Documents (PRDs), Discover portal, mixed telecom documents | GSMA | 158,006 |
| IETF RFC series | NetoAI | 100,751 |
| Industry whitepapers | Khalifa University | 62,000 |
| O-RAN specifications across working groups 1, 2, 4, 5, 6, 7, 8, 9, 10 | University of Leeds | 58,565 |
| O-RAN documents across working groups | The University of Texas at Dallas | 42,000 |
| **Total raw samples** | | **~1,102,494** |

After a four-stage cleaning pipeline (heuristic filtering, reranker-based semantic filtering, embedding-based semantic filtering, and deduplication), the corpus was reduced to **326,767** higher-confidence examples released across four datasets.

> As of May 2026, the released OTel models have been downloaded over **16 million times** and the project has received **157+ pieces of media coverage** worldwide.

## 📦 Model Zoo

All models are available on [HuggingFace](https://huggingface.co/farbodtavakkoli) and have been full-parameter post-trained on both AMD and NVIDIA GPUs.

### Datasets

| Dataset | Purpose |
|---|---|
| [OTel-LLM](https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM) | Context-grounded instruction tuning for telecom RAG generation |
| [OTel-Embedding](https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding) | Bi-encoder retrieval training (MNRL, 5 hard negatives per anchor) |
| [OTel-Reranker](https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker) | Pointwise cross-encoder reranking with continuous relevance scores |
| [OTel-Safety](https://huggingface.co/datasets/farbodtavakkoli/OTel-Safety) | Abstention training for refusing answers when context is insufficient |

### Language Models (LLMs)

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel-LLM-270M-IT | 270M | gemma-3-270m-it |
| OTel-LLM-0.6B-IT | 0.6B | Qwen3-0.6B |
| OTel-LLM-1B-IT | 1B | gemma-3-1b-it |
| OTel-LLM-1.2B-IT | 1.2B | LFM2.5-1.2B-Instruct |
| OTel-LLM-1.7B-IT | 1.7B | Qwen3-1.7B |
| OTel-LLM-3B-IT | 3B | Mistral-3-3B |
| OTel-LLM-4B-IT | 4B | gemma-3-4b-it |
| OTel-LLM-7B-IT | 7B | OLMo-3-7B |
| OTel-LLM-8.2B-IT | 8.2B | Qwen3-8B |
| OTel-LLM-8.3B-IT | 8.3B | RNJ-1-Instruct |
| OTel-LLM-12B-IT | 12B | gemma-3-12b-it |
| OTel-LLM-14B-IT | 14B | Qwen3-14B |
| OTel-LLM-20B-IT | 20B | GPT-OSS-20B |
| OTel-LLM-20B-Reasoning | 20B | GPT-OSS-20B |
| OTel-LLM-24B-IT | 24B | LFM2-24B-A2B |
| OTel-LLM-27B-IT | 27B | gemma-3-27b-it |
| OTel-LLM-32B-IT | 32B | OLMo-3-32B |

### Embedding Models

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel-Embedding-22M | 22M | all-MiniLM-L6-v2 |
| OTel-Embedding-33M | 33M | BAAI/bge-small-en-v1.5 |
| OTel-Embedding-34M | 34M | all-MiniLM-L12-v2 |
| OTel-Embedding-109M | 109M | all-mpnet-base-v2 |
| OTel-Embedding-300M | 300M | Gemma3-Embedding-300M |
| OTel-Embedding-335M | 335M | BAAI/bge-large-en-v1.5 |
| OTel-Embedding-568M | 568M | BAAI/bge-m3 |
| OTel-Embedding-600M | 600M | Qwen3-Embedding-0.6B |
| OTel-Embedding-4B | 4B | Qwen3-Embedding-4B |
| OTel-Embedding-8B | 8B | Qwen3-Embedding-8B |

### Reranker Models

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel-Reranker-0.6B | 0.6B | Qwen3-0.6B |
| OTel-Reranker-4B | 4B | Qwen3-4B |
| OTel-Reranker-8B | 8B | Qwen3-8B |

### Auxiliary Models

These are released alongside the core 30-model OTel baseline family but are discussed separately in the paper appendix.

| Model | Parameters | Base Model | Purpose |
|-------|------------|------------|---------|
| OTel-LLM-8.3B-Safety | 8.3B | RNJ-1-Instruct | Abstention-focused safety variant |
| OTel-LLM-12B-Safety | 12B | gemma-3-12b-it | Abstention-focused safety variant |
| OTel-LLM-8.3B-Classification | 8.3B | RNJ-1 | Classification head for the TeleLogs 5G root-cause-analysis benchmark |
| OTel-LLM-8.3B-QnA | 8.3B | RNJ-1-Instruct | Non-abstention QnA variant for direct context-free telecom QA |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv scalarlm_env
source scalarlm_env/bin/activate

# Install dependencies
pip3 install scalarlm pyyaml python-dotenv

# Set deployment endpoint
export SCALARLM_API_URL={your_endpoint_url}
```

### Training

```bash
# Train a language model
python3 train.py --data_path data/llm_training_sample.json --model_type qwen3

# Train with different model architectures
python3 train.py --model_type gemma3    # Gemma-style tokens
python3 train.py --model_type llama3    # Llama-style tokens
python3 train.py --model_type mistral   # Mistral-style tokens

# Train an embedding model
python3 train.py --training_mode embedding --data_path data/embedding_training_sample.json
```

### Inference

```bash
# Inference from a fine-tuned language model
python3 inference.py --model_name {model_id} --model_type qwen3

# Inference with different model types
python3 inference.py --model_name {model_id} --model_type gemma3
python3 inference.py --model_name {model_id} --model_type llama3

# Embedding inference
python3 inference.py --inference_mode embedding --test_data_path data/embedding_training_sample.json
```

### Evaluation

Reproduce the held-out OTel results for any of the three benchmarked model families with a single command:

```bash
# LLM correctness via LLM-as-judge (90/10 split, seed 42)
python eval.py --mode llm --model farbodtavakkoli/OTel-LLM-1.2B-IT

# Embedding NDCG@10 (90/10 split, seed 42)
python eval.py --mode embedding --model farbodtavakkoli/OTel-Embedding-300M

# Reranker MRR@10 (95/5 split, seed 42)
python eval.py --mode reranker --model farbodtavakkoli/OTel-Reranker-0.6B
```

Embedding and reranker modes need only `HF_TOKEN`. LLM mode also requires `OPENAI_API_KEY` (the LLM-as-judge defaults to `gpt-4.1`; override with `--judge gpt-4o-mini`). Place both tokens in a `dev.env` file at the repo root — see `dev.env.example`.

### View Training Logs

```bash
scalarlm logs --model={model_id}
```

## ⚙️ Configuration Options

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/llm_training_sample.json` | Path to training data |
| `--training_mode` | `language_model` | Training mode: `language_model` or `embedding` |
| `--model_type` | `qwen3` | Model architecture: `gemma3`, `qwen3`, `llama3`, `rnj-1`, `olmo3`, `mistral`, `gpt-oss_reasoning`, `gpt_oss_it` |
| `--gradient_on_output_only` | `True` | Compute gradient only on output tokens (input masked) |
| `--sample_fraction` | `1.0` | Fraction of dataset to use (0.0-1.0), useful for testing |
| `--max_steps` | `10` | Maximum training steps |
| `--learning_rate` | `0.0005` | Learning rate |
| `--batch_size` | `2` | Batch size per GPU |
| `--max_token_block_size` | `1730` | Maximum token block size |
| `--r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha scaling factor |
| `--lora_dropout` | `0.05` | LoRA dropout rate |
| `--target_modules` | `['q_proj', 'k_proj', 'v_proj', 'o_proj']` | LoRA target modules |
| `--steps_per_checkpoint` | `100` | Steps between checkpoints |
| `--use_lora` | `True` | Enable LoRA fine-tuning (disable for Tokenformer) |
| `--full_parameter_training` | `True` | Enable full parameter training |
| `--optimizer_type` | `adamw` | Optimizer: `adamw`, `sgd`, `rmsprop` |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--gradient_checkpointing` | `True` | Enable gradient checkpointing to reduce memory |
| `--distribution_strategy` | `fsdp` | Distribution strategy: `fsdp` or `ddp` |
| `--attn_implementation` | `flash_attention_2` | Attention: `flash_attention_2`, `sdpa`, `eager` |
| `--freeze_keywords` | `vision_model, vision_tower, ...` | Comma-separated keywords to freeze parameters |
| `--upload_to_hf` | `False` | Upload final model to HuggingFace Hub |
| `--hf_repo_id` | `farbodtavakkoli/scalarlm-test` | HuggingFace repo ID for upload |
| `--gpus` | `1` | Number of GPUs per Kubernetes pod |
| `--nodes` | `2` | Number of nodes for distributed training |

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `gemma3` | Model architecture for token formatting |
| `--inference_mode` | `language_model` | Inference mode: `language_model` or `embedding` |
| `--max_tokens` | `500` | Maximum tokens for generation |
| `--batch_size` | `2` | Batch size for inference |

## 📊 Training Data Format

### Language Model Data

```json
{
  "prompt": "Your input question or context",
  "completion": "Expected model response",
  "reasoning": "Optional reasoning trace (for reasoning models)"
}
```

### Embedding Data

```json
{
  "anchor": "Query text",
  "positive": "Relevant passage",
  "negative_1": "Hard negative 1",
  "negative_2": "Hard negative 2",
  "negative_3": "Hard negative 3",
  "negative_4": "Hard negative 4",
  "negative_5": "Hard negative 5"
}
```

## 🌐 Infrastructure

- **Compute**: AMD and NVIDIA GPUs
- **Framework**: ScalarLM (GPU-agnostic)

## Citation

```bibtex
@misc{otel2026,
  title  = {OTel: Open Telco AI Datasets, Benchmarks, and Models},
  author = {Tavakkoli, Farbod and others},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://huggingface.co/farbodtavakkoli}
}
```

## 📬 Contact

If you have any technical questions, please feel free to reach out to farbod.tavakkoli@att.com or farbodtavakoli@gmail.com
