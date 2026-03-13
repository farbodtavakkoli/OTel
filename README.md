# OTel Family of Models — LLM, Embedding & Reranker Suite

<p align="center">
  <a href="https://huggingface.co/farbodtavakkoli">🤗 HuggingFace</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-llm">LLM Collection</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-embedding">Embedding Collection</a> •
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-reranker">Reranker Collection</a>
  <a href="https://huggingface.co/farbodtavakkoli/datasets">Datasets</a>
</p>

This repository has been built on top of **ScalarLM**, a GPU-agnostic, open-source framework for training, inference, and deployment. This repository contains the training and inference code for the OTel models, a collaborative effort to build industry-standard AI models for the global telecommunications sector.

## 🎯 Project Overview

The OTel models are fine-tuned language models, embedding models, and rerankers optimized for RAG and agentic applications in telecommunications. Our models are trained high-quality data points curated by 200+ telecom domain experts from organizations including AT&T, GSMA, Purdue University, Khalifa University, University of Leeds, Yale University, and others.

**Training Data Sources:**
- GSMA Permanent Reference Documents
- 3GPP Specifications
- O-RAN Documentation
- RFC Series
- eSIM, terminals, security, networks, roaming, APIs
- Industry whitepapers and telecom academic papers

## 📦 Model Zoo

All models are available on [HuggingFace](https://huggingface.co/farbodtavakkoli) and have been full-parameter trained on both AMD and NVIDIA GPUs.

### Language Models (LLMs)

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel_LLM_270M_IT | 270M | gemma-3-270m-it |
| OTel_LLM_0.6B_IT | 0.6B | Qwen3-0.6B |
| OTel_LLM_1.2B_IT | 1.2B | LFM2.5-1.2B-Instruct |
| OTel_LLM_1B_IT | 1B | gemma-3-1b-it |
| OTel_LLM_1.7B_IT | 1.7B | Qwen3-1.7B |
| OTel_LLM_3B_IT | 3B | Mistral-3-3B |
| OTel_LLM_4B_IT | 4B | gemma-3-4b-it |
| OTel_LLM_7B_IT | 7B | OLMo-3-7B |
| OTel_LLM_8.2B_IT | 8.2B | Qwen3-8B |
| OTel_LLM_8.3B_IT | 8.3B | RNJ-1-Instruct |
| OTel_LLM_8.3B_Safety | 8.3B | RNJ-1-Instruct |
| OTel_LLM_12B_IT | 12B | gemma-3-12b-it |
| OTel_LLM_12B_Safety | 12B | gemma-3-12b-it |
| OTel_LLM_14B_IT | 14B | Qwen3-14B |
| OTel_LLM_20B_IT | 20B | GPT-OSS-20B |
| OTel_LLM_20B_Reasoning | 20B | GPT-OSS-20B |
| OTel_LLM_24B_IT | 24B | LFM2-24B-A2B |
| OTel_LLM_27B_IT | 27B | gemma-3-27b-it |
| OTel_LLM_32B_IT | 32B | OLMo-3-32B |

### Embedding Models

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel_Embedding_22M | 22M | all-MiniLM-L6-v2 |
| OTel_Embedding_33M | 33M | BAAI/bge-small-en-v1.5 |
| OTel_Embedding_34M | 34M | all-MiniLM-L12-v2 |
| OTel_Embedding_109M | 109M | all-mpnet-base-v2 |
| OTel_Embedding_300M | 300M | Gemma3-Embedding-300M |
| OTel_Embedding_335M | 335M | BAAI/bge-large-en-v1.5 |
| OTel_Embedding_568M | 568M | BAAI/bge-m3 |
| OTel_Embedding_600M | 600M | Qwen3-Embedding-0.6B |
| OTel_Embedding_4B | 4B | Qwen3-Embedding-4B |
| OTel_Embedding_8B | 8B | Qwen3-Embedding-8B |

### Reranker Models

| Model | Parameters | Base Model |
|-------|------------|------------|
| OTel_Reranker_0.6B | 0.6B | Qwen3-0.6B |
| OTel_Reranker_4B | 4B | Qwen3-4B |
| OTel_Reranker_8B | 8B | Qwen3-8B |

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
  "negative_1": "Hard negative passage 1",
  "negative_2": "Hard negative passage 2"
}
```

## 🌐 Infrastructure

- **Compute**: TensorWave for AMD GPUs and Azure for Nvidia GPUs
- **Framework**: ScalarLM (GPU-agnostic)

## 📬 Contact

If you have any technical questions, please feel free to reach out to farbod.tavakkoli@att.com or farbodtavakoli@gmail.com

Authors: [Farbod Tavakkoli](https://www.linkedin.com/in/farbodtavakkoli/),  [Gregory Diamos](https://www.linkedin.com/in/gregory-diamos-1a8b9083/), [Roderic Paulk](https://www.linkedin.com/in/roderic-paulk-64a30718/), [Jorden Terrazas](https://www.linkedin.com/in/jorden-terrazas-4a440714a/).
