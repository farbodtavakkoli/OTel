# ScalarLM — Open Source Training & Inference Stack

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ScalarLM is an open-source platform for **closed-loop LLM experimentation** — running a model and post-training it against live feedback in the same deployment, across GPU hardware from any vendor, at scales from a single workstation to multi-node Kubernetes clusters.

It is **CC-0 licensed**. You can fork it, publish with it, build on it, and ship it without restriction or attribution.

Maintained and sponsored by [TensorWave](https://tensorwave.com) and [RelationalAI](https://www.relational.ai)

---

## Why ScalarLM

Most training and inference infrastructure is designed for one direction of work: either serving a fixed model, or producing a new one. Closing the loop — using a live model's outputs to drive the next round of post-training — typically requires stitching together separate systems with different APIs, checkpoint formats, and scheduling assumptions.

ScalarLM is built around that loop as the primary use case. A single deployment exposes both an inference endpoint and a training endpoint. You can query the running model, construct training signal from the results, and submit a post-training job against the same deployment without touching infrastructure. The updated checkpoint is picked up automatically at the next inference request.

This makes it well-suited for research in online learning, RLHF pipelines, self-improvement, and any setting where the boundary between inference and training needs to be thin.

---

## Architecture

ScalarLM composes three production-grade components, each responsible for a distinct part of the stack:

- **vLLM** handles live inference with PagedAttention, continuous batching, and high token throughput. Each deployment exposes an OpenAI-compatible endpoint backed by vLLM.
- **Megatron-LM** handles distributed training, providing tensor and pipeline parallelism for scaling across multiple GPUs and nodes. Training jobs are dispatched via a Slurm scheduler running inside the Kubernetes deployment.
- **Hugging Face Hub** is the model source and, optionally, the model sink. Any Hub-hosted model can be deployed; post-training checkpoints can be pushed back to the Hub automatically at the end of a training run.

The seam between them is a shared checkpoint system. vLLM loads from a checkpoint; Megatron writes to one. The inference pod does not need to restart for the update to take effect.

See the [Architecture page](https://www.scalarlm.com/architecture/) for a full diagram.

---

## GPU Agnostic

ScalarLM runs on **NVIDIA and AMD GPUs** without code changes. The training stack is built on PyTorch, and the inference stack inherits vLLM's hardware support. Deployments at TensorWave run on AMD MI300X hardware; the same Helm charts and `ml/` directory work on NVIDIA A100 and H100 clusters.

---

## 🚀 Quick Start

### 1. Clone and start

```bash
git clone git@github.com:tensorwavecloud/scalarlm.git
cd scalarlm
./scalarlm up
```

This brings up the ScalarLM development server on `localhost:8000` with an OpenAI-compatible API.

### 2. Send your first request

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

### 3. Use the Python client

```python
import scalarlm

scalarlm.api_url = "http://localhost:8000"

llm = scalarlm.SupermassiveIntelligence()

dataset = ["What is 1 + 1?", "What is 2 + 2?"]
results = llm.generate(prompts=dataset)
print(results)
```

### 4. Load a different model

Edit `values.yaml` and set the `model` field, then restart:

```bash
./scalarlm up
```

---

## CLI Commands

```
./scalarlm up              # Start ScalarLM server
./scalarlm benchmark       # Run performance benchmarks
./scalarlm llm-logs        # View LLM logs
./scalarlm llm-ls          # List available models
./scalarlm llm-plot        # Plot training metrics
./scalarlm llm-squeue      # View training queue status
./scalarlm test            # Run tests
./scalarlm build-image     # Build Docker image
```

Install the pip CLI for monitoring:

```bash
pip install scalarlm
export SCALARLM_API_URL="http://<your-deployment-ip>"
scalarlm [-h] {logs,plot,ls,squeue} ...
```

---

## Supported Models

ScalarLM deploys any Hugging Face-hosted model. The following have been validated and are ready to use by setting the `model` field in `values.yaml`:

| Model | Parameters | Architecture | Context | License | Notes |
|---|---|---|---|---|---|
| `google/gemma-3-4b-it` | 4B | Dense | 128K | Gemma ToU | Default deployment; production-tested |
| `google/gemma-3-27b-it` | 27B | Dense | 128K | Gemma ToU | |
| `Qwen/Qwen2-32B-Instruct` | 32B | Dense | 128K | Apache 2.0 | Production-tested |
| `Qwen/Qwen3.5-35B-A3B` | 35B total / 3B active | Hybrid MoE + Gated Delta Networks | 256K (1M via YaRN) | Apache 2.0 | Native thinking mode; multimodal |
| `Qwen/Qwen3.5-122B-A10B` | 122B total / 10B active | Hybrid MoE + Gated Delta Networks | 1M | Apache 2.0 | Requires multi-GPU |
| `openai/gpt-oss-120b` | 117B total / 5.1B active | MoE | 131K | Apache 2.0 | Fits single 80GB GPU; verified on AMD |
| `openai/gpt-oss-20b` | 21B total / 3.6B active | MoE | 131K | Apache 2.0 | Fits 16GB GPU; strong tool use and reasoning |
| `nvidia/Nemotron-3-Super-120B` | 120B total / 12B active | Hybrid Mamba-Transformer MoE | 1M | NVIDIA Open | Optimized for multi-agent workloads |
| `EssentialAI/rnj-1-instruct` | 8B | Dense (Gemma 3 variant) | 32K | Apache 2.0 | Strong agentic coding and STEM; designed for post-training |

**MoE models** (Qwen3.5, gpt-oss, Nemotron 3 Super) activate only a fraction of total parameters per token — they run significantly faster and cheaper than their total parameter count implies. ScalarLM's vLLM backend supports MoE natively on both NVIDIA and AMD hardware.

**Multi-GPU models** above 30B active parameters typically require sharding. Set `inference_gpus` in `values.yaml` and enable sharding in the vLLM Helm chart. See the [Kubernetes Deployment](https://www.scalarlm.com/kubernetes/) page for details.

---

## Docker Support

### Prebuilt Containers

| Target | Container | Latest Release |
|---|---|---|
| NVIDIA BLACKWELL | `gdiamos/scalarlm-nvidia-12.0:latest` | `gdiamos/scalarlm-nvidia-12.0:v1.0` |
| NVIDIA HOPPER | `gdiamos/scalarlm-nvidia-8.0:latest` | `gdiamos/scalarlm-nvidia-8.0:v1.0` |
| NVIDIA HOPPER | `gdiamos/scalarlm-nvidia-8.6:latest` | `gdiamos/scalarlm-nvidia-8.6:v1.0` |
| NVIDIA ADA | `gdiamos/scalarlm-nvidia-7.5:latest` | `gdiamos/scalarlm-nvidia-7.5:1.0` |
| ARM | `gdiamos/scalarlm-arm:latest` | `gdiamos/scalarlm-arm:v1.0` |
| AMD MI300 | `gdiamos/scalarlm-amd-mi300:latest` | `gdiamos/scalarlm-amd-mi300:v1.0` |
| x86 CPU | `gdiamos/scalarlm-cpu:latest` | `gdiamos/scalarlm-cpu:v1.0` |

### Quick Docker Start

```bash
./scalarlm up cpu        # CPU version
./scalarlm up nvidia     # NVIDIA GPU version
./scalarlm up amd        # AMD GPU version
```

---

## Designed for Experimentation

The training pipeline runs from a local `ml/` directory that is packaged and shipped to the cluster automatically with each job submission. You can modify the training loop, optimizer, loss function, or data loader locally — with your normal editor and version control — and the cluster picks it up without a Docker rebuild or a redeployment.

Current production deployments include **Gemma 3 4B Instruct**, **Gemma 3 Embedding 300M**, and **Qwen2 32B Instruct**, running on multi-GPU Kubernetes clusters with live inference endpoints.

---

## Project Structure

```
scalarlm/
├── .github/workflows/   # CI workflows
├── cmd/                 # CLI entry points
├── deployment/          # Helm charts and deployment configs
├── docs/                # Documentation source
├── frontend/            # Frontend assets
├── infra/               # ScalarLM infrastructure layer
├── ml/                  # Training loop, optimizer, dataset loading
├── scripts/             # Utility scripts
├── sdk/                 # Python SDK (pip install scalarlm)
├── test/                # Unit and integration tests
├── Faq.md               # Frequently asked questions
└── README.md            # This file
```

---

## What It Is Not

ScalarLM is not a managed service or a training-as-a-service product. It is infrastructure you deploy and own. There is no scheduler-as-a-service, no auto-scaling, and no hosted model registry beyond what Hugging Face provides. If you want a fully managed experience, this is probably not the right tool. If you want to understand and control every layer of the stack, it is.

---

## Get Started

| I want to… | Start here |
|---|---|
| Understand the full system design | [Architecture](https://www.scalarlm.com/architecture/) |
| Run my first inference or training job | [Quick Start](https://www.scalarlm.com/quick-start/) |
| Customize the training loop | [Custom Training](https://www.scalarlm.com/training/) |
| Deploy to my own Kubernetes cluster | [Kubernetes Deployment](https://www.scalarlm.com/kubernetes/) |
| Browse the full docs | [scalarlm.com](https://www.scalarlm.com) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`./scalarlm test`)
5. Commit and push, then open a Pull Request

---

## 📄 License

ScalarLM is licensed under **CC-0** (public domain). See [LICENSE](LICENSE) for details.

---

## Acknowledgments

ScalarLM is inspired by the work of **Seymour Roger Cray** (1925–1996), "the father of supercomputing."

Built with:
- [vLLM](https://github.com/vllm-project/vllm) — high-performance LLM inference
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — large-scale distributed training
- [Hugging Face Hub](https://huggingface.co/) — model registry and transformers
- [PyTorch](https://pytorch.org/) — deep learning framework
