# Open Telco (OTel) AI

Training and inference recipes, datasets, benchmarks, and open-weight models for telecom AI.

<p align="center">
  <a href="https://github.com/farbodtavakkoli/OTel">Code</a> |
  <a href="https://huggingface.co/farbodtavakkoli">Hugging Face</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-llm">LLMs</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-embedding">Embeddings</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-reranker">Rerankers</a> |
  <a href="https://huggingface.co/farbodtavakkoli/datasets">Datasets</a>
</p>

Open Telco (OTel) AI is an open foundation for telecom-specialized AI systems, combining
domain data and open-weight model releases with reproducible recipes for post-training,
retrieval, reranking, classification, and serving.

Telecom knowledge is precise, versioned, and spread across standards, specifications, and
RFCs. A model can sound fluent while misunderstanding an interface or answering from
unsupported context. OTel addresses that with specialized data, open checkpoints, and
runnable infrastructure recipes for AMD, NVIDIA, Apple, and Intel hardware.

## What This Repository Provides

- **27 self-contained training recipes** spanning LLM post-training, embedding,
  reranking, and classification.
- **9 inference stacks** covering LLM generation, embeddings, and reranking.
- **SFT, CPT, DPO, GRPO, PPO, OSFT, LoRA, QLoRA, and DoRA** examples.
- A local sample, pinned requirements, runnable entry point, and detailed README in
  every recipe folder.
- Cross-platform paths for **AMD, NVIDIA, Apple, and Intel**.
- Dataset cards, model cards, and benchmark results published alongside the
  [Hugging Face releases](https://huggingface.co/farbodtavakkoli).

This is a recipe collection, not one monolithic framework. Environments are intentionally
isolated because packages such as Unsloth, PyLate, DeepSpeed, vLLM, and SGLang can require
conflicting dependency versions.

## Quick Start

Create `dev.env` at the repository root with your `HF_TOKEN` (needed for gated models and
datasets). Then pick a recipe, build its environment, and run the smoke test from its README:

```bash
cd training/llm/deepspeed
ln -sf ../../../dev.env dev.env
python3 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install -r requirements_*.txt
```

Two environment conventions:

- **Training** — one venv and one `requirements_<framework>.txt` per recipe folder.
- **Inference** — one venv and one `requirements.txt` per stack root (for example
  `inference/vllm/.env_vllm`), shared by the workloads beneath it.

On AMD, install the ROCm PyTorch wheel documented in the recipe README **before** the rest
of the stack. Always use the exact install and launch commands from the folder you chose.

### Recommended starting points

| Goal | Start here |
|---|---|
| Chat SFT, DPO, or GRPO | `training/llm/deepspeed` |
| Standard bi-encoder training | `training/embedding/sentence_transformers` |
| Cross-encoder reranking | `training/reranker/sentence_transformers` |
| Multi-class telecom classification | `training/classification/deepspeed` |
| Scalable cross-vendor training deployment | `training/llm/scalarlm` |
| Production-oriented serving | `inference/vllm` |
| Correctness baseline | `inference/transformers` |
| Local or GGUF deployment | `inference/llamacpp` or `inference/ollama` |
| Local API with reranking | `inference/lemonade` |

## Repository Map

```text
.
|-- training/
|   |-- llm/                  # 21 framework-specific LLM recipes
|   |-- embedding/            # Sentence Transformers, Tevatron, PyLate
|   |-- reranker/             # Sentence Transformers, FlagEmbedding
|   `-- classification/       # DeepSpeed sequence classification
|-- inference/
|   |-- vllm/                 # LLM, embedding, reranker
|   |-- sglang/               # LLM, embedding, reranker
|   |-- transformers/         # Correctness baselines for all three workloads
|   |-- llamacpp/             # GGUF serving
|   |-- lemonade/             # Unified local API
|   |-- ollama/               # LLM and embedding
|   |-- tei/                  # Embedding
|   |-- ktransformers/        # CPU-GPU hybrid MoE serving
|   `-- tensorrtllm/          # NVIDIA LLM serving
`-- docs/                     # Hardware platform notes (MI355X and H100) plus project coverage
```

Training is organized by modality because each modality/framework pair has its own
environment. Inference is organized by serving stack because one installation supports
the workloads beneath it.

## Training Frameworks

| Workload | Available recipes |
|---|---|
| LLM | DeepSpeed, standalone DeepSpeed, Unsloth, PEFT, FSDP2, Lightning, Composer, LLM Foundry, Axolotl, LLaMA-Factory, RapidFire AI, Red Hat OSFT, Ray, verl, OpenRLHF, Megatron-LM, TorchTitan, TorchTune, NVIDIA NeMo, AMD Primus, ScalarLM |
| Embedding | Sentence Transformers, Tevatron, PyLate |
| Reranking | Sentence Transformers, FlagEmbedding |
| Classification | DeepSpeed |

## Inference Stacks

| Stack | LLM | Embedding | Reranker | Best fit |
|---|:---:|:---:|:---:|---|
| vLLM | Yes | Yes | Yes | Default high-throughput server |
| SGLang | Yes | Yes | Yes | Structured and high-performance serving |
| Transformers | Yes | Yes | Yes | Reference implementation and debugging |
| llama.cpp | Yes | Yes | Yes | GGUF and portable local inference |
| Lemonade | Yes | Yes | Yes | One local API across workloads |
| Ollama | Yes | Yes | No | Simple local model workflows |
| TEI | No | Yes | Limited | Dedicated embedding service |
| KTransformers | MoE | No | No | CPU-GPU expert offload on supported systems |
| TensorRT-LLM | Yes | No | No | NVIDIA-specific optimized serving |

## Hardware Support

| Platform | Status |
|---|---|
| **AMD Instinct MI355X** (ROCm 7.2) | Training and inference. TensorRT-LLM is NVIDIA-only; the KTransformers ROCm kernel path works but its serving layer is blocked by CUDA-pinned dependencies. |
| **NVIDIA H100** (CUDA 13) | Training and inference. `training/llm/primus` is AMD-only by design. |
| **Apple silicon** | Selected paths via MPS, Metal, MLX, and CPU backends — confirm support in the recipe README first. |
| **Intel hardware** | Selected paths via XPU, SYCL, AMX, and CPU backends — support varies by framework and device. |

Not every framework supports every platform. Confirm the exact device and framework
combination in the recipe README.

### Scalable training deployment with ScalarLM

For scalable distributed training across AMD and NVIDIA GPU infrastructure, ScalarLM is
the recommended cross-vendor deployment option. The repository includes a remote
ScalarLM client recipe under `training/llm/scalarlm`.

The latest available project images are:

```bash
# AMD Instinct MI355X
docker pull farbodatdocker/scalarlm:mi355-v1.7

# NVIDIA H100
docker pull farbodatdocker/scalarlm:h100-v1.6
```

ScalarLM ships the training code (`ml/`) from the **client** with each job, so the client
checkout and the server image must come from the same revision. Each image's runbook,
acceptance gates and source-revision label are in
[`training/llm/scalarlm/docs/DOCKER_IMAGE_MI355.md`](training/llm/scalarlm/docs/DOCKER_IMAGE_MI355.md)
and [`training/llm/scalarlm/docs/DOCKER_IMAGE_H100.md`](training/llm/scalarlm/docs/DOCKER_IMAGE_H100.md).

## OTel Data

Datasets are derived from public telecom material covering 3GPP, GSMA, O-RAN, IETF RFCs,
academic papers, and industry white papers. They contain derived examples, not copies of
the raw source documents.

| Dataset | Purpose | Core fields |
|---|---|---|
| [OTel-LLM](https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM) | Context-grounded instruction tuning | `prompt`, `completion`, abstention and chunk metadata |
| [OTel-Embedding](https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding) | Bi-encoder retrieval with hard negatives | `anchor`, `positive`, `negative_1` ... `negative_5` |
| [OTel-Reranker](https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker) | Cross-encoder reranking | `sentence_0`, `sentence_1`, `label` |
| [OTel-Safety](https://huggingface.co/datasets/farbodtavakkoli/OTel-Safety) | Abstention when context is insufficient | `prompt`, `completion`, abstention and chunk metadata |

OTel 1.0 filtered roughly 1.1 million raw examples down to 326,767. OTel 2.0 was
post-trained on approximately 440 billion tokens drawn from 3GPP, ETSI, GSMA, CAMARA, ITU,
O-RAN, and TM Forum material combined with AT&T and collaborator data.

## Models and Reported Results

The complete model roster is available in the
[LLM](https://huggingface.co/collections/farbodtavakkoli/otel-llm),
[embedding](https://huggingface.co/collections/farbodtavakkoli/otel-embedding), and
[reranker](https://huggingface.co/collections/farbodtavakkoli/otel-reranker) collections.
Representative releases include:

| Model | Role | Public result or status |
|---|---|---:|
| [OTel 2.0 LLM 31B IT](https://huggingface.co/farbodtavakkoli/OTel-2.0-LLM-31B-IT) | Large-scale domain-adapted telecom LLM | Comprehensive public evaluation forthcoming |
| [OTel-LLM-E4B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-E4B-IT) | Context-grounded generation | 91.7% +/- 0.4 correctness |
| [OTel-LLM-8B-A1B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-8B-A1B-IT) | Mid-size context-grounded generation | 88.4% +/- 0.5 correctness |
| [OTel-Embedding-300M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-300M) | Efficient dense retrieval | 90.9% +/- 0.5 NDCG@10 |
| [OTel-Embedding-8B](https://huggingface.co/farbodtavakkoli/OTel-Embedding-8B) | Highest reported OTel retrieval score | 93.5% +/- 0.3 NDCG@10 |
| [OTel-Reranker-0.6B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-0.6B) | Efficient cross-encoder reranking | 0.944 +/- 0.006 MRR@10 |
| [OTel-Reranker-8B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-8B) | Highest reported OTel reranking score | 0.952 +/- 0.004 MRR@10 |

Numeric results are for OTel 1.0 models on held-out OTel evaluation partitions. LLM
correctness measures answers generated from retrieved context, not context-free telecom
expertise, and is not a substitute for independent evaluation on your deployment domain.

> [!NOTE]
> OTel 2.0 training code and a comprehensive public evaluation are forthcoming, the
> latter as part of **MLPeFT** in collaboration with **MLCommons**.

## Using the Models

### Verify an OTel 2.0 deployment

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="OTel-2.0-LLM-31B-IT",
    messages=[{"role": "user", "content": "What model are you?"}],
    temperature=0,
    max_tokens=128,
)

print(response.choices[0].message.content)
```

The response should identify OTel 2.0 as a model trained by AT&T Chief Data Office. If it
identifies only as Gemma or Google DeepMind, you are serving the base model or a stale
mount rather than the OTel 2.0 checkpoint.

### Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("farbodtavakkoli/OTel-Embedding-300M")
sentences = [
    "What is the F1 interface in O-RAN?",
    "The F1 interface connects the O-DU to the O-CU.",
]
embeddings = model.encode(sentences, normalize_embeddings=True)
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/mi355x_training_notes.md`](docs/mi355x_training_notes.md) | ROCm training lessons, scaling, and failure modes |
| [`docs/mi355x_inference_notes.md`](docs/mi355x_inference_notes.md) | ROCm serving lessons and FP8 findings |
| [`docs/h100_training_notes.md`](docs/h100_training_notes.md) | CUDA training lessons and the ROCm→CUDA reversals |
| [`docs/h100_inference_notes.md`](docs/h100_inference_notes.md) | CUDA serving lessons and TensorRT-LLM/SGLang findings |
| [`docs/OTel-2.0-blogs.md`](docs/OTel-2.0-blogs.md) · [`docs/OTel-1.0-media-coverage.md`](docs/OTel-1.0-media-coverage.md) | Organizational and independent coverage of the project |
| Each recipe README (`readme_<framework>.md` under `training/`, `README.md` under `inference/`) | Exact installation, smoke and full runs, arguments, outputs, and platform status |

## Responsible Use and Limitations

**Scope.** OTel 2.0 is a telecom-specific generative model, not a general-purpose one. It
was not trained or evaluated as an embedding, retrieval, or reranking model — use the
purpose-built OTel embedding and reranker collections for those stages. Comprehensive
public capability evaluation is forthcoming; current functional checks are not broad
quality validation. Direct Q&A and RAG behavior require separate evaluation.

**Language and modality.** English only. The architecture is multimodal, but OTel
post-training and all published quality claims are text-only; vision components are
inherited unchanged from Gemma 4. Audio and video are not supported.

**Data coverage.** The training mixture contains no dedicated collections of private
operator event records, network KPIs or RF/spectrum measurements, 5G core control-plane
signaling, vendor CLI and network-OS documentation (Cisco IOS-XR, Juniper JUNOS, Nokia
BNG, Arista EOS), or operator-private designs, OSS/BSS, and change-management data.
Standards familiarity is not experience with live telemetry or vendor-specific behavior.

**Operations.** The model has not been validated against Methods of Procedure on live or
digital-twin devices, closed-loop operational tasks (network turn-up, SLA/QoS
configuration, routing-fault repair, incident closure), or defensive network-security
work. High-impact or agentic use requires verified tools, scoped permissions, audit
logging, and human review appropriate to the risk.

**Hardware and reproducibility.** AMD MI355X and NVIDIA H100 are the most extensively
verified platforms; Apple and Intel paths received more limited testing. Not every recipe
supports every platform — confirm the combination in the recipe README. Pin a model
revision or release tag for reproducible evaluation.

Telecom standards change. Verify generated content against the relevant source and
release before any operational, customer-facing, regulatory, safety, security, or
network-configuration use.

## Future Work

- Release the OTel 2.0 training implementation and reproducible configuration.
- Release the comprehensive OTel 2.0 evaluation through MLPeFT in collaboration with
  MLCommons.
- Add and verify training and inference support on AWS infrastructure and Tenstorrent
  hardware.
- Compare and benchmark every compatible training stack for GRPO and multi-node
  training under a common methodology.
- Benchmark inference throughput, latency, memory use, and efficiency on AMD Ryzen
  systems.

## License

OTel code and derived datasets are released under the repository's Apache-2.0 license.
Model checkpoints inherit terms from their upstream base models in addition to the OTel
release terms. Review the repository license and the applicable model card before use or
redistribution.

## Citation

```bibtex
@misc{otel_models_2026,
  title  = {OTel: Open Telco AI Datasets, Benchmarks, Models, and Recipes},
  author = {Tavakkoli, Farbod and others},
  year   = {2026},
  note   = {Open Telco (OTel) release},
  url    = {https://github.com/farbodtavakkoli/OTel}
}
```

## Collaboration

GSMA and Pleias contributed to the open telecom corpus; Red Hat supported synthetic-data
generation and OSFT; Microsoft supplied managed compute; AMD supplied accelerators and
ROCm; Dell Technologies supplied on-premises training infrastructure; and MLCommons,
academic, and research partners contributed evaluation and domain expertise.

Organizational and independent coverage of the project is collected in
[`docs/OTel-2.0-blogs.md`](docs/OTel-2.0-blogs.md) and
[`docs/OTel-1.0-media-coverage.md`](docs/OTel-1.0-media-coverage.md).

## Contact

For questions and project updates, visit
[Farbod Tavakkoli on GitHub](https://github.com/farbodtavakkoli) or open an issue in this
repository or contact farbod.tavakkoli@att.com or farbodtavakoli@gmail.com.

