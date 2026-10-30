# Open Telco (OTel) AI

Training and inference recipes, datasets, benchmarks, and open-weight models for telecom AI.

OTel research includes a [NeurIPS 2026 Spotlight paper](https://arxiv.org/pdf/2610.07766) in the Evaluations & Datasets Track and a Breakthrough Impact paper accepted to the [ACM AI Leadership Summit 2026](https://arxiv.org/abs/2608.15436). See [selected blog and media coverage](https://github.com/farbodtavakkoli/OTel/blob/main/docs/OTel-selected-coverage.md) of OTel.


<p align="center">
  <a href="https://arxiv.org/pdf/2610.07766">Paper (NeurIPS 2026 Spotlight)</a> |
  <a href="https://arxiv.org/abs/2608.15436">Paper (ACM AI Leadership Summit 2026 Breakthrough Impact)</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-20-llm">OTel-2.0-LLM</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-10-llm">OTel-1.0-LLM</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-10-embedding">OTel-1.0-Embedding</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-10-reranker">OTel-1.0-Reranker</a> |
  <a href="https://huggingface.co/farbodtavakkoli/datasets">Datasets</a>
</p>

Open Telco (OTel) AI is an open ecosystem of telecom datasets, benchmarks, models, and
reproducible training and inference recipes. Its mission is to give telecom operators the
freedom to choose their models, software, hardware, and infrastructure. With contributions
from across the industry, OTel provides a shared AI foundation that operators can build,
adapt, and deploy on their own terms.

OTel brings together specialized language, embedding, reranking, decision, classification,
and safety models with workflows for data curation, post-training, evaluation, benchmarking, and serving. The ecosystem supports multiple software stacks across hardware from AMD, NVIDIA, AWS, Apple, Tenstorrent, and Intel, enabling reproducible experimentation and deployment across platforms.

## What This Repository Provides

- **Multiple training recipes** spanning LLM post-training, embedding, reranking,
  and classification.
- **Multiple inference stacks** covering LLM generation, embeddings, and reranking,
  with workload support varying by stack.
- **SFT, CPT, DPO, GRPO, PPO, OSFT, LoRA, QLoRA, and DoRA** examples.
- Recipe documentation, dependency requirements, runnable entry points, and
  sample workflows with installation and verification instructions.
- Training and inference support on multiple software stacks across hardware from **AMD, NVIDIA, AWS, Apple, Tenstorrent, and Intel.**
- Telecom datasets, model releases, and reported evaluation results documented
  in the repository and accompanying
  [Hugging Face model and dataset cards](https://huggingface.co/farbodtavakkoli).
  

For the shared GRPO benchmark harness, see the [GRPO methodology guide](docs/grpo_method.md).

This is a recipe collection, not one monolithic framework. Environments are intentionally
isolated because packages such as Unsloth, PyLate, DeepSpeed, vLLM, and SGLang can require
conflicting dependency versions.

## Environment Setup

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
| Judge based GRPO across stacks | `training/llm/grpo_common`, then `training/llm/verl/grpo` or `training/llm/scalarlm` |
| Standard bi-encoder training | `training/embedding/sentence_transformers` |
| Cross-encoder reranking | `training/reranker/sentence_transformers` |
| Multi-class telecom classification | `training/classification/deepspeed` |
| Scalable cross-vendor training deployment | `training/llm/scalarlm` |
| Production-oriented serving | `inference/vllm` |
| Correctness baseline | `inference/transformers` |
| Local or GGUF deployment | `inference/llamacpp` or `inference/ollama` |
| Local API with reranking | `inference/lemonade` |
| Apple silicon (Mac) inference | `inference/mlx` |
| Tenstorrent Blackhole Galaxy serving | `inference/tenstorrent` |

## Repository Map

```text
.
|-- training/
|   |-- llm/                  # Framework-specific LLM recipes
|   |   |-- <framework>/grpo/ # GRPO benchmark stack for that framework
|   |   `-- grpo_common/      # shared judge, reward, data and ROCm base image for GRPO
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
|   |-- tensorrtllm/          # NVIDIA LLM serving
|   |-- mlx/                  # Apple silicon (Metal) LLM, embedding, reranker
|   `-- tenstorrent/          # Tenstorrent tt-inference-server LLM serving
`-- docs/                     # OTel paper, hardware platform notes (MI355X and H100), project coverage
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
| MLX | Yes | Yes | Yes | Apple silicon: served LLM, in-process embedding and reranking |
| Tenstorrent | Yes | No | No | Tenstorrent hardware through tt-inference-server (vLLM on TT-Metalium) |

## Hardware Support

| Platform | Status |
|---|---|
| **AMD Instinct MI355X** (ROCm 7.2) | Training and inference. TensorRT-LLM is NVIDIA-only; the KTransformers ROCm kernel path works but its serving layer is blocked by CUDA-pinned dependencies. |
| **NVIDIA H100** (CUDA 13) | Training and inference. `training/llm/primus` is AMD-only by design. |
| **AWS Trainium2** | Documented training and inference path through ScalarLM using Neuron PyTorch and the vLLM Neuron plugin. Requires a locally built image; runtime adapter loading and GRPO are not supported on this path. See the [Trainium2 guide](training/llm/scalarlm/readme_scalarlm.md#trainium2). |
| **Apple silicon** | Inference through `inference/mlx`. Training paths via MPS/CPU are selected. Confirm support in the recipe README first. |
| **Intel hardware** | Selected paths via XPU, SYCL, AMX, and CPU backends — support varies by framework and device. |
| **Tenstorrent Blackhole Galaxy** | LLM inference through `inference/tenstorrent`. Training is not covered. |

Not every framework supports every platform. Confirm the exact device and framework
combination in the recipe README.

### Scalable training deployment with ScalarLM

ScalarLM provides scalable distributed training across AMD and NVIDIA GPU infrastructure,
with an additional documented training and inference path for AWS Trainium2. See the
[ScalarLM recipe](training/llm/scalarlm/readme_scalarlm.md) for platform requirements and limitations.

ScalarLM ships the training code (`ml/`) from the client with each job, so the client
checkout and server image must come from the same revision. Image versions, deployment
commands, and acceptance gates are maintained in the
[MI355X runbook](training/llm/scalarlm/docs/DOCKER_IMAGE_MI355.md),
[H100 runbook](training/llm/scalarlm/docs/DOCKER_IMAGE_H100.md), and
[Trainium2 guide](training/llm/scalarlm/readme_scalarlm.md#trainium2).

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

Explore the model releases in the
[OTel-2.0-LLM](https://huggingface.co/collections/farbodtavakkoli/otel-20-llm),
[OTel-1.0-LLM](https://huggingface.co/collections/farbodtavakkoli/otel-10-llm),
[OTel-1.0-Embedding](https://huggingface.co/collections/farbodtavakkoli/otel-10-embedding), and
[OTel-1.0-Reranker](https://huggingface.co/collections/farbodtavakkoli/otel-10-reranker) collections.
Representative releases include:

| Model | Role | Public result or status |
|---|---|---:|
| [OTel-2.0-LLM-31B-IT](https://huggingface.co/farbodtavakkoli/OTel-2.0-LLM-31B-IT) | Large-scale domain-adapted telecom LLM | Comprehensive public evaluation forthcoming |
| [OTel-LLM-E4B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-E4B-IT) | Context-grounded generation (post-paper) | 91.7% +/- 0.4 correctness |
| [OTel-LLM-8B-A1B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-8B-A1B-IT) | Mid-size context-grounded generation (post-paper) | 88.4% +/- 0.5 correctness |
| [OTel-LLM-27B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-27B-IT) | Highest LLM score reported in the paper | 87.8% +/- 0.5 correctness |
| [OTel-Embedding-300M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-300M) | Efficient dense retrieval | 90.4% +/- 0.6 NDCG@10 |
| [OTel-Embedding-8B](https://huggingface.co/farbodtavakkoli/OTel-Embedding-8B) | Highest reported OTel retrieval score | 93.1% +/- 0.4 NDCG@10 |
| [OTel-Reranker-0.6B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-0.6B) | Efficient cross-encoder reranking | 0.938 +/- 0.007 MRR@10 |
| [OTel-Reranker-8B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-8B) | Highest reported OTel reranking score | 0.947 +/- 0.005 MRR@10 |

Numeric results are for OTel 1.0 models on held-out OTel evaluation partitions, as reported
in the [OTel paper](https://arxiv.org/pdf/2610.07766); standard errors use 1,000 bootstrap
resamples. `OTel-LLM-E4B-IT` and `OTel-LLM-8B-A1B-IT` were released after the paper and use
the same protocol, but are not part of the paper's results. LLM correctness is the average
of two judges (GPT-4o mini and Claude Sonnet 3.5) on answers generated from retrieved
context, not context-free telecom expertise, and is not a substitute for independent
evaluation on your deployment domain.

## Using the Models

### Smoke Test an OTel 2.0 Deployment

After starting an OpenAI compatible server, use its configured model identifier in the
request below. This checks that the endpoint responds, not which checkpoint it loaded.

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

Model self identification is not reliable checkpoint verification. Confirm the loaded
checkpoint path and pinned revision in the server configuration and startup logs, and
check that the served model identifier matches your request. A response naming OTel or
Gemma alone does not establish which weights are loaded.

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

## Technical Documentation

| Document | Contents |
|---|---|
| [GRPO methodology](docs/grpo_method.md), [findings](docs/grpo_findings.md), and [porting](docs/grpo_porting.md) | Shared benchmark harness, stack comparison results, and ROCm to CUDA porting |
| [MI355X training notes](docs/mi355x_training_notes.md) | ROCm training lessons, scaling, and failure modes |
| [MI355X inference notes](docs/mi355x_inference_notes.md) | ROCm serving lessons and FP8 findings |
| [H100 training notes](docs/h100_training_notes.md) | CUDA training lessons and porting from ROCm |
| [H100 inference notes](docs/h100_inference_notes.md) | CUDA serving lessons and TensorRT-LLM/SGLang findings |
| [ScalarLM Trainium2 guide](training/llm/scalarlm/readme_scalarlm.md#trainium2) | Neuron environment, local image build, deployment, and limitations |
| Each recipe README (`readme_<framework>.md` under `training/`, `README.md` under `inference/`) | Exact installation, smoke and full runs, arguments, outputs, and platform status |

## Responsible Use and Limitations

The following capability and data limitations apply to the OTel 2.0 LLM unless otherwise
stated. Consult individual model cards for limitations specific to each release.

**Scope.** OTel 2.0 is a telecom-specific generative model, not a general-purpose one. It was not trained or evaluated as an embedding, retrieval, or reranking model — use the
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

**Operations.** The model has not been validated against Methods of Procedure on live or digital-twin devices, closed-loop operational tasks (network turn-up, SLA/QoS
configuration, routing-fault repair, incident closure), or defensive network-security
work. High-impact or agentic use requires verified tools, scoped permissions, audit
logging, and human review appropriate to the risk.

**Hardware and reproducibility.** AMD MI355X and NVIDIA H100 are the most extensively
verified platforms; Apple and Intel paths received more limited testing. Not every recipe supports every platform — confirm the combination in the recipe README. Pin a model revision or release tag for reproducible evaluation.

Telecom standards change. Verify generated content against the relevant source and
release before any operational, customer-facing, regulatory, safety, security, or
network-configuration use.


## License

OTel code and derived datasets are released under the repository's Apache-2.0 license.
Model checkpoints inherit terms from their upstream base models in addition to the OTel
release terms. Review the repository license and the applicable model card before use or
redistribution.

## Citation

If you use OTel datasets, models, benchmarks, or recipes, please cite:

```bibtex
@inproceedings{tavakkoli2026otel,
  title     = {OTel: Open Telco AI Datasets, Benchmarks, and Models},
  author    = {Tavakkoli, Farbod and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://github.com/farbodtavakkoli/OTel}
}
```

## Collaboration

GSMA and Pleias contributed to the open telecom corpus; Red Hat supported synthetic-data
generation and OSFT; Microsoft supplied managed compute; AMD supplied accelerators and
ROCm; Dell Technologies supplied on-premises training infrastructure; and MLCommons,
academic, and research partners contributed evaluation and domain expertise.

Organizational and independent coverage of the project is collected in
[`docs/OTel-2.0-coverage.md`](docs/OTel-2.0-coverage.md) and
[`docs/OTel-1.0-coverage.md`](docs/OTel-1.0-coverage.md).

## Contact

For questions and project updates, visit
[Farbod Tavakkoli on GitHub](https://github.com/farbodtavakkoli) or open an issue in this
repository or contact farbod.tavakkoli@att.com or farbodtavakoli@gmail.com.
