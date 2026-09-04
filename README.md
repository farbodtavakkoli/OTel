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

Open Telco (OTel) AI is an open foundation for building telecom-specialized AI systems. It combines
domain data and model releases with reproducible recipes for post-training, retrieval,
reranking, classification, evaluation, and serving.

This codebase provides compatible model training and inference paths for
AMD, NVIDIA, Apple, and Intel hardware. Support and verification depth vary by framework
and platform, as documented below and in each recipe README.

Support for model training and inference on AWS and Tenstorrent hardware will be added
soon.

## Why OTel

Telecom knowledge is precise, versioned, and distributed across standards,
specifications, RFCs, white papers, research, and institutional expertise. A model can
sound fluent while misunderstanding an interface, missing a condition, or answering
from unsupported context. OTel addresses that gap with specialized data, transparent
evaluation, open-weight checkpoints, and runnable infrastructure recipes.

OTel has evolved in three stages:

| Stage | Objective | What changed |
|---|---|---|
| **OTel 1.0** | Establish a shared telecom training and evaluation foundation | Released datasets and model families for retrieval, reranking, context-grounded generation, classification, and abstention |
| **OTel 2.0** | Train telecom knowledge more deeply and at much larger scale | Expanded to a 31B model, broader instruction and direct-Q&A data, and a hundreds-of-billions-of-tokens training mixture |
| **Toward OTel 2.5** | Make the family more current, capable, efficient, and deployable | Address the current limitations documented in the OTel 2.0 model card, including gaps in evaluation, data coverage, telecom-specific tools, operational validation, languages, modalities, and hardware support |

The progression is not simply *small model -> larger model*. It is **shared foundation
-> deeper domain intelligence -> continuously improving, deployable telecom AI**.

## What This Repository Provides

- **27 self-contained training recipes** spanning LLM post-training, embedding,
  reranking, and classification.
- **9 inference stacks** covering LLM generation, embeddings, and reranking.
- **SFT, CPT, DPO, GRPO, PPO, OSFT, LoRA, QLoRA, and DoRA** examples.
- A local sample, pinned requirements, runnable entry point, and detailed README in
  every recipe folder.
- Cross-platform paths for **AMD, NVIDIA, Apple, and Intel**, with extensive MI355X and
  H100 execution evidence and preliminary Apple and Intel coverage.
- Dataset cards, model cards, and benchmark results published alongside the
  [Hugging Face releases](https://huggingface.co/farbodtavakkoli).

This is a recipe collection, not one monolithic framework. Environments are intentionally
isolated because packages such as Unsloth, PyLate, DeepSpeed, vLLM, and SGLang can require
conflicting dependency versions.

## Quick Start

Choose a recipe, create its environment, and run the documented smoke test against the
included sample data:

```bash
# From the repository root, create dev.env and add HF_TOKEN when a gated
# model or dataset requires it.

cd training/llm/deepspeed
ln -sf ../../../dev.env dev.env
python3 -m venv .env_deepspeed
source .env_deepspeed/bin/activate
pip install -r requirements_*.txt
```

Each recipe folder ships exactly one `requirements_<framework>.txt`, which is what the
wildcard resolves to. Use the exact install and launch command in the selected folder's
README. For AMD, install the documented ROCm PyTorch wheel first. Inference environments live at the
stack root, such as `inference/vllm/.env_vllm`, and are shared by that stack's
workloads.

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
`-- docs/                     # Hardware campaign notes (MI355X and H100)
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

## Extensive Hardware Verification and Cross-Platform Support

Taken together, the repository's recipes can train models and run inference on AMD,
NVIDIA, Apple, and Intel hardware. Verification depth differs by platform. "Verified"
means a repository path was executed with recorded commands, environment details,
outputs, and a verdict, rather than being listed solely from an upstream compatibility
claim.

| Platform | Training and inference coverage | Verification status |
|---|---|---|
| **AMD Instinct MI355X** | Training and inference campaigns on 8 x MI355X with ROCm 7.2.4 | **Extensive.** TensorRT-LLM is NVIDIA-only. The KTransformers ROCm kernel path was tested, but its serving layer is blocked by CUDA-pinned dependencies. |
| **NVIDIA H100** | Every applicable training folder and inference stack was exercised on H100 | **Extensive.** Training used real 1-GPU smoke runs and measured 2-GPU sharding where applicable. Any 8-GPU figure is a projection unless a folder explicitly states otherwise. `training/llm/primus` is AMD-only by design. |
| **Apple silicon** | Selected training and inference paths through compatible MPS, Metal, MLX, and CPU backends | **Preliminary.** Apple coverage is not yet comparable to the full AMD and NVIDIA campaigns. Confirm support in the selected recipe before use. |
| **Intel hardware** | Selected training and inference paths through compatible XPU, SYCL, AMX, and CPU backends | **Preliminary.** Intel coverage is not yet comparable to the full AMD and NVIDIA campaigns. Support varies by framework and device. |

Not every framework supports every platform. Hardware mentioned only from upstream
documentation remains labeled as unverified in the per-folder README.

### Scalable training deployment with ScalarLM

For scalable distributed training across AMD and NVIDIA GPU infrastructure, ScalarLM is
the recommended cross-vendor deployment option. The repository includes a remote
ScalarLM client recipe under `training/llm/scalarlm`.

The latest available project images are:

```bash
# AMD Instinct MI355X
docker pull farbodatdocker/scalarlm:mi355-v1.6

# NVIDIA H100
docker pull farbodatdocker/scalarlm:h100-v1.5
```

The ScalarLM client path has been checked on MI355X and H100. Client-side verification
does not by itself establish a complete server deployment or performance result. Consult
the ScalarLM recipe and image notes for the current server-side status.

Note that ScalarLM ships the training code (`ml/`) from the **client** with each job, so the
client checkout and the server image must come from the same revision. The MI355X image
runbook, acceptance gates and source-revision label are in
[`training/llm/scalarlm/DOCKER_IMAGE_MI355.md`](training/llm/scalarlm/DOCKER_IMAGE_MI355.md).

## OTel Data

The OTel 1.0 source corpus draws on public telecom material and contributor-provided
examples covering 3GPP, GSMA, O-RAN, IETF RFCs, academic papers, industry white papers,
Wikipedia, and web-derived telecom content. Released datasets contain derived examples
rather than copies of the raw source documents.

| Dataset | Purpose | Core fields |
|---|---|---|
| [OTel-LLM](https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM) | Context-grounded instruction tuning | `prompt`, `completion`, abstention and chunk metadata |
| [OTel-Embedding](https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding) | Bi-encoder retrieval with hard negatives | `anchor`, `positive`, `negative_1` ... `negative_5` |
| [OTel-Reranker](https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker) | Cross-encoder reranking | `sentence_0`, `sentence_1`, `label` |
| [OTel-Safety](https://huggingface.co/datasets/farbodtavakkoli/OTel-Safety) | Abstention when context is insufficient | `prompt`, `completion`, abstention and chunk metadata |

The OTel 1.0 pipeline reduced roughly 1.1 million raw examples to 326,767
higher-confidence examples through heuristic and semantic filtering, reranking,
embedding comparisons, and deduplication.

For OTel 2.0, GSMA provided an initial corpus of approximately 15 billion raw tokens
assembled from 3GPP, ETSI, GSMA, CAMARA, ITU, O-RAN, and TM Forum material. AT&T combined
that corpus with additional AT&T and collaborator data, processed more than 1 trillion
tokens, and post-trained OTel 2.0 on more than 400 billion tokens. The current model card
reports approximately 440 billion training tokens. See the organization reports below
for the attributed data, compute, and infrastructure claims.

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

The numeric results above are for OTel 1.0 models on held-out OTel evaluation partitions.
LLM correctness measures answers generated from retrieved context and must not be
interpreted as unrestricted, context-free telecom expertise. The primary results are not
a substitute for independent evaluation on the intended deployment domain.

> [!NOTE]
> OTel 2.0 training code will be released soon. A comprehensive public OTel 2.0
> evaluation release is also forthcoming as part of **MLPeFT**, in collaboration with
> **MLCommons**. The targeted checks currently described in the model card are not a
> comprehensive capability evaluation.

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

The response should identify OTel 2.0 as a model trained by AT&T Chief Data Office. If
it identifies only as Gemma or Google DeepMind, verify that the OTel 2.0 checkpoint,
rather than the base model or a stale mount, is being served.

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
| [`docs/h100_training_notes.md`](docs/h100_training_notes.md) | CUDA training evidence and cross-framework lessons |
| [`docs/h100_inference_notes.md`](docs/h100_inference_notes.md) | CUDA serving evidence and TensorRT-LLM/SGLang findings |
| Each recipe README (`readme_<framework>.md` in the recipe folder) | Exact installation, smoke and full runs, arguments, outputs, evidence, and verdict |

## Responsible Use and Limitations

### Model and evaluation scope

- OTel 2.0 is a telecom-specific generative model. It should not be treated as a
  general-purpose model for unrelated fields.
- It was not trained or evaluated as an embedding, retrieval, or reranking model. Use
  the purpose-built OTel embedding and reranker collections for those stages.
- Comprehensive public OTel 2.0 capability evaluation is forthcoming. Any current
  functional or packaging checks should not be read as broad quality validation.
- Direct Q&A and RAG behavior require separate evaluation. Strong results in one setting
  do not establish strong performance in the other.
- Telecom standards change. Check responses against the relevant source, version, and
  release. RAG quality also depends on ingestion, chunking, retrieval, reranking, prompt
  design, and source freshness.

### Language and modality

- English is the current target language.
- OTel 2.0 is architecturally multimodal, but OTel post-training and published quality
  claims are text-only. Its vision components are inherited unchanged from Gemma 4 and
  have not received telecom-specific training.
- Image input has received functional smoke testing, not a general or telecom vision
  benchmark. Network diagrams, spectrum plots, equipment images, and scanned documents
  require task-specific evaluation. Audio and video are not supported.

### Data coverage

The OTel 2.0 training mixture does not include dedicated collections of:

- Private operator event records, including user activity, failures, anomalies, IMS
  events, or RADIUS authentication records.
- Network KPIs, 5G performance metrics, PIM interference data, RF measurements,
  spectrum data, field-test results, or signal heatmaps.
- 5G core control-plane and inter-network-function signaling.
- IETF RFCs as a dedicated OTel 2.0 corpus.
- Vendor CLI and network operating system documentation for platforms such as Cisco
  IOS-XR, Juniper JUNOS, DNOS, Nokia BNG, or Arista EOS.
- Operator-private network designs, customer or equipment configurations, OSS/BSS data,
  incident-management systems, change-management systems, and approval workflows.

Standards familiarity must not be interpreted as experience with live telemetry,
operator-private records, or vendor-specific behavior.

### Tool use and network operations

- The mixture includes general-purpose instruction-following and tool-calling examples,
  but it does not include telecom-specific MCP, tool-calling, or instruction-following
  examples.
- The model has not been validated against Methods of Procedure on live or digital-twin
  network devices. Command sequences, expected outputs, checkpoints, and rollback steps
  have not been established as correct or safe.
- Closed-loop operational tasks have not been benchmarked with outcome-based scoring.
  This includes network turn-up, SLA or QoS configuration, routing-fault repair, and
  autonomous incident closure.
- Defensive network-security operations, including traffic analysis, firewall or eBPF
  construction, anomaly detection, and DDoS mitigation, have not been validated.
- High-impact or agentic use requires verified external tools, validated schemas,
  retrieval where appropriate, scoped permissions, safeguards, source attribution,
  audit logging, and human review appropriate to the risk.

### Hardware and reproducibility

- AMD MI355X and NVIDIA H100 received the most extensive repository verification. Apple
  and Intel training and inference paths received more limited testing and should not be
  assumed to have equivalent framework, scale, or performance coverage.
- Not every recipe supports every hardware platform. Confirm the exact device and
  framework combination in the recipe README.
- OTel 2.0 weights may be updated. Pin a model revision, checkpoint hash, or release tag
  for reproducible evaluation and production deployment.

Generated content must be independently verified before operational, customer-facing,
regulatory, safety, security, or network-configuration use.

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

Contributors supplied different parts of the system. GSMA and Pleias contributed to the
open telecom corpus; Red Hat supported synthetic-data generation and OSFT; Microsoft
supplied managed compute for large-scale data processing; AMD supplied accelerators and
ROCm; Dell Technologies supplied on-premises training infrastructure; and MLCommons,
academic, and research partners contributed evaluation and domain expertise.

No single organization began with every required element: data rights, standards
expertise, model engineering, compute, evaluation, and distribution. The collaboration
is therefore part of the technical design, not only the project history.

## Selected Coverage and Technical Background

One primary or high-value source is included per organization where possible:

- **GSMA:** [OTel 2.0 release and Open Telco AI leaderboard](https://www.gsma.com/newsroom/article/atts-otel-2-0-is-now-live-the-largest-and-best-performing-open-source-model-built-for-telecoms/)
- **AT&T:** [The tokenomics equation and OTel 2.0](https://about.att.com/blogs/2026/the-tokenomics-equation.html)
- **Microsoft:** [Scaling the trillion-token data and compute workflow](https://azure.microsoft.com/en-us/blog/att-and-microsoft-scale-trillion-token-workloads-with-microsoft-foundry-and-amd/)
- **AMD:** [Training efficiency on AMD Instinct infrastructure](https://www.amd.com/en/resources/case-studies/att-achieves-94-efficiency-for-ai-training-with-amd.html)
- **Dell Technologies:** [Bringing OTel 2.0 to scale](https://www.dell.com/en-us/blog/otel-2-0-dell-technologies-at-t-and-amd-bring-open-telco-ai-to-scale/)
- **Red Hat:** [Training a model for an industry](https://www.redhat.com/en/blog/open-telco-ai-training-model-industry)
- **The Wall Street Journal:** [AT&T's open-weight AI strategy](https://www.wsj.com/cio-journal/why-at-t-is-betting-big-on-open-weight-ai-a0ea03b1)
- **The Information:** [AT&T is using open-source models to curb Anthropic bills](https://www.theinformation.com/newsletters/applied-ai/t-using-open-source-models-curb-anthropic-bills)
- **Fierce Network:** [AT&T's tokenomics strategy](https://www.fierce-network.com/cloud/open-models-are-driving-atts-ai-tokenomics-strategy)
- **Yahoo Finance:** [AT&T and the "token apocalypse"](https://finance.yahoo.com/technology/ai/articles/t-t-says-not-scared-231933381.html)

## Contact

For questions and project updates, visit
[Farbod Tavakkoli on GitHub](https://github.com/farbodtavakkoli) or open an issue in this
repository or contact farbod.tavakkoli@att.com or farbodtavakoli@gmail.com.
