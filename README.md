# Open Telco (OTel) AI: Datasets, Benchmarks, and Models

<p align="center">
  <a href="https://huggingface.co/farbodtavakkoli">Hugging Face</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-llm">LLM Collection</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-embedding">Embedding Collection</a> |
  <a href="https://huggingface.co/collections/farbodtavakkoli/otel-reranker">Reranker Collection</a> |
  <a href="https://huggingface.co/farbodtavakkoli/datasets">Datasets</a> |
  <a href="https://github.com/farbodtavakkoli/OTel/blob/main/docs/media_coverage.md">Media Coverage</a>
</p>

Open Telco (OTel) AI is an open telecom AI resource for training, evaluating, and deploying retrieval, reranking, context-grounded generation, and abstention models for telecommunications. The release includes derived telecom datasets, held-out evaluation partitions, model cards, dataset cards, Croissant metadata, training and inference code, and a family of full-parameter post-trained telecom models.

The project is designed around the telecom RAG stack:

1. Embedding models retrieve relevant telecom passages.
2. Rerankers re-score retrieved query-passage pairs.
3. LLMs generate grounded answers from the retrieved context.
4. Safety variants abstain when the retrieved context is insufficient or off-topic.

As of May 3, 2026, the released OTel models had more than 16 million downloads, and the Open Telco AI project had received 157+ pieces of media coverage worldwide.

## Highlights

| Area | Current best reported result | Model |
|---|---:|---|
| Context-grounded LLM correctness | 91.7% +/- 0.4 | `OTel-LLM-E4B-IT` |
| Embedding retrieval | 93.5% +/- 0.3 NDCG@10 | `OTel-Embedding-8B` |
| Reranking | 0.952 +/- 0.004 MRR@10 | `OTel-Reranker-8B` |
| Strong mid-size LLM | 88.4% +/- 0.5 correctness | `OTel-LLM-8B-A1B-IT` |
| Efficient embedding pick | 90.9% +/- 0.5 NDCG@10 at 300M params | `OTel-Embedding-300M` |
| Low-latency LLM pick | 74.4% +/- 0.7 correctness at 1.2B params | `OTel-LLM-1.2B-IT` |

Results are reported on held-out OTel evaluation partitions. LLM results measure context-grounded answer generation from retrieved context and should not be interpreted as unrestricted context-free telecom QA performance.

## Datasets

The OTel source corpus contains public telecom documents and contributor-provided telecom examples spanning 3GPP specifications, GSMA documents, O-RAN documents, RFCs, academic papers, industry whitepapers, Wikipedia, and web-derived telecom material. The released datasets contain derived training/evaluation examples, not the raw source documents.

| Dataset | Purpose | Key fields |
|---|---|---|
| [OTel-LLM](https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM) | Context-grounded instruction tuning for telecom RAG generation | `prompt`, `completion`, `abstention`, chunk-count metadata |
| [OTel-Embedding](https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding) | Bi-encoder retrieval training with hard negatives | `anchor`, `positive`, `negative_1` through `negative_5` |
| [OTel-Reranker](https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker) | Cross-encoder reranking supervision | `sentence_0`, `sentence_1`, `label` |
| [OTel-Safety](https://huggingface.co/datasets/farbodtavakkoli/OTel-Safety) | Abstention training when context is insufficient | `prompt`, `completion`, `abstention`, chunk-count metadata |

### Data Sources

| Source | Contributor | Raw samples |
|---|---|---:|
| arXiv telecom papers, 3GPP standards, telecom Wikipedia, telecom Common Crawl pages | Yale University | 681,172 |
| GSMA Permanent Reference Documents, Discover portal, mixed telecom documents | GSMA | 158,006 |
| IETF RFC series | NetoAI | 100,751 |
| Industry whitepapers | Khalifa University | 62,000 |
| O-RAN specifications across working groups 1, 2, 4, 5, 6, 7, 8, 9, 10 | University of Leeds | 58,565 |
| O-RAN documents across working groups | The University of Texas at Dallas | 42,000 |
| Total raw samples | | ~1,102,494 |

After heuristic filtering, reranker-based semantic filtering, embedding-based semantic filtering, and deduplication, the retained corpus contains 326,767 higher-confidence examples.

Each released dataset includes a dataset card and Croissant metadata with Responsible AI fields for data limitations, biases, sensitive-information considerations, use cases, social impact, synthetic-data status, and provenance.

## Model Zoo

All models are available under [farbodtavakkoli on Hugging Face](https://huggingface.co/farbodtavakkoli). Models are full-parameter post-trained from open base checkpoints on OTel-derived telecom data.

### Language Models

| Model | Params | Base model | OTel score | Delta |
|---|---:|---|---:|---:|
| [OTel-LLM-270M-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-270M-IT) | 270M | gemma-3-270m-it | 31.2% +/- 1.2 | +9.0 pp |
| [OTel-LLM-0.6B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-0.6B-IT) | 0.6B | Qwen3-0.6B | 59.0% +/- 0.9 | +10.0 pp |
| [OTel-LLM-1B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-1B-IT) | 1B | gemma-3-1b-it | 57.3% +/- 0.9 | +9.0 pp |
| [OTel-LLM-1.2B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-1.2B-IT) | 1.2B | LFM2.5-1.2B-Instruct | 74.4% +/- 0.7 | +8.0 pp |
| [OTel-LLM-1.7B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-1.7B-IT) | 1.7B | Qwen3-1.7B | 61.3% +/- 0.8 | +8.5 pp |
| [OTel-LLM-3B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-3B-IT) | 3B | Mistral-3-3B | 64.4% +/- 0.8 | +7.5 pp |
| [OTel-LLM-4B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-4B-IT) | 4B | gemma-3-4b-it | 73.2% +/- 0.7 | +7.0 pp |
| [OTel-LLM-E4B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-E4B-IT) | 4.5B | gemma-4-E4B-it | 91.7% +/- 0.4 | +9.3 pp |
| [OTel-LLM-7B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-7B-IT) | 7B | OLMo-3-7B | 63.4% +/- 0.8 | +6.0 pp |
| [OTel-LLM-8B-A1B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-8B-A1B-IT) | 8B | LFM2.5-8B-A1B | 88.4% +/- 0.5 | +7.2 pp |
| [OTel-LLM-8.2B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-8.2B-IT) | 8.2B | Qwen3-8B | 66.4% +/- 0.7 | +6.0 pp |
| [OTel-LLM-8.3B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-8.3B-IT) | 8.3B | RNJ-1-Instruct | 79.6% +/- 0.6 | +7.1 pp |
| [OTel-LLM-12B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-12B-IT) | 12B | gemma-3-12b-it | 83.3% +/- 0.5 | +5.0 pp |
| [OTel-LLM-14B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-14B-IT) | 14B | Qwen3-14B | 66.2% +/- 0.7 | +5.5 pp |
| [OTel-LLM-20B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-20B-IT) | 20B | GPT-OSS-20B | 66.4% +/- 0.7 | +5.0 pp |
| [OTel-LLM-20B-Reasoning](https://huggingface.co/farbodtavakkoli/OTel-LLM-20B-Reasoning) | 20B | GPT-OSS-20B | 71.7% +/- 0.7 | +6.5 pp |
| [OTel-LLM-24B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-24B-IT) | 24B | LFM2-24B-A2B | 79.5% +/- 0.6 | +4.5 pp |
| [OTel-LLM-27B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-27B-IT) | 27B | gemma-3-27b-it | 88.2% +/- 0.4 | +3.7 pp |
| [OTel-LLM-32B-IT](https://huggingface.co/farbodtavakkoli/OTel-LLM-32B-IT) | 32B | OLMo-3-32B | 71.3% +/- 0.7 | +4.5 pp |

LLM score is LLM-as-judge correctness on held-out OTel-LLM evaluation examples. Correctness is judged against the retrieved context and reference answer.

### Embedding Models

| Model | Params | Base model | OTel NDCG@10 | Delta |
|---|---:|---|---:|---:|
| [OTel-Embedding-22M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-22M) | 22M | all-MiniLM-L6-v2 | 84.3% +/- 0.7 | +60.2 pp |
| [OTel-Embedding-33M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-33M) | 33M | bge-small-en-v1.5 | 86.9% +/- 0.6 | +55.5 pp |
| [OTel-Embedding-34M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-34M) | 34M | all-MiniLM-L12-v2 | 85.1% +/- 0.7 | +55.9 pp |
| [OTel-Embedding-109M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-109M) | 109M | all-mpnet-base-v2 | 87.8% +/- 0.6 | +49.3 pp |
| [OTel-Embedding-300M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-300M) | 300M | Gemma3-Embedding-300M | 90.9% +/- 0.5 | +18.6 pp |
| [OTel-Embedding-335M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-335M) | 335M | bge-large-en-v1.5 | 89.7% +/- 0.5 | +38.0 pp |
| [OTel-Embedding-568M](https://huggingface.co/farbodtavakkoli/OTel-Embedding-568M) | 568M | bge-m3 | 90.1% +/- 0.5 | +32.9 pp |
| [OTel-Embedding-0.6B](https://huggingface.co/farbodtavakkoli/OTel-Embedding-0.6B) | 600M | Qwen3-Embedding-0.6B | 90.5% +/- 0.4 | +10.8 pp |
| [OTel-Embedding-4B](https://huggingface.co/farbodtavakkoli/OTel-Embedding-4B) | 4B | Qwen3-Embedding-4B | 92.2% +/- 0.4 | +9.7 pp |
| [OTel-Embedding-8B](https://huggingface.co/farbodtavakkoli/OTel-Embedding-8B) | 8B | Qwen3-Embedding-8B | 93.5% +/- 0.3 | +9.6 pp |

### Reranker Models

| Model | Params | Base model | OTel MRR@10 | Delta |
|---|---:|---|---:|---:|
| [OTel-Reranker-0.6B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-0.6B) | 0.6B | Qwen3-0.6B | 0.944 +/- 0.006 | +0.598 |
| [OTel-Reranker-4B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-4B) | 4B | Qwen3-4B | 0.948 +/- 0.005 | +0.541 |
| [OTel-Reranker-8B](https://huggingface.co/farbodtavakkoli/OTel-Reranker-8B) | 8B | Qwen3-8B | 0.952 +/- 0.004 | +0.535 |

### Auxiliary Models

These models are released alongside the RAG-oriented family and should be interpreted according to their task-specific evaluation setting.

| Model | Params | Base model | Purpose |
|---|---:|---|---|
| [OTel-LLM-8.3B-Safety](https://huggingface.co/farbodtavakkoli/OTel-LLM-8.3B-Safety) | 8.3B | RNJ-1-Instruct | Abstention-focused safety variant |
| [OTel-LLM-12B-Safety](https://huggingface.co/farbodtavakkoli/OTel-LLM-12B-Safety) | 12B | gemma-3-12b-it | Abstention-focused safety variant |
| [OTel-LLM-8.3B-Classification](https://huggingface.co/farbodtavakkoli/OTel-LLM-8.3B-Classification) | 8.3B | RNJ-1 | TeleLogs 5G root-cause-analysis classification |
| [OTel-LLM-8.3B-QnA](https://huggingface.co/farbodtavakkoli/OTel-LLM-8.3B-QnA) | 8.3B | RNJ-1-Instruct | Non-abstention QnA variant for context-free telecom QA |

## Quick Start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r OTel/requirements.txt
```

If you are working directly from the `OTel/` repository root rather than this workspace root, use:

```bash
pip install -r requirements.txt
```

### Use an OTel LLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "farbodtavakkoli/OTel-LLM-8.3B-IT"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = """You are a precise telecom assistant in a RAG pipeline.
Use only the retrieved context to answer.

User Question
What is the purpose of the F1 interface in O-RAN?

Retrieved Contexts
CONTEXT 1
The F1 interface connects the O-RAN Distributed Unit (O-DU) to the O-RAN Central Unit (O-CU).

Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Use an OTel Embedding Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("farbodtavakkoli/OTel-Embedding-300M")

sentences = [
    "What is the F1 interface in O-RAN?",
    "The F1 interface connects the O-RAN Distributed Unit to the O-RAN Central Unit.",
]

embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)
```

### Use an OTel Reranker

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "farbodtavakkoli/OTel-Reranker-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
)

query = "What is the F1 interface?"
documents = [
    "The F1 interface connects O-DU to O-CU in O-RAN architecture.",
    "5G networks may use millimeter-wave frequencies.",
]

pairs = [[query, doc] for doc in documents]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    scores = model(**inputs).logits.squeeze()
print(scores)
```

## Training and Evaluation Code

The executable code in this workspace is under `OTel/`.

```bash
cd OTel

# Train a language model
python3 train.py --data_path data/llm_training_sample.json --model_type qwen3

# Train an embedding model
python3 train.py --training_mode embedding --data_path data/embedding_training_sample.json

# Run inference
python3 inference.py --model_name farbodtavakkoli/OTel-LLM-8.3B-IT --model_type qwen3
```

To reproduce held-out OTel evaluation results, use `eval.py` from inside `OTel/`:

```bash
# Embedding NDCG@10
python3 eval.py --mode embedding --model farbodtavakkoli/OTel-Embedding-300M

# Reranker MRR@10
python3 eval.py --mode reranker --model farbodtavakkoli/OTel-Reranker-0.6B

# LLM correctness via LLM-as-judge
python3 eval.py --mode llm --model farbodtavakkoli/OTel-LLM-1.2B-IT
```

Embedding and reranker evaluation require a Hugging Face token. LLM evaluation also requires an OpenAI API key for the judge model. See `OTel/dev.env.example`.

## Representative Dataset Rows

### OTel-LLM

```json
{
  "anchor": "How can a cell be considered to be operating in MBSFN mode for 3.84/7.68 Mcps TDD?",
  "completion": "A cell shall be considered to be operating in MBSFN mode when individual scrambling codes are assigned to all timeslots via the IE \"TDD MBSFN Information\".",
  "abstention": false,
  "n_positive_chunks": 1,
  "n_negative_chunks": 4
}
```

### OTel-Embedding

```json
{
  "anchor": "During the Measurement ID Coordination test case, what is the relationship between the U-Plane data and the F1 logs?",
  "positive": "F1 logs recorded in the Protocol Analyzer and the Test UE or UE emulator show that all downlink U-Plane data recorded in the F1 logs is correctly received, and all uplink U-Plane data transmitted by the Test UE or emulated UE is recorded in the F1 logs.",
  "negative_count": 5
}
```

### OTel-Reranker

```json
{
  "sentence_0": "The Fronthaul Gateway can translate FH protocol from an O-DUx with split option 7-2 to an O-RUy with split option 8.",
  "sentence_1": "Fronthaul Gateway that can translate FH protocol from an O-DUx with split option x to an O-RUy with split option y, with currently available option 7-2 to 8.",
  "label": 1.0
}
```

### OTel-Safety

```json
{
  "anchor": "How are SCP domains structured and grouped in the SCP trust domain solution?",
  "completion": "I do not have enough information based on the provided context to answer your question.",
  "abstention": true,
  "n_positive_chunks": 0,
  "n_negative_chunks": 5
}
```

## Repository Layout

```text
.
|-- OTel/                         # Training, inference, and evaluation code
|-- dataset_cards/                # Dataset card drafts for released OTel datasets
|-- dataset_examples.md           # Representative rows from each released dataset
|-- neurips_ready_paper/          # Paper, plots, checklist, model-card generator
|-- plots/                        # Main performance and project plots
|-- OTel_croissant_metadata_clean/ # Croissant metadata for the released datasets
```

## Training Recipe

| Item | Value |
|---|---|
| Framework | ScalarLM |
| Training method | Full-parameter post-training / fine-tuning |
| Optimizer | AdamW, 8-bit |
| Learning-rate schedule | Cosine decay with warmup |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Random seed | 42 |
| Maximum sequence length | 1500 tokens |
| Precision | BF16 |
| Attention | Flash Attention 2 |
| Distributed training | Fully Sharded Data Parallel |
| Gradient checkpointing | Enabled |
| Epochs | 3 for LLM/embedding models; 2 for rerankers |
| Compute | AMD MI300X/MI325X/MI355X and NVIDIA A100/H100 GPUs |

## Intended Use

OTel is intended for telecom RAG research and deployment:

- Retrieve relevant telecom chunks from standards, RFCs, whitepapers, O-RAN documents, GSMA documents, academic papers, and related sources.
- Rerank retrieved chunks before generation.
- Generate context-grounded telecom answers from retrieved evidence.
- Train and evaluate abstention behavior when context is insufficient or off-topic.

## Limitations and Responsible Use

- OTel models are domain-specific to telecommunications and should not be treated as general-purpose models.
- The current release is English-only and primarily text-centric.
- Main results are reported on held-out OTel evaluation partitions rather than a fully independent external benchmark suite.
- LLM results evaluate context-grounded RAG behavior, not unrestricted context-free QA.
- Aggregate scores can hide subdomain variation; O-RAN retrieval appears comparatively strong, while academic-paper and GSMA PRD examples need further curation.
- Generated telecom content should be verified before operational, customer-facing, regulatory, safety, or network-configuration use.
- Users must comply with both the OTel release license and the upstream base-model licenses or terms.

## License

The OTel datasets release derived QA/retrieval/reranking examples under Apache-2.0. The model releases are based on different upstream base checkpoints; users should review and comply with the terms of each base model as well as the OTel release license.

## Citation

```bibtex
@misc{otel_models_2026,
  title  = {OTel: Open Telco AI Datasets, Benchmarks, and Models},
  author = {Tavakkoli, Farbod and others},
  year   = {2026},
  note   = {Open Telco (OTel) model release},
  url    = {https://huggingface.co/farbodtavakkoli}
}
```

## Contact

For technical questions, contact farbod.tavakkoli@att.com or farbodtavakoli@gmail.com.
