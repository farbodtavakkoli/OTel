# Standalone EmbeddingGemma-2 fine-tuning on Trainium

Copy this folder to a Trainium host. Its four files are the entire application:

- `train_embedding_standalone.py`: preparation, Neuron adaptation, training,
  evaluation and verified export in one script; no sibling Python imports.
- `requirements_embedding.txt`: pinned Python dependencies.
- `OTel_embedding_sample_100.jsonl`: 100-row installation/integration sample.
- `readme_embedding.md`: this guide.

The recipe uses ordinary `MultipleNegativesRankingLoss` wrapped in
`MatryoshkaLoss`, with equally weighted dimensions 768/512/256/128. It uses
native TorchNeuron (`torch.neuron`, `torch.compile(backend="neuron")`), not XLA.
There is no GradCache, gradient checkpointing, distributed gathering, AWS API,
SSM, systemd or campaign dependency. Only `google/embeddinggemma-2` is supported
by this tested model profile; other models require separate validation.

## Environment prerequisites

The folder does not bundle the Neuron driver, runtime, compiler or model weights.
Use Linux x86_64, Python 3.12 and a compatible Trainium host. The established
reference environment is Trn2 LNC2, Neuron runtime 2.35, compiler 2.28 and native
TorchNeuron source revision `50b14171affaa365f1df6aa015232427336f6a4d`.
The Transformers source revision is pinned in the requirements file.

The native wheel used for this recipe is
`torch_neuronx-2.13.3.0.1+50b1417.dev-cp312-cp312-linux_x86_64.whl`, built against
CPU PyTorch 2.13 and the matching runtime headers on Ubuntu 24.04.
Its reference SHA-256 is
`0dd71baca4ccd9f1e7b5462811bda7a3217b90de01b05824968e97b8defff292`.
Obtain that compatible wheel, or build the pinned source using its upstream
build instructions and matching runtime headers. A different build must be
revalidated. The public prebuilt wheel previously required a newer glibc than
this host; substituting an arbitrary `torch-neuronx` release is not supported.

With the compatible wheel available, create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps /path/to/compatible/torch_neuronx-2.13.3.0.1+50b1417.dev-cp312-cp312-linux_x86_64.whl
python -m pip install -r requirements_embedding.txt \
  --extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip check
```

Expose the installed matching runtime library and compiler executables using
`LD_LIBRARY_PATH` and `PATH` if they are not in the system defaults. These paths
are installation-specific and are not hardcoded by the script. Stop other
accelerator workloads before starting; this script does not manage services or
reserve cores from unrelated processes.

Torchvision and Pillow are required by the model's processor-loading path even
when vision/audio towers are disabled for text-only training.

## Run

CPU semantic self-test (no model download or accelerator use):

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python train_embedding_standalone.py --self_test
```

Fine-tune from the pinned pretrained base with a new optimizer:

```bash
python train_embedding_standalone.py \
  --model_name google/embeddinggemma-2 \
  --train_file OTel_embedding_sample_100.jsonl \
  --output_dir ./outputs/first-run \
  --batch_size 3 --epochs 1 --lr 1e-5 \
  --max_seq_length 512 --eval_max_seq_length 1024 \
  --matryoshka_dims 768 512 256 128 --num_negatives 1
```

Use a new output directory each time. Existing directories are rejected to
protect results. No checkpoint resume is performed. Paths are configurable;
there is no dependency on an `/opt/trainium` directory. Default input is resolved
relative to the script, so the script can be launched from another directory.
A Hugging Face token is needed only if your model access requires it.

The first step compiles graphs and can take several minutes. Compilation time
is not steady-state throughput. Only the model forward is compiled; shapes are
static and weight loading is synchronous. Weights/Adam states remain FP32;
training and evaluation encoding use BF16 autocast. Retrieval ranking uses CPU
FP32 cosine similarity.

## Data and arguments

JSONL input contains `anchor`, `positive` and `negative_1` through `negative_5`.
`--num_negatives` explicitly controls how many clean source-ordered negatives
are retained (default 1, to match the validated small-memory recipe).
Increasing it changes activation memory and the training objective; reduce the
batch size if needed. Rows with too few clean negatives are audited and skipped.

Query/positive connected groups and punctuation variants are kept together.
Seeded hash buckets allocate approximately 90% train, 5% validation, 5% test;
small samples need not match those ratios exactly. With the shipped sample and
seed 42, the split is 93/3/4, with a 599-document corpus. Test queries are never
evaluated or used to select a checkpoint. The corpus can contain their documents.
The three validation queries are an integration check, not a quality benchmark.

| Argument | Default | Purpose |
|---|---|---|
| `--model_revision` | `914f7f89142e33e77833254d9c9b90c3cef7303b` | Pin pretrained base |
| `--model_cache_dir` | Hugging Face cache under home | Downloaded model files |
| `--compile_cache_dir` | `<output_dir>/compiled` | Neuron compilation cache |
| `--batch_size` | 3 | Contrastive training batch |
| `--eval_batch_size` | 1 | Loss evaluation batch |
| `--retrieval_batch_size` | 2 | Retrieval encoding batch |
| `--epochs` / `--lr` | 1 / 1e-5 | Training schedule |
| `--max_seq_length` | 512 | Fixed training length |
| `--eval_max_seq_length` | 1024 | Fixed retrieval evaluation length |
| `--matryoshka_dims` | 768 512 256 128 | All dimensions trained each step |
| `--num_negatives` | 1 | Explicit hard negatives per row |
| `--eval_corpus_size` | 20000 | Additional sampled distractors |
| `--seed` | 42 | Data split and training seed |
| `--query_prompt`, `--document_prompt` | Model retrieval prefixes | Explicit text prefixes |
| `--max_steps` | -1 | Optional step cap; must allow actual updates beyond warmup |
| `--prepare_only` | Off | Audit and split data without ML dependencies |

## Outputs and verification

Each output directory contains configuration, environment versions, source/data
hashes, prepared splits, baseline metrics, per-step training logs, checkpoints,
`final_model/` and `result.json`. The best epoch checkpoint is selected using
full-width validation NDCG@10. Baseline and export evaluation cover every
Matryoshka dimension. A single-epoch run therefore selects its epoch-end model.

Success requires real weight changes, finite weights and training logs, exact
export/reload tensor hashes, and Neuron retrieval evaluation of the reload.
`result.json` reports `status: completed` only after these checks. It reports
whether full-width NDCG improved; successful execution does not guarantee a
better retriever. The script never publishes or deploys the output.

The pretrained and exported model configuration is text-only. Vision and audio
towers are excluded. Small validation changes should be investigated with a
larger fixed evaluation set rather than tuning repeatedly on three queries.

## Sample provenance

Source: `farbodtavakkoli/OTel` commit
`4cfef3a3ea27c9a49c57fbe609c40143f3b14350`, file
`training/embedding/sentence_transformers/OTel_embedding_sample_100.jsonl`.
SHA-256: `d936f660d5d06e24b7579714bc211cc04d2254f8f8c2e70df1ccd148e350bb53`.

## Standalone test completed — October 29, 2026

Only these four application files were copied into a new directory on a
`trn2.3xlarge`. A new Python virtual environment was installed using this
requirements file and the documented compatible native wheel. `pip check`
passed. The script ran with Python `-I`, from outside its own directory, using
separate model and compiler caches. Its recorded import path contains only the
standard library and the new virtual environment, with no old application path.

The first clean-environment attempt identified missing torchvision/Pillow
processor dependencies. Both are now included in the manifest and setup command.
The corrected run completed one epoch: 93 training rows, 31 optimizer steps,
batch 3, one hard negative, 512-token training and 1024-token evaluation.
Elapsed time was 925.70 seconds, including baseline, compilation and verification.

Verification passed: 389 of 413 model-state tensors changed, all weights and
training logs were finite, selected/exported/reloaded tensor hashes matched,
and the reloaded model encoded on `neuron:0` at every configured dimension.
The selected checkpoint was step 31. No held-out test evaluation or deployment
was performed. Serving remained stopped and the accelerator was idle afterward.

| Dimension | Baseline NDCG@10 | Fine-tuned NDCG@10 |
|---|---:|---:|
| 768 | 1.000000 | 0.876977 |
| 512 | 1.000000 | 1.000000 |
| 256 | 1.000000 | 0.876977 |
| 128 | 1.000000 | 0.876977 |

These three-query validation scores demonstrate operation, not quality
improvement. The validation results above are from the original isolated Trainium run; this
source contribution does not include checkpoints, compiler caches, or machine-local logs.
