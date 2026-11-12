# `training/trainium/embeddinggemma-2` — Trainium embedding fine-tuner

`train_embedding_standalone.py` fine-tunes `google/embeddinggemma-2` as a dense retrieval bi-encoder on query, positive-passage, and hard-negative rows. It uses `MultipleNegativesRankingLoss` (dot product, scale 20) wrapped in `MatryoshkaLoss`, producing embeddings at dimensions 768, 512, 256, and 128 from one model.

This package is standalone: copy this directory to a compatible Trainium host and provide the path to an existing native TorchNeuron environment. It has no dependency on another source folder, a specific EC2 instance, AWS account, region, hostname, systemd service, or `/opt/trainium` path.

**Validated profile:** Linux x86_64, Python 3.12, Trn2 LNC2, Neuron runtime 2.35, native PyTorch 2.14.0, `torch-neuronx` 2.14.3.0.1+b1b6c66, compiler 2.28.7631.0+8f7b0cf5, and NKI 0.7.0+37713544100.gfede04bb. Other profiles require separate validation.

## Files

- `train_embedding_standalone.py` — thin CLI and workflow sequencing.
- `embedding_training_config.py` — validated model defaults, split constants, cache layout, and dependency names.
- `prepare_embedding_data.py` — JSONL parsing, grouping, splitting, negative filtering, corpus sampling, qrels, and evaluation specifications.
- `embedding_model.py` — model loading, static-shape adaptation, loss construction, metrics, tensor checks, and CPU semantic tests.
- `train_embedding_on_trainium.py` — Trainium preflight, trainer classes, evaluation, fitting, checkpoint selection, export, and reload verification.
- `embedding_run_artifacts.py` — argument validation, run directories, environment checks, logging, source hashes, and artifact hashes.
- `requirements_embedding.txt` — pinned application dependencies for the primary native 2.14 profile.
- `constraints_neuron_2_14.txt` — validated native 2.14 package versions.
- `OTel_embedding_sample_100.jsonl` — 100-row integration sample and default `--train_file`.
- `setup_embedding_environment.py` — creates the application environment from a validated native TorchNeuron base environment.
- `run_embedding_training.sh` — selects the environment and obtains a secure cooperative accelerator lock.
- `tests/test_embedding_package.py` — lightweight failure-path and runner-security tests.

## Setup

Copy this directory to the Trainium host and enter it:

```bash
cd /path/to/embeddinggemma-2
```

Create a fresh application environment from an existing validated native TorchNeuron 2.14 environment:

```bash
python3.12 setup_embedding_environment.py --base-env /path/to/native-neuron-environment
```

`setup_embedding_environment.py` creates `.venv`, installs the pinned application dependencies, adds the base environment's native Neuron packages, runs `pip check`, and executes the CPU semantic self-test. It records the resolved environment and setup log inside `.venv`. The base environment remains unchanged and must remain available.

Use another destination if required:

```bash
python3.12 setup_embedding_environment.py \
  --base-env /path/to/native-neuron-environment \
  --venv /path/to/recipe-venv

RECIPE_VENV=/path/to/recipe-venv bash run_embedding_training.sh --preflight \
  --output_dir outputs/preflight-01
```

The setup rejects an existing destination and an unvalidated native dependency profile instead of modifying it. This release supports the native 2.14 profile only.

Torchvision and Pillow are required because the model processor imports its multimodal processing path even when this recipe disables the vision and audio towers and trains on text only.

## Data

`--train_file` accepts JSONL with one retrieval example per line:

```json
{
  "anchor": "query text",
  "positive": "relevant passage",
  "negative_1": "hard negative passage",
  "negative_2": "optional additional hard negative"
}
```

The package supports `negative_1` through `negative_5`. `--num_negatives` controls how many clean, source-ordered negatives are used. Rows missing an anchor, positive, or enough valid negatives are recorded in the data audit and excluded.

The preparation stage:

- normalizes and hashes query and document identities;
- groups connected query-positive components and punctuation variants;
- assigns groups to approximately 90% train, 5% validation, and 5% test using the configured seed;
- removes known positives and held-out positives from training negatives;
- creates a fixed sampled retrieval corpus and evaluation specification; and
- records source, preprocessing, split, and evaluation hashes.

The shipped sample validates the package only. With seed 42 it produces 93 training, 3 validation, and 4 test rows with a 599-document corpus. Three validation queries are too few to measure production retrieval quality.

Use `--prepare_only` to inspect data without loading the model:

```bash
bash run_embedding_training.sh \
  --prepare_only \
  --train_file OTel_embedding_sample_100.jsonl \
  --output_dir outputs/data-check-01
```

## Run

Stop serving and other accelerator workloads before running. Separate ports or processes do not isolate Trainium cores or device memory.

### Package tests

These tests require no model download or Trainium accelerator:

```bash
python3 tests/test_embedding_package.py
```

### CPU semantic self-test

This verifies loss composition, gradients, finite values, optimizer updates, and invalid Matryoshka dimensions:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .venv/bin/python train_embedding_standalone.py --self_test
```

### Trainium preflight

Preflight verifies dependency consistency, the model processor, native `torch.neuron` availability, and a compiled forward/backward/update operation. It does not fine-tune the full model or prove that arbitrary batch sizes fit.

```bash
bash run_embedding_training.sh --preflight --output_dir outputs/preflight-01
```

### Fine-tune

Use a new output directory for every run:

```bash
bash run_embedding_training.sh \
  --model_name google/embeddinggemma-2 \
  --train_file OTel_embedding_sample_100.jsonl \
  --output_dir outputs/train-01 \
  --batch_size 3 \
  --epochs 1 \
  --lr 1e-5 \
  --max_seq_length 512 \
  --eval_max_seq_length 1024 \
  --matryoshka_dims 768 512 256 128 \
  --num_negatives 1
```

The first training step compiles static Neuron graphs and can take several minutes. Compilation latency is not steady-state training throughput.

The model initially loads on CPU, moves to `neuron`, and compiles its forward path with:

```python
torch.compile(model.forward, backend="neuron", dynamic=False)
```

Training and evaluation encoding use BF16 autocast. Model weights and AdamW states remain FP32. Retrieval ranking moves embeddings to CPU and calculates FP32 cosine similarity.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | `google/embeddinggemma-2` | Required model ID in this validated release |
| `--model_revision` | `914f7f89142e33e77833254d9c9b90c3cef7303b` | Pinned pretrained revision |
| `--train_file` | `OTel_embedding_sample_100.jsonl` | Input JSONL |
| `--output_dir` | `outputs/embeddinggemma2` | New run directory |
| `--model_cache_dir` | `~/.cache/huggingface/hub` | Downloaded model cache |
| `--compile_cache_dir` | `<output_dir>/compiled` | Neuron backend workspace and compilation caches |
| `--device` | `neuron` | Native training device |
| `--batch_size` | `3` | Per-device contrastive training batch |
| `--eval_batch_size` | `1` | Loss-evaluation batch |
| `--retrieval_batch_size` | `2` | Retrieval encoding batch |
| `--epochs` | `1` | Training epochs |
| `--lr` | `1e-5` | AdamW learning rate |
| `--max_steps` | `-1` | Optional training-step cap |
| `--max_seq_length` | `512` | Static training sequence length |
| `--eval_max_seq_length` | `1024` | Static retrieval sequence length |
| `--matryoshka_dims` | `768 512 256 128` | Dimensions trained at every step |
| `--num_negatives` | `1` | Hard negatives retained per row |
| `--eval_corpus_size` | `20000` | Additional hash-ranked distractors |
| `--query_prompt` | `task: search result \| query:` | Query retrieval prefix |
| `--document_prompt` | `title: none \| text:` | Passage retrieval prefix |
| `--seed` | `42` | Split and training seed |
| `--prepare_only` | off | Prepare and audit data without training |
| `--preflight` | off | Validate the model and native Neuron path |
| `--self_test` | off | Run the CPU loss and gradient test |

`--num_negatives` and `--batch_size` directly affect activation and similarity-matrix memory. Reduce the batch size first if device memory is exhausted.

## Output

Each successful training directory contains:

- `config.json`, `environment.json`, and `source_hashes.json`;
- `data/` with audited splits, qrels, corpus, and evaluation specification;
- `baseline.json` with pretrained retrieval metrics at every dimension;
- `training_steps.json` and `run.log`;
- `checkpoints/` with at most two epoch checkpoints;
- `final_model/` with the selected exported model; and
- `result.json` with final metrics and verification evidence.

The best epoch is selected by full-width validation NDCG@10. Success requires:

- at least one optimizer step and a nonzero learning rate;
- changed model tensors;
- finite tensors and training logs;
- exact tensor-hash agreement after export and reload; and
- successful Neuron retrieval evaluation of the reloaded model.

`result.json` reports `status: completed` only after these checks pass. Completion proves that training and export worked; it does not guarantee that retrieval quality improved. The test split is not used for checkpoint selection. The script does not publish or deploy the model.

## Notes

- `HF_DEACTIVATE_ASYNC_LOAD=1` disables asynchronous weight loading for Trainium compatibility.
- `TORCH_NEURONX_DEBUG_DIR` and the Neuron/Inductor cache variables are
  directed under the run-specific compile-cache directory. This avoids
  collisions with root-owned SDK defaults under `/tmp` when an unprivileged
  user runs the recipe on a reused host.
- Preprocessing pads every batch to its configured static sequence length.
- The loss does not use GradCache, gradient checkpointing, or cross-device gathering.
- `dataloader_drop_last=True`; the number of training steps is based only on complete duplicate-free batches.
- Existing output directories are rejected so previous results are preserved.
- No checkpoint resume is performed; every invocation starts from the pinned pretrained model with a new optimizer.
- A Hugging Face token is required only when model access requires one.
- Model and dependency downloads require outbound network access unless already cached.
- The runner lock coordinates only instances of this runner. It cannot stop an unrelated serving or training process from using the accelerator.
- The lock is stored under `$XDG_RUNTIME_DIR`, or `/tmp/trainium-recipe-$UID` when that variable is unset. The directory must be owned by the current user with mode `0700`. Set `TRAINIUM_RECIPE_RUNTIME_DIR` only to another private directory with the same ownership and permissions.

## Sample provenance

The bundled sample comes from `farbodtavakkoli/OTel` commit `4cfef3a3ea27c9a49c57fbe609c40143f3b14350`:

```text
training/embedding/sentence_transformers/OTel_embedding_sample_100.jsonl
```

SHA-256: `d936f660d5d06e24b7579714bc211cc04d2254f8f8c2e70df1ccd148e350bb53`

## Recorded standalone validation

The native PyTorch 2.14 training workflow completed one epoch on a
`trn2.3xlarge` with 93 training rows, batch size 3, one hard negative,
512-token training, and 1024-token evaluation. It completed 31 optimizer steps
in 997.46 seconds, including baseline evaluation, compilation, training,
export, reload, and verification.

The validation changed 389 of 413 model-state tensors. All weights and logs were finite, and selected, exported, and reloaded tensor hashes matched.

| Dimension | Baseline NDCG@10 | Fine-tuned NDCG@10 |
|---|---:|---:|
| 768 | 1.000000 | 0.876977 |
| 512 | 1.000000 | 1.000000 |
| 256 | 1.000000 | 0.876977 |
| 128 | 1.000000 | 0.876977 |

These scores come from three validation queries and demonstrate operation, not a statistically reliable quality improvement.

On 2026-11-01, the documented workflow was repeated from a clean copy of this
directory on a `trn2.3xlarge`. `setup_embedding_environment.py`, all five package tests,
`pip check`, the CPU semantic self-test, and the Trainium preflight passed. The
fine-tuning command above then completed all 31 steps, exported the selected
checkpoint, reloaded it, and reproduced the recorded metrics. The first
optimizer step, including graph compilation, took 497.74 seconds; subsequent
steps took approximately 0.29–0.34 seconds.

On 2026-11-02, the modular package was tested as an unprivileged new user from
a root-owned, read-only source copy on a `trn2.3xlarge`. Seven package tests,
environment setup, `pip check`, the CPU semantic self-test, and Trainium
preflight passed. The first preflight found that the SDK default
`/tmp/neuron_backend` belonged to another user; the package now directs
`TORCH_NEURONX_DEBUG_DIR` into the run-specific compile-cache directory.

The corrected package then completed the documented one-epoch run: 31 optimizer
steps in 929.12 seconds, 389 of 413 tensors changed, all weights and logs were
finite, and export/reload hashes matched. Reloaded evaluation ran on `neuron`.
The first optimizer step took 465.33 seconds including compilation; subsequent
steps took approximately 0.29–0.30 seconds. The metrics reproduced the table
above. Host-specific evidence is kept outside this standalone directory.

After the source files received the descriptive names documented in the
**Files** section and tests moved under `tests/`, the package was validated
again from a clean, read-only copy on November 2, 2026. A new application
environment, model cache, compilation cache, and output directory were created
under an unprivileged user. All seven package tests, setup checks, CPU semantic
self-test, and Trainium preflight passed. The full run completed 31 steps in
946.46 seconds, changed 389 of 413 tensors, and passed export, reload, hash,
finite-value, and reloaded-Neuron evaluation checks.
