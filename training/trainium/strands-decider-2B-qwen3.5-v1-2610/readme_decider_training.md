# Strands Decider full-parameter training on Trainium

A standalone, single-host training recipe for
`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`.
Before training, we incorporate the released LoRA updates into the decoder weights
and remove the separate adapter modules. We then fine-tune all 1,884,976,448
parameters of the resulting model, including embeddings and the released pointer head.
It preserves 24 decoder layers (18 Gated DeltaNet and 6 attention layers), option
scoring, and head dropout 0.05. No language-model head or generation loop is added.

Merging means replacing each adapted weight matrix with `W_base + scale * B @ A`,
where `A` and `B` are the released LoRA matrices. Their learned adjustment becomes
part of the ordinary decoder weights; no separate LoRA parameters remain.
Preparation checks predictions before and after this merge. Full-parameter
training then updates the combined weights and the existing pointer head.

## Files

- `readme_decider_training.md` — setup, usage and validation guide for this recipe.
- `setup_trainium_environment.py` — creates the application venv and installs the
  pinned dependencies and upstream Decider package using an existing Neuron base.
- `requirements_decider_training.txt` — pinned application dependencies to install.
- `constraints_neuron.txt` — version constraints for the preinstalled Neuron framework.
- `run_decider_training.sh` — configures the runtime and caches, then launches
  preparation, training, verification or probes; training uses four Neuron ranks.
- `prepare_decider_dataset.py` — generates synthetic examples, downloads/converts
  the OTel fixture, and audits prepared choice-format JSONL.
- `prepare_decider_model.py` — downloads the pinned model, merges LoRA into the
  decoder, and checks architecture and merge/reload prediction parity.
- `train_decider_full.py` — trainer entry point; sets up FSDP and compilation,
  coordinates training and resume, and exports the full trained model.
- `decider_training_step.py` — performs one optimizer update, including microbatch
  loss scaling, backward, gradient validation and global norm clipping.
- `decider_checkpoints.py` — saves/restores rank checkpoints, optimizer and RNG
  state; manages completion markers and verifies export-to-shard equality.
- `decider_model_io.py` — saves and loads full decoder weights, pointer head,
  tokenizer and model configuration.
- `decider_recipe_utils.py` — shared input validation, strict tokenization,
  hashing, manifests and training provenance.
- `verify_decider_training.py` — reloads the trained model, checks architecture
  and export equality, and compares baseline/trained evaluation metrics.
- `tests/probe_trainium_compilation.py` — small compiled Neuron forward/backward
  and AdamW update check; launched by the runner's `preflight` command.
- `tests/probe_trainium_gated_delta.py` — checks gated-delta outputs and gradients
  on Neuron against a CPU reference; launched with `gdn-probe`.
- `tests/probe_trainium_fsdp.py` — checks a four-rank FSDP AdamW update against a
  CPU reference; launched with `fsdp-probe`.
- `tests/compare_decider_checkpoints.py` — compares final safetensors from
  uninterrupted and resumed runs for exact equality; launched with `compare`.
- `tests/test_decider_recipe.py` — dependency-free data, tokenizer, architecture
  and manifest contract tests.
- `tests/test_decider_dataset_download.py` — offline download retry and failure
  tests using mocked HTTP transport.
- `tests/test_decider_model_preparation.py` — offline Hub download failure,
  output-preservation and rerun tests.
- `tests/test_decider_training.py` — CPU numerical, checkpoint/resume and mocked
  distributed orchestration tests.

## Supported environment

Use an existing Trainium2 host with **four logical Neuron devices at LNC=2**.
The tested campaign host was `trn2.3xlarge` (96 GiB aggregate accelerator memory;
24 GiB per logical-device memory domain). Four-rank FSDP2 distributes training
state. Aggregate memory is not one device allocation pool. Trn1/XLA is unsupported.

Install the AWS Deep Learning AMI for Neuron, Ubuntu 24.04, SDK 2.33 native profile:
Python 3.12, torch 2.14.0, torch-neuronx 2.14.3.0.1+b1b6c66,
neuronx-cc 2.28.7631.0+8f7b0cf5, NKI 0.7.0+37713544100.gfede04bb,
driver 2.31.7.0 and runtime 2.35.30.0-d2725ccb8.
Select the image for your AWS region; no account-specific AMI ID is embedded.
A different native SDK requires revalidation, especially the internal RNG API.

Provide git, internet access for setup/downloads, a private writable disk and
sufficient CPU RAM. The original run used a trn2.3xlarge host. Allow at least 70 GiB
free disk for downloads, merged weights, a final model and one resumable checkpoint;
each retained checkpoint adds roughly 23 GB. Check space before longer runs.
The base Neuron environment and its source directories must remain available.
Setup creates an application venv and exposes base packages without upgrading them.

## Quick start

Run as an ordinary user. Keep this source directory separate from the writable
work directory; source can be read-only. Commands work with paths containing spaces.

```bash
PACKAGE="$PWD"  # Run this assignment from the extracted recipe folder.
export DECIDER_WORK="$HOME/decider-work"
python3 setup_trainium_environment.py \
  --base-python /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python \
  --work "$DECIDER_WORK"
python3 "$PACKAGE/prepare_decider_dataset.py" synthetic --output "$DECIDER_WORK/data/train.jsonl"
python3 "$PACKAGE/prepare_decider_dataset.py" audit --source "$DECIDER_WORK/data/train.jsonl"
bash "$PACKAGE/run_decider_training.sh" preflight
bash "$PACKAGE/run_decider_training.sh" gdn-probe
bash "$PACKAGE/run_decider_training.sh" fsdp-probe
bash "$PACKAGE/run_decider_training.sh" prepare \
  --data "$DECIDER_WORK/data/train.jsonl" \
  --output "$DECIDER_WORK/results/prepare" --pad-to 768
bash "$PACKAGE/run_decider_training.sh" train \
  --merged "$DECIDER_WORK/results/prepare/merged" \
  --data "$DECIDER_WORK/data/train.jsonl" \
  --output "$DECIDER_WORK/results/smoke" \
  --steps 1 --epochs 1 --checkpoint-every 1
bash "$PACKAGE/run_decider_training.sh" verify \
  --final "$DECIDER_WORK/results/smoke/final" \
  --checkpoint "$DECIDER_WORK/results/smoke/step-1" \
  --merged "$DECIDER_WORK/results/prepare/merged" \
  --architecture "$DECIDER_WORK/results/prepare/architecture.json" \
  --data "$DECIDER_WORK/data/train.jsonl" \
  --training-data "$DECIDER_WORK/data/train.jsonl" \
  --split training --output "$DECIDER_WORK/results/smoke-verification.json"
```

The preflight is a synthetic compiled optimizer update, not a model test. GDN and
FSDP probes are small numerical checks. `prepare` performs CPU merge/reload parity
and checks every prompt for excessive length. `train` executes the actual model
on Neuron with no CPU training fallback. `verify` independently reloads weights,
checks the architecture and exact checkpoint export, and compares CPU evaluation
metrics. A passing synthetic test does not guarantee a full model fits.

## Your dataset and longer runs

Use UTF-8 JSONL, one choice example per line:

```json
{"kind":"choice","state":"Text to classify","instructions":"Choose a category","options":[["a","Description of category a"],["b","Description of category b"]],"label":0,"task":"my_task"}
```

Labels index the canonical options. The upstream collator shuffles options and
remaps the label for training. This package supports text-only choice examples,
2–32 distinct options, uniform weights, no instruction variants, and up to 64 MiB
of input per file. It rejects missing fields, duplicate normalized states, and
oversized prompts. It does not truncate. Different option counts can compile
additional graphs. Group near-duplicates/documents before splitting your own data.

The optional original 10-row fixture is downloaded on demand, checked against
SHA-256 `94ad9da34e2d11f34e65d1513c1e962c871d50b105bb3b994e933b77d49c4202`:

```bash
"$DECIDER_WORK/venv/bin/python" "$PACKAGE/prepare_decider_dataset.py" otel --output "$DECIDER_WORK/data/otel.jsonl"
```

It is sourced from farbodtavakkoli/OTel and is not bundled. Review its content and
terms before redistribution. Both supplied fixtures are pipeline smoke tests;
training-set scores are not evidence of generalization.
Download failures report the source URL and retry guidance, then exit nonzero.
Responses are closed and no output file is created on a failed download.
The optional downloader uses pinned `urllib3==2.8.0` from the application venv.
Its GET retry policy permits at most four attempts across connection, HTTP and
body-read failures. Only connection errors, timeouts, interrupted reads, HTTP 429
and HTTP 5xx are retried. The first retry is immediate; subsequent delays use
exponential backoff (factor 1 second) with up to 0.5 seconds of jitter, capped at
8 seconds. Connect/read timeouts are 10/60 seconds per attempt. `Retry-After` is
disabled to keep the delay bounded and avoid implicit retries of other statuses.
Redirects, certificate failures, other HTTP errors and checksum failures are not
retried. Body reads retain the 1 MiB limit.

For a larger run, pass `--epochs`, `--steps` (maximum total updates), `--batch-size`
(global real examples, default 5), `--learning-rate` (default 5e-6), `--pad-to`,
`--seed` and `--checkpoint-every`. Samples run in file order each epoch; shuffle
or group your file before the run if required. Epoch boundaries retain the final
partial batch. One example per rank per microbatch; dummy slots have zero loss.
Loss scaling compensates for FSDP averaging, followed by global norm clipping at 1.
AdamW uses betas (0.9,0.95), weight decay 0, no scheduler, FP32 parameter/optimizer
storage, BF16 torso computation and FP32 pointer scoring. Activation checkpointing
and per-layer compilation are enabled. Calibration temperatures are reset to 1.

This bounded recipe exports the **last completed step**, not a validation-selected
best model. Use disjoint validation data with `verify --split validation`, choose
your recipe using validation only, then run a frozen test once. Pass the training
file through `--training-data`; exact normalized state overlap is rejected for
validation/test. That check cannot detect semantic or document-level leakage.
Recalibrate on a separate calibration split before relying on confidence scores.

## Resume and model loading

Resume into a new output directory with the same initialization, source, dataset,
versions, topology and numerical settings. Increase `--steps`/`--epochs` as needed:

```bash
bash "$PACKAGE/run_decider_training.sh" train \
  --merged "$DECIDER_WORK/results/prepare/merged" \
  --data "$DECIDER_WORK/data/train.jsonl" \
  --resume "$DECIDER_WORK/results/smoke/step-1" \
  --output "$DECIDER_WORK/results/continued" --steps 2 --epochs 1
```

Checkpoints contain rank-local parameters, AdamW state, RNG state and a recipe
signature; completion metadata is mandatory. They are version-specific. Pointer
head dropout uses a deterministic training-position seed to support exact resume.
The shared checkpoint module retains the existing on-disk fields. Resume signatures
also cover the extracted checkpoint and training-step modules. Source changes,
including module renames and this refactor, reject continuation from an earlier source signature;
use the matching original source when resuming an older checkpoint.
Use `tests/compare_decider_checkpoints.py` to compare final safetensors from matched uninterrupted
and resumed runs. Allow disk space for both runs before doing this test.

Load an exported model with this package's `decider_model_io.load_full(Path(...))`.
**Do not use the upstream default loader**: this full-backbone format differs
from the adapter release and must restore the trained local torso.

## Validation evidence

The renamed, refactored recipe passed a fresh ordinary-user test on November 2,
2026 (UTC), using a read-only source folder and new application environment,
upstream checkout, model cache, compilation cache and output directory.
All 37 local tests and all three hardware probes passed. One full-model AdamW
update completed on four Neuron ranks with all 326 distributed parameter tensors
changed, including embeddings and the pointer head. Peak device allocation was
about 13.8 GB per rank.

Independent CPU reload verified finite weights, unchanged architecture and exact
equality with the saved checkpoint shards. The synthetic fixture contains ten
rows; the one-step quick start trains on its first five rows. Evaluation over all
ten rows changed loss from 0.8731 to 0.2477 and accuracy from 70% to 90%. These
fixture metrics are not an independent quality evaluation. Setup took 25 seconds,
model preparation 101 seconds, training including compilation/checkpoint/export
380 seconds, and verification 50 seconds in this run; timings are observations,
not guarantees.

The first attempt caught a module-rename regression in the `--data` CLI option
and setup's data directory name. Both were fixed before the successful fresh
restart, and a CLI regression test was added. The packaged resume path has CPU
unit-test coverage but has not been separately hardware-tested since the refactor.
Detailed validation and dependency-audit reports are retained outside the recipe
source. This run did not create a release ZIP or deploy a serving model.

Preparation creates its output directory only after the pinned Hub snapshot
downloads successfully. Connectivity, access and missing-revision failures exit
nonzero with actionable guidance; cached downloads remain available for a retry
with the same output path. Existing output directories are preserved and rejected
before downloading. Later model-preparation failures can still leave an output
directory.

## Local tests

Run the dependency-free contract tests with:

```bash
python3 "$PACKAGE/tests/test_decider_recipe.py"
```

Run all four local test modules using the application environment:

```bash
"$DECIDER_WORK/venv/bin/python" -B -m unittest discover \
  -s "$PACKAGE/tests" -p 'test_*.py'
```

These tests run without a Neuron process group or model download. The hardware
probes and checkpoint comparison are separate commands and are not collected by
this test pattern. Production training explicitly initializes the Neuron backend.

## Failure diagnostics

Failures exit nonzero. Capture stdout/stderr to a private work log; rank logs are
inside each run directory. No existing output is silently overwritten. Keep
compiler and failure logs for diagnosis; do not publish private data in them.
The reference-PyTorch Gated DeltaNet warning is expected on this native path;
do not install CUDA-only flash-linear-attention to suppress it. Native compiler
debug, HLO, NEFF and local caches are explicitly placed beneath the private work
directory to avoid collisions with root-owned defaults from earlier jobs.

## References

- https://github.com/strands-labs/strands-decider/tree/13faf5214f7cbea07d005278f1aeaf3973611eba/training
- https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/tree/216d1ccb46a9e9b036eb9187b248db98480f77d4
- https://github.com/aws-neuron/torch-neuronx/tree/d9d8423807874b11c8d59242b09378e68bc9a504
- https://huggingface.co/blog/hf-torchneuron
- https://aws.amazon.com/blogs/machine-learning/train-qwen3-on-aws-trainium-with-torchtitan/
- https://urllib3.readthedocs.io/en/2.8.0/reference/urllib3.util.html#urllib3.util.Retry

The AWS examples inform native device/compilation setup, not the Decider loss or
architecture. DecisionFormer informed the initial full-weight investigation; this
package does not import its classifier head or change the upstream optimizer family.
