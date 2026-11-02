# Standalone Trainium cross-encoder reranker fine-tuner

`train_reranker_standalone.py` fine-tunes
`nvidia/llama-nemotron-rerank-1b-v2` with explicit binary cross entropy on raw
logits, using one positive and five hard negatives per query. It runs a native
TorchNeuron training loop with `model.to("neuron")`, baseline evaluation,
epoch checkpoint selection by NDCG@10, and independent export/reload checks.
The model is pinned to revision `828765652b05bd439c9789d2a6d093db1caa1443`.

**Tested hardware profile:** Trainium2, one visible logical device, LNC2; Linux
x86_64 and Python 3.12. This package adapts the presentation of the
[OTel sentence-transformers sample](https://github.com/farbodtavakkoli/OTel/tree/59ed2bc34ae4a5e243e7601272bcc908778da28d/training/reranker/sentence_transformers).
Its CUDA/NCCL launcher and GPU dependencies do not apply to this native backend.

## Files

- `train_reranker_standalone.py` — thin command-line entry point and orchestration.
- `reranker_data.py` — dataset validation, deterministic splitting and audit output.
- `reranker_service.py` — model adaptation, metrics, training, checkpoint export and verification.
- `requirements_reranker.txt` — categorized pins from the tested environment.
- `readme_reranker.md` — prerequisites, setup and execution guide.
- `OTel_reranker_sample_100.jsonl` — unchanged 100-row sample; default `--data`.

The shared source folder contains only these six files. Environments, caches,
logs and checkpoints are created at runtime; no output folder is shipped.

## Setup

### Native TorchNeuron prerequisites

The tested software profile uses PyTorch `2.14.0`, native TorchNeuron
`2.14.3.0.1+b1b6c66`, compiler `2.28.7631.0+8f7b0cf5`, NKI
`0.7.0+37713544100.gfede04bb`, runtime `2.35.30.0-d2725ccb8`, and Python 3.12.
The application retains Transformers commit
`50e35cc0ca18ec19b80653690de553e6decf74f5`.
Choose a current Neuron image or host environment that supplies compatible
versions; image identifiers differ by region and release. Use a Trn2 host for
this tested recipe. Other Trainium generations and software profiles require
separate validation. A non-root user with access to the Neuron devices can run
the package. Git and outbound access to GitHub, Hugging Face and the listed
package indexes are required. Setup does not create or modify cloud resources.
A compatible Neuron system driver/runtime and the native base environment must
already be installed. This profile does not require the older locally built
TorchNeuron 2.13 wheel.

See the [native TorchNeuron guide](https://huggingface.co/blog/hf-torchneuron)
and [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html).
Do not substitute a legacy XLA package for this native backend.

From inside this folder, run Bash on the compatible Linux Trainium host:

```bash
set -euo pipefail
recipe_dir="$(pwd -P)"
base_env="${NEURON_BASE_ENV:?Set NEURON_BASE_ENV to the compatible native TorchNeuron environment}"
work_dir="${RERANKER_WORK_DIR:-$HOME/reranker-work}"
base_env="$(cd "$base_env" && pwd -P)"
mkdir -p "$work_dir/tmp" "$work_dir/pip-cache" "$work_dir/weights" "$work_dir/compiled"
work_dir="$(cd "$work_dir" && pwd -P)"
export recipe_dir base_env work_dir
export TMPDIR="$work_dir/tmp" PIP_CACHE_DIR="$work_dir/pip-cache"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$base_env/bin/python" - <<'PYSETUP'
import importlib.metadata as m
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

base = Path(os.environ["base_env"]).resolve()
work = Path(os.environ["work_dir"]).resolve()
expected = {"torch": "2.14.0", "torch-neuronx": "2.14.3.0.1+b1b6c66",
            "neuronx-cc": "2.28.7631.0+8f7b0cf5", "nki": "0.7.0+37713544100.gfede04bb"}
versions = {name: m.version(name) for name in expected}
assert sys.version_info[:2] == (3, 12), sys.version
assert versions == expected, versions
venv = work / "venv"
assert not venv.exists(), "Choose a new work directory; environment already exists"
subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
site = sysconfig.get_paths()["purelib"]
# addsitedir also registers the AMI's editable native backend installation.
(venv / "lib/python3.12/site-packages/neuron-base.pth").write_text(
    f"import site; site.addsitedir({site!r})\n")
(work / "base-environment.json").write_text(json.dumps(
    {"base": str(base), "site": site, "versions": versions}, indent=2) + "\n")
PYSETUP
source "$work_dir/venv/bin/activate"
export PATH="$work_dir/venv/bin:$base_env/bin:$PATH"
python -m pip install -r "$recipe_dir/requirements_reranker.txt"
python -m pip check
python -m pip freeze > "$work_dir/environment.lock.txt"
python -c 'import torch, torch_neuronx; print(torch.__version__, torch.neuron.is_available())'
```

The application environment references the base through `site.addsitedir` and
installs application packages locally; it does not upgrade the shared base.
Keep the base environment and its editable source installation available at the
same paths. The base is an explicit external prerequisite, not another model's
application code. The requirements retain the previous environment pins except
for the new native backend, matching torchvision 0.29, and grpcio-tools 1.76
compatible with the application protobuf pin. Set `HF_TOKEN` in the calling
environment only when Hugging Face Hub authentication is needed. Do not place
tokens in this folder, command examples, logs or generated configuration.

## Data

Each JSONL row contains an `anchor`, a `positive`, and `negative_1` through
`negative_5`. Extra columns, including `answer`, are ignored:

```json
{"anchor":"query","positive":"relevant passage","negative_1":"irrelevant passage","negative_2":"irrelevant passage","negative_3":"irrelevant passage","negative_4":"irrelevant passage","negative_5":"irrelevant passage"}
```

Each query expands into `1 + --n_neg` pairs, labeled 1.0 for the positive and
0.0 for negatives. Inputs use NVIDIA's exact template
`question:{query} \n \n passage:{document}`, left padding, and truncation at
`--max_len`. Empty text, contradictory labels, and query or document overlap
between the seeded training/validation splits cause an error. JSON lines must
be objects. A full run with only one planned optimizer update is rejected before
model loading because its warmup learning rate would be zero.

The sample comes from OTel commit `2243c637c6b3d3d875b798deb947ab53b601a0d4`.
SHA-256: `b64c55a59e7071e8909bdcf46446de16b717f249bd22b9562429bc7a48a3c649`.
Default splitting yields 99 training queries, 594 pairs and one validation query.
There is no test set. One validation query checks execution, not general quality.
The script resolves its default dataset beside itself, regardless of working directory.

## Run

Data preparation uses only the Python standard library and can run locally:

```bash
python3 train_reranker_standalone.py --prepare_only --out /tmp/reranker-preparation
```

On a compatible, available Trainium device, continue in the setup shell:

```bash
export NEURON_RT_VISIBLE_CORES=0 NEURON_LOGICAL_NC_CONFIG=2 OMP_NUM_THREADS=4
export HF_HOME="$work_dir/weights/hf-home" XDG_CACHE_HOME="$work_dir/xdg-cache"
export NKI_ENABLE_TRACE_CACHE=0
unset PYTHONPATH
cd "$work_dir"
python -I -u "$recipe_dir/train_reranker_standalone.py" \
  --weights "$work_dir/weights" --cache "$work_dir/compiled" \
  --out "$work_dir/smoke-01" --smoke > "$work_dir/smoke-01.log" 2>&1
```

Smoke runs the full model with two optimizer updates, evaluation, checkpoint
export and reload. It uses physical microbatches with no scheduler warmup.
It is separate from the reference's `--test_mode` (a 200-row cap), which is not
implemented here. Full two-epoch training:

```bash
python -I -u "$recipe_dir/train_reranker_standalone.py" \
  --weights "$work_dir/weights" --cache "$work_dir/compiled" \
  --out "$work_dir/full-01" > "$work_dir/full-01.log" 2>&1
```

Paths may contain spaces; relative `RERANKER_WORK_DIR` values are resolved at setup.
The trainer directs native compiler debug artifacts to `--cache/debug`, avoiding
shared `/tmp/neuron_backend` permissions from another user.
Use a new `--out` for every run; existing directories are refused. This recipe
runs one process; it has no distributed launcher or checkpoint-resume support.

### Batch geometry

`--batch` is the logical training batch (64); `--microbatch` is the physical
training/evaluation batch (4). Two logical batches accumulate before an update:

```text
pairs             = training_queries × (1 + n_neg)
logical_batches   = floor(pairs / batch)
updates_per_epoch = ceil(logical_batches / gradient_accumulation)
nominal_update    = batch × gradient_accumulation = 128 pairs
```

Defaults drop 18 pairs per epoch and yield five updates per epoch; the final
update contains 64 pairs. Microbatch mean losses are weighted by their share
of the actual update group. This preserves nominal batch and loss weighting;
bitwise or gradient equivalence to a physical batch of 64 has not been proved.
The default full run uses AdamW (betas 0.9/0.999, epsilon 1e-8, weight decay 0),
clipping at 1.0, BF16 weights, gradient checkpointing and a linear scheduler
with 10% warmup rounded up. Ten optimizer calls include one at zero learning rate.

## Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--data` | Sibling sample JSONL | Input rows |
| `--out` | `output` | New output directory, relative to the working directory |
| `--weights` | `./weights` | Model download cache |
| `--cache` | `./compiled` | Native compiler cache and debug-artifact root |
| `--max_len` | `1024` | Padded/truncated sequence length |
| `--batch` | `64` | Logical training batch |
| `--microbatch` | `4` | Physical batch; must divide logical batch |
| `--gradient_accumulation` | `2` | Logical batches per update |
| `--epochs` | `2` | Full-run epochs |
| `--lr` | `1e-5` | Peak learning rate |
| `--n_neg` | `5` | Negative columns per query |
| `--eval_frac` | `0.003` | Held-out fraction |
| `--seed` | `42` | Split and training shuffle seed |
| `--prepare_only` | off | Audit and split without model downloads or training |
| `--smoke` | off | Two full-model optimizer updates and verification |

The model and revision are constants because the compatibility adaptations are
model-specific. Unlike the GPU reference, this recipe does not offer a generic
`--model` switch. `HF_HOME` is an environment variable; `--weights` explicitly
controls the model download cache.

## Output

Under `--out`, preparation writes `config.json`, `audit.json`, and `split.json`.
The split manifest contains SHA-256 row identifiers derived from required
training fields; it does not copy dataset text or ignored extra columns.
Saved configuration and environment evidence omit host-specific absolute paths,
cloud identifiers and credentials.
Training adds environment/placement evidence, token lengths, mask comparison,
`baseline.json`, `steps.json`, epoch metrics, `checkpoint-<epoch>/`, and `final/`.
Each checkpoint contains weights, tokenizer and custom model code; optimizer
resume state is not saved. There are no TensorBoard logs; stdout is JSON step
records, and the examples redirect it to a log outside the source folder.

`final/` contains the selected checkpoint (highest validation NDCG@10; ties
keep the earlier epoch). `result.json` with `status: completed` confirms finite
checks, changed parameters, independent tensor hash verification and reloaded
inference comparison. Exceptions inside preparation/training write
`failure.json`; abrupt process termination may only appear in the shell log.

## Notes

- Public-user validation passed in the tested software profile with a
  non-root user, read-only source folder,
  paths with spaces, a relative work directory, and `python -I` outside the
  source directory. No Hugging Face credentials were needed for the sample run.
  Both the two-update smoke and full two-epoch command passed. The full run
  made 10 optimizer calls (nine at nonzero learning rate), changed 123 tensors,
  and passed finite checks, checkpoint hash equality and reloaded Neuron
  inference. `neuron` resolved to `neuron:0` throughout. The full training,
  selection and reload timer was 235.19 seconds, excluding initial load and
  baseline.
- The refactored six-file package was replayed from a clean copy on November
  2, 2026. A new application environment passed `pip check`, isolated
  `python -I` preparation processed all 100 rows, and the two-update full-model
  smoke changed 123 tensors and passed checkpoint export, independent reload,
  tensor-hash equality and reloaded Neuron inference. Machine-readable validation evidence is retained outside this standalone
  package. The subsequent default two-epoch job completed
  all 10 optimizer updates and passed the same artifact and reload checks.
- Historical baseline and selected NDCG@10/MRR@10 were 1.0 on one held-out
  query. No quality improvement was demonstrated. Evaluation uses raw logits,
  optimistic ties (only strictly greater scores outrank the positive), and a
  cutoff of 10. It does not report MAP as the GPU sample does.
- The retained static bidirectional attention-mask adaptation has a CPU forward
  parity check. Its necessity versus the unmodified native path and gradient
  equivalence were not established. The save-time compatibility workaround
  preserves NVIDIA's `temperature` score scaling while filtering it from
  Transformers generation-parameter validation.
- Select a filesystem with capacity for downloaded weights, environment,
  compiler caches, every epoch checkpoint and final export. Generated artifacts
  are excluded from the sharing layout. No `.gitignore` is included.
- The trainer imports only this file's helpers and installed dependencies.
  Cloud account, region, image and instance identifiers are intentionally not
  part of the package.
