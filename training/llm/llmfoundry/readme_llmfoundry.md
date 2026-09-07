# `training/llm/llmfoundry` — YAML-driven training with LLM Foundry

## 1. Overview & when to use

YAML-driven continued pre-training and instruction finetuning with
[LLM Foundry](https://github.com/mosaicml/llm-foundry), the MosaicML/Databricks codebase
that trained MPT and DBRX. Foundry sits on top of [Composer](https://github.com/mosaicml/composer)
(see `training/llm/composer` in this repo) and adds the pieces a pre-training run actually
needs: a StreamingDataset pipeline that reads shards straight from object storage, a
finetuning dataloader with sequence packing, in-context-learning eval, and a callback that
writes Hugging Face checkpoints as you go. You configure all of it in one YAML and launch
with `composer scripts/train/train.py <yaml>`.

Files in this folder:
- `train_llm_llmfoundry.py` — thin launcher: picks a recipe YAML, applies per-run overrides, checks the env, execs `composer`.
- `yamls/finetune_chat_sft.yaml` — instruction finetuning from an HF checkpoint on local chat JSONL; wired to the shipped sample.
- `yamls/continued_pretrain.yaml` — continued pre-training / domain adaptation on MDS-converted text.
- `data/OTel_LLM_sample_10.jsonl` — 10-row chat sample so the SFT recipe is self-contained.
- `requirements_llmfoundry.txt` — install paths (Docker / venv / ROCm), and what needs a git source.
- `readme_llmfoundry.md` — this file.

> **Tested topology:** Smoke-tested on **1x AMD Instinct MI355X (gfx950, ROCm 7.2)** and
> then on **8x MI355X with FSDP `FULL_SHARD` (ROCm 7.2.4 / RCCL)** — see
> [§8a MI355X (ROCm 7.2) — tested](#8a-mi355x-rocm-72--tested) for the exact
> install, the run command and the outcome (**works with changes**), and the
> [8-GPU subsection](#8-gpu-run-8x-mi355x-rocm-724) for the
> multi-GPU launch, the batch geometry Foundry logged and the per-rank VRAM delta that
> proves sharding. **NVIDIA H100 (CUDA 13.0, driver 580) is also verified** on
> **1x H100 80GB (single-GPU SFT smoke, `flash_attention_2`)** — see
> [§8b NVIDIA H100](#8b-nvidia-h100-cuda-130-driver-580) for the
> install, the torch reconciliation and the run evidence (**works with changes** — the
> change is the model, not the code). Written against the LLM Foundry upstream README,
> `scripts/train/README.md`, the shipped `yamls/finetune/*.yaml`, and the
> `TrainConfig` / `finetuning/tasks.py` / `hf_checkpointer.py` sources on `main` for the
> pinned versions below. Targets a single node of 8x H100 80GB (the support matrix upstream lists
> A100 and H100 on torch 2.7 / CUDA 12.8). YAML schemas here move between minor releases —
> always run `--dry-run` first and diff against `scripts/train/yamls/finetune/` in your
> checkout before a long job.

### An honest note on the Mosaic stack

LLM Foundry's development slowed sharply after the Databricks acquisition. The README's
headline models (MPT-7B/30B, DBRX) are from 2023–2024, the repo now largely serves
Databricks Mosaic AI Training, and community activity is far below what you get around the
Hugging Face ecosystem. **This is not the default choice for fine-tuning a chat model.** For
that, `training/llm/unsloth`, `training/llm/fsdp`, `training/llm/peft`, or the recipe-driven
`training/llm/llamafactory` / `training/llm/axolotl` will be faster to stand up and much easier
to get help with.

What LLM Foundry still does better than anything else in this repo:

- **Continued pre-training on data that does not fit on disk.** The MDS/StreamingDataset
  path (shard once, stream from S3/GCS/OCI to any cluster size, deterministic resumption
  mid-epoch) is genuinely production-grade and is the reason to be here. If you are feeding
  a model billions of tokens of domain text, this is the strongest option in the repo.
- **Device-count-agnostic configs.** `global_train_batch_size` fixes the optimization math;
  `device_train_microbatch_size: auto` handles execution. The same YAML gives the same
  training result on 8 or 64 GPUs, which makes runs reproducible across clusters.
- **HF checkpoints written during training.** The `hf_checkpointer` callback emits a
  ready-to-serve Hugging Face folder at intervals — no post-hoc conversion step, no
  half-day discovering your `.pt` will not load.
- **Sequence packing with `packing_ratio: auto`.** It profiles your dataset and picks the
  highest packing ratio with near-zero waste. On short instruction data this is a large,
  free throughput win.

If your run is "8 GPUs, a few hundred thousand chat rows, ship it this week", use another
folder. If it is "adapt a base model to 20B tokens of our corpus, then instruction-tune it,
reproducibly", this is the right tool.

## 2. Install

LLM Foundry is used from a **git checkout**, because the launcher, data-prep and conversion
scripts live in `scripts/` and are not part of the wheel. Point the launcher at the
checkout with `--foundry-dir` (default `./llm-foundry`).

### NVIDIA (CUDA)

Upstream strongly recommends their Docker image:

```bash
docker pull mosaicml/llm-foundry:2.7.0_cu128-latest
```

Note the trap: the `llm-foundry` images ship the **dependencies only**, not the package.
Inside the container (or in a venv, if you are going bare):

```bash
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e ".[gpu]"        # or `pip install -e .` with no NVIDIA GPU
```

Without Docker, install `cmake packaging torch` first — `setup.py` requires them to be
present before it runs. See `requirements_llmfoundry.txt` for both paths plus the
`flash-attn --no-build-isolation` and TransformerEngine (FP8) notes.

### AMD / ROCm (beta upstream)

Unlike Composer, LLM Foundry documents an AMD path: the upstream README has an
"AMD (BETA support)" section, backed by MosaicML's published MI250 training work
([mosaicml.com/blog/amd-mi250](https://www.mosaicml.com/blog/amd-mi250), and the follow-up
Databricks post on training at scale with MI250s). The documented setup is a venv install
**without** the `[gpu]` extra (it is CUDA-only), a ROCm torch build, and the ROCm
flash-attention fork:

```bash
python3 -m venv llmfoundry-venv-amd && source llmfoundry-venv-amd/bin/activate
pip install cmake packaging torch
git clone https://github.com/mosaicml/llm-foundry.git
cd llm-foundry
pip install -e .
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# then install ROCm flash-attention per the fork's instructions
```

Honest caveats, straight from upstream: it is beta, they state they are not actively
testing AMD systems, there is no Docker image "where everything works perfectly", and they
needed package pin adjustments (e.g. `numpy==1.23.5`). The ROCm 5.4.2 index URL in their
instructions is dated — expect to adapt versions. This folder's YAMLs also default to
`attn_implementation: flash_attention_2`, which on ROCm requires that fork to actually be
installed; drop the line to fall back to the model's default attention.

> **Do not follow the recipe above verbatim on a modern ROCm box — it does not work.**
> The order shown (`pip install cmake packaging torch`, then `pip install -e .`, then
> the ROCm torch) is exactly backwards for a 2026 stack: `pip install -e .` re-resolves
> `torch>=2.7.0,<2.7.1` from PyPI and installs the **CUDA** build plus the whole
> `nvidia-*` runtime, and the ROCm index in the last line (`rocm5.4.2`) no longer exists.
> More fundamentally, torch 2.7 has **no ROCm 7.x wheel at all**, so the pin cannot be
> satisfied on ROCm 7.2 — see §8a for the staged install that does work, and
> `requirements_llmfoundry.txt` PATH C-ROCm7.2 for the copy-pasteable commands.

## 3. Environment & secrets

Put a `dev.env` in this folder:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`train_llm_llmfoundry.py` loads it with `load_dotenv("dev.env")` and the `composer`
launcher passes the parent environment to every rank. Both shipped YAMLs set
`use_auth_token: true` on the model block, which is what makes Foundry read `HF_TOKEN` when
pulling a gated checkpoint such as Llama. The same token is used by
`convert_composer_to_hf.py` if you pass `--hf_repo_for_upload`.

`dev.env` is git-ignored at the repo root. **Never commit a token**; rotate it on the Hub if
one ever lands in a commit.

## 4. Data

The two recipes take different data, because they are different jobs.

### 4a. Instruction finetuning (`--recipe sft`) — chat JSONL

Same format as the rest of this repo, one conversation per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The folder ships a 10-row sample at `data/OTel_LLM_sample_10.jsonl` and
`yamls/finetune_chat_sft.yaml` points at it, so the smoke run works with no data prep.

Foundry's `finetuning` dataloader accepts the `messages` schema **natively** — it detects
the example type, renders each turn with `tokenizer.apply_chat_template`, and builds labels
from it. Rules it enforces (`llmfoundry/data/finetuning/tasks.py`):

- Roles must be one of `user`, `assistant`, `system`, `tool`.
- The **last** message must be `assistant`.
- The same role may not repeat back-to-back.
- Each message has exactly two keys: `role` and `content`.
- **A chat example must have `messages` as its only top-level key.** `_get_example_type`
  rejects rows with extra columns. The shipped sample carries metadata columns besides
  `messages` (`unmask`, `flow`, `source_id`, `source_repo`, `source_spec_id`,
  `source_version`), so the YAML sets
  `preprocessing_fn: llmfoundry.data.finetuning.tasks:messages_format_preprocessor` —
  an upstream-provided function that extracts just the `messages` key. Keep that line for
  any JSONL with metadata columns; drop it if your rows are pure `{"messages": [...]}`.

The alternative schema, if you prefer flat pairs, is `{"prompt": "...", "response": "..."}`
(two keys, both strings). Foundry picks whichever it sees; do not mix them in one file.
Accepted file extensions: `.jsonl`, `.json`, `.csv`, `.parquet`.

To swap in your own data: put your JSONL somewhere, then either edit
`variables.data_local` / `variables.train_file` in the YAML or pass
`--data-local /path/to/dir --extra variables.train_file=my.jsonl`. The eval loader reuses
the train split as a documented smoke choice (the 10-row sample has no held-out split);
point it at a real validation file for any serious run.

Which tokens generate loss is controlled by two dataset keys, not by the data:

| Setting | Values | Meaning |
|---|---|---|
| `target_prompts` | `none` (default), `all`, `length>=XX` | Whether user turns are training targets |
| `target_responses` | `last` (default), `all` | Whether every assistant turn is a target, or only the final one |

For multi-turn conversations where you want to learn from every assistant reply, set
`target_responses: all`. The defaults (`none` / `last`) supervise only the final answer.

### 4b. Continued pre-training (`--recipe pretrain`) — MDS shards

The `text` dataloader does not read JSONL; it reads MosaicML StreamingDataset (`.mds`)
shards of pre-concatenated, fixed-length token blocks. Convert first:

```bash
cd llm-foundry/scripts

# From a HF dataset (C4 shown; --concat_tokens packs to full-length blocks, no padding)
python data_prep/convert_dataset_hf.py \
  --dataset allenai/c4 --data_subset en \
  --out_root ./my-mds-data --splits train val \
  --concat_tokens 4096 --tokenizer meta-llama/Llama-3.1-8B \
  --eos_text '<|end_of_text|>'
```

`--out_root` can be an object-store URI (`s3://...`, `gs://...`, `oci://...`). Then set
`variables.data_remote` to that URI and `variables.data_local` to a fast local cache
directory; Foundry streams remote to local automatically. If `data_remote` is blank, the
shards must already exist in `data_local`.

Sanity-check the shards before training:

```bash
python ../llmfoundry/data/text_data.py --local_path ./my-mds-data --split val
```

### 4c. Optional: MDS for finetuning data too

Chat JSONL can also be converted to MDS, which is worth doing once your instruction set is
large or lives in object storage:

```bash
python data_prep/convert_finetuning_dataset.py \
  --dataset json --data_files /path/to/train.jsonl \
  --splits train \
  --skip-preprocessing \
  --out_root s3://my-bucket/my-sft-data
```

`--skip-preprocessing` is the flag to use when your rows are already in `messages` (or
`prompt`/`response`) form — without it the script demands a registered or explicit
`--preprocessor` and errors out. Chat rows are written to a `messages` column. Then swap the
`hf_name`/`hf_kwargs` block in the SFT YAML for:

```yaml
    remote: s3://my-bucket/my-sft-data
    local: /tmp/mds-cache/
    split: train
```

## 5. Run

Some commands below use two placeholders — set them once to suit your machine:

```bash
# Set these to suit your machine
export OUTPUT_DIR=/path/to/outputs     # checkpoints and run artifacts
export HF_HOME=/path/to/hf_cache       # Hugging Face model cache
```

Always dry-run first — it prints the exact `composer` command and validates that the
launcher, the checkout and the config all exist:

```bash
python3 train_llm_llmfoundry.py --recipe sft --dry-run
```

Smoke run against the shipped sample (the YAML already points at `./data`):

```bash
python3 train_llm_llmfoundry.py --recipe sft --gpus 8 \
  --max-duration 20ba --run-name sft-smoke
```

Instruction finetuning on real data:

```bash
nohup python3 train_llm_llmfoundry.py \
  --recipe sft \
  --foundry-dir ./llm-foundry \
  --gpus 8 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data-local /path/to/data \
  --extra variables.train_file=train.jsonl \
  --max-seq-len 4096 \
  --max-duration 3ep \
  --global-batch-size 64 \
  --lr 1e-5 \
  --save-folder ./llmfoundry_run/checkpoints \
  --run-name llama31-8b-sft \
  > train_llm_llmfoundry.log 2>&1 &

tail -f train_llm_llmfoundry.log
```

Continued pre-training:

```bash
nohup python3 train_llm_llmfoundry.py \
  --recipe pretrain \
  --model meta-llama/Llama-3.1-8B \
  --data-local ./my-mds-data \
  --data-remote s3://my-bucket/my-mds-data \
  --max-duration 2000ba \
  --global-batch-size 512 \
  --lr 5e-6 \
  > train_llm_llmfoundry.log 2>&1 &
```

Anything not exposed as a flag goes through `--extra` as raw `key=value` overrides (this is
OmegaConf dotted-path syntax, and it must come last):

```bash
python3 train_llm_llmfoundry.py --recipe sft --extra \
  train_loader.dataset.packing_ratio=auto \
  train_loader.dataset.target_responses=all \
  fsdp_config.activation_checkpointing=false \
  eval_interval=100ba
```

Equivalent raw command, if you would rather skip the wrapper:

```bash
composer llm-foundry/scripts/train/train.py \
  yamls/finetune_chat_sft.yaml \
  variables.data_local=/path/to/data \
  max_duration=3ep
```

**What "working" looks like:** the launcher echoes the `composer ...` line; Foundry logs the
fully resolved config (so you can confirm your overrides landed), builds the tokenizer,
loader and model, warns if FSDP was requested on a single GPU, and then prints per-batch
blocks like:

```
[batch=12/500]:
    Train LanguageCrossEntropy: 1.8421
    Train Perplexity: 6.3095
    Train loss/train/total: 1.8421
```

plus throughput from `speed_monitor` and an ETA from `runtime_estimator`. With
`device_train_microbatch_size: auto` you may see it catch an OOM, halve the microbatch and
carry on — that is the feature, not a failure. For SFT, loss should drop noticeably in the
first few hundred batches. For continued pre-training it moves much more slowly; watch for
*spikes*, which is why `kill_loss_spike` is enabled in that recipe.

## 6. Arguments

Launcher flags (`python3 train_llm_llmfoundry.py --help`):

| Arg | Default | Meaning |
|---|---|---|
| `--recipe` | `sft` | `sft` (instruction finetuning) or `pretrain` (continued pre-training) |
| `--config` | none | Explicit YAML path, bypassing `--recipe` |
| `--foundry-dir` | `./llm-foundry` | Path to the llm-foundry checkout (needs `scripts/train/train.py`) |
| `--gpus` | autodetect | Processes to launch; omit and the `composer` launcher autodetects |
| `--model` | YAML value | HF repo id or local path (sets `variables.model_name`) |
| `--data-local` | YAML value | Data dir (sft; YAML default `./data`) or MDS cache root (pretrain; YAML default `./my-mds-data`) |
| `--data-remote` | YAML value | MDS object-store URI (pretrain) |
| `--max-seq-len` | YAML value | Sets `variables.max_seq_len` |
| `--max-duration` | YAML value | Composer Time string: `3ep`, `2000ba`, `10000000tok` |
| `--global-batch-size` | YAML value | Sets `global_train_batch_size` |
| `--device-microbatch-size` | YAML value | int or `auto`, sets `device_train_microbatch_size` |
| `--lr` | YAML value | Sets `optimizer.lr` |
| `--save-folder` | YAML value | Composer checkpoint directory |
| `--run-name` | YAML value | Sets `run_name` |
| `--extra` | none | Everything after this is forwarded verbatim as `key=value` overrides |
| `--dry-run` | off | Print the command and exit |

Note that `--save-folder` sets the *Composer* checkpoint directory only. The
`hf_checkpointer` callback has its own path; move it with
`--extra callbacks.hf_checkpointer.save_folder=...` if you relocate the run.

Key YAML fields:

| Field | Meaning |
|---|---|
| `model.name` | `hf_causal_lm` to load HF weights; `mpt_causal_lm` for the MPT architecture |
| `model.pretrained_model_name_or_path` | The checkpoint you are starting from |
| `model.attn_implementation` | `flash_attention_2` for HF models that support it |
| `model.init_device` | `mixed` = rank 0 loads real weights, others on meta, FSDP syncs. Avoids N CPU copies |
| `train_loader.name` | `finetuning` (prompt/response or chat) vs `text` (streaming pre-training blocks) |
| `train_loader.dataset.preprocessing_fn` | Optional row transform; here, the upstream `messages` extractor (see 4a) |
| `train_loader.dataset.packing_ratio` | `auto` profiles and packs sequences; big throughput win on short rows |
| `target_prompts` / `target_responses` | Which spans generate loss (see section 4a) |
| `global_train_batch_size` | The optimization math — fixed regardless of GPU count |
| `device_train_microbatch_size` | The execution — `auto` finds the largest that fits |
| `max_duration`, `eval_interval`, `save_interval` | Composer Time strings: `3ep`, `2000ba`, `10000000tok` |
| `precision` | `amp_bf16` (use `amp_fp8` only with TransformerEngine layers on H100) |
| `fsdp_config.*` | Sharding, `mixed_precision: PURE`, activation checkpointing, `state_dict_type` |
| `callbacks.hf_checkpointer` | Writes a Hugging Face folder during training |
| `save_num_checkpoints_to_keep` | Set it, or you will fill the disk |

## 7. Output

```
llmfoundry_run/
  checkpoints/
    ep1-ba500-rank0.pt              # Composer checkpoints (model+optimizer+schedule)
    latest-rank0.pt
  hf_checkpoints/
    ba500/                          # from the hf_checkpointer callback
      config.json, model-*.safetensors, tokenizer files
```

With `fsdp_config.state_dict_type: sharded` (the continued-pretrain recipe) each save is a
*directory* containing `.metadata` plus one `.distcp` per rank, and `load_path` must point
at that directory rather than a file. Sharded is what gives you elastic resumption — save
on 8 GPUs, resume on 16.

If you did not enable `hf_checkpointer`, convert a Composer checkpoint by hand:

```bash
python llm-foundry/scripts/inference/convert_composer_to_hf.py \
  --composer_path llmfoundry_run/checkpoints/latest-rank0.pt \
  --hf_output_path ./final_model \
  --output_precision bf16
  # --hf_repo_for_upload user-org/repo-name   # needs a write-enabled HF_TOKEN
```

Then the usual smoke test:

```bash
python llm-foundry/scripts/inference/hf_generate.py \
  --name_or_path ./final_model \
  --max_new_tokens 256 \
  --prompts "Summarise the following incident report:"
```

Resume a run by setting `load_path` in the YAML (or via `--extra load_path=...`). Foundry
also auto-resumes when `run_name`, `save_folder` and `save_latest_filename` are all set and
`save_overwrite` is false — convenient on a preemptible cluster, surprising if you did not
expect it, so pass `save_overwrite=true` when you deliberately want a fresh run in the same
folder.

## 8. Hardware support & evidence

**Other hardware (upstream claims — not verified here):** none claimed beyond NVIDIA CUDA and AMD ROCm (beta).


- **NVIDIA — supported.** The upstream README's support matrix lists A100-40GB/80GB and
  H100-80GB on torch 2.7.0 / CUDA 12.8 as supported, and states the codebase "has been
  tested with PyTorch 2.4 with NVIDIA A100s and H100s". Docker images
  (`mosaicml/llm-foundry:2.7.0_cu128-latest`, plus an `_aws` EFA variant) are
  NVIDIA-only.
- **AMD / ROCm — beta upstream, and the beta is stale.** The README carries an
  "AMD (BETA support)" install section (venv, `pip install -e .`, ROCm torch, ROCm
  flash-attention fork) and links MosaicML's MI250 training blog
  ([mosaicml.com/blog/amd-mi250](https://www.mosaicml.com/blog/amd-mi250)); Databricks
  later published "Training LLMs at Scale with AMD MI250 GPUs". The README also says
  other devices, including AMD cards, "may work" but are not actively tested, and that no
  fully-working AMD Docker image exists. **Take that section as evidence that Foundry
  *can* run on ROCm, not as install instructions**: its `rocm5.4.2` wheel index is dead,
  its step order re-installs CUDA torch over your ROCm one, and the `torch>=2.7.0,<2.7.1`
  pin in `setup.py` has no ROCm 7.x wheel. §8a below is a run on current hardware.
- **AMD MI355X (gfx950) / ROCm 7.2 — verified here, works with changes.** See §8a.
- **NVIDIA H100 80GB (CUDA 13.0, driver 580) — verified here, works with changes.**
  Single-GPU SFT smoke, `flash_attention_2` active (it builds and runs on H100, unlike
  ROCm). The "changes" are not code or YAML edits — they are the *model* (Foundry's
  `transformers<4.52` pin can't load the LFM2 arch) and small batch/seq overrides. See §8b.
- **Intel Gaudi — experimental**, on the `habana_alpha` branch only, with a Gaudi2
  blog post from Databricks.
- **Maintenance status:** headline models (MPT 2023, DBRX 2024) and the "Latest News"
  section date from 2023–2024; post-acquisition the repo primarily serves Databricks
  Mosaic AI Training. Factor that into how much community support you expect.

Sources: LLM Foundry README (`main`) — support matrix, Docker table,
"AMD (BETA support)" and "Intel Gaudi" sections; `llmfoundry/data/finetuning/tasks.py`
(chat-schema validation, `messages_format_preprocessor`); mosaicml.com/blog/amd-mi250.

## 8b. NVIDIA H100 (CUDA 13.0, driver 580)

Verified on **1x NVIDIA H100 80GB HBM3** (Hopper cc 9.0), driver **580.173.02**, CUDA 13.0,
Python 3.12.3. Single-GPU SFT smoke, **`flash_attention_2` active**. Summary at the bottom.

**Unlike the ROCm path, no `--no-deps` gymnastics and no torch pin-fight are needed on
CUDA.** Foundry's `torch>=2.7.0,<2.7.1` pin resolves to a real CUDA wheel from PyPI
(`torch 2.7.0+cu126`) that runs fine on this cu13/driver-580 box. Two things *do* differ
from the MI355X recipe, and both are documented below: (1) the **model** — Foundry pins
`transformers<4.52`, which does not know the `lfm2` architecture, so `LiquidAI/LFM2.5-350M`
cannot load — use `Qwen/Qwen3-0.6B` instead; (2) **`flash_attention_2` is kept** (the
YAML default) rather than overridden to `sdpa`.

### Environment / versions (what the install resolved to)

| Piece | Version |
|---|---|
| GPU | 1x NVIDIA H100 80GB HBM3, Hopper cc(9,0), native bf16/FP8 |
| Driver / CUDA | 580.173.02 / 13.0 |
| torch | **2.7.0+cu126** (Foundry/Composer pin-resolved from PyPI; runs on driver 580) |
| Base-case torch | `pip install torch` alone gives `2.13.0+cu130`; the Foundry deps downgrade it to `2.7.0+cu126` (expected — see reconciliation note) |
| triton | 3.3.0 (the CUDA build shipped WITH torch 2.7.0 — do **not** blind-uninstall it) |
| mosaicml (Composer) | 0.32.1 (provides the `composer` CLI) |
| llm-foundry | 0.23.0.dev0 (git `0cdb2f4`, editable checkout, `--no-deps`) |
| transformers | 4.51.3 (Foundry pins `>=4.51.0,<4.52`) |
| datasets / mosaicml-streaming / numpy | 3.6.0 / 0.12.0 / 2.1.3 |
| flash-attn | **2.7.4.post1** (prebuilt `cu12torch2.7` wheel — builds/runs on H100) |
| accelerate | 1.8.1 |

**torch reconciliation (the H100 gotcha, mild here).** `pip install torch` gives
`2.13.0+cu130`; installing the Foundry dependency closure (which carries
`torch>=2.7.0,<2.7.1`) then **downgrades it to `2.7.0+cu126`**. That is expected and
correct — the cu126 build runs on the cu13/driver-580 box (bf16 matmul verified on-GPU).
Do **not** try to force `2.13.0+cu130` back; nothing in Foundry/Composer supports it and
`pip check` is clean at `2.7.0`. This is the mirror image of the ROCm path (§8a), where the
same pin *cannot* be satisfied and you must install `--no-deps` to protect the ROCm wheel.

### Install (CUDA / H100) — the commands that worked

Upstream's primary NVIDIA recipe is `git clone … && pip install -e ".[gpu]"`. That works on
this box; the only wrinkle is that on Python 3.12 the old PyPI `llm-foundry` **wheel**
(≤0.22.0) drags in `pathtools`, which imports the removed `imp` module and fails to build —
so install from the **git checkout**, not the wheel. The recipe below installs the 0.23.0.dev0
dependency closure by name (so torch resolves cleanly), then the checkout `--no-deps`:

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry && source .env_llmfoundry/bin/activate
ln -sf ../../../dev.env dev.env                 # HF_TOKEN
pip install -U pip 'setuptools<81' wheel        # setuptools<81 => pkg_resources (Foundry imports it)
pip install torch numpy                         # base check: torch 2.13.0+cu130, bf16 matmul OK on GPU

# Foundry 0.23.0.dev0 runtime closure by name. Drop mosaicml's [wandb] extra (it pulls an
# ancient wandb -> pathtools -> imp), pin a modern wandb explicitly instead. torch pin -> 2.7.0+cu126.
pip install \
  'mosaicml[libcloud,oci,gcs,mlflow]>=0.32.1,<0.33' 'wandb>=0.18,<0.19' \
  'mlflow>=2.14.1,<3.0' 'accelerate>=0.25,<1.9' 'transformers>=4.51.0,<4.52' \
  'mosaicml-streaming>=0.12.0,<0.13' 'torch>=2.7.0,<2.7.1' 'datasets>=3.3.2,<3.7' \
  fsspec==2023.6.0 sentencepiece==0.2.0 einops==0.8.1 'omegaconf>=2.2.3,<3' 'slack-sdk<4' \
  'mosaicml-cli>=0.6.10,<1' onnx==1.18.0 onnxruntime==1.22.0 'boto3>=1.21.45,<2' \
  'huggingface-hub[hf_xet]>=0.30.0,<0.34' 'beautifulsoup4>=4.12.2,<5' 'tenacity>=8.2.3,<10' \
  'catalogue>=2,<3' 'typer<1' GitPython==3.1.44 'python-dotenv>=1.0.1'

# The checkout supplies scripts/train/train.py (NOT in the wheel). --no-deps => protects torch.
cd .env_llmfoundry && git clone --depth 1 https://github.com/mosaicml/llm-foundry.git
cd llm-foundry && pip install --no-deps -e . && cd ../..

# flash-attn: the pip build fails ONLY because PIP_CACHE_DIR on tmpfs triggers a cross-device
# rename ("Invalid cross-device link"). The prebuilt wheel it names downloads fine — grab it:
curl -sSL -o /tmp/fa.whl \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install --no-deps /tmp/fa.whl

# RE-VERIFY torch after everything:
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# -> 2.7.0+cu126 12.6 True
```

`pip check` reports **No broken requirements found**. (If you prefer the fully documented
upstream path, `pip install -e ".[gpu]"` on the checkout also resolves to this same torch
2.7.0 and pulls flash-attn 2.7.4.post1 via the `[gpu]` extra — use `--no-build-isolation`
and set `PIP_CACHE_DIR` off tmpfs to dodge the cross-device wheel-cache error above.)

**Model note (the real deviation).** `LiquidAI/LFM2.5-350M` is `model_type: lfm2` with no
remote code; transformers 4.51.3 (Foundry's ceiling) does **not** recognize it
(`lfm2 in CONFIG_MAPPING == False` → *"Transformers does not recognize this architecture"*).
Bumping transformers past 4.52 to get LFM2 would break Foundry's pin. So the smoke used
`Qwen/Qwen3-0.6B` — `qwen3` arch, loads under 4.51.3, ships a real chat template, 0.6 B.
(`Qwen/Qwen3-Reranker-0.6B` is also cached and *is* `qwen3`, but its chat template is a
reranker yes/no judge that discards the `user`/`assistant` turns → every row renders to ~73
tokens with a 0-token response and the loader drops all of them. Don't use a reranker
checkpoint for chat SFT.)

### Smoke command

```bash
export HF_DATASETS_CACHE=/dev/shm/dscache_llmfoundry     # datasets .arrow on tmpfs
export MASTER_PORT=29647                                # a distinct port (never 29500)
set -a; . ./dev.env; set +a                             # HF_TOKEN

CUDA_VISIBLE_DEVICES=7 python3 train_llm_llmfoundry.py --recipe sft \
  --foundry-dir .env_llmfoundry/llm-foundry \
  --gpus 1 --model Qwen/Qwen3-0.6B \
  --max-seq-len 2048 --max-duration 40ba \
  --global-batch-size 2 --device-microbatch-size 1 \
  --run-name h100-smoke-flash \
  --extra model.attn_implementation=flash_attention_2 model.init_device=cpu \
          save_folder=null eval_interval=1000ba console_log_interval=2ba \
          callbacks.hf_checkpointer.save_folder=$OUTPUT_DIR/llmfoundry/hf_checkpoints
```

Only `CUDA_VISIBLE_DEVICES` is set (no `HIP_VISIBLE_DEVICES` — that is ROCm-only).
`init_device=cpu` is required on a single GPU: the YAML's `init_device: mixed` assumes FSDP,
and Composer reverts FSDP→DDP on 1 GPU (`UserWarning: FSDP is not applicable for single-GPU
training. Reverting to DDP.`), where `mixed` has no sharded ranks to sync from.

### Expected output

Batch geometry: `global_train_batch_size=2`, `device_train_microbatch_size=1`, `n_gpus=1`
→ **`device_train_grad_accum=2`**. 40 logged batches over the 9 surviving rows ≈ 4.4 epochs
of real optimizer stepping (well past a trivial 0–1 steps). The loader logs the drop:

```
tasks.py:1032: UserWarning: Dropped 1 examples where the prompt was longer than 2048 ...   # 9/10 rows kept
```

Loss (train `LanguageCrossEntropy`, `console_log_interval=2ba`) — monotone-ish down:

```
[batch=1/40]  Train metrics/train/LanguageCrossEntropy: 2.0920
[batch=20/40] Train metrics/train/LanguageCrossEntropy: 0.5302
[batch=28/40] Train metrics/train/LanguageCrossEntropy: 0.0839
[batch=40/40] Train metrics/train/LanguageCrossEntropy: 0.0039
[Eval batch=3/3] Eval metrics/eval/LanguageCrossEntropy: 0.0154 | LanguagePerplexity: 1.0155 | TokenAccuracy: 0.9994
Train throughput/tokens_per_sec: 10016.4385   Train memory/peak_reserved_mem: 21.5820
llmfoundry.command_utils.train: Done.          # EXIT=0
```

GPU residency check (`nvidia-smi` filtered by PID, sampled from **inside** the running job)
— map the target GPU's UUID to its index first, then confirm the VRAM holder is your own
rank:

```
$ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  <pid>, <gpu-uuid>, 21888 MiB
$ pgrep -af scripts/train/train.py
  <pid> .../composer -n 1 ... train.py     # launcher
  <pid> .../python3        ... train.py     # rank 0 == the 21888 MiB holder above, on GPU 7
```

`peak_reserved_mem 21.58 GB` on an 80 GB card — a 0.6 B full fine-tune with DecoupledAdamW
leaves plenty of headroom; a larger model or longer `max_seq_len` would scale from here.

### Checkpoint

Composer/Foundry's `hf_checkpointer` writes a servable HF folder at
`<save_folder>/huggingface/ba40/` (sharded `model-0000{1,2}-of-00002.safetensors` + config +
tokenizer, ~1.5 GB) **at fit-end regardless of `save_interval`** — `save_folder=null` only
suppresses the *Composer* `.pt` checkpoints, not this HF export. Keep run outputs under
`$OUTPUT_DIR`, not the repo.

### Multi-GPU (not exercised on H100)

A 2- or 8-GPU pass would
drop `--gpus 1`/`init_device=cpu`, restore the YAML's `init_device: mixed` + `fsdp_config`
(`FULL_SHARD`), bump `--global-batch-size` to a multiple of the world size, use a distinct
`MASTER_PORT`, and expect a per-rank VRAM drop as FSDP shards (mirror the §8a 8-GPU run).
The 9-row sample is too small to shard usefully — point `--data-local` at a real dataset.

### Summary — works with changes

LLM Foundry **trains on 1x H100** on the stock CUDA path: torch resolves to `2.7.0+cu126`
with no override, `flash_attention_2` builds and runs, loss falls 2.09 → 0.004 and the model
saves. The only changes vs. the MI355X recipe are (1) **the model** — Foundry's
`transformers<4.52` pin can't load LFM2, so use a `qwen3`/`llama`-family checkpoint; and
(2) `flash_attention_2` is **kept** (on ROCm it had to become `sdpa`). Everything else
— launcher, YAML, batch geometry — carried over unchanged. Two Python-3.12 packaging traps
to know: install Foundry from the **git checkout** not the PyPI wheel (`pathtools`/`imp`),
and add **`setuptools<81`** for `pkg_resources`.

## 8a. MI355X (ROCm 7.2) — tested

**This path works with changes.** The SFT recipe in this folder trains on an AMD Instinct
MI355X. Foundry's own code needed no patching — the changes are all in *how you install
it* (its `torch` pin is unsatisfiable on ROCm 7.x) and four YAML overrides on the command
line. Nothing here required editing `yamls/finetune_chat_sft.yaml`, so the NVIDIA defaults
in that file are untouched.

Host: 8x AMD Instinct MI355X (gfx950, 288GB), ROCm 7.2.4, Ubuntu, Python 3.12.3.
Tested on **one** GPU (`HIP_VISIBLE_DEVICES=7`); multi-GPU FSDP on ROCm was **not**
exercised — see "Not covered" below.

Versions that worked: `torch 2.11.0+rocm7.2`, `torchvision 0.26.0+rocm7.2`,
`torchaudio 2.11.0+rocm7.2`, `mosaicml` (Composer) `0.32.1`, `llm-foundry 0.23.0.dev0`
(git `0cdb2f4`), `transformers 4.51.3`, `mosaicml-streaming 0.12.0`, `datasets 3.6.0`,
`numpy 2.1.3`.

### The root problem: an unsatisfiable torch pin

`llm-foundry/setup.py` on `main` pins `torch>=2.7.0,<2.7.1`, and Composer 0.32.1 — the
newest release — pins `torch<2.7.1,>=2.6.0` and `torchvision<0.22.1,>=0.21.0`. The ROCm
7.2 wheel index only publishes torch 2.11.0 / 2.12.x / 2.13.0. **No wheel can satisfy
both**, so a plain `pip install -e .` silently resolves torch from PyPI:

```
$ pip install -e . --dry-run          # inside the checkout, with ROCm torch already installed
Collecting torch<2.7.1,>=2.7.0 (from llm-foundry==0.23.0.dev0)
  Using cached torch-2.7.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata
Collecting nvidia-cudnn-cu12==9.5.1.17 (from torch<2.7.1,>=2.7.0->llm-foundry==0.23.0.dev0)
```

That is the CUDA build plus the entire `nvidia-*` stack landing on a machine with no
NVIDIA GPU, replacing your ROCm torch. The fix is to install the dependency closure
*without* the torch-family pins, then install Composer and Foundry with `--no-deps`.

### Install that worked (copy-pasteable)

```bash
cd training/llm/llmfoundry
python3 -m venv .env_llmfoundry
source .env_llmfoundry/bin/activate
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7      # never set these to ""
export PIP_CACHE_DIR=/path/to/pip_cache                  # off tmpfs; see the flash-attn note
pip install -U pip setuptools wheel

# 1. ROCm torch FIRST — torchvision/torchaudio too, so nothing drags in a CUDA wheel later
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/rocm7.2

# 2. A constraints file makes any accidental torch swap a loud failure instead of a silent one
printf 'torch==2.11.0+rocm7.2\ntorchvision==0.26.0+rocm7.2\ntorchaudio==2.11.0+rocm7.2\n' \
  > /tmp/rocm_constraints.txt

# 3. Composer's dependencies MINUS torch/torchvision/torchaudio, then Composer itself
pip install -c /tmp/rocm_constraints.txt \
  apache-libcloud coolname databricks-sdk google-cloud-storage importlib-metadata \
  'mlflow>=2.14.1,<3.0' 'mosaicml-cli>=0.5.25,<0.8' 'numpy<2.3.0,>=1.21.5' oci \
  'packaging<25.1,>=21.3.0' 'pillow<12,>=10.3.0' psutil py-cpuinfo 'pynvml<12,>=11.5.0' \
  pyyaml requests tabulate==0.9.0 'torch_optimizer<0.4,>=0.3.0' \
  'torchmetrics<1.7.5,>=1.0' tqdm 'wandb<0.19,>=0.13.2'
pip install --no-deps mosaicml==0.32.1

# 4. Foundry's dependencies MINUS torch and MINUS mosaicml (installed above)
pip install -c /tmp/rocm_constraints.txt \
  'accelerate>=0.25,<1.9' 'transformers>=4.51.0,<4.52' 'mosaicml-streaming>=0.12.0,<0.13' \
  'datasets>=3.3.2,<3.7' fsspec==2023.6.0 sentencepiece==0.2.0 einops==0.8.1 \
  'omegaconf>=2.2.3,<3' 'slack-sdk<4' onnx==1.18.0 onnxruntime==1.22.0 'boto3>=1.21.45,<2' \
  'huggingface-hub[hf_xet]>=0.30.0,<0.34' 'beautifulsoup4>=4.12.2,<5' 'tenacity>=8.2.3,<10' \
  'catalogue>=2,<3' 'typer<1' GitPython==3.1.44 python-dotenv

# 5. The checkout (kept inside the git-ignored venv dir so `git status` stays clean)
cd .env_llmfoundry
git clone --depth 1 https://github.com/mosaicml/llm-foundry.git
cd llm-foundry && pip install --no-deps -e .     # --no-deps is what protects the ROCm torch
```

`pip check` afterwards reports exactly three expected metadata complaints
(`llm-foundry`/`mosaicml` want torch `<2.7.1`, `mosaicml` wants torchvision `<0.22.1`).
They are cosmetic — nothing at runtime asserts on the torch version. Verify with
`python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"` →
`2.11.0+rocm7.2 AMD Instinct MI355X`. **Never `pip install flash-attn` here.** If pip ever
does replace torch, put it back with
`pip install --force-reinstall --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2`.

### Run that worked

Model substituted: `Qwen/Qwen3-0.6B` instead of the YAML's `meta-llama/Llama-3.1-8B-Instruct`
— small, ungated, and the point of the smoke test is the ROCm stack, not the checkpoint.
The Llama default is unchanged in the YAML.

```bash
source .env_llmfoundry/bin/activate
export $(grep -v '^#' ../dev.env | xargs)          # HF_TOKEN
export MASTER_PORT=29720                            # 29500 is often taken on a shared box

python3 train_llm_llmfoundry.py --recipe sft \
  --foundry-dir .env_llmfoundry/llm-foundry \
  --gpus 1 --model Qwen/Qwen3-0.6B \
  --max-seq-len 2048 --max-duration 20ba \
  --global-batch-size 2 --device-microbatch-size 1 \
  --run-name mi355x-smoke \
  --extra model.attn_implementation=sdpa model.init_device=cpu \
          save_folder=null eval_interval=1000ba console_log_interval=5ba \
          callbacks.hf_checkpointer.save_interval=1000ba \
          callbacks.hf_checkpointer.save_folder=$OUTPUT_DIR/llmfoundry/hf_checkpoints
```

Expected output (the run should finish with exit code 0):

```
[batch=1/20]:
	 Train metrics/train/LanguageCrossEntropy: 2.0903
	 Train metrics/train/LanguagePerplexity: 8.0871
[batch=20/20]:
	 Train metrics/train/LanguageCrossEntropy: 0.5299
	 Train throughput/tokens_per_sec: 14793.7086
	 Train memory/peak_reserved_mem: 21.9950
[Eval batch=3/3] Eval on eval data:
	 Eval metrics/eval/LanguageCrossEntropy: 1.8331
	 Eval metrics/eval/TokenAccuracy: 0.6154
```

`rocm-smi` sampled mid-run on the assigned GPU:

```
GPU[7]	: GPU use (%): 98
GPU[7]	: GPU Memory Allocated (VRAM%): 10
GPU[7]	: Current Socket Graphics Package Power (W): 315.0
```

Loss falls 2.09 → 0.53 over 20 batches on a 10-row sample (that is memorisation, which is
the expected signal at this scale), `amp_bf16` is used throughout, and the
`hf_checkpointer` callback wrote a servable HF folder (`config.json`, sharded
`model-*.safetensors`, tokenizer files) to `<save_folder>/huggingface/ba20/` — note the
extra `huggingface/` level the callback inserts under the path you give it.

### Required changes, and why

| Change | Why |
|---|---|
| Staged `--no-deps` install (above) | `torch>=2.7.0,<2.7.1` has no ROCm 7.x wheel; the normal route installs CUDA torch |
| `model.attn_implementation=sdpa` | The YAML default `flash_attention_2` aborts at model build: `ValueError: use_flash_attention_2 is set to True, but flash-attention 2 is not installed.` FA2 has no ROCm build here and must not be pip-installed. `sdpa` is torch's own kernel and works on gfx950 |
| `model.init_device=cpu` | Single-GPU only, **not** a ROCm issue: Foundry drops FSDP on one GPU (`FSDP is not applicable for single-GPU training. Reverting to DDP.`) and then `init_device: mixed` raises `NotImplementedError: Using init_device 'mixed' is only supported with FSDP`. Keep `mixed` for real multi-GPU runs |
| `save_folder=null` | Storage, not ROCm: Composer's `CheckpointSaver` writes a checkpoint at *fit end* regardless of `save_interval`, and for a 0.6B model that is a **6.7 GB** `.pt` (fp32 weights + Adam state). Disable it for smoke tests, or expect the file |
| `--global-batch-size 2` | Unrelated to hardware: the shipped `global_train_batch_size: 64` with `drop_last: true` yields **zero** batches from the 10-row sample |

Non-blocking quirks observed: `pynvml` is a hard dependency of Composer and installs fine
on ROCm — nothing calls into it on this path; Composer's dist init logs
`ProcessGroupNCCL ... Guessing device ID based on global rank` (RCCL under the NCCL name,
harmless on one rank); and `destroy_process_group() was not called` is printed at exit on
every run, including successful ones.

### 8-GPU run (8x MI355X, ROCm 7.2.4)

**This path works as documented.** The same SFT recipe scales from 1 to 8 MI355X with real FSDP
`FULL_SHARD` over RCCL. No code patch, no new package, no extra pin — the only changes
versus the 1-GPU run are configuration (batch geometry that divides by 8, a bigger data
slice, and `init_device: mixed` put *back*, which is what it was written for). Both runs
exited **0** with all 8 ranks finishing and no teardown hang.

Host: 8x MI355X (gfx950, 288 GB), ROCm 7.2.4, torch `2.11.0+rocm7.2`, Composer 0.32.1,
llm-foundry 0.23.0.dev0 — identical venv to the 1-GPU test.

#### Launch command

```bash
cd training/llm/llmfoundry
source .env_llmfoundry/bin/activate
# If the venv's bin/activate pins HIP_VISIBLE_DEVICES=7 from a 1-GPU session,
# overriding it AFTER sourcing is mandatory or you silently train on one GPU.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -a; . ./dev.env; set +a            # HF_TOKEN
python3 -c "import torch; assert torch.cuda.device_count()==8"

composer -n 8 --master_port 29750 \
  .env_llmfoundry/llm-foundry/scripts/train/train.py \
  yamls/finetune_chat_sft_8gpu.yaml
```

`--master_port` matters on a shared box: Composer defaults to 29500 and a collision shows
up as a rendezvous hang, not a clear error. `train_llm_llmfoundry.py --gpus 8 --config
yamls/finetune_chat_sft_8gpu.yaml` produces the same command line (it has no
`--master_port` flag; use the `MASTER_PORT` env var with the launcher).

#### Parallelism, as Foundry logged it

```
composer.core.state: Automatically setting data_parallel_shard to have parallelization degree 8.
n_gpus: 8
global_train_batch_size: 32
device_train_microbatch_size: 1
device_train_grad_accum: 4
config_utils.py:539: UserWarning: Setting `sync_module_states = True` for FSDP.
                                  This is required when using mixed initialization.
```

`data_parallel_shard = 8` is the line that proves FSDP is real — with `NO_SHARD` or DDP
Composer reports `data_parallel_replicate` instead. World size 8, `FULL_SHARD`,
`mixed_precision: PURE`, activation checkpointing on, `state_dict_type: full`.

#### FSDP VRAM delta — the proof sharding actually happened

Two back-to-back 8-rank runs, identical geometry, only `fsdp_config.sharding_strategy`
changed (`memory/peak_reserved_mem`, GB, per rank):

| Run | sharding_strategy | peak_reserved_mem / rank |
|---|---|---|
| 8 GPU | `FULL_SHARD` | **8.23 GB** |
| 8 GPU | `NO_SHARD` (replicated, DDP-equivalent) | **18.81 GB** |
| 1 GPU (§8a above) | FSDP reverted to DDP | 22.00 GB |

**10.58 GB/rank saved** at the same batch geometry — the optimizer state and parameter
shards are genuinely split 8 ways. Reproduce the comparison with
`... yamls/finetune_chat_sft_8gpu.yaml fsdp_config.sharding_strategy=NO_SHARD`.

#### What a healthy run looks like (20ba, `FULL_SHARD`, exit 0)

```
[batch=1/20]:  Train metrics/train/LanguageCrossEntropy: 1.9195
[batch=6/20]:  Train metrics/train/LanguageCrossEntropy: 1.6595
               Train throughput/tokens_per_sec: 62282.6574
               Train throughput/device/tokens_per_sec: 7785.3322
[batch=12/20]: Train metrics/train/LanguageCrossEntropy: 0.4316
[batch=20/20]: Train metrics/train/LanguageCrossEntropy: 0.0773
               Train metrics/train/LanguagePerplexity:   1.0803
               Train memory/peak_reserved_mem:           8.2292
               Train time/sample: 608          # 20ba x 32 global batch, minus warmup accounting
```

Finite, monotonically falling loss; ~50-63k tokens/s aggregate (~6.2-7.9k/s/device).
The loss reaching 0.08 is **memorisation of a 10-row sample replicated 128x** — see the
data note below. This is a pipeline proof, not a learning result.

#### GPU evidence, sampled in-band and cross-checked against the job's own PIDs

Run the `rocm-smi` sampler **inside** the job (a background loop under the same
machine-wide `flock`), so every sample is contemporaneous with the training processes.
A healthy `rocm-smi --showpids` sample looks like:

```
PID     PROCESS NAME  GPU(s)  VRAM USED
<pid>   python3       1       4780736512     <- rank 0
<pid>   python3       1       5233676288
<pid>   python3       1       5239967744
<pid>   python3       1       5227384832
<pid>   python3       1       5174956032
<pid>   python3       1       5082681344
<pid>   python3       1       5101555712
<pid>   python3       1       5091069952     <- 8 PIDs, ~4.8-5.2 GB each
```

`pgrep -af scripts/train/train.py` in the **same** sample:

```
<pid> .../bin/composer -n 8 --master_port 29750 .../scripts/train/train.py yamls/finetune_chat_sft_8gpu.yaml
<pid> .../python3 .../scripts/train/train.py yamls/finetune_chat_sft_8gpu.yaml
... (the remaining 7 ranks composer forked)
```

The eight PIDs holding VRAM should be exactly the eight ranks `composer -n 8` forked —
not another tenant's job on a shared box.

**Utilisation needed a longer run to measure.** In the 20ba run `rocm-smi --showuse`
reported **0% busy on every GPU** in most samples: 20 steps is a few seconds of burst and
the instantaneous poll lands in the gaps. Do not read that as an idle GPU, and do not
report it as a pass either — re-run at `max_duration=150ba` with the sampler at 3 s (same
config otherwise). Across 39 samples x 8 GPUs:

```
busy%:  87:1   92:1   93:3   94:3   95:1   96:14   97:27   98:32   99:71   100:53
one sample            GPU[1..7]: GPU use (%): 100      (GPU[0] polled 0 in this sample)
                      power:  272 / 315 / 326 / 317 / 319 / 320 / 322 / 328 W
--showpids            8 python3 PIDs, ~5.6-7.8 GB VRAM each
pgrep -af (same sample)
   <pid> .../bin/composer -n 8 --master_port 29750 .../train.py yamls/finetune_chat_sft_8gpu.yaml max_duration=150ba
   <pid>..<pid+7> .../train.py yamls/finetune_chat_sft_8gpu.yaml max_duration=150ba   <- the 8 ranks
```

All eight GPUs sit at 96-100% under load at 272-328 W (idle is ~140 W), and the 8 VRAM
holders are again the `composer` launcher's own children. The 150ba run peaked at 7.98 GB/rank and
drove cross-entropy 1.9195 → 0.0010 by batch 25 (full memorisation of the replicated
sample, as expected).

#### What differed from the 1-GPU run

| Change | Why |
|---|---|
| **New file `yamls/finetune_chat_sft_8gpu.yaml`** | The 1-GPU YAML is left untouched (it keeps the NVIDIA defaults). The new one bakes in the §8a ROCm fixes plus 8-rank geometry |
| `HIP_VISIBLE_DEVICES=0,...,7` exported **after** `source bin/activate` | If `bin/activate` still carries a `=7` pin from a 1-GPU session, the run is silently single-GPU without the override |
| `global_train_batch_size: 32` (was 2) | Must be divisible by the 8 ranks; Foundry aborts in config validation otherwise. 32 = 8 ranks x 4, with `device_train_microbatch_size: 1` giving grad accum 4 |
| `init_device: mixed` **restored** (1-GPU needed `cpu`) | `mixed` requires FSDP; at 8 ranks FSDP is real, so the §8a workaround is no longer needed — and `mixed` is what stops 8 CPU copies of the weights at startup |
| Data: 1280-row replicated slice **outside the repo** | The shipped 10-row sample cannot feed 8 ranks at global batch 32 with `drop_last: true` — zero batches. Write it to `$OUTPUT_DIR/llmfoundry/gpu8/data/OTel_LLM_sample_1280.jsonl` (10 rows x 128); the repo sample stays unchanged |
| `save_folder` and the `hf_checkpointer` callback omitted entirely | Nothing large should land for a smoke run. Verified: 0 bytes of weights on disk afterwards |
| `max_seq_len: 2048`, `attn_implementation: sdpa` | Carried forward unchanged from §8a |

**Sample survival:** of 1280 rows, Foundry logged
`Dropped 128 examples where the prompt was longer than 2048, the prompt or response was
empty, or the response was all padding tokens` — **1152 rows survived** (exactly one of
the ten source conversations exceeds 2048 tokens, so 1 in 10 replicas is dropped).
Always read that warning: it is the only place the loader tells you how much data it kept.

**No MDS/StreamingDataset here** — the SFT recipe reads plain JSONL through the HF `json`
builder, so the shard-per-rank starvation issue does not apply to this path. It would
apply to `yamls/continued_pretrain.yaml`, which is still untested.

**No requirements change.** `requirements_llmfoundry.txt` is unchanged: the 8-GPU path
needed no new package and no new pin. `pynvml` remains an inert Composer dependency, and
`destroy_process_group() was not called` is still printed at exit on every rank — cosmetic,
same as on 1 GPU.

### Not covered

- **Multi-node** FSDP (this was a single 8-GPU node) and FSDP checkpoint *saving* at 8
  ranks — `state_dict_type: full` is set in the config but no checkpoint was written, so
  the monolithic-save path is still unverified on gfx950.
- The `pretrain` recipe (`yamls/continued_pretrain.yaml`) and the MDS/StreamingDataset
  conversion scripts — not run. Note it carries the same `attn_implementation:
  flash_attention_2` default and needs the same `sdpa` override on ROCm.
- FP8 (`amp_fp8`) — needs TransformerEngine, which is NVIDIA-only. MI355X FP8 is not
  reachable through this path.

## 9. Notes

- **The YAML is the program.** `scripts/train/train.py` is four lines that call
  `train_from_yaml`; the dataclass behind it (`llmfoundry.utils.config_utils.TrainConfig`)
  defines every legal top-level key. Unknown keys are a hard error — Foundry refuses to run
  a config with a typo in it, and tells you to put anything custom under `variables:`. That
  strictness is a feature; it is also why copying a snippet from an old blog post fails.
- **`fsdp_config` at the top level, `parallelism_config` inside Composer.** Composer's
  Trainer takes `parallelism_config={'fsdp': ..., 'tp': ...}` (the flat `fsdp_config=`
  kwarg is gone). Foundry's *YAML* still uses a top-level `fsdp_config:` block and folds it
  into `parallelism_config` in `llmfoundry/command_utils/train.py`. Both are current; do not
  "modernise" the YAML.
- **`variables:` plus `${...}` interpolation is how one value reaches five places.**
  `variables.max_seq_len` feeds the top-level `max_seq_len`, the tokenizer's
  `model_max_length`, and both dataloaders. The launcher's overrides target
  `variables.*` for exactly this reason — setting the leaf keys individually is how configs
  drift out of sync.
- **`global_train_batch_size` is math, `device_train_microbatch_size` is systems.** Foundry
  derives gradient accumulation from the two plus the world size, so the same YAML produces
  the same optimization on any GPU count. Changing the microbatch changes speed and memory,
  never the result.
- **`init_device: mixed` is the one to know for large models.** Rank 0 materialises real
  weights while the other ranks initialise on the meta device; FSDP then syncs module states
  outward. Without it, every rank loads a full copy into CPU RAM at startup and an 8-GPU box
  can OOM before training begins. Foundry forces `sync_module_states: true` and defaults
  `load_monolith_rank0_only: true` when you use it.
- **Sequence packing is where the SFT throughput is.** `packing_ratio: auto` profiles the
  dataset and picks the highest ratio with near-zero waste. If you enable it for an MPT
  model you must also set `attn_uses_sequence_id: true` so attention does not bleed across
  packed examples; for HF models with FlashAttention-2 the loader handles the boundaries.
- **The two dataloaders are not interchangeable.** `finetuning` builds prompt/response or
  chat turns and applies loss masking; `text` yields undifferentiated fixed-length token
  blocks with no masking at all. Continued pre-training wants `text` — masking prompts in a
  corpus you are trying to model is exactly wrong.
- **`hf_checkpointer` is the callback that saves you a day.** It writes a real HF folder
  during training, so if the run dies at hour 30 you still have a servable model. Set its
  `precision: bfloat16` unless you specifically want fp32 files (the default is `float32`,
  which doubles the size on disk).
- **Continued pre-training LRs are small.** `5e-6` with a 100-batch warmup in the shipped
  config, versus `1e-5` for SFT. Reusing a from-scratch pre-training LR on an already
  converged checkpoint is the standard way to destroy its instruction-following ability.
- **Rough memory rule of thumb, from upstream:** with FULL_SHARD, activation checkpointing
  and a decoupled optimizer, total cluster memory in GB should exceed `12 x N` where N is
  billions of parameters. 8x H100 80GB = 640GB, so a 70B full fine-tune is feasible but
  tight; below that, drop `device_train_microbatch_size` first.
