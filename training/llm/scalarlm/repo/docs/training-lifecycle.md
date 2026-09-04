# Training Lifecycle

This document walks a training job end-to-end: from `SupermassiveIntelligence.train(...)` in a user's notebook to post-training model registration with the live vLLM engine. It is the deep-dive companion to §3.4 and §4 of `architecture.md`.

The lifecycle has six phases:

1. **Client-side packaging** — SDK builds a tarball and multipart upload.
2. **Server-side ingestion** — API unpacks the upload, hashes it into a job directory, writes `config.yaml`.
3. **SLURM submission** — Control plane generates an sbatch command and launches it.
4. **Worker-side training** — The sbatch job runs `cray_megatron.main`, which drives the `TrainingLoop` until `max_steps` or timeout.
5. **Checkpointing and status** — Rank 0 writes `checkpoint_*.pt` and maintains `status.json` throughout.
6. **Post-training registration** — A periodic control-plane task discovers the new checkpoint and registers it with vLLM, closing the loop.

Every piece of coordination happens through the job directory. No direct RPC runs from the training job back to the control plane — the control plane polls, the training process writes files.

---

## 1. Client-Side Packaging (SDK)

### 1.1 Entry point

```python
import scalarlm
llm = scalarlm.SupermassiveIntelligence(api_url="http://host:8000")
llm.train(
    data=[{"input": "...", "output": "..."}, ...],   # list, file path, or bytes stream
    model_name="my-run",
    train_args={"max_steps": 500, "learning_rate": 3e-4, "gpus": 2},
)
```

`SupermassiveIntelligence.train` (`sdk/masint/api/supermassive_intelligence.py:11`) is a synchronous wrapper around `AsyncCray.train`, which delegates to `submit_training_job` in `sdk/masint/engines/cray/submit_training_job.py:20`.

### 1.2 Data normalization

`make_data_file` (`submit_training_job.py:146`) accepts three input shapes and normalizes all of them to a local file:

| Input type | Handling |
|---|---|
| `str` (path) | Opened directly; assumed to be JSON Lines. |
| `io.BufferedIOBase` (stream) | Copied to a `tempfile.NamedTemporaryFile`. |
| `list` (of dicts) | Written to a temp file with `jsonlines.Writer`. |

Zero-length files are rejected (`check_for_zero_length_file`).

### 1.3 Archive construction

`make_training_archive` (`submit_training_job.py:86`) builds a gzip tar with two members:

- `dataset.jsonlines` — the normalized training data
- `ml/` — the local `ml/` directory, if one is found by `find_ml_dir`

The inclusion of `ml/` is what makes the training stack user-editable: `find_ml_dir` (`submit_training_job.py:127`) first checks `./ml` relative to the caller's CWD, then falls back to the SDK-relative `cray/ml`. The user's current `training_loop.py`, `dataset loaders`, `adapters`, etc. travel with the job. If no `ml/` exists locally the server falls back to the `ml/` baked into its container.

Filesystem metadata is stripped (`tar_info_strip_file_info`) so the archive is reproducible: same data + same code → same tarball bytes → same `dataset_hash` on the server → same job directory.

### 1.4 Streaming multipart upload

The upload is streamed, not buffered in memory:

- `make_multipart_writer` constructs an `aiohttp.MultipartWriter` with two parts:
  - `file` → 64 KB-chunked `file_sender` generator reading the tarball
  - `params` → JSON-encoded `train_args`
- `get_content_length` (`submit_training_job.py:44`) walks the writer once with a counting sink to precompute `Content-Length`, so the server can reject over-size uploads early.
- `upload_async` (`submit_training_job.py:27`) POSTs to `POST /v1/megatron/train`.

The default upload cap is **10 GB** (`max_upload_file_size` in `default_config.py:19`). The request body cap is 2× that.

---

## 2. Server-Side Ingestion

### 2.1 Streaming parse

`upload_training_data` (`infra/cray_infra/training/upload_training_data.py:26`) is the server-side counterpart. It uses `streaming_form_data.StreamingFormDataParser` so incoming bytes never pile up in memory:

- `FileTarget(temp_filepath, validator=MaxSizeValidator(max_file_size))` streams the tar to disk.
- `ValueTarget()` accumulates the JSON params into memory (they're small).
- `MaxBodySizeValidator` aborts with HTTP 413 on overflow.

Errors are returned as semantic HTTP statuses:

| Error | Status | Meaning |
|---|---|---|
| `MaxBodySizeException` | 413 | Total body exceeded 2× `max_upload_file_size`. |
| `ValidationError` (from `MaxSizeValidator`) | 413 | File part alone exceeded `max_upload_file_size`. |
| Missing `file` multipart | 422 | Uploader omitted the dataset. |
| `ClientDisconnect` | logged warning | Client aborted mid-upload; job dir not created. |

### 2.2 Content-addressed job directory

After the tar lands on disk:

```
file_hash     = sha256(tar contents)                      # upload_training_data.py:57
train_args["dataset_hash"]   = file_hash
train_args["job_directory"]  = training_job_directory / sha256(json.dumps(train_args))
                                                           # get_job_directory:154
```

The job directory path is the SHA-256 of the **full, augmented** `train_args` dict — which already contains `dataset_hash`. This gives two important properties:

1. **Idempotent resubmission.** Re-running the same SDK call against the same server will produce the same directory. `launch_training_job` detects this case with `job_already_exists` (`launch_training_job.py:46`) and returns the existing `status.json` without re-submitting. Safe to retry.
2. **Dataset + hyperparams identify a run.** Change the learning rate, get a new directory. Change the dataset, get a new directory. Same inputs, same job.

### 2.3 Layout after ingestion

```
/app/cray/jobs/{job_hash}/
├── dataset.jsonlines          # from tar
├── ml/                        # from tar (client's training code)
│   ├── cray_megatron/
│   ├── adapters/
│   └── tokenformer/
├── config.yaml                # written from train_args + defaults
├── train_job_entrypoint.sh    # copied + templated from default script
├── status.json                # seeded at QUEUED
├── slurm-{jobid}.out          # SLURM stdout (later)
└── checkpoint_{step}.pt       # written by rank 0 during training
```

`make_training_directory` (`launch_training_job.py:54`) writes the augmented `train_args` to `config.yaml`. This is what `get_job_config()` reads from inside the sbatch job (see §4.1).

---

## 3. SLURM Submission

### 3.1 Wait for SLURM

Before submitting, `wait_for_slurm` (`launch_training_job.py:32`) polls `squeue` for up to `slurm_wait_time` (default 30 s). This matters during container boot: `scripts/start_one_server.sh` starts `slurmctld`/`slurmd` before the API process, but the first training request can still arrive before SLURM has finished initializing.

### 3.2 Building the sbatch command

`create_slurm_run_command` (`launch_training_job.py:78`) assembles the flags from three sources: `train_args`, live cluster introspection via `scontrol show nodes`, and configuration defaults.

| Flag | Source | Logic |
|---|---|---|
| `--ntasks-per-node` | `get_tasks_per_node` (L142) | `clamp(train_args["gpus"], 1, get_max_gpu_count_from_slurm())` |
| `--gres=gpu:N` | same N, only if cluster has GPUs | Detected by scanning `scontrol show nodes` for `Gres=gpu`. |
| `--nodes` | `get_node_count` (L174) | `min(train_args["nodes"], max_node_count_from_slurm)` |
| `--cpus-per-task` | `get_cpu_per_task` (L198) | `CPUTot / max(tasks_per_node, max_gpus)` |
| `--time` | `get_train_time_limit` | **Per-slice** cap: `min(train_args["timeout"], max_train_time) + extra_training_seconds`. Not the user's total budget — see §5.4. |
| `--signal=B:TERM@N` | `signal_grace_seconds` config | SLURM sends SIGTERM to the batch shell `N` seconds (default 300, i.e. 5 min) before `--time` runs out, so the trainer can checkpoint and trigger auto-relaunch (§5.4). `B:` targets the batch shell rather than the job step. |
| `--output` | fixed | `{job_dir}/slurm-%j.out` |
| `--job-name` | fixed | `basename(job_directory)` — i.e. the job hash |

`get_train_time_limit` returns a **per-slice** walltime, not the user's total training budget. `train_args["timeout"]` is the user's total time across all slices; each individual SLURM job is capped at `max_train_time` (the cluster's hard cap on a single job — `slurmctld` won't run anything longer). Long jobs accept that SLURM cuts them into slices and a fresh slice is queued on timeout. `extra_training_seconds` (default 300 s) is a small buffer on top so the trainer can finish a checkpoint without colliding with the SIGKILL deadline. See §5.4 for the relaunch coordination.

Walltime format is `DD-HH:MM:SS` via `format_timedelta` (L241).

### 3.3 Entrypoint templating

`get_train_job_entrypoint` (`launch_training_job.py:117`) copies the configured `train_job_entrypoint` script (default `/app/cray/scripts/train_job_entrypoint.sh`) **into the job directory** and string-replaces `REPLACE_CONFIG_PATH` with the per-job `config.yaml`. Two consequences:

1. Each job owns a copy of its entrypoint — useful for restarts, auditing, and changing the global script mid-production without affecting running jobs.
2. The entrypoint exports `CRAY_TRAINING_JOB_CONFIG_PATH` so `get_job_config` can find the config at runtime (see `get_job_config.py:19` — it asserts the env var is set).

### 3.4 Submission

`run_sbatch` (`launch_training_job.py:248`):

1. Scrubs `PMI*` environment variables from the parent (they'd confuse the new MPI world).
2. Writes `status.json` = `{"status": "QUEUED", "start_time": <now>}`.
3. Runs `sbatch` with `cwd=job_directory`.
4. Parses `"Submitted batch job (\d+)"` from stdout (`get_job_id_from_sbatch_output`).
5. Re-writes `status.json` with `{"job_id": <slurm id>}` merged in (write_job_status merges, never overwrites).
6. On non-zero exit, writes `{"status": "FAILED", "output": ...}` and returns.

The API response (`TrainResponse`) is the merged status dict plus a `deployed: false` flag — it's not deployed yet because the checkpoint doesn't exist.

---

## 4. Worker-Side Training

### 4.1 Entry

SLURM runs `{job_dir}/train_job_entrypoint.sh` on the allocated node(s). The bash script is intentionally trivial — it sets the MPI/GPU env vars, exports `CRAY_TRAINING_JOB_CONFIG_PATH={job_dir}/config.yaml`, and `exec`s mpirun:

```bash
exec mpirun --allow-run-as-root python "${LOCAL_DIRECTORY}/ml/cray_megatron/main.py" "$@"
```

The `exec` is load-bearing. Because slurm's `--signal=B:TERM@N` (§3.2) targets the batch shell by PID, replacing bash with mpirun in place means slurm's SIGTERM lands on mpirun directly — no bash trap, no PID juggling. mpirun's standard SIGTERM forwarding then propagates the signal to every rank's python process. Keeping no custom code on the `sbatch → mpirun → main.py` path avoids cross-system behavior drift (different mpirun forks, different slurm versions, different bash versions).

The training process itself is `ml/cray_megatron/main.py:29`:

```python
def main():
    harness = TrainingHarness()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()
    try:
        setup_logging()
        setup_signal_handler(harness)         # SIGTERM/SIGCONT → stop_flag.request_stop()
        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
    except Exception as e:
        print_exception()
        harness.update_status(status=TrainingJobStatus.FAILED,
                              metadata={"error": str(e)})
        raise e
    finalize_mpi()
```

Three points:

- **Harness first.** `TrainingHarness` is constructed before `try` so even setup failures can write `FAILED`.
- **HF token injection.** `get_hf_token()` decrypts `hf_encrypted_token` from `default_config.py:66` with the Fernet key at `:67`, exports it as `HUGGING_FACE_HUB_TOKEN`. Gives `from_pretrained` access to gated models without shipping plaintext credentials in the job config.
- **Signal handling.** The handler (registered for both `SIGTERM` and `SIGCONT`) sets a module-level latch in `ml/cray_megatron/megatron/stop_flag.py` and returns; it deliberately does **not** call `sys.exit()`. The training loop polls the latch at each step boundary, breaks cleanly, and `TrainingLoop.train()`'s post-loop checkpoint runs before the process exits. Without this, killing the trainer mid-step would force the next slice to redo work back to the previous `steps_per_checkpoint` boundary. The trainer never resubmits itself; queuing the next slice when a TRAINING/QUEUED job falls out of `squeue` is the API server's job (§5.4, §6.3).

### 4.2 Status lifecycle

`TrainingJobStatus` (`infra/cray_infra/training/training_job_status.py:4`):

```python
class TrainingJobStatus(str, Enum):
    QUEUED    = "QUEUED"
    TRAINING  = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
```

The transitions:

```
  sbatch submitted       MegatronTrainer.train_loop()    TrainingLoop.train() returns
        │                           │                              │
        ▼                           ▼                              ▼
     QUEUED ──────────►  TRAINING (metadata: max_steps)  ──►  COMPLETED (only when no signal)
                                    │
                                    │ every step:
                                    │   status=TRAINING, metadata=history  (update_history)
                                    │
                                    │ SIGTERM (slurm slice timeout) or SIGCONT (preempt):
                                    │   stop_flag set → loop drains, checkpoints
                                    │   → _finalize_slice updates accumulated_train_seconds
                                    │   → status stays as last-written (TRAINING)
                                    │   → restart_megatron_jobs (§6.3) requeues
                                    │
                                    │ any exception:
                                    └─►  FAILED (metadata: error/output)
```

`MegatronTrainer.train_loop` (`megatron_trainer.py:21`) writes `TRAINING` with `max_steps` before running, and `COMPLETED` on success — but **only** when `stop_flag.was_stop_requested()` is false. This is the status discipline that drives §5.4: a signal-induced exit leaves status at `TRAINING`, which the `restart_megatron_jobs` reconciler picks up as "should be running but isn't in squeue" and resubmits. Without this guard, `MegatronTrainer` would write `COMPLETED` and the reconciler would (correctly) leave the job alone. Every training step calls `update_history`, which writes `TRAINING` again with an updated `history` list.

### 4.3 The `TrainingHarness`

`ml/cray_megatron/megatron/training_harness.py:16`:

```python
class TrainingHarness:
    def update_status(self, status, metadata={}):
        current = get_status()                # read whole status.json
        current["status"] = status
        current.update(metadata)
        save_status(current)                  # @main_rank_only

    def checkpoint(self, checkpoint_state, checkpoint_name):
        job_config = get_job_config()
        path = os.path.join(job_config["job_directory"], checkpoint_name)
        torch.save(checkpoint_state, path)
```

- `save_status` is decorated with `@main_rank_only` (`collectives/main_rank_only.py:26`) — only rank 0 actually writes the file. Other ranks are no-ops.
- `update_status` is read-modify-write, not overwrite. Metadata accumulates across calls (e.g. `start_time`, `job_id`, `max_steps`, `history` can coexist).
- The harness never issues HTTP. All external visibility is through `status.json` + checkpoint files in the shared job directory.

Note: there are two `TrainingHarness` classes — one in `ml/cray_megatron/megatron/training_harness.py` (the one used by the training loop) and a thinner one in `infra/cray_infra/training/training_harness.py` used by `training_job_context.py` for context-manager-style runs. The ml/ version is authoritative for normal training; the infra/ context is used by smaller utility jobs.

### 4.4 `MegatronTrainer` and `TrainingLoop`

`MegatronTrainer` (`megatron_trainer.py:14`) is a thin shell — sets status to `TRAINING`, prints the ASCII logo, runs the loop, sets status to `COMPLETED`.

The real work is in `TrainingLoop` (`ml/cray_megatron/megatron/training_loop.py:28`):

```python
def train(self):
    self.model_manager = get_model_manager()
    self.training_state.model_info = self.model_manager.load_model()
    self.training_loop()
    self.checkpoint()                    # always checkpoint at the end
    self._finalize_slice()               # persists accumulated_train_seconds (§5.4)
```

`training_loop` (L45) in plain sequence:

1. `model.train()` + `get_max_steps()` + `get_gradient_accumulation_steps()` (default 4).
2. Build `AdamW` optimizer (`get_optimizer`, L396) with `learning_rate` from job config.
3. Build `LinearLR` scheduler decaying from 1.0 → 0.0 over `max_steps` (`get_scheduler`, L412).
4. `if does_any_checkpoint_exist(): resume_from_checkpoint()` — see §5.2.
5. Build `DataLoader` wrapping model + tokenizer.
6. For each `step` from `starting_step` to `max_steps`:
   - `optimizer.zero_grad()`
   - For `accum_step in range(gradient_accumulation_steps)`:
     - `batch = next(data_iterator)`
     - `training_step_accumulate` → forward, NaN check, loss/accum scale, backward, sync loss
     - NaN-on-forward → skip backward, zero the loss tensor, continue
     - NaN-in-accumulated-loss → break the inner loop, skip optimizer step, bump `nan_steps`
   - `model.backward_sync()` — force gradient all-reduce across ranks
   - `optimizer_step()` — `clip_grad_norm_(max=gradient_clip_value)` then `optimizer.step()` + `scheduler.step()`
   - `update_history(avg_loss)` — appends `{step, loss, epoch, time}`, capped at `training_history_length` via `remove_closest_entry`
   - Callbacks: `on_step_end` → **`TimeoutCallback`** (L341) checks wall clock vs. `timeout`, **`CheckpointCallback`** (L359) saves every `steps_per_checkpoint` steps
   - If `should_stop_training`: break
7. Final `self.checkpoint()` outside the loop.

NaN handling is notable: a NaN forward is tolerated once, but a NaN after full accumulation skips the optimizer step entirely so momentum doesn't get poisoned. `nan_steps` is persisted in the checkpoint and reported in final logs.

### 4.5 Cross-rank coordination

- `sync_loss` (L209): if `get_size() > 1`, `allreduce_op(loss)` + divide → average across ranks. Used purely for logging; gradients are synced separately via `backward_sync()`.
- `@main_rank_only` is applied to:
  - `save_status` (harness)
  - `save_checkpoint` (TrainingLoop L259)
  - `update_history` (TrainingLoop L280)
  - `print_training_step_info`, `print_microbatch_info`

This pattern means every rank runs the model forward/backward and allreduces, but only rank 0 touches the shared job directory.

---

## 5. Checkpointing and Resume

### 5.1 Save

`TrainingLoop.checkpoint` (`training_loop.py:248`):

```python
def checkpoint(self):
    model = self.training_state.model_info["model"]
    if hasattr(model, "unwrap_model"):
        model_state_dict = model.unwrap_model()          # FSDP: gather full state
    else:
        model_state_dict = filter_checkpoint(model.model, model.model.state_dict())
    self.save_checkpoint(model_state_dict)
```

`save_checkpoint` (L259, `@main_rank_only`) assembles:

```python
{
    "model_state_dict":      model_state_dict,
    "optimizer_state_dict":  optimizer.state_dict(),
    "scheduler_state_dict":  scheduler.state_dict(),
    "step":                  current_step,
    "epoch":                 epoch,
    "nan_steps":             nan_steps,
}
```

and saves to `checkpoint_{step}.pt` via `TrainingHarness.checkpoint(...)` → `torch.save`.

Two callers write checkpoints:

1. **Periodic** — `CheckpointCallback.on_step_end` (L365) fires every `steps_per_checkpoint` steps (default 100), but skips step 0.
2. **Final** — `TrainingLoop.train()` always calls `self.checkpoint()` after the loop exits, so a completed run always produces a fresh checkpoint with the terminal `step == max_steps`.

After saving, `delete_old_checkpoints` (`models/get_latest_checkpoint_path.py:32`) enforces retention: sorts by step, keeps the most recent `max_checkpoints_to_keep` (default 3), removes the rest. This keeps disk usage bounded across long runs without forcing the user to garbage-collect manually.

### 5.2 Resume

`does_any_checkpoint_exist()` gates the resume path. If true, `resume_from_checkpoint` (`training_loop.py:140`) is called *before* the DataLoader is constructed:

```python
checkpoint = torch.load(latest_checkpoint_path, weights_only=True)
self.training_state.current_step = checkpoint["step"]
self.training_state.epoch        = checkpoint["epoch"]
self.training_state.nan_steps    = checkpoint.get("nan_steps", 0)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
self.training_state.history = self.training_harness.get_status()["history"]
```

Two details worth calling out:

- `strict=False` on `load_state_dict`: tolerates adapter layers being added or removed between runs (LoRA ↔ tokenformer, or adapter shape changes). The base model weights match; newly introduced adapter params start from init.
- `history` is restored from `status.json`, not from the checkpoint, so that history always reflects the filesystem source of truth rather than a possibly-stale snapshot in the weights file.

Resume is automatic: the training loop simply loops `for step in range(starting_step, max_steps)`, where `starting_step = training_state.current_step` after resume. No CLI flag, no `--resume` option — it just works.

### 5.3 `latest` model alias

`get_latest_model()` (used by `get_training_job_info` when the hash is literally `"latest"`) returns the most recently trained model. The `vllm_model_manager.VLLMModelManager` keeps models in a `SortedDict` keyed by `get_start_time(model_dir)`, so "latest" is well-defined. Combined with the SDK's `model_name=None` fallback, clients that don't want to track hashes can always ask for "latest".

### 5.4 Long jobs across multiple SLURM slices

The user's `train_args["timeout"]` is the total wall-clock budget for the run. SLURM, however, caps any single job at `max_train_time` (cluster-wide hard ceiling, default 24 h in `default_config.py`). When the user asks for ten days of training, the job runs as a chain of slices, each ≤ `max_train_time`, with a transparent checkpoint-and-resume at every boundary. The user sees one job; SLURM sees many sequential ones.

**The work splits across two existing pieces**, each owning a single concern:

1. **The trainer (in-slice)** — exits the slice cleanly when SLURM is about to kill it, with the latest checkpoint on disk and the total elapsed budget persisted to `status.json`.
2. **The API server's `restart_megatron_jobs` reconciler (§6.3) (cross-slice)** — notices when a `TRAINING`/`QUEUED` job has fallen out of `squeue` and resubmits it via `start_slurm_job(config)`. This task already existed for crash recovery and SIGCONT preemption; long jobs reuse it.

There is **no separate auto-relaunch dispatch path** inside the trainer. The same reconciler that has always handled crashed and preempted jobs handles slice timeouts, kept reliable by careful status discipline in the trainer.

**Trainer responsibilities (per slice)**

1. **At submit time**, the API server sets `--time = min(train_args["timeout"], max_train_time) + extra_training_seconds` and `--signal=B:TERM@signal_grace_seconds` (§3.2). The first `--time` field is the per-slice cap; `signal_grace_seconds` (default 300 s — 5 min) is the warning window SLURM gives before the hard kill.
2. **During the loop**, `TimeoutCallback` compares `accumulated_seconds_at_slice_start + (now - slice_start)` against the user's total `train_args["timeout"]` at each step boundary and stops the loop when total elapsed exceeds the user budget. `accumulated_seconds_at_slice_start` is loaded from `status.json` at slice start (zero for the first slice).
3. **When SLURM sends SIGTERM** (`signal_grace_seconds` before `--time`), the bash batch script's `exec mpirun` means slurm's signal lands directly on mpirun (§4.1); mpirun's standard SIGTERM forwarding reaches each rank's python process, and the handler sets the `stop_flag` latch. The training loop polls the latch at each step boundary, breaks cleanly, and `TrainingLoop.train()`'s post-loop `self.checkpoint()` runs before the process exits. Without the SIGTERM handler, slurm would SIGKILL the trainer at the hard `--time` and the next slice would redo work back to the previous `steps_per_checkpoint` boundary.
4. **`_finalize_slice`** updates `status.json` with the new `accumulated_train_seconds`. The harness's read-modify-write `update_status` preserves the other keys (`job_id`, `start_time`, `history`, …).
5. **`MegatronTrainer` only writes `COMPLETED` when no signal was received.** This is the status discipline that drives the reconciler:
   - Loop ran to `max_steps`, or `TimeoutCallback` exhausted the user's total budget → no signal → status = `COMPLETED` → reconciler leaves it alone. Done.
   - SIGTERM (slurm slice timeout) or SIGCONT (preempt) cut the slice short → status stays `TRAINING` → reconciler picks it up.

**Reconciler responsibilities (cross-slice)** — full detail in §6.3. Every `megatron_refresh_period` (default 30 s) it diffs job-directory `status.json` files (filtered to `TRAINING`/`QUEUED`) against `squeue` and calls `start_slurm_job(config)` for anything missing. `start_slurm_job` rebuilds the sbatch invocation from the current server config (live `scontrol`, current `signal_grace_seconds`, etc.) so config edits between slices take effect on the next slice.

**Resume.** A new slice starts mpirun → `python main.py` → `TrainingLoop.train()`. `does_any_checkpoint_exist()` → `resume_from_checkpoint` (§5.2) loads model / optimizer / scheduler state. `_load_accumulated_seconds()` pulls `accumulated_train_seconds` from `status.json` into `TrainingState` so `TimeoutCallback` keeps honoring the user's total budget across slices.

**When no relaunch happens.** Three correct cases:
- `max_steps` reached → `COMPLETED` → not picked up.
- User's total `train_args["timeout"]` exhausted (TimeoutCallback fired) → `COMPLETED` → not picked up.
- Exception during training → `FAILED` → not picked up.

**Latency.** SIGTERM checkpoint → process exit → up to one `megatron_refresh_period` (default 30 s) of reconciler lag → new sbatch. Tiny next to multi-hour slice durations; operators who want shorter relaunch latency tune `megatron_refresh_period` downward.

**What can defeat clean handoff.** If a checkpoint takes longer than `signal_grace_seconds` to write, SLURM hits the hard `--time` limit and SIGKILLs the trainer mid-checkpoint. The previous checkpoint is still on disk (so the next slice resumes from there, losing the work done after that checkpoint); status stays `TRAINING` so the reconciler still requeues. Tune `signal_grace_seconds` upward for very large models if checkpoint writes routinely exceed 5 min.

---

## 6. Post-Training Registration — Closing the Loop

The FastAPI app's lifespan hook (`api/fastapi/tasks/add_megatron_tasks.py:22`) installs a periodic task that runs every `megatron_refresh_period` seconds (default 30):

```python
@repeat_every(seconds=megatron_refresh_period)
async def run_megatron_tasks():
    await register_megatron_models()
    await restart_megatron_jobs()
    await register_megatron_workers()
    await clear_acked_requests_from_queue()
```

All four are training-related:

### 6.1 `register_megatron_models`

`infra/cray_infra/training/register_megatron_models.py:16` scans the jobs directory for any subdirectory containing a `*.pt` file and registers each one with the in-memory `VLLMModelManager` (`vllm_model_manager.py:9`).

`VLLMModelManager` is a SortedDict keyed by job start time, so "newest trained model" is `O(1)`. `find_model` (L29) implements three-tier lookup:

1. Is it the base model? → return it.
2. Is it already registered? → return it.
3. Does `{training_job_directory}/{model_name}` exist and contain `.pt` files? → auto-register and return.

That third branch lets you do `llm.generate(prompts, model_name="<job_hash>")` even before the refresh period has fired — the model manager will pick it up on demand.

On the vLLM side, the registration is plumbed through the runtime-LoRA update path enabled by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` (set in `infra/cray_infra/one_server/main.py:4` at process start). For Tokenformer adapters, the adapter cache (`tokenformer_cache_capacity`, default 2) controls how many live in memory at once; inactive adapters are lazy-loaded from disk.

### 6.2 `register_megatron_workers`

`register_megatron_workers.py:3` is a one-liner that delegates to `discover_clusters()` (`infra/cray_infra/slurm/discovery/discover_clusters.py`). It detects multi-node SLURM topologies so future jobs can be submitted with correct `--nodes` counts.

### 6.3 `restart_megatron_jobs`

`infra/cray_infra/training/restart_megatron_jobs.py:20` is the **single reconciler** that handles every kind of "job should be running but isn't" — crashes, container restarts, SIGCONT preemption, and slurm slice timeouts (§5.4). Each tick:

1. Scan all job dirs with `status.json` in `{QUEUED, TRAINING}`.
2. Query `squeue` for currently-running job names.
3. For every status.json that is `QUEUED`/`TRAINING` but *not* in `squeue`, call `start_slurm_job(config)` to resubmit.
4. If any SLURM jobs are running, hit `/v1/health/keepalive` to keep the pod alive.

Because job directories are content-addressed (§2.2), a resubmission lands in the same directory and the resume-from-checkpoint path in §5.2 picks up where it left off. `start_slurm_job` rebuilds the sbatch invocation from current server config, so live config edits (e.g. tuning `signal_grace_seconds` or `max_train_time`) take effect on the next slice.

This task is what makes the trainer in §5.4 only need to handle the in-slice half cleanly — it never has to call `sbatch` itself.

### 6.4 `clear_acked_requests_from_queue`

Not training-specific, but runs in the same refresh loop. Removes acked inference requests from the SQLite work queue so it doesn't grow unbounded.

---

## 7. Observability During a Run

Three ways to watch a job:

| What | How | Source |
|---|---|---|
| Current status | `GET /v1/megatron/train/{job_hash}` | `get_training_job_info.py:21` — reads `status.json` + `config.yaml`, attaches `deployed` flag from the vLLM model manager |
| Live stdout/stderr | `GET /v1/megatron/train/logs/{model_name}` (SSE) | `training_logs_generator` streams `slurm-{id}.out` |
| Loss curve | `scalarlm plot {model}` | SDK CLI, reads `history` from the status endpoint and plots |

The `history` list capped at `training_history_length` (default 1024) is what makes `scalarlm plot` cheap: the server-side array never exceeds 1024 entries regardless of run length, because `remove_closest_entry` (L421) evicts the most tightly-spaced entry when adding new ones would exceed the cap. You get a uniformly-sampled curve over the full run without quadratic memory.

`GET /v1/megatron/squeue` and `/v1/megatron/gpu_count` / `/node_count` surface cluster-level state directly.

---

## 8. Cancellation and Deletion

- `POST /v1/megatron/cancel/{job_hash}` → `cancel.py` → `scancel` on the stored `job_id`. `status.json` gets rewritten to `FAILED` (or `QUEUED` if the signal is intercepted as preemption).
- `POST /v1/megatron/delete/{job_hash}` → `delete.py` removes the job directory. Cancel the SLURM job first if still running; otherwise the periodic `restart_megatron_jobs` will helpfully re-launch it for you.

Canceling does not unregister the model from vLLM unless the entire directory is deleted — in-memory `VLLMModelManager.register_model` does not proactively unload.

---

## 9. Config Reference (per-job)

`JobConfig` (`infra/cray_infra/util/default_job_config.py`) defines everything the training loop reads. Authoritative fields:

| Field | Default | Meaning |
|---|---|---|
| `job_directory` | — | Set by server; `/app/cray/jobs/{hash}`. |
| `training_data_path` | — | Set by server; `{job_dir}/dataset.jsonlines`. |
| `dataset_hash` | — | Set by server; content hash of the uploaded dataset. |
| `llm_name` | `meta-llama/Llama-3.2-1B-Instruct` | HF Hub model ID to start from. |
| `max_steps` | 100 | Number of optimizer steps. |
| `learning_rate` | 3e-3 | AdamW `lr`. |
| `batch_size` | 1 | Per-rank micro-batch. |
| `gradient_clip_value` | 1.0 | `clip_grad_norm_` max. |
| `gradient_accumulation_steps` | 4 | Accumulate before optimizer step. |
| `max_token_block_size` | 16 × 10⁶ | Dataset tokenizer chunking limit. |
| `training_mode` | `language_model` | Or `embedding`. |
| `distribution_strategy` | `fsdp` | Or `ddp`, or none. |
| `steps_per_checkpoint` | 100 | `CheckpointCallback` period. |
| `max_checkpoints_to_keep` | 3 | `delete_old_checkpoints` retention. |
| `gpus` | 1 | `sbatch --ntasks-per-node` (clamped by SLURM). |
| `nodes` | 1 | `sbatch --nodes` (clamped by SLURM). |
| `adapter_type` | `tokenformer` | Or `lora`. |
| `lora_config` | `{r:32, alpha:32, dropout:0.1, target_modules:"all-linear"}` | Used when `adapter_type=lora`. |
| `timeout` | 14400 (4 h) | User's **total** wall-clock budget across all SLURM slices — enforced by `TimeoutCallback` against accumulated elapsed (§5.4). Each individual SLURM slice is capped at the server's `max_train_time`; this can be larger than that, and the trainer will chain slices automatically. |
| `training_history_length` | 1024 | `history` list cap. |

Server-side knobs that interact with the timeout (in `default_config.py`):

| Field | Default | Meaning |
|---|---|---|
| `max_train_time` | 86400 (24 h) | Per-slice cap on `--time`. `train_args["timeout"]` longer than this becomes a multi-slice run (§5.4). |
| `extra_training_seconds` | 300 (5 min) | Buffer added on top of `--time` so the trainer's final checkpoint has room before the hard SLURM kill. |
| `signal_grace_seconds` | 300 (5 min) | Window between SLURM's SIGTERM and the hard `--time` deadline. Must be smaller than `max_train_time` and larger than one training step + checkpoint write. Tune up for very large models. |

Any subset of these can be passed in `train_args={...}`; unspecified fields default as above. The server merges them with server-derived fields (`job_directory`, `training_data_path`, `dataset_hash`) before writing `config.yaml`.

---

## 10. Edge Cases and Invariants

**Idempotent submissions.** Identical `train_args` + identical dataset bytes → identical job directory → no-op resubmit (returns existing status). Change any byte and you get a fresh directory.

**Crash safety.** Training process crashes: status stays at whatever was last written; the next `restart_megatron_jobs` tick resubmits; the sbatch worker resumes from the latest checkpoint. The API pod crashes: nothing happens to running SLURM jobs; they keep writing status.json. On API restart, the periodic task reads current state from disk.

**Preemption and SLURM slice timeout** (both routes converge — see §5.4). SLURM sends a signal (`SIGCONT` for preempt, `SIGTERM` `signal_grace_seconds` before `--time` for slice timeout) → handler sets `stop_flag` → loop drains and checkpoints cleanly → `_finalize_slice` persists `accumulated_train_seconds` → `MegatronTrainer` leaves status as `TRAINING` (no `COMPLETED` write when a signal was seen) → `restart_megatron_jobs` (§6.3) re-submits on its next tick → next slice resumes from checkpoint with `accumulated_train_seconds` carried forward so `TimeoutCallback` honors the user's total budget. Distinct SLURM job IDs across slices is by design; users key on the job-directory hash, not the SLURM ID.

**Checkpoint always final.** `TrainingLoop.train()` calls `self.checkpoint()` after the loop exits (whether via `max_steps`, `TimeoutCallback`, or `should_stop_training`). You can never finish a run without a fresh checkpoint.

**NaN tolerance.** A single NaN forward is absorbed; a full-accumulation NaN is a dropped step with `nan_steps` incremented. Neither kills the run.

**History uniformity.** `remove_closest_entry` keeps the 1024-entry history roughly uniform in step-space, so plots look smooth over runs from 100 to 10M steps with no storage cost growth.

**Closed-loop freshness.** A `generate` request submitted 30 seconds after training completes already routes through the new adapter — the refresh period is the worst-case staleness. For sub-period latency, clients can specify `model_name={job_hash}` directly and `find_model` will auto-register on the first call.
