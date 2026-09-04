# Publish to HuggingFace

A modal on the train-detail page that takes a completed training job
and uploads it to a HuggingFace repository. Two modes:

1. **PEFT adapter** — slice the LoRA tensors out of the checkpoint
   and publish in HF's standard adapter shape
   (`adapter_config.json` + `adapter_model.safetensors`). Small, fast,
   HF-loadable via `peft.PeftModel.from_pretrained(base, repo)`.
2. **Merged full model** — fold the LoRA delta into the base model
   with `merge_and_unload`, save the merged `*ForCausalLM` weights +
   tokenizer, push as a self-contained model repo loadable via
   `transformers.AutoModelForCausalLM.from_pretrained(repo)`.

Both modes produce **HF-standard artifacts** that third parties can
load. The earlier draft of this doc proposed pushing the raw
`checkpoint_<step>.pt` as Mode 1 — that file is a `torch.save` dump
of training state and isn't loadable by `transformers` or `peft`,
so it offered no actual value as an HF artifact and was scrapped.

Anchors to existing UI conventions in
[`docs/ui-design.md`](../../docs/ui-design.md): modal pattern (§6.1,
training submit), destructive-action confirmation (§4.3), live-progress
component (§5.5, log pane), credential handling (§6.4 — secrets are
never persisted to the browser, only forwarded once per request).

## Motivation

The merge pipeline already exists as a CLI:
[`ml/adapters/merge_lora_and_push.py`](../../ml/adapters/merge_lora_and_push.py).
This doc covers the UI flow and the SLURM-job orchestration that
runs the upload off-process. Operators don't want to SSH and
remember argparse flags every time they ship a model.

## Non-goals

- **OAuth-based "log in with HuggingFace."** v1 takes a user-pasted
  access token, holds it in component state for the duration of the
  modal, forwards it once via `sbatch --export=...,HF_TOKEN=<value>`,
  then drops it. The token never lands on disk in either pod, never
  on a CLI argv (which would show up in `ps`), never in a log line.
- **Repo browser / dataset card editor.** The HF web UI handles those
  better. We just push files and surface the resulting repo URL.
- **Per-step push.** You can publish at any time; the modal always
  acts on the latest checkpoint by default. Earlier checkpoints are
  selectable from a dropdown but not multi-selected.
- **Asynchronous publish queue independent of training.** Each
  publish is a one-shot SLURM job tagged to a single training-job
  hash. There is no separate "publish board" UI; the train-detail
  page is the single surface.
- **Mobile-optimized layout.** Same constraint as the rest of the
  app per `ui-design.md` non-goals.

## Where it runs

Both modes execute as a single-task SLURM job dispatched onto a
megatron pod. Three reasons:

1. **The base model is already cached on the megatron pod.** It pulled
   the base from HF when training started; mode 2's
   `from_pretrained(base_name)` hits the local HF cache instead of
   re-downloading ~62 GB.
2. **Megatron pods have GPU + RAM headroom.** mode 2's `merge_and_unload`
   wants ~2× the base size in RAM briefly; mode 1's adapter conversion
   is small but tens of MB to GB upload still shouldn't block the API
   pod's HTTP worker for 30+ min.
3. **SLURM gives lifecycle for free.** Cancel = `scancel`, retry = a
   new job in a new directory, logs already follow the
   per-job-output convention, no new orchestration to write.

The API pod's role shrinks to: validate the request, write a job
script, `sbatch` it, and act as a thin proxy for status/log/cancel
endpoints.

## Surface

### Trigger

The train-detail action row grows a **Publish to HF** button next
to "Open in Chat" / "Cancel" / "Delete". Enabled only when
`status === "COMPLETED"`. For `TRAINING`/`QUEUED` it's disabled with
a title tooltip explaining why; for `FAILED` it stays disabled.

### Modal layout

```
┌─ Publish "first-model" to HuggingFace ──────────────────── [×] ┐
│                                                                │
│  Mode                                                          │
│   ◉ PEFT adapter                                               │
│      Small, HF-standard adapter repo.                          │
│      Loadable: `PeftModel.from_pretrained(base, repo)`         │
│   ○ Merged full model                                          │
│      Large self-contained model repo. Required for             │
│      off-platform inference. ~30–60 GB for a 31B base.         │
│                                                                │
│  Repository                                                    │
│   ┌─────────────────────────────┐                              │
│   │ myorg/my-adapter            │                              │
│   └─────────────────────────────┘                              │
│   [ ] Private                                                  │
│                                                                │
│  HuggingFace token                                             │
│   ┌─────────────────────────────┐  hf_*** (paste; not saved)   │
│   │ ●●●●●●●●●●●●●●●●●●●●●●●●●●  │                              │
│   └─────────────────────────────┘                              │
│   Get one at huggingface.co/settings/tokens · Write access     │
│   required for new repos.                                      │
│                                                                │
│  Checkpoint                                                    │
│   [latest (step 100, 14m ago)        ▾]                        │
│                                                                │
│  ▷ Advanced                                                    │
│       Override lora_alpha · custom commit message              │
│                                                                │
│  ─────────────────────────────────────────────────────────     │
│                                          [ Cancel ] [ Publish ]│
└────────────────────────────────────────────────────────────────┘
```

### Inline form rules

- **Mode** is a radio. Default = "PEFT adapter" since most users want
  a small portable artifact.
- **Repository** is `<owner>/<name>`, validated client-side with a
  loose regex (letters/digits/`._-` on both sides, exactly one slash).
  HF is the authority on whether the name is actually valid;
  client-side validation catches obvious typos only.
- **Private** checkbox; defaults off. Existing repo visibility wins
  on the server side regardless.
- **Token** is `<input type="password">` with a "show" toggle. Held
  only in component state; cleared on modal close. Inputs carry
  `data-1p-ignore` / `data-lpignore` to deter password-manager
  autofill that would otherwise persist the token.
- **Checkpoint** dropdown lists every `checkpoint_<step>.pt` with its
  step number and "<n>{m,h,d} ago" timestamp. Defaults to latest.
- **Advanced** is a collapsed disclosure containing:
  - `lora_alpha` numeric override (mode = merged full model only —
    Mode 1 already encodes the actual alpha into the adapter config).
  - `commit_message` free-form text.

### Progress states

While a publish job is in flight, the modal swaps its body for a
phase indicator. Phases come from the publish job's `status.json`,
which the merge CLI writes via `--status-file`:

| phase | shown as |
|---|---|
| `queued` | "Waiting for SLURM…" |
| `validating` | "Reading checkpoint…" |
| `loading_base` | "Loading base model `<name>`…" (mode = merge only) |
| `merging` | "Merging LoRA into base…" (mode = merge only) |
| `saving` | "Saving …" |
| `uploading` | "Uploading to `<repo>`…" |
| `done` | "Done. → `huggingface.co/<repo>`" with a copy button |
| `error` | error message + last phase |

The UI also tails `<publish_dir>/publish.log` underneath the phase
indicator for debugging.

### Cancellation

The Cancel button on an in-flight publish hits
`POST /publish/cancel`, which `scancel`s the SLURM job. The merge
job's `__finally__` is best-effort: a partially-uploaded HF commit is
left on the repo (HF doesn't have transactional commits), and the
operator can clean up via huggingface.co. Document this on the
button's title tooltip:

> Cancels the SLURM job. A partial upload may remain on huggingface.co.

## API

Four endpoints under `/v1/megatron/train/{job_hash}/publish`.

### `POST /v1/megatron/train/{job_hash}/publish`

Submits the SLURM publish job. Body:

```json
{
  "mode": "adapter" | "merged",
  "repo_id": "myorg/my-adapter",
  "private": false,
  "hf_token": "hf_xxxxxxxxxxxxxxx",
  "checkpoint": "checkpoint_100.pt",
  "lora_alpha": 32,
  "commit_message": "..."
}
```

Returns immediately:

```json
{
  "publish_job_id": "<slurm_job_id>",
  "publish_dir": "<job_dir>/publish_<unix_ts>",
  "status": "QUEUED"
}
```

Responds 409 if a publish for this job_hash is already in flight
(its `status.json` is in `queued` / `validating` / `loading_base` /
`merging` / `saving` / `uploading`). Force-retry: cancel the running
publish first.

**Token handling.** The handler extracts `hf_token` from the body,
stages an `sbatch --export=ALL,HF_TOKEN=<value>,...` argv, and never
touches the value again. Specifically:
- Token is **not** written into the entrypoint script.
- Token is **not** written into status.json.
- Token is **not** logged: any logger.info on the request body
  scrubs the field via the existing `truncate_dict` helper.
- Token is **not** in the SLURM job's argv (which would show in
  `scontrol show jobid` / `ps`); it goes through the env-export
  path instead.

### `GET /v1/megatron/train/{job_hash}/publish/status`

Returns the current `<publish_dir>/status.json`. Shape:

```json
{
  "publish_job_id": "...",
  "mode": "adapter" | "merged",
  "phase": "queued" | "validating" | ... | "done" | "error",
  "started_at": 1761369000.123,
  "completed_at": null,
  "repo_url": null,
  "error": null,
  "uploaded_bytes": 1234567,
  "total_bytes": 5234567
}
```

If no publish has ever run for this job_hash, returns 404. The UI
polls every 2 s while the phase is non-terminal.

### `GET /v1/megatron/train/{job_hash}/publish/logs`

NDJSON tail of `<publish_dir>/publish.log` — same protocol as
`/health/logs/{service}` (byte-offset resume, scroll-back via
`before_byte_offset`/`before_count`). Reuses
`streamNdjsonLogOnce` on the client.

### `POST /v1/megatron/train/{job_hash}/publish/cancel`

`scancel <publish_job_id>`. Updates `status.json` phase to `error`
with `reason: "cancelled"`.

## On-disk layout

```
<job_dir>/
  publish_<unix_ts>/
    entrypoint.sh             # sbatch script; never contains the token
    status.json               # phase + progress; rewritten atomically
    publish.log               # merge/upload stdout+stderr
    merged/                   # only in mode=merged; staged before upload
    adapter/                  # only in mode=adapter; staged before upload
```

Each publish gets its own subdirectory keyed by submission timestamp
so retries don't clobber each other. Old publish dirs aren't
auto-cleaned; operators can prune by hand or via a future watchdog.

## Backend pieces

```
ml/adapters/merge_lora_and_push.py     # already exists; gets:
                                        #   --status-file <path>
                                        #   --mode adapter|merged
                                        #   adapter mode emits PEFT
                                        #   adapter_config.json +
                                        #   adapter_model.safetensors
infra/cray_infra/training/launch_publish_job.py   # new — sbatch wrapper
infra/cray_infra/training/publish_job_entrypoint.sh # template, copied
infra/cray_infra/training/publish_status.py        # status.json reader
infra/cray_infra/api/fastapi/generate/publish_logs.py # log tail (reuse)
infra/cray_infra/api/fastapi/routers/megatron_router.py # 4 new routes
```

## UI files

```
ui/src/api/publish.ts                       # request/poll/log hooks
ui/src/routes/train/PublishToHFModal.tsx    # the modal
ui/src/routes/train/TrainDetail.tsx         # adds the trigger button
```

## Security

- Token is held only in component state; not in localStorage or
  IndexedDB. Wiped on modal close.
- Token forwarded over `POST` body (not query string) to keep it out
  of access logs. Route handler logs the request after scrubbing the
  field via `truncate_dict`-style masking.
- Server side, token enters the SLURM job via env-var export only.
  Never reaches the entrypoint script's argv, status.json, or
  publish.log. The merge CLI reads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
  from env (already supported).
- Operator-supplied default credentials: if `HF_TOKEN` is set in the
  pod's cray-config.yaml, the modal's token field shows "(use
  deployment credentials)" placeholder; submitting blank uses the
  pod default.

## States

| state | trigger | message |
|---|---|---|
| disabled trigger | job_status != COMPLETED | tooltip explains why |
| validation error | bad repo_id, missing token | inline red text |
| submitting | POST /publish in flight | spinner on Publish |
| queued | status.phase = queued | "Waiting for SLURM…" |
| running | non-terminal phase | phase indicator + log tail |
| success | status.phase = done | green check + repo URL + close |
| error | status.phase = error | red banner + log tail + retry |
| network failure | fetch rejects | red banner with offline copy |

## Testing

- **Unit (pytest):** `launch_publish_job` builds the right argv;
  status.json reader returns the right shape; merge CLI's
  `--status-file` writes phase markers.
- **Unit (pytest):** the new adapter-mode path of `merge_lora_and_push`
  produces a directory containing `adapter_config.json` and
  `adapter_model.safetensors` whose tensor names reload via
  `peft.PeftModel.from_pretrained` against a tiny stub base.
- **Unit (vitest):** form validation, state machine over
  status-poll responses, log tail rendering. (Phase 0 already has
  the regex + label-formatter coverage.)
- **e2e:** out of scope. Real HF uploads can't be hermetic.

## Phased rollout

1. **Phase 0 (shipped):** `GET /checkpoints` endpoint, modal form
   shell, trigger button. No submit; banner explains preview state.
2. **Phase 1:** `--mode adapter|merged` + `--status-file` in the
   merge CLI; new launcher (`launch_publish_job.py`); `POST /publish`
   submits a SLURM job; `GET /publish/status` reads status.json.
   Modal wires submit → poll-status → display phase. Cancel hits
   `POST /publish/cancel`.
3. **Phase 2:** `GET /publish/logs` byte-offset tail; modal wires
   the log tail underneath the phase indicator. Streamed
   `uploaded_bytes` / `total_bytes` progress where the merge CLI
   knows it (HF callback hook).
4. **Phase 3:** Advanced options pane (`lora_alpha` override,
   custom commit message). Operator-supplied default-credentials
   placeholder.
