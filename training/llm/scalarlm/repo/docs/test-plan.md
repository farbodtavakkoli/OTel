# ScalarLM Test Plan (CPU Targets)

## Status

Design. First revision of a unified test plan targeting the **CPU x86_64 and CPU arm64** container images. The goal is a suite that a contributor can run on a laptop — no GPU, no multi-node cluster, no large-model downloads — and that still exercises every major user-visible contract in the system.

This plan refers heavily to the existing design docs; each test section cross-links the specific contract it exercises:

| Reference | Covers |
|---|---|
| [architecture.md](./architecture.md) | Process topology, routes, directory layout |
| [inference-queue.md](./inference-queue.md) | SQLite queue, worker, file idempotency |
| [training-lifecycle.md](./training-lifecycle.md) | Upload → sbatch → harness → register flow |
| [adapters.md](./adapters.md) | Tokenformer / LoRA surgery and vLLM-side manager |
| [gpu-aware-mpi.md](./gpu-aware-mpi.md) | Collectives, SHM channel, MPI lifecycle |
| [configuration.md](./configuration.md) | Server / job / SDK config resolution |
| [ui-design.md](./ui-design.md) | React SPA, routes, streaming |

---

## 1. Goals and Non-Goals

### Goals

- **Every public contract has a test.** The FastAPI routes enumerated in architecture.md §3.2, the SDK methods on `SupermassiveIntelligence`, the bashly CLI commands in `cmd/bashly.yml`, the PyPI CLI commands in `sdk/masint/cli/main.py`, and every UI route in `ui/src/App.tsx` must each have at least one test that fails if the contract regresses.
- **Docker is the only dependency.** Every test — unit, component, e2e, UI, collectives — runs inside a container. A contributor with Docker installed and nothing else on the host can run the full suite. No host-side `pip install`, no host-side `npm install`, no host-side Python interpreter requirement.
- **Reproducible on a laptop.** The full "CPU-fast" profile runs on a dev machine with Docker and ~16 GB RAM in under 10 minutes. No GPU driver, no HF gated-model access, no external network beyond pip / npm mirrors at image-build time.
- **Fail informatively.** A failing test names the exact file+line it's protecting (e.g. "regression of `launch_training_job.py:222` walltime clamp"). No `assert status_code in [200, 400, 404, 422, 500]` — existing integration tests lean on that pattern and it hides regressions.
- **Deterministic.** No reliance on timing beyond explicit polling with a budget; no reliance on HF downloads inside the hot path (pre-populate a cache fixture); no reliance on `persistqueue` SQLite ordering beyond what the library guarantees.
- **Three layers, one plan.** Big-picture end-to-end ⇄ mid-level component ⇄ unit/sanity. Every mid-level test is a lens on what a big-picture test only implicitly covers; every unit test is a lens on what a mid-level test only implicitly covers. The split exists so a failure at layer N points at layer N+1 of test to inspect.

### Non-Goals

- **Multi-GPU correctness.** FSDP sharding, NCCL/RCCL, tensor-parallel inference, and ROCm/CUDA kernel parity are out of scope for the CPU plan. They stay in the hardware-specific `benchmark/` and `deployment/` suites.
- **Real multi-node MPI.** `gpu_aware_mpi` is tested via its standalone `multiprocessing.shared_memory` mirror (already in `test/collectives/test_shm_channel.py`) plus optional single-process `mpirun -n 2` wrappers. Anything larger is a cluster concern.
- **Long training jobs.** Max **10 optimizer steps** anywhere in the suite. Convergence and perplexity are not asserted — the plan verifies the *lifecycle* works, not that the model learns anything.
- **Gated / large HF models.** No `meta-llama/*`, `google/gemma-3-*`, `Qwen/*`, `openai/gpt-oss-*`. The supported-models table in the README is for deployments, not tests.
- **UI visual regression / pixel snapshots.** Out of scope for v1. Component behavior (Zod schema round-trip, SSE parser, tar builder) is unit-tested.

---

## 2. Scope of CPU Targets

Two images out of `Dockerfile`:

| Target | Build | Runtime |
|---|---|---|
| `cpu` (x86_64) | `./scalarlm build-image cpu` → `gdiamos/scalarlm-cpu:latest` | `docker-compose up cray` |
| `cpu` (arm64) | Same Dockerfile, built on arm64 host (Apple Silicon, Grace, Graviton) | `DOCKER_PLATFORM=linux/arm64 docker-compose up cray` |

Both images contain:
- PyTorch CPU wheels
- vLLM built with `VLLM_TARGET_DEVICE=cpu`
- `gpu_aware_mpi` built with the CPU backend (OpenMPI, no CUDA/ROCm)
- SLURM (`munged` + `slurmctld` + `slurmd`) running from `scripts/start_slurm.sh`
- The ScalarLM UI bundle from `ui/dist/`
- No CUDA/ROCm userspace libs

What is *not* in the CPU images:
- NVIDIA / AMD kernels (vLLM runs on CPU with reduced throughput)
- `cudaHostRegister` path in the SHM channel (no-op; everything is host memory anyway)
- GPU-resident tensors

This shape is the ceiling for what a CPU test can exercise. Anything below that ceiling is fair game.

---

## 3. Test Fixtures and CPU Constraints

### 3.1 Tiny model fixtures

Four models are cheap enough to load and run on CPU:

| Fixture | Size | Provenance | Use |
|---|---|---|---|
| `tiny-random/gemma-4-dense` | ~2 MB | Current `default_config.py` default | vLLM + inference tests |
| `masint/tiny-random-llama` | ~2 MB | Commented in `default_config.py` | Training tests (language-model mode) |
| `yujiepan/qwen3-moe-tiny-random` | ~5 MB | Commented in `default_config.py` | MoE training lifecycle (no convergence asserted) |
| `prajjwal1/bert-tiny` | ~17 MB | Upstream HF | Embedding-mode training (`training_mode: embedding`) |

All four have permissive licenses and ship with their own tokenizer. They must be **pre-downloaded into the image's `/root/.cache/huggingface`** via a fixture-download step in the Dockerfile's test stage, so that no test blocks on a network call.

### 3.2 Synthetic datasets

```python
# test/fixtures/tiny_dataset.py
TINY_LM = [{"input": f"q{i}", "output": f"a{i}"} for i in range(16)]
TINY_EMBED = [{"query": f"q{i}", "positive": f"p{i}", "negative": f"n{i}"} for i in range(16)]
TINY_CHAT = [{"messages": [{"role":"user","content":f"q{i}"},
                           {"role":"assistant","content":f"a{i}"}]} for i in range(16)]
```

16 examples is enough to: split cleanly across 2 "ranks" for DataLoader tests, produce at least one epoch boundary at `max_steps=3` with `gradient_accumulation_steps=1`, and not stress the dataset tokenizer.

### 3.3 Short-step budget

Every training test uses:

```python
TRAIN_ARGS_MINIMAL = {
    "max_steps": 3,                         # not 100
    "learning_rate": 1e-6,                  # below anything that matters
    "batch_size": 1,
    "gradient_accumulation_steps": 1,       # not 4 — we want one optimizer step to match one "step"
    "gpus": 1,
    "nodes": 1,
    "steps_per_checkpoint": 2,              # hit CheckpointCallback once
    "max_checkpoints_to_keep": 2,
    "timeout": 120,                         # 2 minutes is plenty for 3 steps on CPU + tiny model
    "distribution_strategy": "none",        # single-rank path — see §3.5
    "llm_name": "masint/tiny-random-llama",
}
```

Any test asserting convergence is wrong. Tests assert: loss is a finite float, step counter advances, checkpoint file exists, status transitions QUEUED → TRAINING → COMPLETED.

### 3.4 Server-start fixtures

Three shapes for bringing up the control plane, all running inside the CPU
container:

| Fixture | What it starts | When to use |
|---|---|---|
| `asgi_testclient` | `app` from `api/fastapi/main.py` under `httpx.AsyncClient(transport=ASGITransport(app))`, with `VLLMModelManager` and `find_model` monkey-patched to a stub, work-queue pointed at a `tmp_path` SQLite file. Runs in-process within the container. | Routing, request-validation, queue bookkeeping — no real vLLM, no sbatch |
| `server_api_only` | `await start_cray_server(server_list=["api"])` (same shape as `test/infra/health.py`) — in-process, but the full FastAPI app | SDK ↔ API HTTP round-trips that need a real ASGI loop but not vLLM |
| `server_full_cpu` | `./scripts/start_one_server.sh` running SLURM + API + vLLM + Generate Worker inside the same container | Full closed-loop E2E: generate → training → register → generate-with-adapter |

All three are driven from inside the CPU container by `pytest`; `./scalarlm
test` spawns the container so the contributor never runs Python directly on
the host. The `asgi_testclient` fixture is the critical one — it keeps ~80 %
of the plan running against a real FastAPI app but without paying for vLLM
initialization or an sbatch round-trip.

### 3.5 Single-rank training path

The existing training loop assumes MPI is initialized. On CPU without `mpirun`, `get_rank()` still works (`gpu_aware_mpi` initializes with a world size of 1), and collectives reduce to no-ops. But to keep the CPU test path honest, we add `distribution_strategy: none` handling in `apply_distribution_strategy.py` that skips FSDP/DDP entirely and wires `backward_sync` to a no-op. This exists in the codebase (`distribution/no_distribution.py`) and should be the default for CPU tests.

### 3.6 Stub vLLM adapter

For tests in `asgi_testclient`, we provide a `StubVLLMModelManager`:

```python
class StubVLLMModelManager:
    def __init__(self):
        self._models = {"tiny-random/gemma-4-dense": "tiny-random/gemma-4-dense"}
    def find_model(self, name):
        return self._models.get(name) or self._models.get("tiny-random/gemma-4-dense")
    def get_registered_models(self):
        return list(self._models.values())
    def register_model(self, path): ...
```

Plus a `StubGenerateWorker` coroutine that polls `/v1/generate/get_work`, *fakes* the vLLM response (echoes the prompt back), and `POST /v1/generate/finish_work`. This exercises everything on the queue path (push, fill_work_queue, update_and_ack, poll_for_responses) without requiring a real engine.

---

## 4. Layer 1 — Big-Picture (End-to-End) Tests

These tests sit above the system and assert contracts the user sees. Each one touches multiple components; a failure means "something user-visible broke, drill into the mid-level tests to find what."

### 4.1 OpenAI-compatible chat (synchronous)

**Contract** (architecture.md §3.2.1, §12.1): `POST /v1/chat/completions` returns a valid OpenAI-shape response against the base model. Same for `/v1/completions`.

**Tests:**
- `test_chat_completions_ok_shape` — 200 response, `choices[0].message.content` non-empty string, `usage.total_tokens` is a positive int.
- `test_chat_completions_streaming` — `stream: true` returns `text/event-stream`, emits ≥ 1 `data:` chunk before `[DONE]`, chunks are valid JSON with `choices[0].delta`.
- `test_completions_ok_shape` — legacy completions endpoint, same assertions.
- `test_chat_unknown_model_404` — `model: "nope/nope"` returns 404 with `detail` present.
- `test_openai_tokenize` — `POST /v1/openai/tokenize` returns `{tokens: [int, ...]}` for a known string (length roughly matches expected tokenizer output).

**Fixture:** `server_full_cpu` with `SCALARLM_MODEL=tiny-random/gemma-4-dense`. No need for the queue path.

### 4.2 Generate queue round-trip (async SDK)

**Contract** (inference-queue.md §4): `SupermassiveIntelligence().generate(prompts)` submits through the queue, polls, and returns results in submission order. Covers both paths:

**Tests:**
- `test_generate_small_batch` — 4 prompts, each gets a non-None string response, responses ordered to match input.
- `test_generate_large_batch_upload_path` — 130 prompts (crosses the 128 threshold in `async_cray.py:31`), SDK automatically switches to `/v1/generate/upload`.
- `test_generate_idempotent_resubmit` — submit the same 4 prompts twice back-to-back; second call returns identically (exercises the `fill_work_queue` skip at `get_work_item.py:72` + response-file existence check).
- `test_generate_dedupe_upload` — POST the same file hash via upload twice; second call returns same `request_id` without enqueuing (upload.py:65 dedupe).
- `test_generate_poll_timeout_returns_none` — override `response_timeout=2` in config, submit a prompt the stub worker will never finish (inject a sleep); client gets `response=None`, not an exception.

**Fixture:** `asgi_testclient` + `StubGenerateWorker`. The stub worker echoes prompts back so assertions on response content work.

### 4.3 Training submission + lifecycle

**Contract** (training-lifecycle.md): `llm.train(data, train_args)` uploads, the server writes `config.yaml`, sbatch queues, harness advances QUEUED → TRAINING → COMPLETED, a checkpoint lands, and `list_models` surfaces it.

**Tests:**
- `test_train_submit_creates_job_directory` — POST succeeds, `job_directory` returned, directory exists on disk, `config.yaml` matches the submitted `train_args` merged with required server-side fields (`dataset_hash`, `training_data_path`).
- `test_train_idempotent_resubmit` — second identical submission returns the existing status without re-running sbatch (`launch_training_job.py:46`).
- `test_train_dataset_hash_changes_directory` — modify one byte of the dataset; job directory must change.
- `test_train_short_run_to_completion` — submit with `TRAIN_ARGS_MINIMAL`, poll `get_training_job` until `status=COMPLETED` (or fail after 180 s), assert a `checkpoint_*.pt` file exists and `history` has ≥ 1 entry.
- `test_train_list_models_includes_completed` — after completion, `llm.list_models()` contains the job hash.
- `test_train_status_shape` — the returned dict has `status`, `start_time`, `job_id`, `max_steps` keys.
- `test_train_fails_on_invalid_llm_name` — `llm_name: "__does_not_exist__/__never__"` ends with `status=FAILED` and a non-empty `error` field in `status.json`.
- `test_train_chains_across_slurm_slices` — set `max_train_time=10`, `signal_grace_seconds=3`, and `megatron_refresh_period=2` server-side, submit with `train_args["timeout"]=25` and `max_steps=20`. Poll `status.json` and assert: (a) `job_id` changes at least once across the run (proving the reconciler queued a new sbatch), (b) `accumulated_train_seconds` monotonically increases across slices, (c) terminal status is `COMPLETED` with `step` close to `max_steps`. Verifies the unified chain in training-lifecycle.md §5.4 — SLURM `--time` → SIGTERM → trainer checkpoints → status stays `TRAINING` → `restart_megatron_jobs` notices the job missing from `squeue` → new sbatch → next slice resumes. Allow 90 s.

**Fixture:** `server_full_cpu`. SLURM is already running inside the container from `start_slurm.sh`. Training budget < 60 s on CPU with a tiny random model.

### 4.4 Closed-loop inference with a trained adapter

**Contract** (adapters.md §6, architecture.md §12.5): After a training run completes, the next `generate` request that names the job hash routes through the freshly-loaded adapter — same pod, no restart.

**Tests:**
- `test_closed_loop_generate_with_new_job_hash` — (a) complete a training run (from §4.3), (b) wait up to 2 × `megatron_refresh_period`, (c) call `generate(prompts=["hi"], model_name="<job_hash>")`, assert 200 and a response. Asserts *registration* happens, not quality.
- `test_closed_loop_latest_alias` — same but with `model_name="latest"`, asserts resolution through `get_latest_model`.
- `test_closed_loop_auto_register_before_refresh` — immediately after training completes, before the 30 s refresh tick, `generate(model_name=<hash>)` still succeeds via `VLLMModelManager.find_model` tier 3 (auto-register from disk, adapters.md §6.2 tier 3).

**Fixture:** `server_full_cpu`. Uses real vLLM to load a real tokenformer checkpoint.

### 4.5 Cancel / delete

**Contract** (training-lifecycle.md §8): `POST /v1/megatron/cancel/{hash}` issues `scancel`, `status.json` flips. `POST /v1/megatron/delete/{hash}` removes the job dir.

**Tests:**
- `test_cancel_running_job` — submit a long-looking job (set `max_steps=10000`, tiny model — not going to finish in 2 s), poll until `TRAINING`, cancel, assert status transitions to `FAILED` or `QUEUED` within 10 s.
- `test_delete_removes_directory` — after completion, `llm.delete(hash)` → directory gone, subsequent `get_training_job(hash)` returns 404.
- `test_delete_running_job_without_cancel_restarts` — delete without cancelling first; `restart_megatron_jobs` *would* re-launch. Assert the documented-but-surprising behavior is preserved (training-lifecycle.md §8 note).

### 4.6 Health + metrics

**Contract** (architecture.md §3.2.4, inference-queue.md §6):

**Tests:**
- `test_health_shape` — `GET /v1/health` returns `{api, vllm, slurm}` dict (granular health, ui-design.md §13).
- `test_health_keepalive` — 200 with `{"status":"ok"}`.
- `test_metrics_counters_increment` — take a snapshot, run 3 generate calls, take another; `total_completed_requests` increased by 3, `total_completed_tokens` increased by ≥ 3.
- `test_metrics_token_per_second_after_requests` — after work, `token/s` is finite and positive (doesn't NaN the 0-epoch transition).
- `test_prometheus_endpoint_exposes_known_metrics` — `GET /v1/metrics` returns Prometheus text format containing `queue_depth`, `queue_wait_time_seconds`, `vllm_*`.
- `test_generate_metrics_zero_when_idle` — counters are 0 or absent before any call, not negative or NaN.

### 4.7 Bashly CLI (`./scalarlm`)

**Contract** (architecture.md §6): Top-level developer CLI.

**Tests:** These run *outside* Docker against the locally checked-out repo.
- `test_cli_help_succeeds` — `./scalarlm --help` exit 0, prints known command names (`up`, `test`, `build-image`, `llm`).
- `test_cli_subcommand_help` — `./scalarlm up --help`, `./scalarlm build-image --help` each exit 0 and mention allowed targets (`cpu`/`nvidia`/`arm`/`amd`/`spark`).
- `test_cli_unknown_target_rejected` — `./scalarlm up windows` exits non-zero (bashly validates `allowed`).
- `test_cli_llm_subcommands_exist` — `./scalarlm llm --help` lists `plot`, `logs`, `ls`, `squeue`.
- *(Not tested)* — `./scalarlm up` / `build-image` / `test` actually running Docker. Those belong in CI smoke, not in the unit suite.

### 4.8 PyPI CLI (`scalarlm`)

**Contract** (architecture.md §5.3, `sdk/masint/cli/main.py`): Post-install CLI.

**Tests:** Use `subprocess.run([sys.executable, "-m", "masint.cli.main", ...])` with a running `asgi_testclient`.
- `test_pypi_cli_ls` — `scalarlm ls`, exit 0, output mentions at least the base model.
- `test_pypi_cli_squeue` — `scalarlm squeue`, exit 0, plain output.
- `test_pypi_cli_stats` — `scalarlm stats`, exit 0, prints counters.
- `test_pypi_cli_logs_unknown_model_graceful` — `scalarlm logs --model nope`, non-zero exit, clear error on stderr, no Python traceback.
- `test_pypi_cli_plot_empty_data_graceful` — `scalarlm plot --model <hash with 0 history>` doesn't crash; skip-or-warn.
- `test_pypi_cli_honors_scalarlm_api_url` — `SCALARLM_API_URL=http://127.0.0.1:<port>` environment overrides default (configuration.md §4).
- `test_pypi_cli_cancel_requires_model_flag` — `scalarlm cancel` without `--model` → non-zero exit, argparse error message.

### 4.9 UI route reachability

**Contract** (ui-design.md §1.5): `/`, `/app/`, `/app/*` SPA routes, `/app/api-config.json`, `/app/assets/*`.

**Tests:** Still HTTP-level — these test the FastAPI-side serving, not the React behavior.
- `test_root_redirects_to_app` — `GET /` returns 302 to `/app/`.
- `test_app_serves_index_html` — `GET /app/` returns 200 and HTML starting with `<!doctype html>` (or `<!DOCTYPE html>`).
- `test_app_deep_link_returns_index_html` — `GET /app/train/abc123`, `GET /app/chat/xyz`, etc., all return 200 with index.html (SPA history routing).
- `test_app_api_config_shape` — `GET /app/api-config.json` returns `{api_base, version, default_model, features}` with string values for the first three and an object for `features`.
- `test_app_assets_cache_header` — `GET /app/assets/<any-hashed-js>` has `Cache-Control` containing `max-age=` with a large value (or absent, meaning the mount is healthy; see `setup_ui.py`).
- `test_app_bundle_missing_falls_through_to_404` — with `SCALARLM_UI_BUNDLE_DIR=/tmp/does-not-exist`, `/app/` returns 404 and the API itself still works (`/v1/health` still 200). Regressions here would mean a backend-only image dies at boot.

### 4.10 UI React bundle sanity

Build-time only — no headless browser.
- `test_ui_npm_build_succeeds` — `cd ui && npm run build` exits 0 (CI gate, not a pytest).
- `test_ui_typecheck_clean` — `cd ui && npm run typecheck` exits 0.
- `test_ui_bundle_under_size_budget` — `cd ui && npm run size` exits 0 against the target in ui-design.md §3 (< 5 MB gzipped JS, < 500 KB first-load).
- `test_ui_lint_clean` — `cd ui && npm run lint` exits 0, `--max-warnings 0`.

### 4.11 Observability stack wiring (optional)

These duplicate `test/infra/test_observability_data.py` but tighten it.
- `test_prometheus_can_scrape_scalarlm` — requires the observability compose up; asserts the scrape target is `UP`.
- `test_otel_trace_emitted_on_generate_call` — requires tempo; assert a trace with the expected service name appears within 5 s of a request.

Gate with `@pytest.mark.skipif(not os.getenv("SCALARLM_OBSERVABILITY_UP"))` so they don't run by default.

---

## 5. Layer 2 — Mid-Level Component Tests

Each test targets one component in isolation. When a Layer 1 test fails, these are where to look for the specific root cause.

### 5.1 Inference work queue — `api/work_queue/`

Covers inference-queue.md §2–§5.

- `test_inference_queue_put_then_get_returns_same_payload` — basic `InferenceWorkQueue.put` / `get` round-trip on a `tmp_path` SQLite.
- `test_inference_queue_ack_marks_acked` — after `ack(id)`, `get_if_finished(id)` returns the row; `get()` does not re-serve it.
- `test_inference_queue_unack_timeout_recycles_eligible_rows` — seed an `unack` row with `timestamp` older than `inference_work_queue_ack_timeout`, call `restart_unacked_requests_from_queue`, assert the row is back to `ready` (inference-queue.md §5.2).
- `test_inference_queue_auto_resume_on_reopen` — open queue, `get()` a row (puts in unack), close object, reopen with `auto_resume=True`, assert row is back to `ready`.
- `test_fill_work_queue_skips_when_response_exists` — write a fake `{group}_response.json` to `upload_base_path`, enqueue matching row; `fill_work_queue` must skip it (inference-queue.md §4.3).
- `test_push_into_queue_initializes_in_memory_results` — exercises `get_in_memory_results.py` state initialization.
- `test_acquire_file_lock_serializes_writers` — two coroutines contending for the same `.lock`, exactly one gets in at a time, 30 s timeout respected.
- `test_update_and_ack_writes_response_file_before_ack` — introspect ordering: response file on disk before SQLite ack flips, verified via a fault injection that raises between the two.
- `test_clear_queue_drains_sqlite` — puts 5 rows, `clear_queue` → `len(queue) == 0`.

### 5.2 Work-queue worker loop — `one_server/create_generate_worker.py`

- `test_worker_batch_size_from_kv_cache` — mock `engine_client.get_current_kv_cache_size` → 2048 and `engine_client.model_config.max_model_len` → 256; `get_batch_size` returns `min(2048/256, generate_batch_size) == 8`.
- `test_worker_loads_new_adapter_via_get_adaptors_response` — `get_work` response carries `new_adaptors=["hash1"]`; worker calls `load_lora_adapter` exactly once; `loaded_adaptor_count` increments.
- `test_worker_retries_failed_adapter_load` — `load_lora_adapter` raises first call, worker does not increment `loaded_adaptor_count`; second `get_work` re-delivers and succeeds (adapters.md §6.5).
- `test_worker_sleeps_when_no_work` — empty queue, worker sleeps roughly `inference_work_queue_timeout` + idle 1 s; not spin-looping.
- `test_worker_sets_worker_ready_while_idle` — `ready_worker_idle_start_time` updated on idle entry/exit (consumed by ack-timeout recycling, inference-queue.md §5.2).

### 5.3 VLLMModelManager — `training/vllm_model_manager.py`

- `test_model_manager_finds_base_model` — `find_model("tiny-random/gemma-4-dense")` returns it immediately (tier 1).
- `test_model_manager_returns_registered_model` — register `"hash-a"`, `find_model("hash-a")` returns (tier 2).
- `test_model_manager_auto_registers_from_disk` — create `{jobs_dir}/hash-b/checkpoint_1.pt`, `find_model("hash-b")` returns and auto-registers (tier 3, adapters.md §6.2).
- `test_model_manager_returns_none_for_unknown` — `find_model("totally-fake")` → None → caller's 404.
- `test_sorted_by_start_time` — register three models with different `start_time` files, `get_registered_models()` returns in ascending start_time order.
- `test_concurrent_find_model_no_duplicate_registration` — two asyncio tasks race on `find_model` for the same unknown hash; exactly one `register_model` call actually added.

### 5.4 Periodic tasks — `api/fastapi/tasks/add_megatron_tasks.py`

- `test_register_megatron_models_discovers_new_pt_files` — drop a `.pt` into `{jobs_dir}/h1/`, run the task once, manager now contains `h1`.
- `test_restart_megatron_jobs_resubmits_queued_not_in_squeue` — seed a `status.json` with `QUEUED` + a `job_id` not in `squeue`; stub `start_slurm_job`; assert it was called once per such row (training-lifecycle.md §6.3).
- `test_restart_megatron_jobs_resubmits_training_not_in_squeue` — same shape with `status=TRAINING`. This is the long-jobs path (training-lifecycle.md §5.4): a slice that exited cleanly via SIGTERM leaves status at TRAINING, and the reconciler must queue the next slice without operator intervention.
- `test_restart_megatron_jobs_skips_completed_jobs` — seed `status=COMPLETED` with no slurm job; assert `start_slurm_job` is NOT called. The status discipline in `MegatronTrainer` is what makes the reconciler safe to run unconditionally.
- `test_restart_megatron_jobs_keeps_pod_alive_when_training` — with a `TRAINING` status in squeue, assert a `POST /v1/health/keepalive` call is made.
- `test_clear_acked_requests_gc` — seed 10 acked rows, run task once, 0 rows left in SQLite.
- `test_tasks_recover_from_individual_failure` — one task raises, next tick still runs the others.

### 5.5 Training launch / SLURM command construction — `training/launch_training_job.py`

- `test_create_slurm_run_command_clamps_gpus` — `train_args={"gpus": 999}` against a `scontrol` stub with 1 GPU → `--ntasks-per-node=1`.
- `test_create_slurm_run_command_no_gpu_when_cluster_cpu_only` — `is_gpu_job` false → no `--gres=gpu:*` flag.
- `test_get_train_time_limit_caps_at_max_train_time` — request `timeout=10*86400` (10 days), config `max_train_time=86400` → walltime is 86400 + `extra_training_seconds`. The per-slice cap (training-lifecycle.md §3.2) does NOT shrink the user's total budget — that's enforced separately by `TimeoutCallback`.
- `test_get_train_time_limit_uses_user_value_when_below_cap` — `timeout=600`, `max_train_time=86400` → walltime is 600 + `extra_training_seconds` (small jobs run as a single slice, no relaunch).
- `test_get_train_time_limit_formats_dd_hh_mm_ss` — 3725 s → `"0-01:02:05"` (`format_timedelta`).
- `test_create_slurm_run_command_emits_signal_flag` — argv contains `--signal=B:TERM@<N>` where `N == signal_grace_seconds` from config (training-lifecycle.md §3.2). `B:` prefix is mandatory — targets the batch shell so the entrypoint can call `sbatch` before being killed.
- `test_job_directory_is_content_addressed` — same `train_args` + dataset → same directory; changing one byte → different directory.
- `test_entrypoint_template_replaces_config_path` — `REPLACE_CONFIG_PATH` in the copied script is replaced with the absolute job-config path (training-lifecycle.md §3.3).
- `test_run_sbatch_scrubs_pmi_env` — parent env has `PMI_RANK=3`, subprocess env does not.
- `test_run_sbatch_writes_failed_on_nonzero_exit` — stubbed `subprocess.run` returns exit 1; `status.json` is `FAILED` with `output`.
- `test_upload_training_data_rejects_oversized_body` — `MaxBodySizeValidator` triggers HTTP 413 (training-lifecycle.md §2.1).

### 5.6 TrainingHarness — `ml/cray_megatron/megatron/training_harness.py`

- `test_update_status_merges_metadata` — two consecutive `update_status` calls with different metadata; both keys present in `status.json`.
- `test_save_status_is_main_rank_only` — run under stubbed `get_size()=2, get_rank()=1`, assert file not written; rerun under `get_rank()=0`, file written.
- `test_checkpoint_writes_torch_file` — call `checkpoint({"x": 1}, "checkpoint_2.pt")`, file is readable via `torch.load`.
- `test_get_status_returns_existing_json` — seed a status.json, `get_status` returns the dict verbatim.

### 5.7 Training loop state machine — `ml/cray_megatron/megatron/training_loop.py`

These use the tiny-random-llama model with 3 steps on CPU.

- `test_training_loop_advances_step_counter` — `max_steps=3`, run end-to-end, final `status.json` has `step: 3` (or `step: 2` if 0-indexed — document in the test).
- `test_training_loop_checkpoints_on_completion` — even with `max_steps=3` and `steps_per_checkpoint=100`, a final checkpoint exists (training-lifecycle.md §5.1 invariant).
- `test_training_loop_checkpoint_callback_fires_on_period` — `max_steps=4`, `steps_per_checkpoint=2` → two periodic checkpoints plus one final. Also asserts `delete_old_checkpoints` keeps only `max_checkpoints_to_keep`.
- `test_training_loop_resumes_from_latest_checkpoint` — run once, kill with `SIGCONT`, relaunch; final `step > previous final step` (training-lifecycle.md §5.2).
- `test_training_loop_nan_forward_tolerance` — patch model forward to return NaN once, then valid; run 3 steps, `nan_steps == 1`, training completes.
- `test_training_loop_timeout_callback_stops_on_total_budget` — `train_args["timeout"]=2`, patch forward to sleep 3 s, loop exits early via `TimeoutCallback`, status `COMPLETED` with step < max_steps.
- `test_training_loop_timeout_callback_honors_accumulated_seconds` — pre-seed `status.json` with `accumulated_train_seconds=1.5`, `train_args["timeout"]=2`, patch forward to sleep 0.6 s. Loop must stop on the first step (slice elapsed 0.6 s + prior 1.5 s > 2 s budget), not after 2 s of slice time. Verifies that the budget is total-across-slices, not per-slice (training-lifecycle.md §5.4).
- `test_training_loop_update_history_capped` — set `training_history_length=8`, run 20 steps, history has exactly 8 entries with roughly uniform step coverage (verifies `remove_closest_entry` behavior).
- `test_training_loop_sigterm_checkpoints_cleanly` — send SIGTERM mid-loop, loop exits via `stop_flag`, the post-loop `self.checkpoint()` runs (verifies a fresh `.pt` lands), `_finalize_slice` advances `accumulated_train_seconds > 0`. Status stays at whatever was last written (typically `TRAINING`) — `restart_megatron_jobs` (§6.3) is what queues the next slice (training-lifecycle.md §5.4).
- `test_training_loop_sigcont_same_handling_as_sigterm` — send SIGCONT mid-loop; same behavior as SIGTERM at the trainer level (drain, checkpoint, persist accumulated_seconds). The reconciler picks both up via the same status-missing-from-squeue path.
- `test_finalize_slice_persists_accumulated_seconds_across_slices` — run slice A (`accumulated_train_seconds` ends at ~`A`), simulate relaunch by reusing the same job dir + harness, run slice B (loop starts with `accumulated_seconds_at_slice_start == A`, ends at ~`A+B`). This is the cross-slice invariant that makes the user's total `timeout` honored.
- `test_finalize_slice_preserves_other_status_keys` — pre-seed `status.json` with `job_id`, `start_time`, `history`, and an existing `status`; after `_finalize_slice` runs, those keys are still present and `status` is unchanged (regression guard for the read-modify-write contract; `_finalize_slice` never touches status, only `MegatronTrainer` does).
- `test_megatron_trainer_skips_completed_when_signal_seen` — mock `TrainingLoop.train()` to set `stop_flag` and exit; `MegatronTrainer.train_loop()` does NOT call `update_status(COMPLETED)`. Without this guard, `restart_megatron_jobs` would (correctly) leave the job alone instead of queuing the next slice — this test pins the status discipline that makes §5.4 work.
- `test_megatron_trainer_writes_completed_when_loop_exits_cleanly` — no signal received; `MegatronTrainer.train_loop()` writes `COMPLETED`. Confirms the inverse — the reconciler stops respawning when the job is genuinely done (max_steps reached or user's total budget exhausted via `TimeoutCallback`).

### 5.8 Dataset loading — `ml/cray_megatron/megatron/dataset/`

- `test_load_language_model_dataset_tokenizes` — 16 examples → tokenized tensor; len > 0.
- `test_load_embedding_dataset_triplets` — `training_mode=embedding` path yields (query, positive, negative) triples batched correctly.
- `test_data_loader_epoch_advances` — iterate more than len(dataset), `data_loader.epoch` increments.
- `test_load_dataset_dispatch_by_mode` — `language_model` vs `embedding` vs unknown-mode-raises (`load_dataset.py`).
- `test_max_token_block_size_respected` — large dataset + small `max_token_block_size` → chunks all ≤ block size.

### 5.9 Adapter surgery — `ml/tokenformer/tokenformer_surgeon.py` + `ml/adapters/`

- `test_tokenformer_surgeon_wraps_only_mlp_suffix` — build a `nn.Module` with `model.layers.0.mlp`, `model.layers.0.attention`, `model.lm_head`; surgery wraps exactly the first one.
- `test_tokenformer_p_zero_at_init` — `TokenformerAdapter.tokenformer_p` is all zeros (adapters.md §2.1).
- `test_tokenformer_forward_is_identity_at_init` — `y = model(x)` before and after wrapping are element-wise equal at `p=0`.
- `test_create_tokenformer_model_freezes_expected_modules` — after wrapping, only modules matching the name set in `create_tokenformer_model.py:52-65` have `requires_grad=True`.
- `test_create_lora_model_applies_peft` — adapter_type=lora path produces a PEFT-wrapped model with `unwrap_model` patched.
- `test_filter_checkpoint_keeps_only_trainable` — LoRA path saves only `requires_grad=True` weights.
- `test_train_lm_head_auto_for_small_models` — model with <100M params → `train_lm_head=True`; ≥100M → false.

### 5.10 vLLM-side TokenformerModelManager — `vllm/tokenformer/tokenformer_model_manager.py`

These already have fixture-based tests in `test/integration/vllm/`. The mid-level plan tightens them.

- `test_tokenformer_manager_init_runs_surgery_when_supports_tokenformer` — mock model flagged as `supports_tokenformer` → manager's model has tokenformer layers; otherwise passes through.
- `test_add_adapter_respects_cache_capacity` — `TOKENFORMER_CACHE_CAPACITY=2`, add 3 adapters, first one evicted.
- `test_activate_adapter_preserves_original_tensors` — activate A, activate B, activate A → weights are bitwise-identical to the base model + A (adapters.md §5.3).
- `test_activate_adapter_skips_lora_keys` — load a checkpoint with keys containing `"lora"`, verify they are skipped during activation (adapters.md §9 pitfall).
- `test_from_local_checkpoint_raises_on_missing` — extends existing `test_tokenformer.py:37-45`.

### 5.11 Config loader — `infra/cray_infra/util/`

- `test_config_defaults_when_no_yaml_and_no_env` — clean env + non-existent YAML path → every field equals the Pydantic default.
- `test_config_yaml_overrides_default` — write `model: other-model` into YAML, assert `get_config()["model"] == "other-model"`.
- `test_config_env_overrides_yaml` — set `SCALARLM_MODEL=env-model` with YAML present, env wins.
- `test_config_env_casts_to_field_type` — `SCALARLM_MAX_MODEL_LENGTH="8192"` is an int; `SCALARLM_GPU_MEMORY_UTILIZATION="0.9"` is a float; `SCALARLM_AUTO_RESUME="true"` is True (configuration.md §1.1).
- `test_config_env_reject_invalid_cast` — `SCALARLM_MAX_MODEL_LENGTH="hello"` raises ValueError.
- `test_job_config_requires_three_server_fields` — `JobConfig(**{})` raises because `job_directory` / `training_data_path` / `dataset_hash` are required.
- `test_hf_token_env_wins` — `HF_TOKEN=x` → `get_hf_token()` returns `"x"` regardless of config.
- `test_hf_token_encrypted_fallback` — neither env nor plaintext set → Fernet-decrypted value returned (configuration.md §3.1).

### 5.12 SDK engine — `sdk/masint/engines/async_cray.py`

Mocked `aiohttp.ClientSession`; no server needed.

- `test_async_cray_generate_small_uses_generate_endpoint` — ≤128 prompts → one POST to `/v1/generate`, then polling.
- `test_async_cray_generate_large_uses_upload_path` — 200 prompts → POST to `/v1/generate/upload` + poll `/v1/generate/download` (async_cray.py:31 threshold).
- `test_async_cray_url_builder_honors_module_api_url` — `scalarlm.api_url = "http://other"` affects the URL; env var `SCALARLM_API_URL` likewise (configuration.md §4).
- `test_async_cray_is_finished_raises_on_error_field` — result dict contains `error`, polling raises immediately.
- `test_submit_training_job_streams_tar_no_buffering` — assert `aiohttp.MultipartWriter` with streaming file_sender; peak memory stays below 2× one chunk (64 KB).
- `test_submit_training_job_builds_deterministic_tar` — `tar_info_strip_file_info` → same inputs → same tar bytes (training-lifecycle.md §1.3).
- `test_submit_training_job_includes_local_ml_dir` — when `./ml/` exists, archive contains a `ml/` member; otherwise doesn't.

### 5.13 Periodic task task-level interactions

- `test_megatron_refresh_period_respected` — patch `time.monotonic` + `asyncio.sleep`, one "tick" per `megatron_refresh_period` seconds; no more.
- `test_add_megatron_tasks_shuts_down_on_app_close` — lifespan exit cancels the periodic task cleanly; no warning about un-awaited coroutine.

### 5.14 UI unit tests (non-React)

Run under `node --test` or `vitest` — no React rendering.

- `test_ui_tar_builder_produces_valid_ustar` — `buildTar([{name: "dataset.jsonl", data: ...}])` output is a valid tar when read back by Python's `tarfile.open`.
- `test_ui_zod_schema_matches_python_defaults` — `defaultTrainArgs()` values == corresponding Python `JobConfig` defaults (trainArgsSchema mirrors JobConfig). Cross-language snapshot test.
- `test_ui_zod_schema_rejects_negative_max_steps` — `validateTrainArgs({max_steps: -1})` returns `ok: false` with a field-level error.
- `test_ui_api_config_fallback_when_404` — mock `fetch` to 404; `loadApiConfig()` resolves to the fallback (`/v1`, `unknown`).
- `test_ui_api_error_thrown_on_non_2xx` — `apiFetch` mock returns 500 → throws `ApiError` with `status=500`, `body` populated.
- `test_ui_sse_parser_handles_chunked_data` — synthetic `ReadableStream` yielding `data: {"delta":"a"}\n\ndata: {"delta":"b"}\n\n` → parser emits two chunks in order.
- `test_ui_sse_parser_stops_on_done_sentinel` — `data: [DONE]\n\n` terminates the async iterator.
- `test_ui_rolling_buffer_evicts_oldest` — `useRollingBuffer(N=3)` push 4 items → first is evicted, order preserved.

### 5.15 UI data layer (React component tests)

Use `@testing-library/react` + a fetch mock (`msw` or hand-rolled).

- `test_metrics_page_renders_from_prometheus_fixture` — mount `MetricsPage`, mock `/v1/generate/metrics` with a canned payload, assert throughput numbers render.
- `test_metrics_page_refetch_interval_3s` — with `@tanstack/react-query` + fake timers, verify a second fetch fires at 3 s.
- `test_train_index_shows_running_and_completed` — mock `/v1/megatron/list_models` + `/v1/megatron/squeue`, assert both appear as cards sorted by start_time desc.
- `test_train_detail_renders_loss_chart` — mock status with `history: [{step: 1, loss: 1.2}, ...]`, assert `LossChart` renders an `<svg>` with uPlot.
- `test_submit_modal_submits_valid_form` — fill in required fields + drop a JSONL, click Submit, assert `POST /v1/megatron/train` was called with the expected multipart body.
- `test_submit_modal_blocks_invalid_form` — `max_steps: 0`, submit button disabled or form shows validation error.
- `test_chat_page_streams_assistant_message` — user sends "hi", mocked SSE stream returns `"hello"` over 3 chunks, final DOM shows `"hello"` progressively. Stop button aborts.
- `test_conversation_store_persists_across_reloads` — write a conversation, simulate page reload (teardown + rerender with fresh QueryClient), conversation is re-read from IndexedDB.

---

## 6. Layer 3 — Unit and Sanity Tests

Small, fast, in-process. Any of these failing localizes a bug to a specific function.

### 6.1 Small utilities

- `test_make_api_url_joins_correctly` — `make_api_url("v1/health", api_url="http://x:8000/")` → `http://x:8000/v1/health` (no double slash, no missing slash).
- `test_get_api_base_precedence_order` — matrix covering 6 precedence levels of configuration.md §4.
- `test_quiet_loggers_does_not_override_explicit_levels` — set `filelock` to DEBUG before calling `quiet_noisy_loggers()`; verify it's *not* overwritten (if that's the intended behavior — document).
- `test_format_timedelta_boundary_values` — 0 s, 59 s, 3600 s, 86399 s, 1 day + 1 h.
- `test_get_contents_hash_stable_across_process_restarts` — same input, different PID, identical hash.
- `test_truncate_string_edge_cases` — len=100 (boundary), empty string, multi-byte characters.
- `test_format_timedelta_rejects_negative` — if `extra_training_seconds < 0` somehow trips in, either clamp or raise; document and test.
- `test_group_request_id_path_helpers` — `{id}.json`, `{id}_status.json`, `{id}_response.json` all constructed with the same id.

### 6.2 Pydantic models — `api/fastapi/routers/request_types/`

- `test_generate_request_accepts_list_of_strings_and_chat_messages` — both `prompts: ["hi"]` and `prompts: [{"role":"user","content":"hi"}]` valid.
- `test_generate_request_rejects_negative_max_tokens` — currently accepted; decide and encode (`test_generation_parameter_limits` in `test_pipeline.py:419` shows this is already inconsistent).
- `test_train_request_response_shape` — `TrainResponse(job_status=..., job_config=..., deployed=False)` serializes with all three keys.
- `test_get_work_request_batch_size_positive` — `{batch_size: 0}` or `< 0` rejected with 422.
- `test_finish_work_requests_accepts_empty_list` — no-op, returns 200.
- `test_get_results_request_request_ids_list` — missing field → 422.

### 6.3 Queue path helpers

- `test_get_work_item_returns_none_when_empty` — `in_memory_work_queue` empty and SQLite empty → `(None, None)`.
- `test_make_id_zero_pads_to_nine` — `make_id("abc", 5)` → `"abc_000000005"`.
- `test_fill_work_queue_expands_one_sqlite_row_to_n_items` — enqueue row pointing to a file with 3 prompts; `fill_work_queue` puts 3 items into `in_memory_work_queue`.
- `test_acquire_file_lock_timeout_raises` — hold the lock in one process, second call times out after 30 s (shorten for the test with a config knob, otherwise this takes 30 s).
- `test_compute_flop_count_matches_formula` — seed a fake config (num_hidden_layers, hidden, num_heads, head_size, intermediate, vocab, num_kv_heads), assert the returned count equals the hand-computed sum from inference-queue.md §6.3.

### 6.4 Collectives — `test/collectives/test_shm_channel.py` (already exists; retain)

Existing standalone Python-only mirror is solid. Keep it in the CPU suite. Add:

- `test_shm_channel_growing_capacity` — force a grow by sending first a 100 MB tensor, then a 500 MB tensor; header/capacity reflect the larger size; data integrity preserved (gpu-aware-mpi.md §5.2).
- `test_shm_channel_duplex_concurrent` — two processes A/B, each both sending and receiving on the same pair; confirm no interleaving.
- `test_peer_sync_uses_sendrecv_not_global_barrier` — read the channel setup path with enough processes and pairs that a `MPI_Barrier` would deadlock; confirm it doesn't (gpu-aware-mpi.md §5.3).

### 6.5 `main_rank_only` — `ml/cray_megatron/collectives/main_rank_only.py`

- `test_main_rank_only_runs_on_rank_zero` — under stubbed `get_rank()=0`, decorated function runs, returns value.
- `test_main_rank_only_returns_none_on_other_ranks` — `get_rank()=1`, function skipped, returns None.
- `test_main_rank_only_barriers_bracket_call` — assert `barrier()` called before and after decorated body.
- `test_main_rank_only_reentrant_safe` — inner `@main_rank_only` call inside outer one: inner runs on rank 0 without double-barrier (documented invariant in architecture.md §13).

### 6.6 MPI lifecycle — `gpu_aware_mpi`

- `test_mpi_get_rank_after_finalize_reinitializes` — call `get_rank()`, `finalize_mpi()`, `get_rank()` again. Second call succeeds.
- `test_dtype_size_mapping_covers_every_torch_dtype_the_code_uses` — sweep over all dtypes from `common.h:38`, assert `get_typesize` returns the expected size.
- `test_mpi_request_lifetime_safe_for_isend` — send a tensor via `isend`, drop the Python reference to the tensor, `mpi_wait` still succeeds (gpu-aware-mpi.md §7.1).

### 6.7 Training lifecycle state helpers

- `test_training_job_status_enum_is_stringified` — `TrainingJobStatus.QUEUED == "QUEUED"` (Pydantic needs this).
- `test_write_job_status_merges_not_overwrites` — seed `{start_time: 1.0}`, call `write_job_status("QUEUED", args, {"job_id": "5"})`; file has both fields (`launch_training_job.py:283`).
- `test_get_existing_job_info_attaches_job_directory_and_model_name` — verify `get_existing_job_info` returns a dict with those two keys derived from the path.
- `test_stop_flag_defaults_clear` — fresh import / after `reset()`, `was_stop_requested()` is False, `last_signal()` is None.
- `test_stop_flag_request_sets_latch_and_signal` — `request_stop(signal.SIGTERM)` → `was_stop_requested()` is True, `last_signal() == signal.SIGTERM`.
- `test_stop_flag_request_without_signal_keeps_prior_signal` — call `request_stop(SIGTERM)` then `request_stop()` (no arg); `last_signal()` is still SIGTERM (don't clobber with None).
- `test_stop_flag_distinguishes_sigterm_from_sigcont` — two flag states from two signal types; `_finalize_slice` (training-lifecycle.md §5.4) keys its relaunch decision on this discrimination.

### 6.8 Observability — `cray_infra/observability/` (optional)

- `test_prometheus_counters_registered` — import the metrics module, assert each expected metric exists on the registry.
- `test_init_tracing_no_crash_without_endpoint` — with no OTEL collector reachable, `init_tracing` logs a warning, does not raise.
- `test_structured_log_middleware_emits_json` — capture log output for a single request; it's valid JSON with `event`, `method`, `status_code`, `duration_seconds`.

---

## 7. Test Organization on Disk

```
test/
├── conftest.py                          # shared fixtures (asgi_testclient, tmp_workdir, …)
├── fixtures/
│   ├── __init__.py
│   ├── tiny_dataset.py                  # TINY_LM / TINY_EMBED / TINY_CHAT
│   ├── stub_vllm.py                     # StubVLLMModelManager + StubGenerateWorker
│   └── server_fixtures.py               # server_api_only, server_full_cpu
│
├── unit/                                # Layer 3, in-process, < 1 s each
│   ├── test_api_url.py
│   ├── test_config_override.py
│   ├── test_hf_token.py
│   ├── test_queue_helpers.py
│   ├── test_pydantic_requests.py
│   ├── test_format_timedelta.py
│   ├── test_main_rank_only.py
│   ├── test_stop_flag.py
│   ├── test_tokenformer_surgeon.py
│   └── test_mpi_lifecycle.py
│
├── component/                           # Layer 2, short, scope one component
│   ├── test_inference_work_queue.py
│   ├── test_work_queue_worker.py
│   ├── test_vllm_model_manager.py
│   ├── test_periodic_tasks.py
│   ├── test_launch_training_job.py
│   ├── test_training_harness.py
│   ├── test_training_loop.py            # tiny-random-llama, 3 steps on CPU
│   ├── test_dataset_loading.py
│   ├── test_adapters.py
│   ├── test_tokenformer_manager.py      # vLLM-side
│   └── test_sdk_engine.py
│
├── e2e/                                 # Layer 1, full ASGI or full container
│   ├── test_openai_chat.py
│   ├── test_generate_queue.py
│   ├── test_train_lifecycle.py
│   ├── test_closed_loop.py
│   ├── test_cancel_delete.py
│   ├── test_health_metrics.py
│   ├── test_bashly_cli.py
│   ├── test_pypi_cli.py
│   └── test_ui_routing.py
│
# UI tests live under ui/ itself so the node:24.2.0 container's bind mount
# of /app/ui picks them up without additional wiring. The test files live
# alongside the SPA source; vitest discovers `ui/test/**/*.test.ts(x)`.
# (Previously documented as test/ui/ — moved into ui/test/ for bind-mount
# simplicity. Pytest-side tests still live under test/.)
│
├── collectives/                          # existing; keep as-is
│   └── test_shm_channel.py + friends
│
└── observability/                        # existing; gate with env marker
    └── test_observability.py
```

Notes:

- `test/infra/` (existing) and `test/integration/` (existing) are left in place but gradually folded into `test/e2e/` and `test/component/` as their assertions get tightened. Don't delete them before the replacement lands.
- `test/deployment/` (existing) stays separate — those are smoke tests for live deployments, not part of the CPU suite.
- `test/benchmark/` is untouched — performance suite.

---

## 8. CI Wiring

All profiles invoke `./scalarlm test --level <name>`. Nothing runs outside
Docker. CI needs only: Docker, the repo checkout, and the `scalarlm` script.

| Profile | Invocation | Where | Budget |
|---|---|---|---|
| **fast** | `./scalarlm test --level fast` | Every PR, every push | < 3 min (after first build is cached) |
| **cpu** | `./scalarlm test --level cpu` | Every PR on main-track files; nightly | < 10 min |
| **live-server** | `test/deployment/` against a provisioned CPU container | Release tag, manual trigger | best-effort |

### 8.1 GitHub Actions sketch

```yaml
# .github/workflows/tests.yml
jobs:
  fast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      # Reuse the built image across runs so the fast profile stays fast.
      - uses: docker/build-push-action@v5
        with:
          context: .
          tags: cray:latest
          load: true
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BASE_NAME=cpu
            VLLM_TARGET_DEVICE=cpu
      - run: ./scalarlm test --level fast --no-build yes

  cpu:
    runs-on: ubuntu-latest
    needs: fast
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          tags: cray:latest
          load: true
          cache-from: type=gha
          build-args: |
            BASE_NAME=cpu
            VLLM_TARGET_DEVICE=cpu
      - run: ./scalarlm test --level cpu --no-build yes --workers 1

  cpu-arm:
    runs-on: ubuntu-24.04-arm
    needs: fast
    # Same shape; Docker builds for linux/arm64 natively on an arm runner.

  live-server:
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - # bring up a full stack on a hosted runner, run test/deployment/
```

The `--no-build yes` flag skips the redundant `./scalarlm build-image` inside
`test_command.sh` since CI has already loaded the tagged image via buildx.

### 8.2 Local developer workflow

The entrypoint is `./scalarlm test`. It accepts a `--level` flag that maps
to the layers defined in §4–§6, plus an optional explicit path for a single
test file. Defined in `cmd/bashly.yml` → `cmd/test_command.sh`.

Every stage runs inside Docker. There is no host-side Python or npm step.

```
./scalarlm test                                     # default: --level cpu (full CPU suite)
./scalarlm test --level unit                         # Layer 3 only
./scalarlm test --level component                    # Layer 2 only
./scalarlm test --level e2e --workers 1              # Layer 1, serialized
./scalarlm test --level ui                           # npm lint + typecheck + test
./scalarlm test --level collectives                  # SHM channel tests
./scalarlm test --level fast                         # unit + ui + collectives (CI fast)
./scalarlm test --level all                          # every non-live level

# Opt-in live inference smoke against an already-built hardware image:
./scalarlm test --level live \
  --live-target spark \
  --model Qwen/Qwen3-0.6B \
  --tag cray:latest \
  --no-build yes

# Filter within a level:
./scalarlm test --level component -k work_queue      # pytest -k pattern
./scalarlm test --level component -m "not slow"      # pytest -m expression

# Explicit path (bypasses --level, still runs in the container):
./scalarlm test test/unit/test_config_override.py
./scalarlm test test/component/test_training_loop.py

# Skip the image rebuild:
./scalarlm test --level e2e --no-build yes
```

Level semantics — every row runs in a container:

| `--level` | Container | What runs |
|---|---|---|
| `unit` | scalarlm CPU image | `test/unit/` via pytest |
| `component` | scalarlm CPU image | `test/component/` via pytest |
| `e2e` | scalarlm CPU image | `test/e2e/` via pytest |
| `ui` | `node:24.2.0` | `ui/` — `npm run lint`, `typecheck`, `test` |
| `collectives` | scalarlm CPU image | `test/collectives/test_shm_channel.py` via pytest |
| `fast` | both | `unit` + `collectives` + `ui` |
| `cpu` | both | `fast` + `component` + `e2e` *(default)* |
| `live` | selected ScalarLM image | Provisions a real server, then checks health, model listing, synchronous ScalarLM API, queued SDK, and streaming inference |
| `all` | both | every non-live level; `live` remains opt-in because it downloads and loads a model |

Pytest stages run inside `$tag` (default `cray:latest`, overridable with
`--tag`). The UI stage runs inside `node:24.2.0` with `ui/` bind-mounted, so
the same toolchain that builds the production bundle (per the Dockerfile's
`ui_builder` stage) also runs the tests.

Each stage reports pass/fail independently; `./scalarlm test` exits non-zero
if **any** stage failed, prints a green summary only when all pass.

Minimum host toolchain: **Docker**. That's it — the Python interpreter lives
inside the image, the node toolchain lives inside `node:24.2.0`, and the
`./scalarlm` wrapper regenerates the bashly CLI inside its own container via
`cmd/bashly.sh`.

The live profile creates uniquely named server and test containers plus a
Docker network, waits until both the API and vLLM report healthy, and runs its
pytest smoke checks on that network. `--keyword` / `--mark` are forwarded to
the live pytest process. The harness manages the test process explicitly and
removes both containers and the network on success, failure, SIGINT, or
SIGTERM. Set `SCALARLM_MODEL_CACHE` to override the default persistent cache
at `./models`. On failure, the harness prints the final 400 server-log lines
before cleanup.

### 8.3 Fail-fast and parallelism

- `pytest -n auto --dist=loadgroup` for Layer-2/3; Layer-1 single-process because they share the SQLite queue file and the SLURM daemon.
- Timeouts: `@pytest.mark.timeout(30)` default, overridden to 180 s for training-to-completion tests, 5 s for unit tests.
- Retries: **none** by default. Flakes are bugs to fix, not statistics to dampen. Explicit `@pytest.mark.flaky(reruns=1)` only when an external dependency (HF Hub) is involved, after logging in the test why.

---

## 9. Rollout Phases

Bite-sized so each delivers value on its own.

### Phase 0 — Prep (1 week)

- Add `test/conftest.py` with `asgi_testclient`, `stub_vllm`, `tmp_workdir` fixtures.
- Add `test/fixtures/` tiny datasets and stub classes.
- Pre-download the four tiny models in the Dockerfile CPU test stage.
- Replace the `status_code in [200, 400, 404, 422, 500]` patterns in `test/integration/api/test_scalarlm_api.py` with specific-code assertions (or delete them if a sharper test already exists in this plan).

### Phase 1 — Unit layer (1 week)

Deliver `test/unit/` top-to-bottom. All new tests, all in-process, none need Docker.

### Phase 2 — UI layer (2 weeks)

`test/ui/` plus the CI `fast` profile gating `npm run build`, `lint`, `typecheck`, `size`, `test`. Blocks regressions in the new React SPA before it ships user-visible.

### Phase 3 — Component layer (3 weeks)

`test/component/` — most require the container. Bring up the CPU image in CI; run against it.

### Phase 4 — E2E layer (3 weeks)

`test/e2e/` against the full container, including closed-loop training-to-generate and the UI routes.

### Phase 5 — Polish and cleanup (ongoing)

- Retire or rewrite anything in `test/integration/` / `test/infra/` that duplicates the new coverage.
- Fill in the observability and benchmark gate if/when those projects are load-bearing on a PR.

---

## 10. Known Gaps (Intentional)

Things this plan does **not** address, and the rationale:

- **GPU-backed inference correctness.** Requires hardware; covered by `deployment/` on real clusters.
- **Multi-node SLURM.** Requires a second machine; the SLURM command builder is covered in 5.5 with a `scontrol` stub, the multi-rank execution is not.
- **Real NCCL/RCCL parity.** The `gpu_aware_mpi` plan tests the MPI + SHM paths; collective correctness on NVIDIA/AMD is out of scope.
- **Huge model regressions.** 27B, 32B, 120B model deploys are release-candidate concerns; CPU can't validate them. Manual / live-cluster smoke remains the check.
- **Long training convergence.** A one-step tiny-random run asserts the plumbing works. Learning dynamics are out of scope.
- **Prometheus / Grafana dashboard correctness.** The dashboard JSON can be linted; rendering is a human concern.
- **Security.** Injection, ACLs, auth — ScalarLM has no auth layer (ui-design.md §Non-Goals). Add a security pass when auth lands.

---

## 11. Test Writing Conventions

- **Name tells you what breaks.** `test_fill_work_queue_skips_when_response_exists` beats `test_queue_behavior`.
- **One assert per concept.** Use separate test functions for distinct properties. Parametrize when the shape is the same and only an input varies.
- **Arrange in fixtures, act in the test body, assert once at the end.** Explicit ordering, easy to read.
- **No live HF downloads on the hot path.** Anything that calls `from_pretrained` against an HF model ID goes through the pre-populated cache in the image. A test that starts with `⚠ downloading 1.1 GB` is not a valid CPU test.
- **No `assert status_code in [...]`** — be precise. If the true contract is "either 200 or 404 depending on state," write two tests that each force the relevant branch.
- **Use `tmp_path` / `tmp_path_factory`** for all disk state. No tests write to `/app/cray/jobs/` directly at the test level — always via a fixture that points `training_job_directory` elsewhere.
- **Close what you open.** `async with start_cray_server(...)` or an explicit teardown in `asyncTearDown`. The existing `test/infra/health.py` pattern is the template.
- **No host-side toolchain assumptions.** Tests run inside the scalarlm CPU image (pytest) or `node:24.2.0` (npm). Anything that would require a specific Python or Node version on the contributor's laptop is a bug.
- **Fail loudly for unexpected success.** A test that asserts 404 but gets 200 should report "the route was supposed to 404 — has a new behavior been introduced?" — not quietly skip with `pytest.skip(...)`.

---

## 12. Key File Reference

| Concern | File |
|---|---|
| Existing test runner | `cmd/test_command.sh` |
| Existing unified runner | `test/scripts/run_tests.sh` |
| Existing pytest reqs | `test/requirements-pytest.txt` |
| Existing infra tests | `test/infra/` |
| Existing integration tests | `test/integration/` |
| Existing deployment tests | `test/deployment/` |
| Existing collective tests | `test/collectives/` |
| New shared fixtures | `test/conftest.py` *(to add)* |
| New tiny dataset + stubs | `test/fixtures/` *(to add)* |
| Per-layer test trees | `test/unit/`, `test/component/`, `test/e2e/` *(to add)* |
| UI test entry point | `ui/package.json` `"test"` script *(to add)* |
| CI profiles | `.github/workflows/tests.yml` *(to add)* |
