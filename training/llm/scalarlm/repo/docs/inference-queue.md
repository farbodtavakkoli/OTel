# The Inference Work Queue

ScalarLM's batched inference path is the least standard piece of the system. Where OpenAI-compatible `/v1/chat/completions` flows straight through to vLLM, the `/v1/generate` path detours through a persistent work queue: a SQLite-backed acknowledgement queue on disk, a two-tier in-memory cache for hot reads, and a pull-based worker that batches requests against vLLM's KV cache. This document explains why that exists, what its invariants are, and every component that makes it work.

This is the deep-dive companion to §3.3 and §12.2–12.3 of `architecture.md`.

---

## 1. Why a Queue Exists At All

There are four forces pushing ScalarLM toward a queue instead of a direct `request → vLLM → response` path:

1. **Large-batch inference.** The SDK switches to the queue path whenever `len(prompts) > 128` (`sdk/masint/engines/async_cray.py:31`). 10k-prompt evaluation sweeps and post-training sample generation are first-class use cases, not exotic.
2. **Durability.** vLLM can OOM, the pod can be rolled, the GPU can page. Without a queue, every in-flight batch dies with the process. With the queue on a mounted PVC, requests outlive restarts.
3. **KV-cache-aware batching.** The Generate Worker sizes each pull by asking vLLM how much KV cache is currently free. That feedback loop can't happen if clients push directly — the system needs a buffer to pull from.
4. **Dedup on resubmission.** SHA-256 content hashing of the full request set lets a retrying client land on the same queue slot rather than duplicating work. Every input file path is `{sha256}.json`.

The whole queue infrastructure exists to give these four properties cheaply, in a single pod, without a separate broker.

---

## 2. Storage Model

### 2.1 The SQLite acknowledgement queue

At the lowest level, the queue is a `persistqueue.SQLiteAckQueue` — a crash-safe FIFO with explicit acknowledgement.

- **File:** `/app/cray/inference_work_queue.sqlite` (`default_config.py:46` — `inference_work_queue_path`).
- **Wrapper:** `cray_infra/api/work_queue/inference_work_queue.py:13` — `InferenceWorkQueue`.
- **Payload schema:** Each row stores a dict:
  ```python
  {"path": "/app/cray/inference_requests/{sha256}.json",
   "request_count": N,
   "timestamp": <set on put()>}
  ```
  The individual prompts are **not** in the queue row — they live in the JSON file at `path`. The queue carries only a pointer.

Rows progress through three states:

```
ready  ──put()──►  ready  ──get()──►  unack  ──ack()──►  acked
                                        │
                                        └─nack()/resume_unack_task()─► ready
```

`persistqueue.AckStatus` exposes these as integers so you can filter the full snapshot. `InferenceWorkQueue.get_unacked_requests` (L79) walks the entire queue and returns all rows currently in `unack` — used by the expiry logic in §5.2.

### 2.2 Why pointers, not payloads

Storing prompts in the SQLite row would force the entire batch through SQLite's write path on every enqueue and through the queue's cursor on every peek. Storing a path makes enqueue constant-size and lets the worker read its batch sequentially from a flat JSON file — which is much cheaper for 10k-prompt batches. It also makes acknowledgement trivial: `ack(id)` flips the row state but doesn't touch the prompt file.

The tradeoff is an extra file per batch. Those live under `upload_base_path` (default `/app/cray/inference_requests`):

| File | Written by | Purpose |
|---|---|---|
| `{sha256}.json` | `generate.py` or `upload.py` | Array of request dicts. Keyed by content hash so identical submissions share a file. |
| `{sha256}_status.json` | `push_into_queue.py`, `update_and_ack.py` | `{"status": "in_progress"\|"completed", "current_index": K, "total_requests": N, "work_queue_id": id, "completed_at": t?}` |
| `{sha256}_response.json` | `update_and_ack.py` on completion | Full in-memory results dict serialized to disk, for `download` and re-reads. |

The queue row's `path` is always the `{sha256}.json` file; the other two are derived (`group_request_id_to_{status,response}_path.py`).

### 2.3 Asyncio serialization

`InferenceWorkQueue` wraps every public method in a single `asyncio.Lock` (L16). SQLite itself is process-safe, but `persistqueue` assumes single-threaded usage of a given queue object — the lock is there to serialize FastAPI coroutines that share the singleton. The module-level `get_inference_work_queue()` (L140) lazily constructs the singleton and returns it to every handler.

---

## 3. Two-Tier In-Memory Cache

Between the SQLite row and the Generate Worker there are two in-memory caches, each solving a different problem:

### 3.1 `in_memory_work_queue` — staged prompts

`cray_infra/api/work_queue/get_work_item.py:20` holds a list of `(request, id)` tuples representing the *unpacked* contents of the current JSON file. Workers pull from this list; the module refills from SQLite only when the list is empty.

```python
async def get_work_item(work_queue):
    async with lock:
        if not in_memory_work_queue:
            await fill_work_queue(work_queue)
        if not in_memory_work_queue:
            return None, None
        return in_memory_work_queue.pop(0)
```

`fill_work_queue` (L55) does the pointer-dereference: pulls one row from SQLite, opens the file it points to, expands all N prompts into the staging list, and attaches a synthetic per-prompt ID `{group_request_id}_{index:09d}` (L90 — `make_id`). Critically, it's behind a **file lock** (`acquire_file_lock`) on the JSON file so that a concurrent rewrite can't race with the read.

If `response_path` already exists when `fill_work_queue` dequeues a row, it skips that row entirely (L72) — the response was persisted previously and the row is a stale retry. The worker should never reprocess an already-completed batch.

### 3.2 `in_memory_results` — staging outputs

`cray_infra/api/work_queue/get_in_memory_results.py:4` holds the *in-progress* response aggregator, keyed by group_request_id:

```python
{
  "<group_id>": {
    "results": {
        "<group_id>_000000000": {"response": "...", "is_acked": true},
        "<group_id>_000000001": {"is_acked": false},   # still outstanding
        ...
    },
    "current_index": 42,
    "total_requests": 100,
    "work_queue_id": <pqid>,
  }
}
```

`current_index` is the count of acked results so far. When it reaches `total_requests`, the whole batch is promoted to disk as `{group}_response.json`, the status file flips to `completed`, and the SQLite row is `ack()`'d (see `finish_work_queue_item`, §4.3). On the `ack`, the in-memory entry is cleared.

This cache is what makes per-prompt polling in `poll_for_responses` cheap: partial results are already keyed by ID in memory; no file I/O per poll.

Both caches are protected by module-level `asyncio.Lock`s.

---

## 4. Request Path, End-to-End

Two producers, one queue, one consumer. The producers are `generate` (small-batch JSON) and `upload` (large-batch file); the consumer is the Generate Worker.

### 4.1 Producer: small batch — `generate.py`

`infra/cray_infra/api/fastapi/generate/generate.py:31` handles `POST /v1/generate`:

1. Resolve `model`: user-specified → `config["model"]` → `get_latest_model()` for `"latest"`. Run it through `VLLMModelManager.find_model` (registers lazily from disk if the model is a training job dir containing `.pt`). 404 if still unknown.
2. Build one request dict per prompt with `{prompt, model, max_tokens, temperature, tools, tool_choice, request_type: "generate"}`. Call `Metrics.record_new_request()` for each (queue_depth++, sets epoch_time on the 0→1 transition).
3. `contents_hash = sha256(json.dumps(requests))` — the SHA-256 of the request array itself. This is the `group_request_id`.
4. Write `requests` to `{upload_base_path}/{group_id}.json`.
5. `push_into_queue(len(requests), path)` (`push_into_queue.py:8`):
   - Lazily create the `in_memory_results` entry (with `total_requests=N`).
   - Put `{path, request_count}` into SQLite.
   - Write `{group_id}_status.json` = `{status: in_progress, current_index: 0, total_requests: N, work_queue_id: id}`.
6. `poll_for_responses(group_id)` blocks in-process until results or timeout.

The synchronous-looking return from `/v1/generate` is actually "enqueue, then poll the in-memory cache until done or `response_timeout` (default 60 s)." Timed-out prompts come back with `response=None`, not as an exception.

### 4.2 Producer: large batch — `upload.py`

`infra/cray_infra/api/fastapi/generate/upload.py:32` is the streaming alternative at `POST /v1/generate/upload`. The SDK picks this path when `len(prompts) > 128`.

- Streams the upload via `streaming_form_data.StreamingFormDataParser` straight to disk, so memory never holds the full batch.
- Enforces `max_upload_file_size` (10 GB default) per part and 2× that for the total body, with early HTTP 413 on overflow.
- Hashes the written file with SHA-256 (64 KB chunks), computes `{hash}.json` as the canonical path.
- **Dedup.** If that path already exists, deletes the temp file and returns the existing hash — no re-enqueue. This is the resubmission-is-idempotent property at the inference layer.
- Otherwise moves the temp file into place, counts requests with `get_request_count` (`json.load` + `len`), and calls `push_into_queue`.
- Returns `{request_id: <hash>}`. The SDK then polls `/v1/generate/download` (`download.py`), which serves the completed `{hash}_response.json` once it exists, with exponential backoff while waiting.

### 4.3 Consumer: the Generate Worker

`infra/cray_infra/one_server/create_generate_worker.py:58` is an in-process asyncio task in the API container. It does **not** run vLLM out-of-process over HTTP; it holds a reference to the in-process vLLM `app` (via `server_status.get_app()`, set by `create_vllm.py`) and calls its handler functions directly. This saves one serialization round-trip per batch.

Worker loop (L82):

```
loop forever:
    clear_finished_tasks(tasks)
    batch_size = await get_batch_size(app)       # KV-cache sized; 0 → sleep 100ms
    POST /v1/generate/get_work {batch_size, loaded_adaptor_count}
        ↓
    apply new adaptors (LoadLoRAAdapterRequest against the in-process vLLM)
    if no requests: sleep 1s, continue
    tasks.append(asyncio.create_task(process_requests_task(app, requests)))
```

`process_requests_task` (L274):

- `asyncio.gather` of one `async_generate_task` per request. Each dispatches to `async_chat_completion_task` or `async_completion_task` depending on whether `prompt` is a string (completion) or a dict/list (chat).
- Each task builds a `ChatCompletionRequest`/`CompletionRequest` and calls vLLM's `create_chat_completion` / `create_completion` directly, bypassing HTTP.
- Extracts `response`, `error`, `token_count`, `flop_count` (see §6.3).
- `POST /v1/generate/finish_work` with the results. Dispatch is asynchronous; the worker doesn't await completion before pulling the next batch (it's fire-and-forget into `tasks[]`, cleaned later).

`get_batch_size` (L170) is the load-feedback loop:

```python
current = await engine_client.get_current_kv_cache_size()
max_model_length = engine_client.model_config.max_model_len
batch  = current // max_model_length                     # conservative
return min(batch, config["generate_batch_size"])        # cap at 1024
```

The division uses the runtime engine's `max_model_len` and assumes every request could consume a full context window. That's intentionally pessimistic — it prevents OOM on worst-case long-output batches at the cost of occasionally underfilling when outputs are short. `generate_batch_size` (default 1024) is the hard ceiling.

### 4.4 `/v1/generate/get_work` — the dequeue endpoint

`get_work.py:21` is what the worker calls. Signature-wise it accepts `{batch_size, loaded_adaptor_count}` and returns `{requests: [...], new_adaptors: {...}}`.

- `worker_ready()` / `worker_not_ready()` (`clear_acked_requests_from_queue.py:53–60`) bracket the blocking first pull. They set `ready_worker_idle_start_time` so the expiry logic in §5.2 knows whether a worker is actually available to retry unacked items.
- First pull uses `get_work_item` (blocking up to `inference_work_queue_timeout` — default 30 s). Subsequent pulls up to `batch_size-1` use `get_work_item_no_wait`. This guarantees the worker either gets at least one prompt or waits, but doesn't waste time spinning for more once the in-memory list is empty.
- `get_adaptors(request)` returns any newly-trained Tokenformer/LoRA adapters the worker hasn't loaded yet — piggy-backing adapter discovery on the work-pull round trip.

### 4.5 `/v1/generate/finish_work` — the ack endpoint

`finish_work.py:15`: for each result, merge into the in-memory entry and try to finalize.

`update_and_ack` (`update_and_ack.py:20`) is the commit point. It:

1. Looks up the in-memory entry for the group.
2. If not already acked, bumps `current_index`.
3. Replaces the per-prompt result dict with the finished one and sets `is_acked: true`.
4. If `current_index >= total_requests`, calls `finish_work_queue_item`:
   - Under `acquire_file_lock(response_path)`:
     - Write full aggregate to `{group}_response.json`.
     - Update status file to `{status: completed, completed_at, current_index}`.
   - `inference_work_queue.ack(work_queue_id)` — flips SQLite row from `unack` to `acked`.
   - `clear_in_memory_results(group_id)` — drops the cached aggregate.
5. Metrics: `record_completed_request(token_count, flop_count)` — decrements queue_depth, accumulates time/tokens/flops.

Note the ordering: file-on-disk first, SQLite-ack second, memory-clear last. If the process dies between those steps, the `skip already processed` check in `fill_work_queue` ensures the batch isn't re-served (the response file exists, so the row is skipped even though not acked), and the eventual `clear_acked_data` cleanup catches up.

---

## 5. Reliability Semantics

### 5.1 Crash recovery on startup

`get_inference_work_queue()` (`inference_work_queue.py:140`) constructs the singleton with `auto_resume=True`. That tells `persistqueue.SQLiteAckQueue` to treat any rows left in `unack` state from a previous run as `ready` — they go back into the queue on boot.

Practical effect: if the worker dies mid-batch (vLLM OOM, process crash, pod eviction), the SQLite row the worker had pulled is still `unack`. On next boot, `auto_resume=True` moves it to `ready`, `fill_work_queue` picks it up again, and processing resumes. The individual in-memory results are gone, but the source JSON file is still on disk — the batch is reprocessed from scratch.

Idempotency at the response layer comes from the existence check in `fill_work_queue` (L72) — if `{group}_response.json` already exists, the row is skipped even without an ack.

### 5.2 Ack timeouts

Not every failure crashes the process. Some just stall — a hung GPU, a deadlock inside vLLM, a runaway `max_tokens` batch. For those, the queue has a soft-expiry mechanism in `restart_unacked_requests_from_queue` (`generate/clear_acked_requests_from_queue.py:26`), called every 30 s from the FastAPI lifespan task loop (`add_megatron_tasks.py:33`).

The expiry rule has **two** conditions that both need to be true before a stuck request is recycled:

```python
time_since_submit       > config["inference_work_queue_ack_timeout"]   # default 300s
ready_worker_idle_time  > config["inference_work_queue_idle_time"]     # default 5s
```

The first is the obvious one: the request has been pulled for too long. The second is subtler — it checks that a worker has actually been sitting idle for at least 5 seconds recently (via `worker_ready()`/`worker_not_ready()` bookkeeping). You don't want to recycle a long-running batch just because a single worker is saturated; you only recycle when there's spare capacity available to retry.

If both conditions hold, `resume_unack_task(id)` → `queue.nack(id)` puts the row back in `ready`, and some later `get_work` call will pick it up again.

### 5.3 Acked-row cleanup

`clear_acked_requests_from_queue` (L12) calls `queue.clear_acked_data()` which physically deletes acked rows from SQLite. Without this the DB would grow forever. Runs every 30 s in the lifespan loop, so a completed request is fully gone from SQLite within one refresh tick.

### 5.4 Admin nuke

`POST /v1/generate/clear_queue` → `clear_queue.clear_queue()` → `InferenceWorkQueue.clear_queue` (L122) drains the queue by dequeuing and ack'ing everything in a loop, then runs `clear_acked_data`. Response files on disk are not touched — this clears pending work, not history.

---

## 6. Metrics

### 6.1 In-memory counters

`cray_infra/generate/metrics.py:6` — `Metrics` class, singleton via `get_metrics()`. Five counters:

| Field | Updated by | Meaning |
|---|---|---|
| `queue_depth` | `record_new_request()` ++ / `record_completed_request()` -- | Currently outstanding requests. |
| `epoch_time` | Reset to `now` on each 0→N transition | Start of current non-idle epoch. |
| `total_completed_requests` | `record_completed_request()` | Monotonic. |
| `total_completed_tokens` | From `response_data["usage"]["total_tokens"]` | Monotonic. |
| `total_completed_flops` | `compute_flop_count(model_config) * total_tokens` | Monotonic. |
| `total_completed_response_time` | `now - epoch_time` accumulated per completion | Excludes idle periods between epochs. |

### 6.2 What `total_completed_response_time` actually measures

This isn't the sum of per-request latencies — it's the **total time the queue was non-empty**. The epoch starts when `queue_depth` transitions 0→1 and is advanced on every completion by `(now - epoch_time)`. Idle periods between epochs don't count.

The rates derived from it are therefore *utilization-adjusted*:

```python
"token/s":   total_tokens   / total_response_time
"request/s": total_requests / total_response_time
"flop/s":    total_flops    / total_response_time
```

These are "when we were busy, how fast were we" — not "over all of wall time, how much did we do." That's the metric you actually want for sizing; wall-clock throughput in a mostly-idle cluster looks artificially low.

Exposed at `GET /v1/generate/metrics` (`generate/metrics.py`).

### 6.3 FLOP counting

`compute_flop_count` (`create_generate_worker.py:418`) computes forward-pass FLOPs per token for transformer-style models by summing:

- Q/K/V projection: `hidden × (n_heads × head_size)` + `hidden × (n_kv_heads × head_size × 2)`
- Attention product: `2 × n_heads × head_size`
- Output projection: `hidden²`
- MLP up+down: `hidden × intermediate + intermediate × hidden`
- Embedding + output LM head: `2 × hidden × vocab`

All per layer, multiplied by `num_hidden_layers`. Then multiplied by total tokens in the response for a per-request FLOP count.

This is a first-order approximation — it ignores softmax, activations, and non-transformer blocks (Mamba, MoE routing). For gpt-oss and Nemotron 3 Super (hybrid architectures) the numbers will be biased low. Good enough for roofline plots against hardware peak.

---

## 7. Multiple Related Batches, Same Hash

Two callers submit the same batch. What happens?

- **Small-batch (`generate`):** Hash matches → `{hash}.json` path matches → `push_into_queue` writes a second `{hash}_status.json` (overwriting the first) and enqueues a second SQLite row. Both workers will eventually pull *some* row, find the response file already exists from the first completion, skip it in `fill_work_queue`, and move on. Duplicate row becomes a no-op. Not perfectly ideal, but correct.
- **Large-batch (`upload`):** Hash matches → the dedup check in `upload.py:65` catches it before enqueueing. Temp file is deleted; the existing `request_id` is returned. No second row.

The upload path dedups explicitly; the generate path relies on response-file idempotency downstream. Either way, the same inputs never produce duplicated inference work.

---

## 8. File Layout Reference

After a successful `POST /v1/generate` with N prompts, the relevant artifacts are:

```
/app/cray/
├── inference_work_queue.sqlite           # persistqueue DB (rows for this batch
│                                          #   may already be in `acked` state)
└── inference_requests/                   # upload_base_path
    ├── {group_id}.json                   # [{prompt, model, ...}, ...] × N
    ├── {group_id}_status.json            # {status: "completed", current_index: N,
    │                                      #   total_requests: N, completed_at: ...}
    └── {group_id}_response.json          # full in_memory_results serialized
```

`{group_id}` is the SHA-256 of the request array (small-batch) or the SHA-256 of the uploaded file (large-batch).

---

## 9. Concurrency Map

Three kinds of locking are in play; getting them confused is the common pitfall when extending this code.

| Lock | Scope | Lives in | Guards |
|---|---|---|---|
| `InferenceWorkQueue.lock` | Per-queue-object `asyncio.Lock` | `inference_work_queue.py:16` | Single-threaded access to `persistqueue` calls. |
| `get_work_item.lock` | Module-level `asyncio.Lock` | `get_work_item.py:19` | `in_memory_work_queue` staging list. |
| `get_in_memory_results.lock` | Module-level `asyncio.Lock` | `get_in_memory_results.py:3` | `in_memory_results` aggregator. |
| `acquire_file_lock(path)` | Sidecar `.lock` file | `acquire_file_lock.py:7` | Multi-process-safe serialization around `{path}.json` / `{path}_response.json`. Uses `O_CREAT\|O_EXCL`, polls every 100 ms, 30 s timeout. |

Sidecar file locks are the *only* cross-process synchronization primitive. They're used where files could be read by one coroutine and rewritten by another — notably around `fill_work_queue`'s read of `{group}.json` and `finish_work_queue_item`'s write of `{group}_response.json` + status file. Asyncio locks are sufficient for everything in-process.

---

## 10. Configuration Reference

Queue-relevant fields from `infra/cray_infra/util/default_config.py`:

| Field | Default | Meaning |
|---|---|---|
| `inference_work_queue_path` | `/app/cray/inference_work_queue.sqlite` | SQLite DB location. Put on a PVC in Kubernetes. |
| `upload_base_path` | `/app/cray/inference_requests` | Where `{hash}.json` / `_status.json` / `_response.json` live. Also on a PVC. |
| `inference_work_queue_timeout` | 30 (s) | Max time `get_work_item` blocks waiting for SQLite to produce a row. The worker's aiohttp client doubles this as its HTTP timeout. |
| `inference_work_queue_idle_time` | 5 (s) | Minimum idle-worker time before stuck-request recycling is allowed. |
| `inference_work_queue_ack_timeout` | 300 (s) | Max age of an `unack` row before it's eligible for recycling. |
| `generate_batch_size` | 1024 | Ceiling on per-pull batch size (worker side). |
| `max_model_length` | 0 (auto) | Fallback admission cap when vLLM runtime introspection is unavailable; `0` disables the fallback cap. Worker batching uses the runtime engine's `max_model_len`. |
| `response_timeout` | 60 (s) | Max time `poll_for_responses` blocks in `/v1/generate` and `/v1/generate/get_results`. Timed-out prompts return `response=None`. |
| `max_upload_file_size` | 10 GB | Upper bound on upload batch file size. |

---

## 11. Failure Mode Catalog

| Failure | Detection | Recovery |
|---|---|---|
| API pod restart, nothing in flight | — | `auto_resume=True` is a no-op; queue resumes normally. |
| API pod restart, worker had pulled a batch | SQLite row state = `unack` on boot | `auto_resume=True` nacks it back to `ready`; next worker pulls it; idempotent re-processing thanks to response-file check. |
| API pod restart, batch had written `_response.json` but not yet acked SQLite | Response file exists | `fill_work_queue` skips the row on next dequeue (L72), worker never reprocesses. Row eventually cleaned by `clear_acked_data` after a manual ack or falls through the expiry path. |
| Worker hung for >300 s on a single batch | `restart_unacked_requests_from_queue` | Nacks the row if an idle worker is also available; batch restarts. |
| Worker crash mid-gather | `tasks[]` with exceptions → `clear_finished_tasks` logs | Next loop iteration pulls more work. Specific prompts in the crashed gather are lost and rely on the 300 s ack timeout to recycle the parent row. |
| Duplicate submission | SHA-256 collision-free under normal inputs | Upload path dedups pre-enqueue; generate path dedups via `fill_work_queue` skip + in-memory idempotency. |
| Client disconnect mid-upload | `ClientDisconnect` caught in `upload.py` | Logged as warning; temp file abandoned (next gc/tmp cleanup will remove). No queue row created. |
| Oversized upload | `MaxBodySizeException` / `ValidationError` | HTTP 413 returned. Temp file deleted by `streaming_form_data` on exception. |
| Model unknown | `VLLMModelManager.find_model` returns None | HTTP 404 returned from `generate()` before any queue/file work. |
| Poll timeout (`response_timeout` exceeded) | `poll_for_responses` breaks out of loop | Unfinished prompts return `response=None`. Queue row stays in flight; ack-timeout path can still recycle it. Client may re-poll `/v1/generate/get_results`. |

---

## 12. Design Notes and Non-Obvious Choices

**Why in-process vLLM, not HTTP?** The Generate Worker holds a direct reference to the vLLM FastAPI `app` via `server_status.get_app()` and invokes vLLM handler functions directly. The port-8001 HTTP server exists for debugging and for the chat-proxy path; the hot path is in-process. Saves a serialization round-trip per request and lets the worker call `engine_client.get_current_kv_cache_size()` synchronously.

**Why poll, not push?** WebSocket or SSE would reduce polling overhead. But polling over a simple POST endpoint means clients are stateless — any API replica can serve any `/v1/generate/get_results` call and read from the shared filesystem. SSE would bind clients to specific replicas. For single-pod ScalarLM it doesn't matter; for the Helm chart with `api_deployment` scaled past 1, it does.

**Why SHA-256 over UUID4 for group IDs?** Determinism. Identical submissions produce identical paths, which produce free idempotency. A UUID-based scheme would need an explicit "have I seen this payload before?" index.

**Why split `status.json` and `response.json`?** Status is hot-read during polling; response is cold until complete. Separating them keeps the poll-path file small (a few fields vs. the full result set), and lets `download` stream the response file as-is via `FileResponse` without rewriting.

**Why not `FastAPI BackgroundTasks`?** They run after the response is returned and die with the process. They don't survive restarts, don't dedup, and don't throttle. They're a different tool for a different problem.

**Why the `0→1` epoch reset in metrics?** Without it, a cluster that's been idle for 6 hours would accumulate 6 hours of "response time" the next time a request came in, tanking the throughput average. Resetting on each non-idle epoch keeps `token/s` representative of how the system performs when actually under load.
