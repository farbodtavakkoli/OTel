# The OpenAI Chat Completions Queue

ScalarLM's OpenAI-compatible `/v1/chat/completions` endpoint currently flows straight through to vLLM. That works for low and moderate traffic, but breaks under bursts: vLLM accepts requests beyond `max_num_seqs`, the KV cache exhausts, and clients see hangs followed by timeouts.

This document specifies a unified queueing layer that absorbs bursts and stays compatible with a vanilla `AsyncOpenAI()` client — **no `max_retries`, `timeout`, or `stream=True` tuning needed from the user**.

The design has two top-level commitments that distinguish it from the previous draft:

1. **Every inference request — `/v1/chat/completions` and `/v1/generate` alike — goes through the existing SQLite-backed `InferenceWorkQueue`.** There is one queue, one worker, one set of metrics, and one durability guarantee. The OpenAI-compatible path becomes a thin handler in front of the same machinery `/v1/generate` already uses.
2. **`/v1/generate` adopts the same chat-template rendering as the OpenAI path.** Each entry in `/v1/generate`'s `prompts: [...]` batch can carry either `prompt: str` (raw passthrough, unchanged) or `messages: [...]` (conversation turns rendered through the model's chat template). Note the distinction: in `/v1/generate` the top-level array is a *batch of independent requests*, while `messages: [...]` *inside* one of those requests is conversation turns of that single request — same semantics as OpenAI's `chat.completions.create(messages=...)`.

This is the live-traffic addition to `inference-queue.md`. That doc remains the deep-dive on the queue mechanics; this doc covers the new components in front of it (chat template rendering, admission control, coalescing, result routing, the heartbeat transport) and the changes to `/v1/generate`.

---

## 1. Why this exists

Three forces push us toward a queueing layer in front of `chat/completions`:

1. **vLLM has hard caps.** Beyond `max_num_seqs`, vLLM's internal pending queue grows unbounded; eventually KV cache allocation fails and sequences abort with errors. We need flow control upstream of vLLM to prevent that.
2. **Default OpenAI client timeouts are short.** The Python SDK's default request timeout is 600 s. Any path that holds an HTTP connection longer than that fails — and "longer than 10 min" happens easily under sustained load.
3. **The user shouldn't have to tune anything.** A senior engineer who has never read our docs should be able to write `await AsyncOpenAI().chat.completions.create(...)` and get a correct response back, even when the server is under heavy load. No `timeout=`, no `max_retries=`, no `stream=True`.

A fourth force pushes us to consolidate onto the existing queue rather than build a parallel in-memory one:

4. **One queue is cheaper to operate than two.** The existing `InferenceWorkQueue` already has dedup via SHA-256 content hashing, crash recovery via `auto_resume=True`, ack-timeout recycling, KV-cache-aware worker batching, and a metrics surface. Building a second in-memory queue would duplicate every one of those features. Better to share.

The throughput concern that originally motivated a separate path — SQLite's ~1000-row practical ceiling — is addressed by the coalescer (§6), which packs up to **`packing_factor`** requests into a single queue row. Effective queue capacity is `packing_factor × row-count-ceiling`; the parameter is the operator's primary knob for trading per-request latency against queue throughput. Coupled with the dedup and the `clear_acked_data` cleanup that already runs every 30 s, sustained throughput well above the original ceiling becomes routine.

---

## 2. Goals and non-goals

**Goals**

- Vanilla `AsyncOpenAI()` works for bursts of up to ~5k concurrent in-flight requests.
- vLLM never exceeds `max_num_seqs` concurrent sequences (no KV-cache OOM).
- `/v1/chat/completions` and `/v1/generate` share the chat template rendering path — same `messages: [...]` rendering on either endpoint.
- All inference requests share the same SQLite queue, the same worker, the same dedup, the same metrics.
- Latency overhead under light load (queue near empty) is under 5 ms per request.

**Non-goals**

- Million-request submissions. Those go through `/v1/batches` with a polling pattern (separate doc). The OpenAI Batch API SDK already handles that protocol natively.
- HTTP/2 or WebSocket. HTTP/1.1 chunked transfer encoding suffices.
- A second, parallel in-memory queue. Explicitly rejected — see §1 force #4.
- Per-tenant admission isolation (per-API-key quotas, fair-share, etc.). Multi-tenancy is handled out-of-band by deploying a separate ScalarLM instance per client; a single deployment is implicitly single-tenant. The admission controller's threshold protects vLLM from total overload, not tenant from tenant.

---

## 3. Architecture overview

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                  API container (FastAPI)                 │
                     │                                                          │
   ┌──────────┐      │   ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
   │AsyncOpenAI│ HTTP│ ┌►│Chat        │─►│Admission │─►│Coalescer │─►│SQLite  │ │
   │  client  │─────┼─┤ │template    │  │ control  │  │ 10/50 ms │  │Inference  │
   └──────────┘  ▲  │ │ │renderer    │  │          │  │          │  │WorkQueue│ │
                 │  │ │ └────────────┘  └──────────┘  └──────────┘  └────────┘ │
   ┌──────────┐  │  │ │       ▲              │                          │      │
   │ /v1/generate ─┼─┘        │              │ 429 + Retry-After        │      │
   │  caller  │  │  │         │              │ when queue over          │      │
   └──────────┘  │  │         │              │ high-water               ▼      │
                 │  │   messages: [...] OR                       ┌────────────┐│
                 │  │   prompt: str                              │ Generate   ││
                 │  │                                            │ Worker     ││
                 │  │                                            │ (existing) ││
                 │  │                                            └────────────┘│
                 │  │                                                  │       │
   chunked-JSON  │  │                                                  ▼       │
   heartbeat     │  │                                            in-process    │
       (§9)      │  │                                              vLLM        │
                 │  │                                                  │       │
                 │  │   ┌───────────────────────────┐                  │       │
                 └──┼───┤ Result router             │◄─────────────────┘       │
                    │   │ correlation_id → Future   │                          │
                    │   └───────────────────────────┘                          │
                    └──────────────────────────────────────────────────────────┘
```

A request travels through five new stages and lands in two existing ones:

1. **Chat template renderer** (§4) — turns `messages: [...]` into a model-specific prompt string. Skipped for `/v1/generate` callers that pass raw `prompt: str`. Shared module used by both endpoints.
2. **Admission control** (§5) — bounded acceptance based on queue depth. Returns `429` with `Retry-After` if over the high-water threshold; otherwise registers a `correlation_id → asyncio.Future` mapping and hands off.
3. **Coalescer** (§6) — accumulates admitted requests into batches of up to 10 or 50 ms, whichever first; bypasses entirely when the queue is below the bypass threshold so light loads pay no batching tax. The output of one coalescer flush is *one* SQLite row.
4. **SQLite `InferenceWorkQueue`** (§7) — the existing durable queue, unchanged. The coalesced batch's request file is hashed, written under `upload_base_path`, and a single row referencing it is inserted.
5. **Generate Worker** (existing, see `inference-queue.md` §4.3) — pulls KV-cache-sized batches from the queue, dispatches in-process to vLLM, calls `/v1/generate/finish_work` with results.
6. **Result router** (§8) — when the worker reports a result, look up the `correlation_id` and resolve the corresponding future. Per-prompt resolution; the coalesced batch's individual requests resolve as their results land.
7. **Heartbeat transport** (§9) — the HTTP handler returns chunked-JSON with whitespace heartbeats while awaiting its future, then writes the final body.

### 3.1 Streaming requests bypass this path

Requests with `stream=True` skip the new infrastructure entirely. SSE already keeps the connection alive natively (each token delta is a real wire-level event, no idle gap to bridge), so the heartbeat layer adds nothing. And the existing direct-to-vLLM streaming handler is already wired up and working.

The handler at `/v1/chat/completions` branches on `stream` at the very top:

- `stream=True` → existing direct-to-vLLM path (no admission, no coalescing, no heartbeat).
- `stream=False` (default) → new path described in §3.

This means streaming clients do not get the new admission-control backpressure. That's intentional for v1: streaming clients already deal with errors at the SSE layer (the SDK surfaces mid-stream errors to the caller naturally), and the volume of streaming traffic is small enough that the existing direct path can absorb it without queue protection. If that changes, see §13 for a streaming-through-queue follow-up.

---

## 4. Chat template rendering (shared stage)

A single module renders chat-style input into a prompt string for the model. Both `/v1/chat/completions` and `/v1/generate` call into it.

### 4.1 The contract

```python
# infra/cray_infra/api/fastapi/chat_completions/render_chat_template.py

def render_chat_template(
    *,
    model: str,
    messages: list[ChatMessage] | None,
    prompt: str | None,
) -> str:
    """
    If `messages` is given, render via the model's tokenizer chat template.
    If `prompt` is given, return it unchanged (raw passthrough).
    Exactly one of the two must be set; raise ValueError otherwise.
    """
```

Implementation calls `AutoTokenizer.from_pretrained(model).apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. The tokenizer is cached on first use per model; subsequent calls are constant-time hash lookups.

### 4.2 Why a shared stage

- **One source of truth for prompt formatting.** A bug in the rendering (wrong system-prompt placement, missing `add_generation_prompt`) would otherwise have to be fixed in two places.
- **Consistent behavior across endpoints.** A user benchmarking `/v1/generate` with `messages` and then running the same workload through `/v1/chat/completions` gets identical model inputs — no hidden divergence in tokenization.
- **`/v1/generate` becomes more useful.** Callers no longer need to know the model's prompt template. The current friction (read the model card, hand-craft `<|im_start|>...<|im_end|>` envelopes) goes away.

### 4.3 Backward compatibility for `/v1/generate`

The renderer is invoked once per *entry* in `/v1/generate`'s `prompts: [...]` batch — each entry is one independent inference request, and the `messages: [...]` field (when present) on that entry is conversation turns of that single request. The two array dimensions are unrelated; a batch of N entries with K turns each invokes the renderer N times, not N×K times.

Existing entries shaped `{"prompt": "...", ...}` are unaffected — the renderer's raw passthrough is a no-op. New entries can be shaped `{"messages": [...], ...}` and get the same rendering as `/v1/chat/completions`. An entry with both `prompt` and `messages` is rejected with HTTP 400 to prevent silent ambiguity.

The SDK can default to `messages` formatting for chat-style use cases without changing wire compatibility.

---

## 5. Admission control

The FastAPI handler for `/v1/chat/completions`:

```python
async def chat_completions(req: ChatCompletionRequest) -> StreamingResponse:
    rendered_prompt = render_chat_template(
        model=req.model,
        messages=req.messages,
        prompt=None,
    )

    if queue_depth_over_high_water():
        wait = estimate_wait_seconds()
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": str(max(1, int(wait)))},
        )

    correlation_id = uuid4()
    future = asyncio.get_running_loop().create_future()
    result_router.register(correlation_id, future)

    coalescer.submit(rendered_prompt, req, correlation_id)

    return StreamingResponse(
        stream_with_heartbeat(future),
        media_type="application/json",
    )
```

`/v1/generate` performs the analogous flow with its own response shape (and may use the existing polling response instead of the heartbeat — see §11.2).

The high-water threshold reads queue depth directly from `InferenceWorkQueue` and the in-flight `result_router`:

```
ADMIT_HIGH_WATER = ADMIT_FACTOR × max_num_seqs        # default ADMIT_FACTOR = 4
queue_depth_over_high_water := (queue.size() + result_router.in_flight_count) > ADMIT_HIGH_WATER
```

The factor of 4 lets the queue absorb a 4× burst above steady-state capacity before pushing back. Beyond that, the OpenAI SDK's built-in `Retry-After` honor handles spillover automatically — the user sees a slowdown, never a failure, with no client-side configuration.

`estimate_wait_seconds` is a moving average of recent batch completion latencies multiplied by `(queue_depth - max_num_seqs) / max_num_seqs`, padded by 1.5×. Padding prevents retry storms when many clients receive identical `Retry-After` values and synchronize their next attempts.

---

## 6. Coalescer

The coalescer reduces SQLite write amplification under sustained load while paying zero latency tax under light load. Without it, every chat completion request would be a separate SQLite row, hitting the well-known throughput wall (~1k rows in flight before the queue grinds). With it, up to `packing_factor` requests become *one* row.

### 6.1 The rule

```
flush trigger 1: accumulator reaches packing_factor requests       → flush
flush trigger 2: window_ms elapsed since first arrival             → flush
fast path:       queue is below bypass_threshold                   → flush immediately
```

Whichever triggers first wins. The fast path means: while the system is under-loaded (queue + in-flight < `bypass_threshold`), every arrival is dispatched immediately, *no batching delay at all*. The window only applies once concurrency rises.

### 6.2 The parameters

All three are operator-facing knobs in `default_config.py`. The defaults are starting points; tune `packing_factor` first when sizing the system against expected QPS.

| Config field | Default | Rationale |
|---|---|---|
| `chat_coalescer_packing_factor` | 10 | Max requests packed into one queue row. Comfortably under typical `max_num_seqs` (256+); large enough to amortize the per-row SQLite cost, small enough to not delay the first arrival's neighbors. The dominant lever on queue capacity — see §1, force #4. |
| `chat_coalescer_window_ms` | 50 | Max wait before flushing on time. An order of magnitude smaller than typical inference latency (seconds). Invisible at the user level; long enough to actually accumulate batch-mates under realistic arrival rates. |
| `chat_coalescer_bypass_threshold` | 10 | Queue depth below which the time window is skipped entirely. Below this depth, batching gains nothing because the next request might not arrive within the window — so don't pay the latency. Above it, batches form naturally. Defaulted equal to `packing_factor` so the bypass disengages exactly when full batches start forming. |

### 6.3 Sizing `packing_factor`

The two forces:

- **Larger** `packing_factor` → higher queue throughput (more requests per SQLite row), but a longer worst-case wait on the *first* request in a batch (it sits in the accumulator for up to `window_ms` before the row is written, even if it could have shipped sooner).
- **Smaller** `packing_factor` → lower per-request latency tail, but proportionally less throughput headroom. At `packing_factor = 1` the coalescer is a no-op and SQLite is exposed to every request directly.

Recommended sizing: start at the default 10. Move up (toward `max_num_seqs / 8` or so) if `chat_admitted_429_rate` (§13) climbs and queue depth tracks closely with the row-count ceiling. Move down if p99 request latency under light load is higher than acceptable; the bypass threshold should already protect against this, but a misconfigured `bypass_threshold` can leak window-tax into the small-burst case.

### 6.3 What flushes look like

A flush produces one SQLite row plus one JSON file under `upload_base_path`, exactly the same shape `/v1/generate` already produces:

```
{upload_base_path}/{sha256-of-batch}.json   ← list of N rendered request dicts
SQLite row payload:
    {"path": ".../{sha256}.json", "request_count": N, "timestamp": ...}
```

The SHA-256 dedup property of the existing queue carries over for free: two coalesced batches with identical contents (rare but possible under deterministic load tests) produce the same path and dedup at the worker pull stage via `fill_work_queue`'s response-file existence check (`inference-queue.md` §4.4).

### 6.4 Implementation sketch

```python
class Coalescer:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.batch: list[CoalescerEntry] = []
        self.flush_timer: asyncio.Task | None = None

    async def submit(self, rendered_prompt, req, correlation_id):
        async with self.lock:
            self.batch.append(CoalescerEntry(rendered_prompt, req, correlation_id))

            if queue_depth() < self.bypass_threshold:
                await self._flush_locked()
            elif len(self.batch) >= self.packing_factor:
                await self._flush_locked()
            elif self.flush_timer is None:
                self.flush_timer = asyncio.create_task(self._flush_after(self.window_ms / 1000))

    async def _flush_after(self, delay):
        await asyncio.sleep(delay)
        async with self.lock:
            await self._flush_locked()

    async def _flush_locked(self):
        if not self.batch:
            return
        outgoing, self.batch = self.batch, []
        if self.flush_timer is not None:
            self.flush_timer.cancel()
            self.flush_timer = None
        asyncio.create_task(_write_to_queue(outgoing))
```

`_write_to_queue` writes the JSON file, computes its SHA-256, and calls the existing `push_into_queue` helper (`inference-queue.md` §4.1, step 5). Single per-process coalescer; the lock is held only across list ops, never across I/O.

### 6.5 What coalescing does not solve

Coalescing reduces *queue write rate*, not vLLM load. The bound on vLLM load is the existing worker's KV-cache-aware `get_batch_size` (`inference-queue.md` §4.3) — that machinery is unchanged.

---

## 7. SQLite `InferenceWorkQueue` (existing)

This stage is the existing `InferenceWorkQueue` with no changes. The new endpoints share it. Behaviors that already apply and continue to apply:

- **Crash recovery** via `auto_resume=True`. A coalesced batch in flight at pod restart returns to `ready` state and re-runs.
- **Dedup** via SHA-256 content hashing of the batch JSON file.
- **Ack-timeout recycling** for stuck batches (default 300 s).
- **`clear_acked_data` cleanup** every 30 s prevents unbounded SQLite growth.
- **In-memory cache** of unpacked work items (`get_work_item.py`).
- **In-memory results staging** for partial batch completion.

See `inference-queue.md` §3, §5 for the full mechanics. The chat-completions path doesn't introduce any new failure modes at this layer.

---

## 8. Result router

A new module that maps per-prompt completions back to their waiting handlers.

```python
# infra/cray_infra/api/fastapi/chat_completions/result_router.py

class ResultRouter:
    """
    Per-prompt fan-out. The coalescer assigns each request a
    correlation_id (UUID); the handler awaits a Future keyed by that id.
    The worker calls .resolve(id, result) when its result lands. Per-id
    cleanup runs on resolution, on cancellation, and on client disconnect.
    """
    def register(self, correlation_id: UUID) -> asyncio.Future: ...
    def resolve(self, correlation_id: UUID, result: dict) -> None: ...
    def unregister(self, correlation_id: UUID) -> None: ...

    @property
    def in_flight_count(self) -> int: ...
```

The existing `in_memory_results` dict (`inference-queue.md` §3.2) tracks per-prompt completion within a group. The result router subscribes to it: when `update_and_ack` (`inference-queue.md` §4.5) sets a per-prompt `is_acked: true`, the router checks for a registered future under that correlation_id and resolves it.

Implementation: extend `update_and_ack` with one extra call to `result_router.resolve(correlation_id, result)`. Existing `/v1/generate` callers that don't register a future are a no-op (router lookup returns nothing).

The correlation_id is stored in the per-prompt request dict written to disk, so it survives queue restart along with the prompt itself. On crash recovery the router is empty (no registered futures) and the worker still produces the result; the response file is still written, so a client retrying the same request hits the dedup-and-skip path.

---

## 9. Whitespace heartbeats over chunked JSON

The transport that ties everything together. The handler returns a `StreamingResponse` with `media_type="application/json"`; FastAPI sets `Transfer-Encoding: chunked` automatically. The body generator emits a single space every 4 s while waiting and the real body when ready:

```python
async def stream_with_heartbeat(future, *, heartbeat_interval_seconds=4.0):
    try:
        while True:
            try:
                result = await asyncio.wait_for(asyncio.shield(future), timeout=heartbeat_interval_seconds)
                break
            except asyncio.TimeoutError:
                yield b" "
        yield json.dumps(result).encode("utf-8")
    finally:
        result_router.unregister(correlation_id)
```

`asyncio.shield` keeps the future alive across the `wait_for` timeout — the timeout cancels only the *waiter*, not the underlying work.

The helper lives at `infra/cray_infra/api/fastapi/chat_completions/heartbeat.py`.

### 9.1 Why this is invisible to the OpenAI SDK

- httpx (used by both `OpenAI` and `AsyncOpenAI`) treats the `read` timeout as time-between-bytes, not total time. Each space resets it. We pick a 4 s interval to stay well under httpx's default `read` timeout (5 s).
- The OpenAI SDK reads the response body as a single string after chunked transfer terminates. JSON parsers (RFC 7159 §2) ignore leading whitespace, so `[spaces...][body]` parses identically to `[body]`.
- No client-side configuration is needed. `AsyncOpenAI()` with all defaults works.

### 9.2 Failure mode: response-buffering proxies

If the deployment is behind a proxy that buffers responses end-to-end before forwarding (CloudFront with default settings, AWS ALB with response buffering, nginx with `proxy_buffering on`), the heartbeats never reach the client and the connection times out at the proxy's idle timeout (typically 60–300 s).

Mitigations:

1. **Document the deployment requirement.** The Helm chart and `docs/configuration.md` must specify disabling response buffering on the path serving `/v1/chat/completions`. nginx: `proxy_buffering off` for that location. ALB: target groups should serve `chat/completions` directly without buffered intermediate stages.
2. **Detection.** Surface a `chat_completions_apparent_buffering` metric: if request duration is bimodal (clustered near common proxy timeout values like 60 s exactly), suspect buffering.
3. **Fallback.** Heartbeat interval is configurable down to 1 s for misconfigured proxies that tolerate 1-byte-per-second writes but not 15 s gaps.

### 9.3 Failure mode: idle TCP connections

Aggressive corporate firewalls and home NAT may drop idle TCP connections after as little as 30 s. Our 4 s heartbeat is well under that. We do not rely on TCP keepalive (which is OS-level and unreliable); the application-layer space character is sufficient.

### 9.4 Validation against `AsyncOpenAI`

The technique is validated end-to-end against a real `openai.AsyncOpenAI` client over a real localhost socket. See `test/integration/test_chat_heartbeat.py`. An ASGI in-memory transport would skip the very layer (httpx `read` timeout) the heartbeat is designed to defeat, so the test deliberately uses a real uvicorn instance on a free port.

Configuration:

| Parameter | Value |
|---|---|
| httpx `read` timeout | 1.0 s |
| Server-side wait before resolving the future | 3.0 s |
| Heartbeat interval | 0.2 s |
| `max_retries` on the client | 0 (no retry covers the result) |

Two tests:

1. **Positive (`test_heartbeat_keeps_connection_alive_past_read_timeout`)**: handler emits a single space every 0.2 s for 3 s, then yields the JSON body. The `AsyncOpenAI` call completes; `response.choices[0].message.content == "hello"`; wall time is ≥ 2.5 s and > the 1.0 s read timeout, confirming the timeout-reset path actually fired.

2. **Negative control (`test_no_heartbeat_times_out_at_read_timeout`)**: identical wait, no heartbeats. Mounted at a separate base URL (`/v1-baseline/`) so the same SDK client construction reaches it. The call must raise `openai.APITimeoutError` — without this control, the positive case's success could be due to anything (proxy buffering, an unrelated transport quirk).

Both pass. Total suite runtime ≈ 7 s. Reproducer:

```bash
PYTHONPATH=infra pytest test/integration/test_chat_heartbeat.py -v
```

The helper exercised by the test, `stream_with_heartbeat`, is the same module the production handler will use — the test is not a parallel implementation.

---

## 10. `/v1/generate` API updates

The `/v1/generate` endpoint gains chat-template support and the same admission/coalescer/heartbeat treatment as `/v1/chat/completions`. The wire-level changes:

### 10.1 Request shape

Existing (unchanged):

```json
{
  "prompts": [{"prompt": "Hello", "model": "...", "max_tokens": 64}, ...]
}
```

New (additionally accepted):

```json
{
  "prompts": [
    {"messages": [{"role": "user", "content": "Hello"}], "model": "...", "max_tokens": 64},
    ...
  ]
}
```

Validation is per-entry-in-the-batch: each entry in `prompts: [...]` must have exactly one of `prompt` and `messages` set. Mixed batches (some entries with `prompt`, some with `messages`) are allowed; entries are rendered independently. `messages` on an entry, when present, is the conversation turns of *that single entry*'s request — not a slice of the batch.

### 10.2 Wire-compatibility

- A batch where every entry uses `prompt` is byte-for-byte identical to the current behavior. No SDK changes required for callers using the raw passthrough.
- An entry with `messages` invokes the chat template renderer (§4) before queue insertion. The queue payload's `prompt` field is the rendered string; downstream code is unchanged.
- The response shape (`{"results": [...]}` for synchronous, `{"request_id": "..."}` for upload) is unchanged.

### 10.3 SDK changes

The `masint` SDK (`sdk/masint/engines/async_cray.py`) gains a `messages=` parameter on its `generate()` call that maps to the per-prompt `messages` field. The legacy `prompt=` parameter is unchanged. No breaking change.

### 10.4 Why on `/v1/generate` and not just `/v1/chat/completions`

`/v1/generate` is the durable, large-batch path. The use cases that exercise it (10k-prompt evaluation sweeps, post-training sample generation) very often involve chat-style prompts. Forcing those callers to hand-craft prompt templates against each model's tokenizer is a real ergonomic tax and an ongoing source of subtle prompt-formatting bugs. Reusing the renderer eliminates both.

---

## 11. Cancellation and cleanup

### 11.1 Client disconnect

If the client closes the connection mid-wait, FastAPI cancels the handler task. The `try / finally` in `stream_with_heartbeat` runs `result_router.unregister(correlation_id)`, removing the future from the dict. The submitted work continues through the queue and the worker; the result is dropped silently when `result_router.resolve` finds no registered future for the id.

We do not attempt to abort work that's already in flight to vLLM; the wasted compute is small (one prompt's worth) and vLLM's request-abort path is heavier than we need for this case.

### 11.2 `/v1/generate` keeps its existing polling

`/v1/generate`'s synchronous form already polls `in_memory_results` for `response_timeout` (default 60 s). It continues to do so. Existing callers that rely on timed-out prompts coming back as `response=None` keep that behavior; the SDK's polling-with-backoff `download.py` flow is unchanged.

The heartbeat path (§9) applies only to `/v1/chat/completions`. The two endpoints share queue infrastructure but have separate response semantics: `/v1/chat/completions` blocks on a single completion via heartbeat-kept-alive HTTP; `/v1/generate` returns a `request_id` and the SDK polls for results. We may revisit unifying transport in a future doc, but it is not part of this design.

### 11.3 Future never resolves

If the worker hangs on a request, the existing ack-timeout machinery (`inference-queue.md` §5.2) recycles the queue row after 300 s. The result router's future stays unresolved until the recycled row produces a result on retry, at which point it resolves normally. The heartbeat keeps the client connection alive across the recycle.

If the client disconnects before the recycle completes, §11.1 applies.

---

## 12. Configuration

New fields to add to `infra/cray_infra/util/default_config.py`:

| Field | Default | Meaning |
|---|---|---|
| `chat_admit_factor` | 4 | `ADMIT_HIGH_WATER = chat_admit_factor × max_num_seqs`. |
| `chat_coalescer_packing_factor` | 10 | Max requests packed into one SQLite row. Primary throughput knob (§6.3). |
| `chat_coalescer_window_ms` | 50 | Max wait before flushing on time. |
| `chat_coalescer_bypass_threshold` | 10 | If queue depth is below this, skip the time window. |
| `chat_heartbeat_interval_seconds` | 4 | Seconds between whitespace heartbeats. |
| `chat_max_request_seconds` | unset | Optional server-side cap on per-request total time. |

Existing queue-related config (see `inference-queue.md` §10) is unchanged. The OpenAI path inherits `inference_work_queue_path`, `upload_base_path`, `inference_work_queue_ack_timeout`, `inference_work_queue_idle_time`, etc.

---

## 13. Metrics

The OpenAI path adds a parallel set of counters and gauges to the existing `Metrics` singleton at `cray_infra/generate/metrics.py`, exposed through the existing `GET /v1/generate/metrics` endpoint. Adding to the existing surface (rather than a parallel `/v1/chat/metrics`) keeps the Prometheus/Grafana wiring single-sourced. Both endpoints continue to share the queue infrastructure metrics (`queue_depth`, `total_completed_*`) — see `inference-queue.md` §6 — and gain endpoint-scoped variants.

### 13.1 New counters and gauges

| Metric | Type | Update site | Meaning |
|---|---|---|---|
| `chat_in_flight` | Gauge | result router register / unregister | Current count of in-flight chat completion requests (registered futures). |
| `chat_admitted_429_count` | Counter | admission control | Number of chat requests rejected with 429 due to the high-water threshold. |
| `chat_admitted_429_rate` | Derived | `chat_admitted_429_count / chat_total_count` | Ratio of 429 responses to total submissions over the current non-idle epoch (matches the epoch semantics in `inference-queue.md` §6.2). |
| `chat_batch_size_p50` / `p99` | Histogram | coalescer flush | Distribution of coalesced batch sizes per flush, range `[1, packing_factor]`. p50 close to `packing_factor` means coalescing is doing its job; p50 of 1 under load means the bypass is firing too often. |
| `chat_request_duration_p50` / `p99` | Histogram | result router resolve / unregister | Total time from admission to result delivery (or client disconnect). Includes wait time inside the queue. |
| `chat_apparent_buffering_count` | Counter | result router resolve | Heuristic — see §13.2. |
| `generate_*` equivalents | Various | `/v1/generate` handlers | Same six metrics scoped to `/v1/generate`, so the two endpoints can be compared directly. |

### 13.2 Apparent-buffering heuristic

A request that "should" complete in ~50 ms but instead completes at exactly 60.0 s (within a small tolerance) is almost certainly being held by an upstream proxy that flushed on its own idle timeout rather than on our heartbeats — the smoking gun for §9.2.

`chat_apparent_buffering_count` increments when both:

- `|duration - common_proxy_timeout| < buffering_match_threshold_seconds` (default `common_proxy_timeout = 60 s`, `buffering_match_threshold_seconds = 0.5 s`).
- The future was already resolved server-side measurably before that — i.e., the result *would have* been delivered sooner if not buffered.

The metric is a signal, not a strict measurement; false positives are acceptable. Operators correlate against deployment topology (does the request path traverse a 60 s-timeout proxy?) before acting.

### 13.3 Configuration fields

Two additional config fields support the heuristic:

| Field | Default | Meaning |
|---|---|---|
| `chat_buffering_check_proxy_timeout_seconds` | 60 | The known proxy idle timeout to match against. |
| `chat_buffering_match_threshold_seconds` | 0.5 | Tolerance window around that timeout. |

---

## 14. Open questions

Two real ones remain. Items previously on this list — `stream=True` routing, tenant isolation, `/v1/generate` heartbeat migration, tokenizer cache invalidation, the metrics surface — are decided in §3.1, §2, §11.2, §4, and §13 respectively.

1. **Heartbeat interval vs. proxy compatibility.** 4 s is conservative; 15 s would be cheaper if no proxy is in the path. Decision deferred until production traces from a representative deployment confirm the proxy mix.
2. **Per-batch ordering inside a coalesced row.** Currently FIFO. Priority queueing (smaller `max_tokens` first) reduces p99 latency but complicates fair-share. Defer until we see evidence p99 latency under load is the bottleneck.

---

## 15. Out of scope (this document)

- `/v1/batches` (file-upload + polling for huge async submissions). Separate doc.
- `stream=True` chat completions. Existing direct-to-vLLM SSE path; bypasses the new infrastructure entirely (§3.1).
- Training-related queue paths.
- Streaming-through-queue (token-incremental delivery from worker → client over SSE) as a future option if streaming traffic ever needs queue protection. Not in v1.
