# Inference Request Browser

A read-only operator surface for searching, scrolling, and expanding the inference requests already persisted to disk by the existing `/v1/generate` and `/v1/chat/completions` flows.

The scope is deliberately narrow: **file-backed history only**. The live SQLite work queue is not consulted. Items appear in the browser the moment the request batch is written to `upload_base_path` (which happens *before* the queue push for `/v1/generate`, and as part of `enqueue_coalesced_batch` for the chat path), and they remain visible forever — there is no cleanup process for these files today.

This doc is intentionally short. The feature touches one new FastAPI router, one new React route, no schema changes, no new persistence.

---

## 1. Why this exists

When something goes wrong in production, the first question an operator asks is "show me the actual request." Today that requires SSH, `ls -lt /app/cray/inference_requests/`, and `jq` — fine for the author of the system, useless for everyone else. The data is already on disk; the gap is purely the lack of a way to look at it.

Three concrete questions the browser should answer in under five clicks:

1. **What did the last 100 inference requests look like?** Most recent first, with prompt previews and completion status.
2. **Did anyone send a prompt matching `/foo.*bar/i`?** Regex over prompt text and request id.
3. **What was the full request and response for `<hash>`?** Click → expand → see the request batch JSON, the response JSON, and the status file side by side.

Anything beyond that — full-text search across millions of historical requests, deletion, re-running, annotations — is out of scope for v1.

---

## 2. Goals and non-goals

**Goals**

- List view of every request file in `upload_base_path`, newest first, paginated by mtime cursor.
- Click-to-expand inline panel showing the full request, response, and status JSON for one entry.
- Client-side regex filter over the currently-loaded page (request id + prompt previews).
- No new persistence, no new dependencies, no schema migrations.

**Non-goals**

- Searching the live in-flight queue. The queue contents are visible at `/v1/generate/get_results` per-id; merging that into the browser would mean a second, racy data source. Items appear in the browser as soon as the batch is written to disk (before queue push for `/v1/generate`, during enqueue for chat completions), so the staleness window is in milliseconds anyway.
- Server-side regex search. The first cut filters client-side over one page (≤ 200 rows). If operators need cross-page deep search, that's a follow-up endpoint.
- Mutation. Read-only. No deletion, no re-enqueue, no editing.
- Multi-tenant scoping. Same single-tenant assumption as the rest of the API.
- Auth. Inherits whatever auth the rest of `/v1/*` has (currently none on the local deployment).

---

## 3. Data sources

Three files per group request id (= SHA-256 of the request batch contents) live at `upload_base_path` (default `/app/cray/inference_requests/`):

| File | Writer | Shape |
|------|--------|-------|
| `{hash}.json` | `generate.py` line 100, `enqueue_coalesced_batch.py` line 37 | `[ {prompt, model, max_tokens, temperature, tools?, tool_choice?, request_type, correlation_id?}, ... ]` |
| `{hash}_response.json` | `update_and_ack.finish_work_queue_item` line 78 | `{current_index, total_requests, results: {"<hash>_<idx>": {response, ...}}}` |
| `{hash}_status.json` | `push_into_queue` line 19 / `update_and_ack` line 91 | `{status: "in_progress" \| "completed", current_index, total_requests, work_queue_id, completed_at?}` |

**Lifecycle observation:** the files are content-hashed, so identical request batches collapse to one set of files (this is how the existing queue achieves dedup — see `inference-queue.md`). The mtime of `{hash}.json` reflects when the request was *first* submitted; the mtime of `{hash}_response.json` reflects when the work finished. Sorting by `{hash}.json` mtime gives "newest request first" ordering.

**Lifecycle hazard:** nothing in the codebase removes these files. On a busy server the directory grows without bound. The browser must tolerate tens of thousands of files efficiently — see §6.

---

## 4. API surface

Two new endpoints on the existing `generate_router`:

### 4.1 `GET /v1/generate/list_requests`

Query params:

- `cursor: float | null` — `mtime` of the last row from the previous page. Omit on the first page.
- `limit: int = 50` — max rows to return. Capped at 200 server-side.

Response:

```json
{
  "rows": [
    {
      "request_id": "abcd1234...",
      "mtime": 1730328400.123,
      "size_bytes": 1248,
      "request_count": 3,
      "status": "completed",
      "completed_at": 1730328401.456,
      "model": "meta-llama/Llama-3-8B",
      "request_type": "generate",
      "prompt_preview": "What is the capital of France?…",
      "has_response": true
    },
    ...
  ],
  "next_cursor": 1730328380.001,
  "has_more": true
}
```

**Implementation:** `os.scandir(upload_base_path)`, filter to filenames matching `^[0-9a-f]{64}\.json$` (the request files; we skip `_response.json` and `_status.json`). Sort by `entry.stat().st_mtime` descending. Drop entries with mtime ≥ cursor (cursor is exclusive of "already-seen"). Take `limit`. For each, read the request file (just the first entry to populate `prompt_preview`, `model`, `request_type`) and stat the matching status file (cheap — single `os.path.exists` + small JSON load). `prompt_preview` is the first 120 chars of the first entry's `prompt` field.

**Performance:** for ~10k files, scandir + sort costs ~50 ms on commodity SSD. The per-row stats and small JSON reads dominate (~1-2 ms each), so a 50-row page lands in ~150 ms. We do not cache — operator click latency is dominated by network anyway, and avoiding cache invalidation on a directory we don't own is worth more than the saved milliseconds.

**Failure modes:**

- Request file missing prompt field → `prompt_preview = ""`, `model = "unknown"`. Don't 500.
- Status file missing → `status = "unknown"`, `completed_at = null`. The request was written but `push_into_queue` may have crashed before writing status; this is rare but real.
- Response file missing → `has_response = false`. Either still in flight or the worker hasn't finished.
- Request file truncated/corrupt JSON → log a warning, return the row with `prompt_preview = "<unreadable>"`. Skipping the row entirely would hide the existence of a real request from the operator.

### 4.2 `GET /v1/generate/request/{request_id}`

Path param: 64-char hex string. We validate the regex server-side to avoid path traversal — even though `os.path.join` would not let `../foo` escape, treating the id as a hash also prevents the endpoint from being abused as a generic file reader.

Response:

```json
{
  "request_id": "abcd1234...",
  "request": [{ ...full request batch... }],
  "response": { ...full response file or null... },
  "status": { ...full status file or null... },
  "request_mtime": 1730328400.123,
  "response_mtime": 1730328401.456
}
```

**Size cap:** request batches can be huge (up to `max_upload_file_size` = 50 MB by default). The endpoint reads each file with `os.path.getsize` first and refuses to serialize anything > 5 MB per file, replacing the body with `{"error": "too large to display", "size_bytes": N}`. Operators can SSH and `jq` for the truly enormous ones; the browser is for the common case.

---

## 5. UI

New route at `/inference`, added to the top nav between Train and Metrics. Page layout:

```
┌──────────────────────────────────────────────────────────────┐
│  Inference requests                      [refresh] [count]   │  PageHeader
├──────────────────────────────────────────────────────────────┤
│  [ regex filter…                                          ]  │  sticky
├──────────────────────────────────────────────────────────────┤
│  ▸ abcd1234… │ generate │ completed │ 3 prompts │ 12s ago    │  row (collapsed)
│  ▸ ef561782… │ chat     │ in_prog   │ 1 prompt  │ 14s ago    │
│  ▾ 9a8b7c6d… │ generate │ completed │ 5 prompts │ 22s ago    │  row (expanded)
│      Request preview: "Write a haiku about ..."              │
│      [Request JSON ▸] [Response JSON ▸] [Status ▸]           │
│      ┌─────────────────────────────────────────────────────┐ │
│      │ { "prompt": "Write a haiku ...", ... }              │ │
│      │ ...                                                 │ │
│      └─────────────────────────────────────────────────────┘ │
│  ▸ ...                                                       │
│                                                              │
│  [ load more ]                                               │  bottom
└──────────────────────────────────────────────────────────────┘
```

**Tech choices:**

- No `react-window`. Scroll performance is dominated by the rows being fixed height and small (~32 px) until expanded. We can render up to ~5k DOM rows comfortably; pagination caps it well below that. Following `ServiceLogsCard.tsx`'s precedent (it holds 20k log lines without virtualization).
- Regex via `new RegExp(pattern, "i")` in a `try/catch`. Invalid patterns render an inline "Invalid regex" hint and stop filtering. Filter applies to `request_id + " " + prompt_preview + " " + model`.
- Expand state stored per-row in a `Set<string>` keyed by `request_id`. Lazy-fetch `/v1/generate/request/{id}` on first expand via `useQuery({ enabled: expanded })`; the response stays cached for the page lifetime so re-expand is free.
- "Load more" is a button, not infinite scroll. Operators should be in control of how much they're looking at; auto-loading 10k rows because someone bumped the scroll wheel is a footgun.
- Polling: refetch the *first page* every 5 s with `refetchInterval` so freshly-completed requests show up without manual refresh. Pages 2+ are static — they only refetch on explicit user action.

**Component layout:**

- `routes/inference/InferenceBrowserPage.tsx` — page shell, nav registration, header.
- `routes/inference/InferenceRequestList.tsx` — the list, regex filter, load-more.
- `routes/inference/InferenceRequestRow.tsx` — one row + expand button.
- `routes/inference/InferenceRequestDetail.tsx` — the expanded panel; fetches via `useInferenceRequestDetail`.
- `api/inference.ts` — `useInferenceRequestList()` (infinite query) + `useInferenceRequestDetail(id)`.

---

## 6. Performance and scale

**Listing performance.** Expected steady state: a few hundred to a few thousand request files; the `inference_requests/` directory is the same place training uploads live, so it can grow large. Worst observed in dev: ~5k files. `os.scandir` + `sorted(key=lambda e: -e.stat().st_mtime)` on 5k entries is ~30 ms; on 50k it's ~400 ms. We do not paginate before sorting (cursor approach is "skip until mtime < cursor, then take limit"), so the scan cost is per-page. That's acceptable until ~50k files; beyond that we'd want an mtime-indexed lookup. Out of scope for v1.

**Memory.** Each row in the list response is ~300 bytes serialized. A 200-row page is ~60 KB. The detail endpoint is bounded by the 5 MB per-file display cap (§4.2).

**Concurrent reads while files are being written.** `update_and_ack.finish_work_queue_item` writes the response file under `acquire_file_lock`. The browser does not take the lock — a partial response could be read mid-write. JSON parse errors get caught and turned into `{"response": null, "warning": "race during read"}`. Reload-the-row handles it.

---

## 7. Testing

Unit tests on the backend (pytest):

- `test_list_requests_pagination` — 10 files, request 3 rows, then cursor for the rest. Assert no duplicates, full coverage.
- `test_list_requests_handles_missing_files` — request file with no status, request file with no response, status file with no request file (the latter must not appear in the listing — we key off the request file).
- `test_list_requests_skips_corrupt_request_file` — invalid JSON → row appears with placeholder preview, not a 500.
- `test_get_request_returns_all_three_files` — happy path.
- `test_get_request_404_on_missing_request_file` — request id with no file at all.
- `test_get_request_400_on_non_hex_request_id` — `..` and other bad inputs rejected before any filesystem call.
- `test_get_request_truncates_huge_files` — synthetic 6 MB request file → response carries the `too large to display` placeholder.

Frontend (vitest):

- `regex filter` — valid pattern filters rows, invalid pattern shows hint without crashing.
- `expand and collapse` — clicking the row toggles the detail fetch; collapsed rows do not re-fetch.

End-to-end with running stack: out of scope for the PR. The component is read-only and behind the same router as everything else; smoke testing in the dev container is sufficient.

---

## 8. Open questions

None known. If operators ask for cross-page server-side regex search, that's a follow-up endpoint that walks all files and streams matches; it doesn't change the data model or the existing endpoints.

---

## 9. Decisions deferred or rejected

- **Live queue merge.** Rejected for v1 (see §2). The live queue is visible per-id at `/v1/generate/get_results`; a unified view would need to reconcile two data sources where the file system is already the source of truth ~milliseconds later.
- **Server-side regex.** Deferred. Client-side over one page covers the operator's "I just saw a weird thing, let me find it" workflow. Cross-history search is a real feature but a much bigger one.
- **Mutation (delete, re-enqueue).** Rejected. Read-only by design. Delete in particular interacts with the queue's dedup machinery in subtle ways — out of scope.
- **A separate audit log.** Rejected. The files already on disk are the audit log.
