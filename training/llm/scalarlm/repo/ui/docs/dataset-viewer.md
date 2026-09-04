# Dataset viewer

A panel on the train-detail page that lets a user scroll and search the
dataset their training job is running on. Same view for QUEUED, TRAINING,
COMPLETED, and FAILED jobs — the dataset is the single artifact that
exists from the instant a job is submitted.

## Motivation

Today the train-detail page shows config, loss, and logs. When a user is
debugging why a job's loss is high or a completion looks wrong, there's
no way to inspect the data the model actually saw without SSH'ing onto
the box and `head`-ing `/app/cray/jobs/<hash>/dataset.jsonlines`. This
panel closes that loop in the UI.

## Non-goals

- **Editing.** The dataset is immutable once uploaded; the viewer is
  read-only. The hash in `config.yaml` pins it.
- **Row-level statistics** (token counts, duplicate detection, etc.).
  Nice to have later; out of scope now.
- **Full-text search across all jobs.** Scope is one job at a time.
- **Serving the full file as a download.** If a user wants the raw
  bytes they can already `curl` the file from the training host. The
  viewer is for inspection, not export.

## On-disk format (already fixed)

Every training job directory contains `dataset.jsonlines` — one JSON
object per line, `{"input": ..., "output": ...}` shape (see
`sdk/masint/engines/cray/submit_slurm_job.py:22` and
`infra/cray_infra/training/upload_training_data.py:70`). Files are
typically <100 MB but have no hard cap; the viewer must not load the
whole file into memory server-side or into the browser.

## API — `GET /v1/megatron/train/{job_hash}/dataset`

New endpoint in `megatron_router.py`, backed by a new
`infra/cray_infra/training/get_dataset_slice.py`.

**Query params**

| name     | type | default | notes                                                |
|----------|------|---------|------------------------------------------------------|
| `offset` | int  | 0       | 0-based line index to start at.                      |
| `limit`  | int  | 50      | Clamped to [1, 200].                                 |
| `q`      | str  | —       | Optional case-insensitive substring filter.          |

**Response**

```json
{
  "total": 12345,
  "offset": 0,
  "limit": 50,
  "matched": 12345,
  "truncated": false,
  "examples": [
    { "index": 0, "input": "...", "output": "...", "raw": {...} }
  ]
}
```

- `total` is the line count of `dataset.jsonlines`.
- `matched` equals `total` when `q` is empty; otherwise it's the number
  of lines whose raw text contains `q` (case-insensitive). We pre-count
  matches so the UI can render "showing 50 of 2,317 matches".
- `truncated` is `true` if the match scan was stopped early for safety
  (see limits below).
- `raw` preserves any fields besides `input`/`output` so datasets with
  unusual schemas still render something useful.

**Error cases**

- `404` if the job directory doesn't exist.
- `404` with `detail="dataset not found"` if the directory exists but
  `dataset.jsonlines` does not. (Possible for jobs that never finished
  uploading; we surface this rather than returning an empty list.)
- `400` on nonsensical pagination (negative `offset`, `limit > 200`).

**Server-side implementation sketch**

- Read the file line-by-line with `open(..., 'r', encoding='utf-8')`.
- When `q` is unset, take the slice `[offset:offset+limit]` and parse
  only those lines as JSON. Parsing all lines up-front is O(N) for no
  user-visible benefit.
- When `q` is set, iterate through the file keeping a count of matches
  and collecting the `offset`-th through `offset+limit-1`-th match. To
  bound pathological scans we stop after `MAX_MATCH_SCAN_BYTES`
  (default 256 MiB) and set `truncated=true`.
- Malformed JSON lines are skipped but still counted toward `total` —
  the user wants to know something's there.

## UI

New component `ui/src/routes/train/DatasetPanel.tsx`, slotted into
`TrainDetail.tsx` between the `Loss` card and the `LogPane` (logs are
noisier, so the dataset sits above them).

### Fetch layer

Add to `ui/src/api/training.ts`:

```ts
export interface DatasetExample {
  index: number;
  input?: string;
  output?: string;
  raw: Record<string, unknown>;
}

export interface DatasetSlice {
  total: number;
  offset: number;
  limit: number;
  matched: number;
  truncated: boolean;
  examples: DatasetExample[];
}

export function useTrainingDataset(
  jobHash: string | undefined,
  { offset, limit, q }: { offset: number; limit: number; q: string },
) { /* useQuery with keepPreviousData */ }
```

Query key: `["training-dataset", jobHash, offset, limit, q]`. We use
`placeholderData: keepPreviousData` so paging and typing in the search
box don't blank the table.

### Layout

```
┌─ Dataset ────────────────── [search box] ──┐
│  total: 12,345 examples                     │
│  showing 1–50 of 12,345                     │
│  ┌─────────────────────────────────────┐   │
│  │ #0   input ────► output            │   │
│  │ #1   input ────► output            │   │
│  │  …                                  │   │
│  └─────────────────────────────────────┘   │
│  [← prev] [page 1 / 247] [next →]           │
└─────────────────────────────────────────────┘
```

- Each row: a two-column grid with `input` on the left and `output` on
  the right, `index` in the gutter. Rows clamp to a few lines with a
  "show more" affordance that expands the row in place (no modal).
- Search box is debounced 200 ms, trimmed, empty-string means "no
  filter". On search, `offset` resets to 0.
- Pagination is offset-based. Keyboard: `j/k` or arrow keys when the
  panel has focus move row selection; `Shift-J/Shift-K` page next/prev
  (matches the rest of the app's keyboard posture).

### States

- **Pending (first load).** Render a `Skeleton` with 5 row-height
  blocks; keep the search box interactive so the user can start typing
  before the data arrives.
- **Empty total.** "This job's dataset is empty." with a link to docs.
- **404 dataset not found.** "No `dataset.jsonlines` was found for this
  job. This usually means upload failed." — we surface this
  distinctly from "empty".
- **Non-empty total, zero matches.** "No rows matching `<q>`."
- **Truncated.** Banner above the list: "Search stopped after scanning
  256 MiB. Narrow the query for complete results."
- **Error.** Standard `<ErrorState>` with retry.

### Accessibility

- The search box is `<input role="searchbox">` with an
  `aria-controls` pointing at the list region.
- Rows are a `<ul role="list">` with `role="listitem"`; the selected
  row gets `aria-selected="true"` and is the sole tabindex-0 target so
  arrow-key navigation can't escape the panel.

## Security / resource limits

- File IO is read-only and bounded (`MAX_MATCH_SCAN_BYTES`,
  `limit ≤ 200`). No user-supplied path; job hash resolves through
  `get_job_directory_for_hash`, which already guards against
  traversal.
- Response size is bounded by `limit × max-line-length`. Lines in
  `dataset.jsonlines` aren't capped by the server at upload time, so
  individual pathological rows (a 5 MB completion, say) can still
  balloon a page. We truncate any single string field to 4 KiB in the
  response and set `raw.__truncated_fields__` to the list of names we
  clipped. The UI shows "… (truncated at 4 KiB)" inline.

## Testing

- Unit: `test/unit/test_get_dataset_slice.py` — pagination math, `q`
  filtering, truncation flag, malformed-line skipping, non-existent
  dataset → 404.
- Component: none — the endpoint is thin enough that the unit test
  covers it.
- UI: `ui/test/datasetPanel.test.tsx` under vitest/jsdom — renders
  the happy path with a fake API, verifies the "searching resets
  offset" behaviour.

## Rollout

1. Merge endpoint + unit tests (no UI change → safe by itself).
2. Ship `DatasetPanel` behind no flag — the panel is additive and the
   endpoint is a GET, so a pre-endpoint UI gracefully degrades to an
   `ErrorState` with "retry", which is acceptable while the two PRs
   travel between commit and deploy.
3. Document the endpoint in `docs/architecture.md` under "training
   routes".
