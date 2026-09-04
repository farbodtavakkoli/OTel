/**
 * DatasetPanel — paginated, searchable view of a training job's
 * dataset.jsonlines. Fetches GET /v1/megatron/train/{hash}/dataset.
 *
 * Collapsed by default; the underlying query only fires when expanded,
 * so opening the train-detail page costs zero dataset bytes until the
 * user asks for them.
 *
 * See ui/docs/dataset-viewer.md for the design.
 */

import { useEffect, useMemo, useState } from "react";

import { ApiError } from "@/api/client";
import { getApiConfig } from "@/api/config";
import { useTrainingDataset } from "@/api/training";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";

const PAGE_SIZE = 50;
const SEARCH_DEBOUNCE_MS = 200;
const DOWNLOAD_SAMPLE_LIMIT = 100;

interface DatasetPanelProps {
  jobHash: string;
}

export function DatasetPanel({ jobHash }: DatasetPanelProps) {
  const [open, setOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  // Index into the current page's `examples` array (not the global
  // dataset row index). null = no row open.
  const [selectedRow, setSelectedRow] = useState<number | null>(null);

  // Debounce the search input and reset paging when the query changes.
  useEffect(() => {
    const handle = setTimeout(() => {
      const trimmed = searchInput.trim();
      if (trimmed !== q) {
        setQ(trimmed);
        setOffset(0);
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [searchInput, q]);

  // Gate the network call on `open` — passing undefined disables the hook.
  const { data, error, isPending, isFetching, refetch } = useTrainingDataset(
    open ? jobHash : undefined,
    { offset, limit: PAGE_SIZE, q },
  );

  const total = data?.total ?? 0;
  const matched = data?.matched ?? 0;
  const examples = data?.examples ?? [];

  const pageStart = examples.length > 0 ? offset + 1 : 0;
  const pageEnd = offset + examples.length;
  const pageIndex = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil(matched / PAGE_SIZE));
  const hasNext = offset + PAGE_SIZE < matched;
  const hasPrev = offset > 0;

  const is404 =
    error instanceof ApiError &&
    error.status === 404 &&
    typeof error.body === "object" &&
    error.body !== null &&
    "detail" in error.body &&
    String((error.body as { detail: unknown }).detail) === "dataset not found";

  return (
    <Card
      title="Dataset"
      subtitle={
        !open
          ? "Collapsed — click to load"
          : q
          ? `${matched.toLocaleString()} matches of ${total.toLocaleString()} rows`
          : `${total.toLocaleString()} rows`
      }
      action={
        <div className="flex items-center gap-2">
          {/* Download is available even while collapsed: the anchor streams
              through the browser and fetches nothing until clicked, so it
              doesn't break the panel's "no bytes until Show" promise. Once
              open with a known row count we show the richer sample/all
              buttons instead. */}
          {open && total > 0 ? (
            <DownloadButtons jobHash={jobHash} total={total} />
          ) : !open ? (
            <DownloadButtons jobHash={jobHash} />
          ) : null}
          {open && (
            <input
              type="search"
              role="searchbox"
              aria-controls="dataset-panel-list"
              placeholder="Search…"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-64 rounded-md border border-border-subtle bg-bg px-3 py-1 text-sm text-fg placeholder:text-fg-subtle focus:border-accent focus:outline-none"
            />
          )}
          {open && isFetching && !isPending && (
            <span
              className="text-xs text-fg-subtle"
              aria-live="polite"
            >
              …
            </span>
          )}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            {open ? "Hide" : "Show"}
          </button>
        </div>
      }
    >
      {!open ? (
        <p className="text-xs text-fg-muted">
          The dataset panel is collapsed to avoid fetching bytes you don't
          need. Click <span className="font-semibold">Show</span> to load
          the first page and enable search.
        </p>
      ) : isPending ? (
        <DatasetSkeleton />
      ) : is404 ? (
        <EmptyNotice
          title="No dataset.jsonlines for this job"
          body="The dataset file is missing from the job directory. This usually means the upload didn't complete."
        />
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : total === 0 ? (
        <EmptyNotice
          title="Empty dataset"
          body="This job's dataset.jsonlines has no rows."
        />
      ) : matched === 0 ? (
        <EmptyNotice
          title={`No rows match "${q}"`}
          body="Try a shorter query or clear the search box."
        />
      ) : (
        <>
          {data?.truncated && (
            <div className="mb-3 rounded-md border border-warning/30 bg-warning/5 px-3 py-2 text-xs text-warning">
              Search stopped after scanning 256 MiB. Narrow the query for
              complete results.
            </div>
          )}
          <ul
            id="dataset-panel-list"
            role="list"
            className="flex max-h-[480px] flex-col divide-y divide-border-subtle overflow-y-auto rounded-md border border-border-subtle"
          >
            {examples.map((ex, i) => (
              <ExampleRow
                key={ex.index}
                index={ex.index}
                input={ex.input}
                output={ex.output}
                raw={ex.raw}
                truncatedFields={ex.truncated_fields}
                onOpen={() => setSelectedRow(i)}
              />
            ))}
          </ul>
          {selectedRow !== null && examples[selectedRow] && (
            <DatasetExampleModal
              example={examples[selectedRow]}
              hasPrev={selectedRow > 0}
              hasNext={selectedRow < examples.length - 1}
              onPrev={() => setSelectedRow((r) => (r === null ? r : r - 1))}
              onNext={() => setSelectedRow((r) => (r === null ? r : r + 1))}
              onClose={() => setSelectedRow(null)}
            />
          )}
          <div className="mt-3 flex items-center justify-between text-xs text-fg-muted">
            <span>
              Showing {pageStart.toLocaleString()}–
              {pageEnd.toLocaleString()} of {matched.toLocaleString()}
              {q ? " matches" : " rows"}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={!hasPrev || isFetching}
                className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                ← Prev
              </button>
              <span className="font-mono">
                {pageIndex} / {pageCount}
              </span>
              <button
                type="button"
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={!hasNext || isFetching}
                className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Download buttons — plain anchor downloads so the file streams through
// the browser rather than buffering through fetch + Blob (full datasets
// can be GB-scale; loading them into JS memory would OOM the tab).

/** Build the dataset download URL; with `limit` it downloads a sample. */
export function datasetDownloadUrl(
  apiBase: string,
  jobHash: string,
  limit?: number,
): string {
  const base = `${apiBase}/megatron/train/${encodeURIComponent(jobHash)}/dataset/download`;
  return limit !== undefined ? `${base}?limit=${limit}` : base;
}

/** Label for the full-dataset download. Generic when the row count is
 *  unknown (collapsed panel), abbreviated K/M otherwise. */
export function downloadAllLabel(total?: number): string {
  if (total === undefined) return "Download";
  if (total >= 1_000_000) return `Download all (${Math.round(total / 1_000_000)}M rows)`;
  if (total >= 1_000) return `Download all (${Math.round(total / 1_000)}K rows)`;
  return `Download all (${total} rows)`;
}

interface DownloadButtonsProps {
  jobHash: string;
  /** Row count; undefined when the panel is collapsed (count not yet
   *  fetched) — we then render only a single generic "Download" link. */
  total?: number;
}

function DownloadButtons({ jobHash, total }: DownloadButtonsProps) {
  const { api_base } = getApiConfig();
  const showSample = total !== undefined && total > DOWNLOAD_SAMPLE_LIMIT;
  const sampleSize =
    total !== undefined ? Math.min(DOWNLOAD_SAMPLE_LIMIT, total) : 0;

  return (
    <span className="flex items-center gap-1">
      {showSample && (
        <a
          href={datasetDownloadUrl(api_base, jobHash, sampleSize)}
          download
          title={`Download the first ${sampleSize} rows as JSONL`}
          className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
        >
          Download {sampleSize}
        </a>
      )}
      <a
        href={datasetDownloadUrl(api_base, jobHash)}
        download
        title="Download the entire dataset.jsonlines file"
        className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
      >
        {downloadAllLabel(total)}
      </a>
    </span>
  );
}

// ---------------------------------------------------------------------------

interface ExampleRowProps {
  index: number;
  input?: string;
  output?: string;
  raw: Record<string, unknown>;
  truncatedFields?: string[];
  onOpen: () => void;
}

function ExampleRow({
  index,
  input,
  output,
  raw,
  truncatedFields,
  onOpen,
}: ExampleRowProps) {
  // If neither `input` nor `output` is present, fall back to dumping raw JSON
  // so datasets with unusual schemas still render something.
  const hasCanonical =
    typeof input === "string" || typeof output === "string";

  const rawPreview = useMemo(() => {
    try {
      return JSON.stringify(raw, null, 2);
    } catch {
      return String(raw);
    }
  }, [raw]);

  return (
    <li
      role="listitem"
      // Whole row is the click target — full content lives in the modal,
      // not inline. Keep <button> semantics for keyboard reachability
      // (Enter/Space) and a visible focus ring.
      className="cursor-pointer transition-colors hover:bg-bg-hover/60 focus-within:bg-bg-hover/60"
    >
      <button
        type="button"
        onClick={onOpen}
        className="block w-full px-3 py-2 text-left focus:outline-none"
        aria-label={`Open dataset row ${index}`}
      >
        <div className="flex gap-3">
          <span className="shrink-0 font-mono text-[11px] text-fg-subtle">
            #{index}
          </span>
          <div className="min-w-0 flex-1">
            {hasCanonical ? (
              <div className="grid grid-cols-1 gap-2 text-sm md:grid-cols-2">
                <Cell label="input" value={input} />
                <Cell label="output" value={output} />
              </div>
            ) : (
              <pre className="line-clamp-3 whitespace-pre-wrap break-words font-mono text-xs text-fg">
                {rawPreview}
              </pre>
            )}
            {truncatedFields && truncatedFields.length > 0 && (
              <p className="mt-1 text-[11px] text-warning">
                Truncated at 4 KiB: {truncatedFields.join(", ")}
              </p>
            )}
          </div>
        </div>
      </button>
    </li>
  );
}

interface CellProps {
  label: string;
  value: string | undefined;
}

function Cell({ label, value }: CellProps) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <div className="line-clamp-3 whitespace-pre-wrap break-words font-mono text-xs text-fg">
        {value ?? <span className="text-fg-subtle">—</span>}
      </div>
    </div>
  );
}

function DatasetSkeleton() {
  return (
    <div className="flex flex-col gap-2">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton key={i} className="h-12" />
      ))}
    </div>
  );
}

function EmptyNotice({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg px-4 py-6 text-center">
      <p className="text-sm font-medium text-fg">{title}</p>
      <p className="mt-1 text-xs text-fg-muted">{body}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail modal — full input/output for one dataset row, with prev/next
// keyboard nav across rows on the current page.
// ---------------------------------------------------------------------------

interface DatasetExample {
  index: number;
  input?: string;
  output?: string;
  raw: Record<string, unknown>;
  truncated_fields?: string[];
}

interface DatasetExampleModalProps {
  example: DatasetExample;
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}

function DatasetExampleModal({
  example,
  hasPrev,
  hasNext,
  onPrev,
  onNext,
  onClose,
}: DatasetExampleModalProps) {
  const rawText = useMemo(() => {
    try {
      return JSON.stringify(example.raw, null, 2);
    } catch {
      return String(example.raw);
    }
  }, [example.raw]);

  // Esc closes; ←/→ step through neighbors. Listening on document so
  // the user doesn't have to focus a specific element first.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowLeft" && hasPrev) {
        e.preventDefault();
        onPrev();
      } else if (e.key === "ArrowRight" && hasNext) {
        e.preventDefault();
        onNext();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onPrev, onNext, hasPrev, hasNext]);

  const hasInput = typeof example.input === "string";
  const hasOutput = typeof example.output === "string";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Dataset row ${example.index}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-[88vh] w-full max-w-4xl flex-col gap-3 rounded-lg border border-border bg-bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex flex-wrap items-center gap-2 border-b border-border-subtle pb-3">
          <h3 className="text-sm font-semibold text-fg">
            Row{" "}
            <code className="font-mono text-fg-muted">#{example.index}</code>
          </h3>
          {example.truncated_fields && example.truncated_fields.length > 0 && (
            <span className="rounded-md border border-warning/30 bg-warning/5 px-2 py-0.5 text-[11px] text-warning">
              Truncated at 4 KiB: {example.truncated_fields.join(", ")}
            </span>
          )}
          <div className="ml-auto flex items-center gap-1">
            <button
              type="button"
              onClick={onPrev}
              disabled={!hasPrev}
              title="Previous row (←)"
              className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted hover:border-border hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
            >
              ←
            </button>
            <button
              type="button"
              onClick={onNext}
              disabled={!hasNext}
              title="Next row (→)"
              className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted hover:border-border hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
            >
              →
            </button>
            <button
              type="button"
              onClick={onClose}
              title="Close (Esc)"
              className="ml-1 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted hover:border-border hover:text-fg"
            >
              Close
            </button>
          </div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
          {hasInput && (
            <DatasetField
              label="input"
              value={example.input}
            />
          )}
          {hasOutput && (
            <DatasetField
              label="output"
              value={example.output}
            />
          )}
          {!hasInput && !hasOutput && (
            <DatasetField label="row" value={rawText} />
          )}

          <details className="rounded-md border border-border-subtle bg-bg/50">
            <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-fg-muted hover:text-fg">
              Raw JSON
            </summary>
            <pre className="m-0 max-h-[40vh] overflow-auto whitespace-pre-wrap break-all border-t border-border-subtle px-3 py-2 font-mono text-[11px] leading-snug text-fg">
              {rawText}
            </pre>
          </details>
        </div>
      </div>
    </div>
  );
}

function DatasetField({
  label,
  value,
}: {
  label: string;
  value: string | undefined;
}) {
  return (
    <section className="flex min-h-0 flex-col">
      <div className="px-1 pb-1 text-[10px] uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <pre className="m-0 max-h-[60vh] overflow-auto whitespace-pre-wrap break-words rounded-md border border-border-subtle bg-bg px-3 py-2 font-mono text-xs text-fg">
        {value ?? "—"}
      </pre>
    </section>
  );
}
