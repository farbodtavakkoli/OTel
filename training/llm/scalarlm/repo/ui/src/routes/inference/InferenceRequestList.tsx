import { useMemo, useState } from "react";

import {
  compileRegex,
  rowMatches,
  useInferenceRequestList,
  type InferenceListRow,
} from "@/api/inference";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";

import { InferenceRequestRow } from "./InferenceRequestRow";

export function InferenceRequestList() {
  const {
    data,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isPending,
    refetch,
  } = useInferenceRequestList();

  const [pattern, setPattern] = useState("");
  const compiled = useMemo(() => compileRegex(pattern), [pattern]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const allRows: InferenceListRow[] = useMemo(
    () => data?.pages.flatMap((p) => p.rows) ?? [],
    [data],
  );

  const visibleRows = useMemo(() => {
    if (compiled === null || compiled === undefined) return allRows;
    return allRows.filter((row) => rowMatches(row, compiled));
  }, [allRows, compiled]);

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <FilterBar
        pattern={pattern}
        onChange={setPattern}
        invalid={compiled === undefined}
        loadedCount={allRows.length}
        visibleCount={visibleRows.length}
      />

      <div className="rounded-lg border border-border-subtle bg-bg-card">
        {isPending && !data ? (
          <div className="flex flex-col gap-2 p-4">
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-full" />
          </div>
        ) : error ? (
          <div className="p-4">
            <ErrorState error={error} onRetry={refetch} />
          </div>
        ) : visibleRows.length === 0 ? (
          <div className="p-6 text-center text-sm text-fg-muted">
            {allRows.length === 0
              ? "No inference requests on disk yet."
              : "No rows match the current filter."}
          </div>
        ) : (
          <div>
            {visibleRows.map((row) => (
              <InferenceRequestRow
                key={row.request_id}
                row={row}
                expanded={expanded.has(row.request_id)}
                onToggle={() => toggle(row.request_id)}
              />
            ))}
          </div>
        )}
      </div>

      {hasNextPage && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
            className="rounded-md border border-border-subtle bg-bg-card px-4 py-2 text-sm text-fg-muted transition-colors hover:bg-bg-hover hover:text-fg disabled:opacity-50"
          >
            {isFetchingNextPage ? "Loading…" : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}

function FilterBar({
  pattern,
  onChange,
  invalid,
  loadedCount,
  visibleCount,
}: {
  pattern: string;
  onChange: (next: string) => void;
  invalid: boolean;
  loadedCount: number;
  visibleCount: number;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex flex-1 flex-col gap-1">
        <input
          type="text"
          value={pattern}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Regex filter (case-insensitive). Matches request_id, model, request_type, prompt preview."
          className="w-full rounded-md border border-border-subtle bg-bg-card px-3 py-2 text-sm text-fg placeholder:text-fg-subtle focus:border-border focus:outline-none"
          aria-invalid={invalid}
          aria-label="Regex filter"
          spellCheck={false}
        />
        {invalid && (
          <span className="text-xs text-danger">Invalid regex</span>
        )}
      </div>
      <span className="shrink-0 text-xs text-fg-subtle">
        Showing {visibleCount} of {loadedCount} loaded
      </span>
    </div>
  );
}
