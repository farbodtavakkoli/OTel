import { useState } from "react";

import { useHealth } from "@/api/health";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { StatusDot } from "@/components/StatusDot";

export function HealthCard() {
  const { data, error, refetch, isPending, dataUpdatedAt } = useHealth();
  const [rawOpen, setRawOpen] = useState(false);

  const lastProbe =
    dataUpdatedAt > 0 ? relativeSeconds(Date.now() - dataUpdatedAt) : null;

  return (
    <Card
      title="Health"
      subtitle="GET /v1/health"
      action={
        lastProbe && (
          <span className="font-mono text-xs text-fg-subtle">
            probed {lastProbe} ago
          </span>
        )
      }
    >
      {isPending && !data ? (
        <div className="flex gap-3">
          <Skeleton className="h-10 w-28" />
          <Skeleton className="h-10 w-28" />
          <Skeleton className="h-10 w-28" />
        </div>
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : data ? (
        <>
          <div className="flex flex-wrap gap-2">
            {data.components.map((c) => (
              <StatusDot
                key={c.name}
                state={c.state}
                label={c.name}
                title={c.reason ?? undefined}
                onClick={c.state === "down" ? () => setRawOpen(true) : undefined}
              />
            ))}
          </div>
          <button
            type="button"
            onClick={() => setRawOpen(true)}
            className="mt-3 text-xs text-fg-muted underline-offset-2 hover:text-fg hover:underline"
          >
            Show raw response
          </button>
          {rawOpen && (
            <RawHealthModal
              data={data.raw}
              onClose={() => setRawOpen(false)}
            />
          )}
        </>
      ) : null}
    </Card>
  );
}

function RawHealthModal({
  data,
  onClose,
}: {
  data: unknown;
  onClose: () => void;
}) {
  const json = JSON.stringify(data, null, 2);
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex max-h-[80vh] w-full max-w-2xl flex-col rounded-lg border border-border bg-bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border-subtle px-4 py-3">
          <h3 className="text-sm font-semibold">Health response</h3>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => navigator.clipboard.writeText(json)}
              className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs hover:border-border hover:bg-bg-hover"
            >
              Copy
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs hover:border-border hover:bg-bg-hover"
            >
              Close
            </button>
          </div>
        </div>
        <pre className="overflow-auto px-4 py-3 font-mono text-xs leading-relaxed">
          {json}
        </pre>
      </div>
    </div>
  );
}

function relativeSeconds(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h`;
}
