import clsx from "clsx";

import type { InferenceListRow } from "@/api/inference";
import { Badge, type BadgeTone } from "@/components/Badge";

import { InferenceRequestDetail } from "./InferenceRequestDetail";

function statusTone(status: string): BadgeTone {
  if (status === "completed") return "success";
  if (status === "in_progress") return "accent";
  if (status === "unknown") return "neutral";
  return "warning";
}

function relativeTime(epochSeconds: number): string {
  const deltaSec = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (deltaSec < 60) return `${Math.round(deltaSec)}s ago`;
  if (deltaSec < 3_600) return `${Math.round(deltaSec / 60)}m ago`;
  if (deltaSec < 86_400) return `${Math.round(deltaSec / 3_600)}h ago`;
  return `${Math.round(deltaSec / 86_400)}d ago`;
}

export function InferenceRequestRow({
  row,
  expanded,
  onToggle,
}: {
  row: InferenceListRow;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="border-b border-border-subtle last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-4 py-2 text-left transition-colors hover:bg-bg-hover focus:bg-bg-hover focus:outline-none"
        aria-expanded={expanded}
      >
        <span
          aria-hidden
          className={clsx(
            "shrink-0 text-fg-subtle transition-transform",
            expanded && "rotate-90",
          )}
        >
          ▸
        </span>
        <code
          className="shrink-0 font-mono text-xs text-fg-muted"
          title={row.request_id}
        >
          {row.request_id.slice(0, 12)}…
        </code>
        <span className="shrink-0 text-xs text-fg-muted">
          {row.request_type}
        </span>
        <Badge tone={statusTone(row.status)}>{row.status}</Badge>
        <span className="shrink-0 font-mono text-xs tabular-nums text-fg-subtle">
          {row.request_count} {row.request_count === 1 ? "prompt" : "prompts"}
        </span>
        <span className="min-w-0 flex-1 truncate text-xs text-fg-muted">
          {row.prompt_preview || <span className="italic">(no preview)</span>}
        </span>
        <span className="shrink-0 text-xs text-fg-subtle">
          {relativeTime(row.mtime)}
        </span>
      </button>
      <InferenceRequestDetail
        requestId={row.request_id}
        expanded={expanded}
      />
    </div>
  );
}
