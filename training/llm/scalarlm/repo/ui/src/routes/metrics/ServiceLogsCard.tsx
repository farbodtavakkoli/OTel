/**
 * Live tail of the api / vllm / megatron service logs. One live stream
 * at a time (the active tab) to keep the number of concurrent SSE
 * connections bounded. An "Expand" button pops the current tab into a
 * full-screen modal so the operator can see more context without
 * leaving the metrics page.
 *
 * Backend contract: GET /v1/health/logs/{service_name} yields NDJSON
 * {"line": "...", "line_number": N}\n — see
 * infra/cray_infra/api/fastapi/routers/health_router.py:29.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import {
  fetchServiceLogRange,
  tailServiceLogs,
  type LogStreamStatus,
  type ServiceName,
} from "@/api/logs";
import type { LogLine } from "@/api/training";

const SERVICES: { name: ServiceName; label: string; hint: string }[] = [
  { name: "api", label: "API", hint: "FastAPI control plane" },
  { name: "vllm", label: "vLLM", hint: "Inference engine" },
  { name: "megatron", label: "Megatron", hint: "Training + slurmd" },
];

// Max lines held in memory before we start dropping the oldest. 20k lines
// ≈ a few MB for typical log widths; keeps the DOM responsive without
// needing a virtualized list.
const MAX_LINES = 20_000;
const INITIAL_TAIL = 5_000;
// How many lines to pull each time the user scrolls near the top.
const BACKFILL_CHUNK = 2_000;
// px from the top below which we trigger a backfill request.
const TOP_TRIGGER_PX = 120;
const ERROR_REGEX = /\berror\b|traceback|cuda error|oom|killed/i;

export function ServiceLogsCard() {
  const [active, setActive] = useState<ServiceName>("api");
  const [expanded, setExpanded] = useState(false);

  // Close the modal on escape so the expanded view behaves like a dialog.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  return (
    <>
      <section className="flex flex-col gap-0 rounded-lg border border-border-subtle bg-bg-card">
        <header className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-3">
          <h2 className="text-sm font-semibold text-fg">Service logs</h2>
          <div
            role="tablist"
            aria-label="Service"
            className="ml-2 flex items-center gap-1"
          >
            {SERVICES.map((svc) => (
              <button
                key={svc.name}
                type="button"
                role="tab"
                aria-selected={active === svc.name}
                onClick={() => setActive(svc.name)}
                title={svc.hint}
                className={clsx(
                  "rounded-md px-3 py-1 text-xs transition-colors",
                  active === svc.name
                    ? "bg-bg-hover text-fg ring-1 ring-border"
                    : "text-fg-muted hover:bg-bg hover:text-fg",
                )}
              >
                {svc.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="ml-auto rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
            aria-label="Expand log viewer"
          >
            Expand
          </button>
        </header>
        <ServiceLogTail service={active} expanded={false} />
      </section>

      {expanded && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Expanded service logs"
          className="fixed inset-0 z-50 flex flex-col bg-bg/95 backdrop-blur-sm"
        >
          <header className="flex items-center gap-2 border-b border-border-subtle bg-bg-card px-4 py-3">
            <h2 className="text-sm font-semibold text-fg">Service logs</h2>
            <div role="tablist" className="ml-2 flex items-center gap-1">
              {SERVICES.map((svc) => (
                <button
                  key={svc.name}
                  type="button"
                  role="tab"
                  aria-selected={active === svc.name}
                  onClick={() => setActive(svc.name)}
                  className={clsx(
                    "rounded-md px-3 py-1 text-xs transition-colors",
                    active === svc.name
                      ? "bg-bg-hover text-fg ring-1 ring-border"
                      : "text-fg-muted hover:bg-bg hover:text-fg",
                  )}
                >
                  {svc.label}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="ml-auto rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
              autoFocus
            >
              Close (Esc)
            </button>
          </header>
          <div className="min-h-0 flex-1">
            <ServiceLogTail service={active} expanded={true} />
          </div>
        </div>
      )}
    </>
  );
}

interface ServiceLogTailProps {
  service: ServiceName;
  expanded: boolean;
}

function ServiceLogTail({ service, expanded }: ServiceLogTailProps) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [status, setStatus] = useState<LogStreamStatus>("connecting");
  const [autoScroll, setAutoScroll] = useState(true);
  const [query, setQuery] = useState("");
  const [useRegex, setUseRegex] = useState(false);
  const [backfilling, setBackfilling] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // `backfillInFlight` prevents overlapping scrollback fetches; `noMoreAbove`
  // short-circuits once we've reached line 0 so we stop hammering the server.
  const backfillInFlight = useRef(false);
  const noMoreAbove = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    setLines([]);
    setStatus("connecting");
    setAutoScroll(true);
    noMoreAbove.current = false;
    backfillInFlight.current = false;

    tailServiceLogs({
      service,
      signal: controller.signal,
      initialTail: INITIAL_TAIL,
      onLine: (line) =>
        setLines((prev) => {
          // Cap the buffer from the *front* so that scrolling back doesn't
          // immediately trim the rows the user is looking at. Live appends
          // drop the oldest rows; backfilled prepends are guarded below.
          const next =
            prev.length >= MAX_LINES
              ? prev.slice(prev.length - MAX_LINES + 1)
              : prev.slice();
          next.push(line);
          return next;
        }),
      onStatus: setStatus,
    });
    return () => controller.abort();
  }, [service]);

  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines, autoScroll]);

  // Jump to end once the first tail batch arrives, regardless of autoScroll
  // state — we want to land at the tail on initial load of each tab.
  useEffect(() => {
    if (lines.length === 0) return;
    const el = scrollRef.current;
    if (!el) return;
    if (el.scrollHeight <= el.clientHeight) return;
    // Only snap on the *first* non-empty render per service.
    if (el.dataset.snapped === "yes") return;
    el.scrollTop = el.scrollHeight;
    el.dataset.snapped = "yes";
  }, [lines]);

  // Reset the snap flag when we switch services.
  useEffect(() => {
    if (scrollRef.current) delete scrollRef.current.dataset.snapped;
  }, [service]);

  const jumpToEnd = () => {
    setAutoScroll(true);
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  // Scrollback: when the user scrolls near the top, pull the previous
  // BACKFILL_CHUNK lines and prepend them, compensating scrollTop so the
  // viewport stays anchored on the row the user is looking at. We ask
  // the server by byte offset (`before_byte_offset`) rather than line
  // number so the backfill doesn't pay a full-file forward scan.
  const maybeBackfill = async () => {
    const el = scrollRef.current;
    if (!el) return;
    if (backfillInFlight.current || noMoreAbove.current) return;
    if (el.scrollTop > TOP_TRIGGER_PX) return;
    if (lines.length === 0) return;

    const earliest = lines[0];
    const earliestOffset = earliest.byte_offset;
    if (earliestOffset === undefined) {
      // No byte offset available — server is running an older protocol.
      // Give up on scrollback rather than falling back to the slow path.
      noMoreAbove.current = true;
      return;
    }
    if (earliestOffset === 0) {
      noMoreAbove.current = true;
      return;
    }

    backfillInFlight.current = true;
    setBackfilling(true);
    const anchorHeight = el.scrollHeight;

    try {
      const older = await fetchServiceLogRange(
        service,
        earliestOffset,
        BACKFILL_CHUNK,
      );
      if (older.length === 0) {
        noMoreAbove.current = true;
        return;
      }
      setLines((prev) => {
        // Guard against races / duplicate fetches: drop any overlap with
        // the current buffer's first row, using byte offset as the
        // stable identity.
        const firstExisting = prev[0]?.byte_offset ?? Infinity;
        const fresh = older.filter(
          (l) => (l.byte_offset ?? -1) < firstExisting,
        );
        return fresh.concat(prev);
      });
      // Preserve scroll anchor: the prepended rows added scrollHeight,
      // so shift scrollTop by the delta.
      requestAnimationFrame(() => {
        const el2 = scrollRef.current;
        if (!el2) return;
        const delta = el2.scrollHeight - anchorHeight;
        if (delta > 0) el2.scrollTop = el2.scrollTop + delta;
      });
      // If the first backfilled record starts at byte 0, we've reached
      // the beginning of the file.
      if ((older[0].byte_offset ?? -1) === 0) noMoreAbove.current = true;
    } catch {
      // Swallow — next scroll event will retry.
    } finally {
      backfillInFlight.current = false;
      setBackfilling(false);
    }
  };

  const filter = useMemo(() => {
    if (!query) return null;
    if (useRegex) {
      try {
        return new RegExp(query, "i");
      } catch {
        return null;
      }
    }
    return query.toLowerCase();
  }, [query, useRegex]);

  const filtered = useMemo(() => {
    if (!filter) return lines;
    if (typeof filter === "string") {
      return lines.filter((l) => l.line.toLowerCase().includes(filter));
    }
    return lines.filter((l) => filter.test(l.line));
  }, [lines, filter]);

  const jumpToLastError = () => {
    for (let i = lines.length - 1; i >= 0; i--) {
      if (ERROR_REGEX.test(lines[i].line)) {
        const rowKey = lines[i].byte_offset ?? `ln-${lines[i].line_number}`;
        const el = document.querySelector(
          `[data-line="${service}-${rowKey}"]`,
        );
        el?.scrollIntoView({ block: "center", behavior: "smooth" });
        setAutoScroll(false);
        return;
      }
    }
  };

  const downloadLog = () => {
    const blob = new Blob([lines.map((l) => l.line).join("\n")], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${service}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-2">
        <StreamStatusDot status={status} />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Search ${service} logs…`}
          className="w-56 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
        />
        <label className="flex items-center gap-1 text-xs text-fg-muted">
          <input
            type="checkbox"
            checked={useRegex}
            onChange={(e) => setUseRegex(e.target.checked)}
            className="h-3 w-3 accent-accent"
          />
          regex
        </label>
        <span className="text-[10px] text-fg-subtle">
          {filtered.length.toLocaleString()}
          {query ? ` / ${lines.length.toLocaleString()}` : ""} lines
          {backfilling && " · loading older…"}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={jumpToLastError}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            Jump to last error
          </button>
          {!autoScroll && (
            <button
              type="button"
              onClick={jumpToEnd}
              className="rounded-md border border-accent/40 bg-accent/10 px-2 py-1 text-xs text-accent hover:bg-accent/20"
              title="Scroll to the live tail and resume auto-scroll"
            >
              Jump to end
            </button>
          )}
          <button
            type="button"
            onClick={() => setAutoScroll((v) => !v)}
            className={clsx(
              "rounded-md border px-2 py-1 text-xs",
              autoScroll
                ? "border-accent/40 bg-accent/10 text-accent"
                : "border-border-subtle bg-bg text-fg hover:border-border hover:bg-bg-hover",
            )}
          >
            {autoScroll ? "Auto-scroll on" : "Auto-scroll off"}
          </button>
          <button
            type="button"
            onClick={downloadLog}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            Download
          </button>
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const nearBottom =
            el.scrollHeight - el.scrollTop - el.clientHeight < 20;
          if (autoScroll && !nearBottom) setAutoScroll(false);
          if (el.scrollTop <= TOP_TRIGGER_PX) {
            // Fire-and-forget; maybeBackfill guards its own concurrency.
            void maybeBackfill();
          }
        }}
        className={clsx(
          "min-h-[240px] overflow-auto px-2 py-1 font-mono text-[11px] leading-relaxed",
          expanded ? "flex-1" : "max-h-[420px]",
        )}
      >
        {filtered.length === 0 ? (
          <div className="px-2 py-3 text-xs text-fg-subtle">
            {lines.length === 0
              ? "Waiting for log output…"
              : "No lines match the current filter."}
          </div>
        ) : (
          filtered.map((l) => {
            // byte_offset is globally unique per-file and survives
            // prepend/append alike; line_number can collide across a
            // tail + backfill boundary, so use it only as a display
            // label, not as a React key.
            const rowKey = l.byte_offset ?? `ln-${l.line_number}`;
            return (
              <div
                key={rowKey}
                data-line={`${service}-${rowKey}`}
                className={clsx(
                  "flex gap-3 px-2 py-[1px]",
                  ERROR_REGEX.test(l.line) && "bg-danger/10 text-danger",
                )}
              >
                <span className="w-14 shrink-0 select-none text-right text-fg-subtle">
                  {l.line_number}
                </span>
                <span className="whitespace-pre-wrap break-all text-fg">
                  {l.line}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function StreamStatusDot({ status }: { status: LogStreamStatus }) {
  const map: Record<LogStreamStatus, { color: string; label: string }> = {
    connecting: { color: "bg-fg-subtle", label: "connecting" },
    streaming: { color: "bg-success", label: "streaming" },
    reconnecting: { color: "bg-warning animate-pulse", label: "reconnecting" },
    closed: { color: "bg-fg-subtle", label: "closed" },
  };
  const { color, label } = map[status];
  return (
    <span
      title={label}
      className="flex items-center gap-1.5 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted"
    >
      <span className={clsx("h-2 w-2 rounded-full", color)} />
      {label}
    </span>
  );
}
