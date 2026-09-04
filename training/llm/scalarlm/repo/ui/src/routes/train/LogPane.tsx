import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import {
  fetchFullTrainingLog,
  tailTrainingLogs,
  type LogLine,
  type TrainingJobStatus,
} from "@/api/training";

interface LogPaneProps {
  jobHash: string;
  status: TrainingJobStatus;
}

type StreamStatus = "connecting" | "streaming" | "reconnecting" | "closed";

const MAX_LINES = 10_000; // cap memory: ~several MB for typical log widths

const ERROR_REGEX = /\berror\b|traceback|cuda error|oom|killed/i;

export function LogPane({ jobHash, status }: LogPaneProps) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("connecting");
  const [autoScroll, setAutoScroll] = useState(true);
  const [query, setQuery] = useState("");
  const [useRegex, setUseRegex] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef(status);
  statusRef.current = status;

  useEffect(() => {
    if (!jobHash) return;
    const controller = new AbortController();
    setLines([]);
    setStreamStatus("connecting");

    tailTrainingLogs({
      modelName: jobHash,
      signal: controller.signal,
      isStillRunning: () =>
        statusRef.current === "QUEUED" || statusRef.current === "TRAINING",
      onLine: (line) => {
        setStreamStatus("streaming");
        setLines((prev) => {
          const next = prev.length >= MAX_LINES ? prev.slice(prev.length - MAX_LINES + 1) : prev.slice();
          next.push(line);
          return next;
        });
      },
      onStatus: (s) => setStreamStatus(s),
    });

    return () => controller.abort();
  }, [jobHash]);

  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines, autoScroll]);

  const filter = useMemo(() => {
    if (!query) return null;
    if (useRegex) {
      try {
        return new RegExp(query, "i");
      } catch {
        return null;
      }
    }
    const needle = query.toLowerCase();
    return needle;
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
        scrollToLine(lines[i].line_number);
        setAutoScroll(false);
        return;
      }
    }
  };

  const scrollToLine = (lineNumber: number) => {
    const el = document.querySelector(`[data-line="${lineNumber}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  };

  // Download the COMPLETE stitched log from the server, not the in-memory
  // `lines` buffer — that buffer is a rolling MAX_LINES tail, so for long or
  // resumed (multi-slice) jobs it omits everything older than the last
  // MAX_LINES lines. Re-reading from the endpoint yields the full log.
  const downloadLog = async () => {
    if (downloading) return;
    setDownloading(true);
    setDownloadError(null);
    const controller = new AbortController();
    try {
      const text = await fetchFullTrainingLog(jobHash, controller.signal);
      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${jobHash}.log`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(
        err instanceof Error ? err.message : "Failed to download log",
      );
    } finally {
      setDownloading(false);
    }
  };

  return (
    <section className="flex flex-col gap-2 rounded-lg border border-border-subtle bg-bg-card">
      <header className="flex flex-wrap items-center gap-2 border-b border-border-subtle px-4 py-2">
        <h2 className="mr-2 text-sm font-semibold">Logs</h2>
        <StreamStatusDot status={streamStatus} />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter"
          className="w-40 rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
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
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={jumpToLastError}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
          >
            Jump to last error
          </button>
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
            disabled={downloading}
            title={downloadError ?? undefined}
            className={clsx(
              "rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover",
              downloading && "cursor-wait opacity-60",
              downloadError && "border-danger/40 text-danger",
            )}
          >
            {downloading ? "Downloading…" : downloadError ? "Retry download" : "Download"}
          </button>
        </div>
      </header>
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
          if (autoScroll && !nearBottom) setAutoScroll(false);
        }}
        className="max-h-[60vh] min-h-[240px] overflow-auto px-2 py-1 font-mono text-[11px] leading-relaxed"
      >
        {filtered.length === 0 ? (
          <div className="px-2 py-3 text-xs text-fg-subtle">
            {lines.length === 0
              ? "Waiting for log output…"
              : "No lines match the current filter."}
          </div>
        ) : (
          filtered.map((l) => (
            <div
              key={l.line_number}
              data-line={l.line_number}
              className={clsx(
                "flex gap-3 px-2 py-[1px]",
                ERROR_REGEX.test(l.line) && "bg-danger/10 text-danger",
              )}
            >
              <span className="w-12 shrink-0 select-none text-right text-fg-subtle">
                {l.line_number}
              </span>
              <span className="whitespace-pre-wrap break-all text-fg">
                {l.line}
              </span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function StreamStatusDot({ status }: { status: StreamStatus }) {
  const map: Record<StreamStatus, { color: string; label: string }> = {
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
