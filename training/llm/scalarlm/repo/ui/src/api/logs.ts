/**
 * Service-log tailer + scrollback fetcher.
 *
 * Wire protocol: see service_logs_generator.py. The server reads the log
 * file by byte offset; each record carries its own `byte_offset` and
 * `next_offset`. The client tracks the last-seen `next_offset` and uses
 * it as the resume token on reconnect, which makes tail/resume O(new
 * lines) rather than O(total lines).
 *
 *  - First connect:        ?tail=N                     — jump to end fast.
 *  - Subsequent reconnects ?starting_byte_offset=<last.next_offset>
 *  - Scrollback (one-shot) ?before_byte_offset=B&before_count=C
 */

import { getApiConfig } from "./config";
import { streamNdjsonLogOnce, type LogLine } from "./training";

export type ServiceName = "api" | "vllm" | "megatron";

export type LogStreamStatus =
  | "connecting"
  | "streaming"
  | "reconnecting"
  | "closed";

export interface ServiceLogTailOptions {
  service: ServiceName;
  signal: AbortSignal;
  onLine: (line: LogLine) => void;
  onStatus?: (status: LogStreamStatus) => void;
  /** Delay between reconnects once the generator hits EOF. Default 2 s. */
  tailReopenDelayMs?: number;
  /**
   * Trailing lines the first connect requests via `tail=N`. Subsequent
   * reconnects resume from the last record's `next_offset`.
   */
  initialTail?: number;
}

export async function tailServiceLogs({
  service,
  signal,
  onLine,
  onStatus,
  tailReopenDelayMs = 2_000,
  initialTail = 5_000,
}: ServiceLogTailOptions): Promise<void> {
  // Resume state. `nextOffset` is authoritative once we've received at
  // least one record; `lineLabel` is the monotonic label the UI shows
  // in the gutter — we carry it across reconnects so the count keeps
  // growing instead of restarting at 0 on every reopen.
  let nextOffset: number | null = null;
  let lineLabel = 0;
  let firstPass = true;

  while (!signal.aborted) {
    onStatus?.(firstPass ? "connecting" : "reconnecting");
    try {
      const params = new URLSearchParams();
      if (firstPass && initialTail > 0) {
        params.set("tail", String(initialTail));
      } else if (nextOffset !== null) {
        params.set("starting_byte_offset", String(nextOffset));
      }
      params.set("starting_line_number", String(lineLabel));

      const url = buildLogsUrl(service, params);
      await streamNdjsonLogOnce(url, lineLabel, signal, (ln) => {
        onStatus?.("streaming");
        if (typeof ln.next_offset === "number") {
          nextOffset = ln.next_offset;
        }
        if (typeof ln.line_number === "number") {
          lineLabel = ln.line_number + 1;
        }
        onLine(ln);
      });
      firstPass = false;
    } catch (err) {
      if (signal.aborted || (err as { name?: string }).name === "AbortError")
        break;
    }

    onStatus?.("reconnecting");
    await sleep(tailReopenDelayMs, signal);
  }

  onStatus?.("closed");
}

/**
 * One-shot backfill. Returns the `count` lines immediately before
 * `beforeByteOffset` (typically an earlier record's byte_offset), in
 * ascending order. Used when the user scrolls near the top of the
 * loaded buffer.
 */
export async function fetchServiceLogRange(
  service: ServiceName,
  beforeByteOffset: number,
  count: number,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const params = new URLSearchParams({
    before_byte_offset: String(Math.max(0, beforeByteOffset)),
    before_count: String(count),
  });
  const url = buildLogsUrl(service, params);
  const out: LogLine[] = [];
  await streamNdjsonLogOnce(
    url,
    0,
    signal ?? new AbortController().signal,
    (ln) => out.push(ln),
  );
  return out;
}

function buildLogsUrl(service: ServiceName, params: URLSearchParams): string {
  const { api_base } = getApiConfig();
  return `${api_base}/health/logs/${encodeURIComponent(service)}?${params}`;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}
