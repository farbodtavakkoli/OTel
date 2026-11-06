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
  tailReopenDelayMs?: number;
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
