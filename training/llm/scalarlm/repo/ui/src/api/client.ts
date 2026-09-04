import { getApiConfig } from "./config";

/**
 * Error type thrown by apiFetch so callers (and TanStack Query) can distinguish
 * HTTP failures from network failures and branch on status codes.
 */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly url: string,
    message: string,
    public readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ApiFetchOptions extends Omit<RequestInit, "body"> {
  /** Parsed JSON body; pass FormData / Blob / string on `rawBody` instead. */
  json?: unknown;
  rawBody?: BodyInit | null;
  /** Abort the request when this signal fires. */
  signal?: AbortSignal;
}

/**
 * Thin wrapper around fetch that:
 *  - prepends the resolved api_base from api-config.json
 *  - handles JSON bodies
 *  - throws ApiError on non-2xx responses with structured info
 *
 * Streaming endpoints (chat completions, training logs) bypass this helper
 * and use fetch / EventSource directly so they can stream without buffering.
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: ApiFetchOptions = {},
): Promise<T> {
  const { api_base } = getApiConfig();
  const url = path.startsWith("http")
    ? path
    : `${api_base}${path.startsWith("/") ? path : `/${path}`}`;

  const headers = new Headers(options.headers);
  let body: BodyInit | null | undefined = options.rawBody;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }

  const resp = await fetch(url, {
    ...options,
    headers,
    body,
  });

  if (!resp.ok) {
    let parsed: unknown;
    try {
      parsed = await resp.json();
    } catch {
      parsed = await resp.text().catch(() => undefined);
    }
    const detail =
      (parsed as { detail?: string } | null)?.detail ?? resp.statusText;
    throw new ApiError(resp.status, url, `${resp.status} ${detail}`, parsed);
  }

  const contentType = resp.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return (await resp.json()) as T;
  }
  return (await resp.text()) as unknown as T;
}
