import { getApiConfig } from "./config";

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
  json?: unknown;
  rawBody?: BodyInit | null;
  signal?: AbortSignal;
}

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
