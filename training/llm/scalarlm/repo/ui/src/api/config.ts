/**
 * Boot-time runtime config. Fetched from /app/api-config.json, which is served
 * dynamically by FastAPI so the same bundle can run against any deployment.
 */
export interface ApiConfig {
  api_base: string;
  version: string;
  default_model: string;
  features: Record<string, boolean>;
}

const CONFIG_URL = "/app/api-config.json";

// Fallbacks used when the config endpoint is unreachable — e.g. during pure
// static-file dev or if the API hasn't come up yet. These keep the UI
// navigable; real data fetches will still fail with a visible error.
const FALLBACK: ApiConfig = {
  api_base: "/v1",
  version: "unknown",
  default_model: "unknown",
  features: {},
};

let cached: ApiConfig | null = null;
let inFlight: Promise<ApiConfig> | null = null;

export async function loadApiConfig(): Promise<ApiConfig> {
  if (cached) return cached;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    try {
      const resp = await fetch(CONFIG_URL, { cache: "no-cache" });
      if (!resp.ok) {
        console.warn(
          `api-config.json returned ${resp.status}; using fallback`,
        );
        cached = FALLBACK;
        return cached;
      }
      cached = (await resp.json()) as ApiConfig;
      return cached;
    } catch (err) {
      console.warn("Failed to load api-config.json; using fallback", err);
      cached = FALLBACK;
      return cached;
    } finally {
      inFlight = null;
    }
  })();

  return inFlight;
}

/** Synchronous accessor. Returns the fallback until loadApiConfig resolves. */
export function getApiConfig(): ApiConfig {
  return cached ?? FALLBACK;
}
