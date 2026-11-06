export interface ApiConfig {
  api_base: string;
  version: string;
  default_model: string;
  features: Record<string, boolean>;
}

const CONFIG_URL = "/app/api-config.json";

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

export function getApiConfig(): ApiConfig {
  return cached ?? FALLBACK;
}
