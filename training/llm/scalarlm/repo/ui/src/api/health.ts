import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

/**
 * GET /v1/health shape today:
 *   { api: "up", vllm: "up" | {status: "down", reason: string}, all: "up"|"down"|"mixed" }
 *
 * The UI renders one dot per component it knows about. If the backend later
 * adds slurm / training / queue sub-components, they'll surface automatically.
 */
interface HealthResponseRaw {
  api?: string;
  vllm?: string | { status: string; reason?: string };
  all?: string;
  [key: string]: unknown;
}

export type HealthState = "up" | "down" | "unknown";

export interface HealthComponent {
  name: string;
  state: HealthState;
  reason?: string;
}

export interface HealthSnapshot {
  components: HealthComponent[];
  overall: HealthState;
  raw: HealthResponseRaw;
}

function stateFromValue(value: unknown): { state: HealthState; reason?: string } {
  if (value === "up") return { state: "up" };
  if (value === "down") return { state: "down" };
  if (value === "mixed") return { state: "down" };
  if (
    value &&
    typeof value === "object" &&
    "status" in value &&
    typeof (value as { status?: unknown }).status === "string"
  ) {
    const v = value as { status: string; reason?: string };
    return {
      state: v.status === "up" ? "up" : "down",
      reason: v.reason,
    };
  }
  return { state: "unknown" };
}

const EXCLUDED_FROM_DOTS = new Set(["all"]);

export function useHealth() {
  return useQuery<HealthSnapshot>({
    queryKey: ["health"],
    queryFn: async () => {
      const raw = await apiFetch<HealthResponseRaw>("/health");
      const components: HealthComponent[] = [];
      for (const [name, value] of Object.entries(raw)) {
        if (EXCLUDED_FROM_DOTS.has(name)) continue;
        const { state, reason } = stateFromValue(value);
        components.push({ name, state, reason });
      }
      const overall = (() => {
        const rawAll = stateFromValue(raw.all);
        if (rawAll.state !== "unknown") return rawAll.state;
        if (components.some((c) => c.state === "down")) return "down";
        if (components.every((c) => c.state === "up")) return "up";
        return "unknown";
      })();
      return { components, overall, raw };
    },
    refetchInterval: 10_000,
    staleTime: 0,
  });
}
