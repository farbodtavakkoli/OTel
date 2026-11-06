import { readJson, writeJson } from "@/lib/preferences";

const KEY = "sampling.temperature";

export const TEMPERATURE_MIN = 0;
export const TEMPERATURE_MAX = 2;
export const TEMPERATURE_DEFAULT = 0;

const listeners = new Set<() => void>();

let current: number = sanitize(
  readJson<number>(KEY, TEMPERATURE_DEFAULT),
);

function sanitize(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return TEMPERATURE_DEFAULT;
  if (n < TEMPERATURE_MIN) return TEMPERATURE_MIN;
  if (n > TEMPERATURE_MAX) return TEMPERATURE_MAX;
  return n;
}

export function getTemperature(): number {
  return current;
}

export function getTemperatureForRequest(): number | undefined {
  return current === TEMPERATURE_DEFAULT ? undefined : current;
}

export function setTemperature(value: number): void {
  const next = sanitize(value);
  if (next === current) return;
  current = next;
  writeJson(KEY, next);
  for (const l of listeners) {
    try {
      l();
    } catch {
    }
  }
}

export function subscribeTemperature(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
