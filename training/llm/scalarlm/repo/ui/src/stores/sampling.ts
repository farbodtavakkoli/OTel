/**
 * Sampling-parameter store (currently just `temperature`).
 *
 * Default is 0 for deterministic output — the user can bump it from the
 * settings page. Temperature is read from this store at send-time by the
 * chat view, so a slider change takes effect on the next message without
 * needing the view to re-render.
 *
 * Mirrors the shape of stores/theme.ts: a typed value + get/set/subscribe
 * for useSyncExternalStore.
 */

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

/**
 * Temperature to send, or undefined to let the server decide.
 *
 * Diffusion models (dLLMs) reject any temperature other than 1.0 — they
 * denoise a canvas on a fixed schedule, so per-request sampling has no
 * meaning. Sending our default of 0 fails every request against one:
 *
 *   ValueError: The temperature, min_p, seed, ... sampling parameters are
 *   not yet supported with diffusion models.
 *
 * Omitting the field when the user hasn't touched the slider lets the
 * server apply its own default, which works for both model families. An
 * explicitly chosen value is still sent, and will still be rejected by a
 * diffusion model — that error is the honest answer to "set temperature
 * to 0.7 on a model that cannot do it".
 */
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
      // continue
    }
  }
}

export function subscribeTemperature(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
