/**
 * Lightweight JSON-in-localStorage helpers for user preferences. No schema —
 * callers are responsible for validating what comes back, because prior UI
 * versions may have written older shapes that no longer parse. We keep a
 * key-prefix so clearing "ScalarLM local data" (settings page, future work)
 * can wipe just our entries.
 */

const PREFIX = "scalarlm:";

export function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Quota exceeded or localStorage unavailable (private mode on some
    // browsers). Silently drop; preferences are best-effort.
  }
}

export function removeKey(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key);
  } catch {
    // ignore
  }
}
