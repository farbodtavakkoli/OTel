/**
 * Theme store. Three user-facing modes:
 *
 *   "auto"  — follow the OS preference (prefers-color-scheme)
 *   "light" — force light
 *   "dark"  — force dark
 *
 * The resolved theme (light|dark) is applied by toggling the `.dark` class
 * on <html>. CSS variables in styles.css flip accordingly. The store
 * persists the user's chosen mode (not the resolved value) in localStorage,
 * so the OS switching on its own re-resolves `auto` without overwriting the
 * user's explicit pick.
 */

import { readJson, writeJson } from "@/lib/preferences";

const KEY = "theme.mode";

export type ThemeMode = "auto" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

const listeners = new Set<() => void>();
let current: ThemeMode = readJson<ThemeMode>(KEY, "auto");
let mediaQuery: MediaQueryList | null = null;

function notify() {
  for (const l of listeners) {
    try {
      l();
    } catch {
      // continue
    }
  }
}

export function resolveTheme(mode: ThemeMode = current): ResolvedTheme {
  if (mode === "light" || mode === "dark") return mode;
  if (typeof window === "undefined") return "dark";
  const prefersDark =
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export function applyTheme(mode: ThemeMode = current): void {
  const resolved = resolveTheme(mode);
  const root = document.documentElement;
  if (resolved === "dark") root.classList.add("dark");
  else root.classList.remove("dark");
}

export function getThemeMode(): ThemeMode {
  return current;
}

export function setThemeMode(mode: ThemeMode): void {
  current = mode;
  writeJson(KEY, mode);
  applyTheme(mode);
  notify();
}

export function subscribeTheme(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Call once at boot. Applies the saved mode and starts listening for OS-level
 * prefers-color-scheme changes so `auto` remains live.
 */
export function initTheme(): void {
  applyTheme(current);
  if (typeof window !== "undefined" && window.matchMedia) {
    mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (current === "auto") {
        applyTheme("auto");
        notify();
      }
    };
    // addEventListener is supported in all evergreen browsers; the legacy
    // addListener fallback was dropped by MDN in 2020.
    mediaQuery.addEventListener("change", onChange);
  }
}
