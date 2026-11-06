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
    mediaQuery.addEventListener("change", onChange);
  }
}
