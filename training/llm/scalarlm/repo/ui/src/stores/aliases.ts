import { readJson, writeJson } from "@/lib/preferences";

const KEY = "aliases";
type AliasMap = Record<string, string>;

let memoryCache: AliasMap | null = null;
const listeners = new Set<() => void>();

function getMap(): AliasMap {
  if (memoryCache === null) memoryCache = readJson<AliasMap>(KEY, {});
  return memoryCache;
}

function notify() {
  for (const l of listeners) {
    try {
      l();
    } catch {
    }
  }
}

export function getAlias(hash: string): string | undefined {
  const map = getMap();
  const value = map[hash];
  return value && value.trim() ? value : undefined;
}

export function displayName(hash: string, short = 12): string {
  const alias = getAlias(hash);
  if (alias) return alias;
  if (hash.length <= short) return hash;
  return `${hash.slice(0, short)}…`;
}

export function setAlias(hash: string, alias: string): void {
  const map = { ...getMap() };
  const trimmed = alias.trim();
  if (trimmed) {
    map[hash] = trimmed;
  } else {
    delete map[hash];
  }
  memoryCache = map;
  writeJson(KEY, map);
  notify();
}

export function clearAliases(): void {
  memoryCache = {};
  writeJson(KEY, {});
  notify();
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
