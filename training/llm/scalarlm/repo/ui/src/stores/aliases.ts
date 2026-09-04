/**
 * Local nicknames for training-job model hashes. Pure localStorage — no
 * backend persistence yet. When a PUT /v1/megatron/train/{hash}/alias endpoint
 * lands (see docs/ui-design.md §13), the write path can fan out to the server
 * while keeping local state as an immediate-feedback cache.
 *
 * The whole table lives under a single localStorage key so we can subscribe
 * once and render everywhere without per-key machinery.
 */

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
      // continue
    }
  }
}

export function getAlias(hash: string): string | undefined {
  const map = getMap();
  const value = map[hash];
  return value && value.trim() ? value : undefined;
}

/** Display label: alias if set, else the first 12 chars of the hash. */
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
