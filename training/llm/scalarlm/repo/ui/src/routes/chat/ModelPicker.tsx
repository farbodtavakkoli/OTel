import { useMemo, useSyncExternalStore } from "react";

import { useModels, type OpenAIModel } from "@/api/chat";
import { getApiConfig } from "@/api/config";
import { getAlias, subscribe } from "@/stores/aliases";

interface ModelPickerProps {
  value: string;
  onChange: (model: string) => void;
  /** If true, first change on a non-empty conversation warns via the caller. */
  locked?: boolean;
}

/**
 * Sorts a model list into:
 *   1. The deployment's base model (from api-config.json::default_model)
 *   2. Post-trained adapters, most recently registered first
 *   3. Anything else, in upstream order
 *
 * A virtual "latest" entry is prepended so users can always resolve to the
 * newest trained model without knowing its hash. ScalarLM's backend accepts
 * model: "latest" and resolves it via get_latest_model() — see generate.py:51.
 */
export function ModelPicker({ value, onChange, locked }: ModelPickerProps) {
  const { data, isPending, error } = useModels();
  const { default_model } = getApiConfig();

  // Re-render dropdown labels when aliases change. Monotonic signal so
  // getSnapshot returns a stable value until an actual alias edit fires.
  useSyncExternalStore(subscribeAliasVersion, getAliasVersion, getAliasVersion);

  const options = useMemo<SortedModels>(() => {
    if (!data) return { base: undefined, rest: [] };
    return sortModels(data, default_model);
  }, [data, default_model]);

  if (isPending) {
    return (
      <div className="h-8 w-48 animate-pulse rounded-md bg-bg-hover" />
    );
  }
  if (error) {
    return (
      <div className="text-xs text-danger">models unavailable</div>
    );
  }

  return (
    <label className="flex items-center text-xs text-fg-muted">
      <span className="sr-only">Model</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={locked}
        aria-label="Model"
        title="Model"
        className="max-w-[22rem] truncate rounded-md border border-border-subtle bg-bg-card px-2 py-1 font-mono text-xs text-fg hover:border-border focus:border-border focus:outline-none disabled:opacity-60"
      >
        <option value="latest">latest · newest trained model</option>
        {default_model && (
          <option value={default_model}>
            {default_model} · (default)
          </option>
        )}
        {options.base && options.base.id !== default_model && (
          <option value={options.base.id}>
            {options.base.id}
          </option>
        )}
        {options.rest.length > 0 && (
          <optgroup label="Adapters">
            {options.rest.map((m) => (
              <option key={m.id} value={m.id}>
                {labelForModel(m)}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </label>
  );
}

interface SortedModels {
  base: OpenAIModel | undefined;
  rest: OpenAIModel[];
}

function sortModels(
  list: OpenAIModel[],
  base: string,
): SortedModels {
  const baseEntry = list.find((m) => m.id === base);
  const rest = list
    .filter((m) => m.id !== base)
    .slice()
    .sort((a, b) => (b.created ?? 0) - (a.created ?? 0));
  return { base: baseEntry, rest };
}

function labelForModel(m: OpenAIModel): string {
  const alias = getAlias(m.id);
  // Adapter IDs are SHA-256 hashes — shortened for a dense dropdown display.
  const shortId = m.id.length > 20 ? `${m.id.slice(0, 12)}…` : m.id;
  const head = alias ? `${alias} (${m.id.slice(0, 8)})` : shortId;
  if (!m.created) return head;
  const ago = relativeTime(m.created * 1000);
  return `${head} · ${ago}`;
}

// Monotonic signal for useSyncExternalStore. The subscribe wrapper bumps the
// counter before invoking React's listener, so getSnapshot returns a value
// that's === stable between edits (avoiding infinite re-renders) but changes
// when any alias is edited.
let aliasVersion = 0;
function subscribeAliasVersion(listener: () => void): () => void {
  return subscribe(() => {
    aliasVersion++;
    listener();
  });
}
function getAliasVersion(): number {
  return aliasVersion;
}

function relativeTime(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 60_000) return "just now";
  const m = Math.floor(delta / 60_000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}
