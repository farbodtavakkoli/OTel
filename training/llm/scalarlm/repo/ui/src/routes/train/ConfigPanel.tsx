import { useState } from "react";

interface ConfigPanelProps {
  config: Record<string, unknown>;
}

/**
 * Collapsible rendering of the per-job config.yaml. YAML-ish serialization
 * (indented key: value pairs), not a round-trippable YAML writer — purely
 * for display. Nested objects are rendered as JSON for compactness.
 */
export function ConfigPanel({ config }: ConfigPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <section className="rounded-lg border border-border-subtle bg-bg-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold hover:bg-bg-hover"
      >
        <span>Config</span>
        <span className="font-mono text-xs text-fg-subtle">
          {Object.keys(config).length} keys · {open ? "hide" : "show"}
        </span>
      </button>
      {open && (
        <pre className="max-h-96 overflow-auto border-t border-border-subtle px-4 py-3 font-mono text-xs leading-relaxed text-fg">
          {serializeConfig(config)}
        </pre>
      )}
    </section>
  );
}

function serializeConfig(config: Record<string, unknown>): string {
  const keys = Object.keys(config).sort();
  const width = Math.max(...keys.map((k) => k.length));
  return keys
    .map((key) => {
      const padded = key.padEnd(width, " ");
      return `${padded}  ${formatValue(config[key])}`;
    })
    .join("\n");
}

function formatValue(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}
