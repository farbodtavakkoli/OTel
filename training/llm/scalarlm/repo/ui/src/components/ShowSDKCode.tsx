import { useState } from "react";

import { CopyButton } from "./CopyButton";

interface ShowSDKCodeProps {
  /** Python snippet to display. Expected to be complete and runnable. */
  code: string;
  /** Label above the snippet; optional. */
  title?: string;
}

/**
 * Disclosure that reveals a runnable Python SDK snippet equivalent to the
 * current UI state. Every write-side action in the UI can be reproduced via
 * the SDK — surfacing the snippet gives users a one-click path from click-ops
 * to scripted-ops without leaving the page.
 */
export function ShowSDKCode({
  code,
  title = "Equivalent SDK call",
}: ShowSDKCodeProps) {
  const [open, setOpen] = useState(false);

  return (
    <section className="rounded-md border border-border-subtle bg-bg-card">
      <header className="flex items-center justify-between px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-2 text-xs text-fg-muted hover:text-fg"
        >
          <span>{title}</span>
          <span className="text-[10px]">{open ? "hide" : "show"}</span>
        </button>
        {open && <CopyButton value={code} />}
      </header>
      {open && (
        <pre className="overflow-x-auto border-t border-border-subtle px-3 py-2 font-mono text-xs leading-relaxed text-fg">
          {code}
        </pre>
      )}
    </section>
  );
}

/**
 * Build a Python snippet for a submitted training job, exactly equivalent to
 * the train_args dict recorded in job_config. Omits server-filled fields
 * (job_directory, training_data_path, dataset_hash) since they aren't
 * user-authored.
 */
export function trainJobSnippet(
  config: Record<string, unknown>,
  apiUrl: string,
): string {
  const userArgs: Record<string, unknown> = {};
  const skip = new Set([
    "job_directory",
    "training_data_path",
    "dataset_hash",
    "llm_name",
  ]);
  for (const [k, v] of Object.entries(config)) {
    if (skip.has(k)) continue;
    userArgs[k] = v;
  }
  const model = typeof config.llm_name === "string" ? config.llm_name : "";
  const argsPy = formatPyDict(userArgs, 2);
  return [
    `import scalarlm`,
    `scalarlm.api_url = ${JSON.stringify(apiUrl)}`,
    ``,
    `llm = scalarlm.SupermassiveIntelligence()`,
    `llm.train(`,
    `    data=[{"input": "…", "output": "…"}, ...],    # your dataset`,
    model ? `    model_name=${JSON.stringify(model)},` : null,
    `    train_args=${argsPy},`,
    `)`,
  ]
    .filter((l) => l !== null)
    .join("\n");
}

export function generateSnippet(model: string, apiUrl: string): string {
  return [
    `import scalarlm`,
    `scalarlm.api_url = ${JSON.stringify(apiUrl)}`,
    ``,
    `llm = scalarlm.SupermassiveIntelligence()`,
    `results = llm.generate(`,
    `    prompts=["your prompt here"],`,
    `    model_name=${JSON.stringify(model)},`,
    `)`,
  ].join("\n");
}

/**
 * Pretty-print a JSON-serializable value as indented Python. Handles the
 * JSON/Python common subset (strings, numbers, booleans, null, arrays,
 * objects). Python-ifies true/false/None since our train_args stays in
 * that common subset.
 */
function formatPyDict(value: unknown, indent: number): string {
  const render = (v: unknown, depth: number): string => {
    const pad = "    ".repeat(depth);
    const inner = "    ".repeat(depth + 1);
    if (v === null || v === undefined) return "None";
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number") {
      if (!Number.isFinite(v)) return "float('nan')";
      return String(v);
    }
    if (typeof v === "string") return JSON.stringify(v);
    if (Array.isArray(v)) {
      if (v.length === 0) return "[]";
      const items = v.map((x) => `${inner}${render(x, depth + 1)}`);
      return `[\n${items.join(",\n")},\n${pad}]`;
    }
    if (typeof v === "object") {
      const entries = Object.entries(v as Record<string, unknown>);
      if (entries.length === 0) return "{}";
      const items = entries.map(
        ([k, val]) => `${inner}${JSON.stringify(k)}: ${render(val, depth + 1)}`,
      );
      return `{\n${items.join(",\n")},\n${pad}}`;
    }
    return JSON.stringify(v);
  };
  return render(value, indent);
}
