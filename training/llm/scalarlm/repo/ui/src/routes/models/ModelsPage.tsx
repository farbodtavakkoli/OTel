import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { useModels, type OpenAIModel } from "@/api/chat";
import { getApiConfig } from "@/api/config";
import {
  useModelList,
  type ModelListEntry,
  type TrainingJobStatus,
} from "@/api/training";
import { Card } from "@/components/Card";
import { CopyButton } from "@/components/CopyButton";
import { ErrorState } from "@/components/ErrorState";
import { PageHeader } from "@/components/PageHeader";
import { ShowSDKCode, generateSnippet } from "@/components/ShowSDKCode";
import { Skeleton } from "@/components/Skeleton";
import { StatusBadge } from "@/components/StatusBadge";
import { formatDurationSeconds, formatNumber } from "@/lib/format";
import { setAlias } from "@/stores/aliases";
import { useAlias } from "@/stores/useAliases";

/**
 * Two data sources, joined:
 *   /v1/models                — authoritative "loaded in vLLM right now"
 *   /v1/megatron/list_models  — training-job-aware enrichment (step/loss/
 *                               status/deployed); may contain entries not yet
 *                               registered with vLLM (the 30 s refresh window).
 *
 * The page renders the deployment's base model as a standalone card, then
 * every other /v1/models entry as an "Adapter" enriched with job metadata
 * where available. Finally, training-job entries that are finished but not
 * yet in /v1/models appear in a "Pending registration" section so users
 * aren't left wondering where a just-completed job went.
 */
type HealthyAdapter = {
  id: string;
  created: number | undefined;
  owned_by: string | undefined;
  job: ModelListEntry | undefined;
  loaded: true;
};

type PendingAdapter = {
  id: string;
  created: number | undefined;
  owned_by: undefined;
  job: ModelListEntry;
  loaded: false;
};

type AdapterRow = HealthyAdapter | PendingAdapter;

type Filter = "all" | "loaded" | "pending";

export function ModelsPage() {
  const modelsQ = useModels();
  const listQ = useModelList();
  const { default_model } = getApiConfig();
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const externalUrl =
    getApiConfig().api_base === "/v1"
      ? window.location.origin
      : getApiConfig().api_base.replace(/\/v1\/?$/, "");

  const { base, adapters } = useMemo(() => {
    const vllmModels = modelsQ.data ?? [];
    const jobsByName = new Map<string, ModelListEntry>();
    for (const j of listQ.data ?? []) jobsByName.set(j.name, j);

    const baseEntry = vllmModels.find((m) => m.id === default_model);

    const rows: AdapterRow[] = [];

    // Live in vLLM, minus the base model.
    for (const m of vllmModels) {
      if (m.id === default_model) continue;
      rows.push({
        id: m.id,
        created: m.created,
        owned_by: typeof m.owned_by === "string" ? m.owned_by : undefined,
        job: jobsByName.get(m.id),
        loaded: true,
      });
    }

    // Training jobs present in list_models but not (yet) loaded in vLLM.
    // These are "in the 30 s refresh gap" or still running / failed.
    const loadedIds = new Set(vllmModels.map((m) => m.id));
    for (const j of listQ.data ?? []) {
      if (loadedIds.has(j.name)) continue;
      if (j.name === default_model) continue;
      rows.push({
        id: j.name,
        created: j.start_time || undefined,
        owned_by: undefined,
        job: j,
        loaded: false,
      });
    }

    // Newest first; loaded above pending within the same bucket.
    rows.sort((a, b) => {
      if (a.loaded !== b.loaded) return a.loaded ? -1 : 1;
      return (b.created ?? 0) - (a.created ?? 0);
    });

    return { base: baseEntry, adapters: rows };
  }, [modelsQ.data, listQ.data, default_model]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return adapters.filter((a) => {
      if (filter === "loaded" && !a.loaded) return false;
      if (filter === "pending" && a.loaded) return false;
      if (q && !a.id.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [adapters, filter, search]);

  const isPending = modelsQ.isPending && !modelsQ.data;
  const error = modelsQ.error;

  const counts = useMemo(
    () => ({
      all: adapters.length,
      loaded: adapters.filter((a) => a.loaded).length,
      pending: adapters.filter((a) => !a.loaded).length,
    }),
    [adapters],
  );

  return (
    <>
      <PageHeader
        title="Models"
        subtitle="Base model and registered adapters"
      />
      <div className="mx-auto flex max-w-5xl flex-col gap-5 px-6 py-6">
        {isPending ? (
          <Skeleton className="h-40" />
        ) : error ? (
          <ErrorState error={error} onRetry={modelsQ.refetch} />
        ) : (
          <BaseModelCard
            id={default_model}
            base={base}
            externalUrl={externalUrl}
          />
        )}

        <section className="flex flex-col gap-3">
          <header className="flex flex-wrap items-center gap-3">
            <h2 className="text-sm font-semibold tracking-tight">Adapters</h2>
            <span className="font-mono text-xs text-fg-subtle">
              {adapters.length} total
            </span>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              <input
                type="search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by hash or alias"
                className="w-56 rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
              />
              <div className="flex gap-1">
                {(
                  [
                    { v: "all", label: "All" },
                    { v: "loaded", label: "Loaded" },
                    { v: "pending", label: "Pending" },
                  ] as { v: Filter; label: string }[]
                ).map((f) => (
                  <button
                    key={f.v}
                    type="button"
                    onClick={() => setFilter(f.v)}
                    className={
                      filter === f.v
                        ? "flex items-center gap-1.5 rounded-md bg-bg-hover px-2 py-1 text-xs text-fg ring-1 ring-border"
                        : "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
                    }
                  >
                    {f.label}
                    <span className="font-mono text-[10px] text-fg-subtle">
                      {counts[f.v]}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </header>

          {isPending ? (
            <div className="grid grid-cols-1 gap-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-16" />
              ))}
            </div>
          ) : adapters.length === 0 ? (
            <EmptyAdapters />
          ) : filtered.length === 0 ? (
            <div className="rounded-md border border-border-subtle bg-bg/50 p-4 text-center text-sm text-fg-muted">
              No adapters match the current filter.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {filtered.map((a) => (
                <AdapterRowCard
                  key={a.id}
                  adapter={a}
                  externalUrl={externalUrl}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------

function BaseModelCard({
  id,
  base,
  externalUrl,
}: {
  id: string;
  base: OpenAIModel | undefined;
  externalUrl: string;
}) {
  const hfUrl = guessHuggingFaceUrl(id);
  return (
    <Card title="Base model" subtitle={base ? "Loaded in vLLM" : "Not reachable"}>
      <div className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <code className="rounded-md border border-border-subtle bg-bg px-2 py-1 font-mono text-sm text-fg">
            {id || "(unset)"}
          </code>
          {id && <CopyButton value={id} label="Copy" />}
          {hfUrl && (
            <a
              href={hfUrl}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-border-subtle bg-bg-card px-2 py-1 text-xs text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
            >
              Hugging Face ↗
            </a>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Link
            to={`/chat?model=${encodeURIComponent(id)}`}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            Open in Chat
          </Link>
          <Link
            to="/train"
            className="rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
          >
            Train an adapter on top
          </Link>
        </div>

        {id && (
          <ShowSDKCode
            title="Generate via SDK"
            code={generateSnippet(id, externalUrl)}
          />
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------

function AdapterRowCard({
  adapter,
  externalUrl,
}: {
  adapter: AdapterRow;
  externalUrl: string;
}) {
  const alias = useAlias(adapter.id);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  const job = adapter.job;
  const status: TrainingJobStatus = job?.status ?? (adapter.loaded ? "COMPLETED" : "UNKNOWN");
  const progressPct =
    job && job.max_steps > 0 ? Math.min(1, job.step / job.max_steps) : null;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border-subtle bg-bg-card p-4 transition-colors hover:border-border">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {editing ? (
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={() => {
                  setAlias(adapter.id, draft);
                  setEditing(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    setAlias(adapter.id, draft);
                    setEditing(false);
                  } else if (e.key === "Escape") {
                    setEditing(false);
                  }
                }}
                placeholder="Add a nickname…"
                className="min-w-[10rem] rounded-md border border-border bg-bg px-2 py-0.5 text-sm font-semibold text-fg focus:border-accent focus:outline-none"
              />
            ) : (
              <button
                type="button"
                onClick={() => {
                  setDraft(alias ?? "");
                  setEditing(true);
                }}
                className={
                  alias
                    ? "rounded-md px-1 -mx-1 text-left text-sm font-semibold text-fg transition-colors hover:bg-bg-hover"
                    : "rounded-md px-1 -mx-1 text-left font-mono text-sm text-fg transition-colors hover:bg-bg-hover"
                }
                title={alias ? `${alias} · click to rename` : "Click to add a nickname"}
              >
                {alias ??
                  (adapter.id.length > 16 ? `${adapter.id.slice(0, 12)}…` : adapter.id)}
              </button>
            )}
            <StatusBadge status={status} />
            {!adapter.loaded && (
              <span
                className="rounded-md border border-border-subtle bg-bg px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-fg-muted"
                title="Not yet registered with vLLM"
              >
                not loaded
              </span>
            )}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[11px] text-fg-subtle">
            <span title={adapter.id}>{adapter.id}</span>
            <CopyButton value={adapter.id} label="Copy id" size="sm" />
            {adapter.created && (
              <span>· {new Date(adapter.created * 1000).toLocaleString()}</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {adapter.loaded ? (
            <Link
              to={`/chat?model=${encodeURIComponent(adapter.id)}`}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
            >
              Open in Chat
            </Link>
          ) : (
            <span
              title="Adapter isn't registered with vLLM yet. Auto-register runs every 30s."
              className="cursor-not-allowed rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg-subtle"
            >
              Open in Chat
            </span>
          )}
          {job && (
            <Link
              to={`/train/${adapter.id}`}
              className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
            >
              View job
            </Link>
          )}
        </div>
      </div>

      {job && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
          <Metric
            label="Step"
            value={
              job.max_steps
                ? `${job.step} / ${job.max_steps}`
                : formatNumber(job.step)
            }
          />
          <Metric label="EMA loss" value={formatNumber(job.loss)} />
          <Metric
            label="Train time"
            value={formatDurationSeconds(job.train_time)}
          />
          <Metric
            label="Owned by"
            value={adapter.owned_by ?? "—"}
            mono={Boolean(adapter.owned_by)}
          />
        </div>
      )}

      {progressPct !== null && (
        <div className="h-1 w-full overflow-hidden rounded-full bg-bg">
          <div
            className={
              status === "FAILED"
                ? "h-full rounded-full bg-danger"
                : "h-full rounded-full bg-accent"
            }
            style={{ width: `${progressPct * 100}%` }}
          />
        </div>
      )}

      {adapter.loaded && (
        <ShowSDKCode
          title="Generate via SDK"
          code={generateSnippet(adapter.id, externalUrl)}
        />
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <div className="text-fg-subtle">{label}</div>
      <div className={mono ? "font-mono text-fg" : "text-fg"}>{value}</div>
    </div>
  );
}

function EmptyAdapters() {
  return (
    <div className="flex flex-col items-start gap-3 rounded-md border border-border-subtle bg-bg-card p-6">
      <p className="text-sm text-fg-muted">
        No post-trained adapters yet. Every completed training job becomes
        available here once it's registered with vLLM.
      </p>
      <Link
        to="/train"
        className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
      >
        Submit a training job →
      </Link>
    </div>
  );
}

// ---------------------------------------------------------------------------

/**
 * Best-effort HuggingFace URL for a model id. HF ids are `{org}/{repo}` where
 * both halves are non-empty alphanumerics (plus `-`, `_`, `.`). Training-job
 * hashes are a single SHA-256 blob with no slash, so they're excluded. The
 * "tiny-random/…" fixtures are still real HF repos.
 */
function guessHuggingFaceUrl(id: string): string | null {
  if (!id || !id.includes("/")) return null;
  const parts = id.split("/");
  if (parts.length !== 2) return null;
  if (!parts[0] || !parts[1]) return null;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(parts[0])) return null;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(parts[1])) return null;
  return `https://huggingface.co/${id}`;
}
