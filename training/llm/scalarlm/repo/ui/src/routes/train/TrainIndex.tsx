import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { getApiConfig } from "@/api/config";
import { useGpuCount, useNodeCount } from "@/api/capacity";
import { useSqueue } from "@/api/squeue";
import {
  useModelList,
  type ModelListEntry,
  type TrainingJobStatus,
} from "@/api/training";
import { CopyButton } from "@/components/CopyButton";
import { ErrorState } from "@/components/ErrorState";
import { PageHeader } from "@/components/PageHeader";
import { Skeleton } from "@/components/Skeleton";
import { StatusBadge } from "@/components/StatusBadge";
import { formatDurationSeconds, formatNumber } from "@/lib/format";
import { useAlias } from "@/stores/useAliases";

import { SubmitModal } from "./SubmitModal";

const FILTERS: {
  value: TrainingJobStatus | "ALL";
  label: string;
}[] = [
  { value: "ALL", label: "All" },
  { value: "QUEUED", label: "Queued" },
  { value: "TRAINING", label: "Training" },
  { value: "COMPLETED", label: "Completed" },
  { value: "FAILED", label: "Failed" },
];

const RUNNING_SLURM_STATES = new Set(["R", "RUNNING", "COMPLETING"]);
const PENDING_SLURM_STATES = new Set([
  "PD",
  "PENDING",
  "CONFIGURING",
  "REQUEUED",
]);

export function TrainIndex() {
  const { data, error, refetch, isPending } = useModelList();
  const [statusFilter, setStatusFilter] = useState<TrainingJobStatus | "ALL">(
    "ALL",
  );
  const [search, setSearch] = useState("");
  const [submitOpen, setSubmitOpen] = useState(false);

  const counts = useMemo(() => {
    const all = data ?? [];
    return {
      ALL: all.length,
      QUEUED: all.filter((m) => m.status === "QUEUED").length,
      TRAINING: all.filter((m) => m.status === "TRAINING").length,
      COMPLETED: all.filter((m) => m.status === "COMPLETED").length,
      FAILED: all.filter((m) => m.status === "FAILED").length,
    } as Record<TrainingJobStatus | "ALL", number>;
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = search.trim().toLowerCase();
    return data.filter((m) => {
      if (statusFilter !== "ALL" && m.status !== statusFilter) return false;
      if (q && !m.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, statusFilter, search]);

  const hasAnyJobs = (data?.length ?? 0) > 0;

  return (
    <>
      <PageHeader
        title="Train"
        subtitle="Submitted training jobs"
        actions={
          <button
            type="button"
            onClick={() => setSubmitOpen(true)}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            + New training job
          </button>
        }
      />
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6">
        <ClusterStrip />

        {hasAnyJobs && (
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by hash or name"
              className="w-60 rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
            />
            <div className="flex flex-wrap gap-1">
              {FILTERS.map((f) => (
                <button
                  key={f.value}
                  type="button"
                  onClick={() => setStatusFilter(f.value)}
                  className={
                    statusFilter === f.value
                      ? "flex items-center gap-1.5 rounded-md bg-bg-hover px-2 py-1 text-xs text-fg ring-1 ring-border"
                      : "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
                  }
                >
                  <span>{f.label}</span>
                  <span className="font-mono text-[10px] text-fg-subtle">
                    {counts[f.value]}
                  </span>
                </button>
              ))}
            </div>
            <span className="ml-auto font-mono text-xs text-fg-subtle">
              {filtered.length} shown
            </span>
          </div>
        )}

        {isPending && !data ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-28" />
            ))}
          </div>
        ) : error ? (
          <ErrorState error={error} onRetry={refetch} />
        ) : filtered.length === 0 ? (
          <EmptyState hasAny={hasAnyJobs} />
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {filtered.map((m) => (
              <JobCard key={m.name} model={m} />
            ))}
          </div>
        )}
      </div>
      <SubmitModal open={submitOpen} onClose={() => setSubmitOpen(false)} />
    </>
  );
}

/**
 * Always-on context strip showing what the user is submitting against.
 * Pulls from the same hooks as the Metrics page; on a fresh deployment it
 * tells you "0 GPUs / 0 nodes" loudly enough to investigate before submitting.
 */
function ClusterStrip() {
  const gpuCount = useGpuCount();
  const nodeCount = useNodeCount();
  const squeue = useSqueue();

  const running =
    squeue.data?.jobs.filter((j) => RUNNING_SLURM_STATES.has(j.state)).length ??
    0;
  const queued =
    squeue.data?.jobs.filter((j) => PENDING_SLURM_STATES.has(j.state)).length ??
    0;

  return (
    <section className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-border-subtle bg-bg-card px-4 py-3 text-sm">
      <ClusterStat
        label="GPUs"
        value={gpuCount.data}
        loading={gpuCount.isPending}
      />
      <ClusterStat
        label="Nodes"
        value={nodeCount.data}
        loading={nodeCount.isPending}
      />
      <ClusterStat label="Running" value={running} loading={squeue.isPending} />
      <ClusterStat label="Queued" value={queued} loading={squeue.isPending} />
      <Link
        to="/metrics"
        className="ml-auto text-xs text-fg-muted underline-offset-2 hover:text-fg hover:underline"
      >
        View full metrics →
      </Link>
    </section>
  );
}

function ClusterStat({
  label,
  value,
  loading,
}: {
  label: string;
  value: number | undefined;
  loading: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-xs uppercase tracking-wider text-fg-subtle">
        {label}
      </span>
      <span className="font-mono tabular-nums text-fg">
        {loading && value === undefined ? "—" : value ?? "—"}
      </span>
    </div>
  );
}

function JobCard({ model }: { model: ModelListEntry }) {
  const pct =
    model.max_steps > 0 ? Math.min(1, model.step / model.max_steps) : 0;
  const alias = useAlias(model.name);

  return (
    <Link
      to={`/train/${model.name}`}
      className="group flex flex-col gap-3 rounded-lg border border-border-subtle bg-bg-card p-4 transition-colors hover:border-border hover:bg-bg-hover"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div
            className={`truncate text-sm text-fg group-hover:text-accent ${
              alias ? "" : "font-mono"
            }`}
            title={alias ? `${alias} · ${model.name}` : model.name}
          >
            {alias ??
              (model.name.length > 12
                ? `${model.name.slice(0, 12)}…`
                : model.name)}
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-xs text-fg-subtle">
            <span>
              {model.start_time
                ? new Date(model.start_time * 1000).toLocaleString()
                : "—"}
            </span>
            {alias && (
              <span className="font-mono">{model.name.slice(0, 8)}</span>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {model.deployed && (
            <span
              title="Registered with vLLM"
              className="rounded-md bg-success/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-success ring-1 ring-inset ring-success/30"
            >
              deployed
            </span>
          )}
          <StatusBadge status={model.status} />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <div className="text-fg-subtle">Step</div>
          <div className="font-mono text-fg">
            {model.step}
            <span className="text-fg-subtle">
              {model.max_steps ? ` / ${model.max_steps}` : ""}
            </span>
          </div>
        </div>
        <div>
          <div className="text-fg-subtle">EMA loss</div>
          <div className="font-mono text-fg">{formatNumber(model.loss)}</div>
        </div>
        <div>
          <div className="text-fg-subtle">Train time</div>
          <div className="font-mono text-fg">
            {formatDurationSeconds(model.train_time)}
          </div>
        </div>
      </div>

      {model.max_steps > 0 && (
        <div className="h-1 w-full overflow-hidden rounded-full bg-bg">
          <div
            className={`h-full rounded-full ${
              model.status === "FAILED" ? "bg-danger" : "bg-accent"
            }`}
            style={{ width: `${pct * 100}%` }}
          />
        </div>
      )}
    </Link>
  );
}

function EmptyState({ hasAny }: { hasAny: boolean }) {
  if (hasAny) {
    return (
      <div className="rounded-md border border-border-subtle bg-bg/50 p-6 text-center text-sm text-fg-muted">
        No jobs match the current filter.
      </div>
    );
  }
  // First-run state: explanation + copy-paste SDK snippet, side-by-side on
  // wide screens so the wasted vertical space gets put to work.
  const externalUrl = computeExternalUrl();
  const snippet = sdkSnippet(externalUrl);
  return (
    <section className="grid grid-cols-1 gap-4 lg:grid-cols-5">
      <div className="lg:col-span-2 flex flex-col gap-3 rounded-lg border border-border-subtle bg-bg-card p-5">
        <h2 className="text-lg font-semibold text-fg">
          No training jobs yet
        </h2>
        <p className="text-sm leading-relaxed text-fg-muted">
          Submit a job from the form (top right), or drive the same flow from
          the Python SDK on the right. Jobs become available to the chat surface
          automatically once they finish — see{" "}
          <Link to="/chat" className="text-accent hover:underline">
            Chat
          </Link>{" "}
          and{" "}
          <Link to="/models" className="text-accent hover:underline">
            Models
          </Link>{" "}
          to try them.
        </p>
        <ul className="mt-1 list-disc pl-5 text-xs text-fg-subtle">
          <li>One JSON object per line in your dataset.</li>
          <li>
            Identical inputs are deduplicated by content hash — safe to retry.
          </li>
          <li>
            <code className="rounded bg-bg px-1 py-0.5 font-mono text-[11px] text-fg">
              max_steps
            </code>
            ,{" "}
            <code className="rounded bg-bg px-1 py-0.5 font-mono text-[11px] text-fg">
              learning_rate
            </code>
            , adapter type, GPU/node count are all editable on submit.
          </li>
        </ul>
      </div>
      <div className="lg:col-span-3 flex flex-col gap-2 rounded-lg border border-border-subtle bg-bg-card p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs uppercase tracking-wider text-fg-subtle">
            Python SDK
          </span>
          <CopyButton value={snippet} label="Copy" />
        </div>
        <pre className="overflow-x-auto rounded-md border border-border-subtle bg-bg p-3 font-mono text-xs leading-relaxed text-fg">
          {snippet}
        </pre>
      </div>
    </section>
  );
}

function computeExternalUrl(): string {
  const cfg = getApiConfig();
  if (cfg.api_base === "/v1") return window.location.origin;
  return cfg.api_base.replace(/\/v1\/?$/, "");
}

function sdkSnippet(apiUrl: string): string {
  return `import scalarlm
scalarlm.api_url = ${JSON.stringify(apiUrl)}

llm = scalarlm.SupermassiveIntelligence()
llm.train(
    data=[{"input": "What is 2+2?", "output": "4"}],
    train_args={"max_steps": 100, "learning_rate": 3e-4},
)`;
}
