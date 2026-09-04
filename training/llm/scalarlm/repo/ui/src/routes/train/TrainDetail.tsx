import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import {
  useCancelJob,
  useDeleteJob,
  useRestartJob,
  useTrainingJob,
} from "@/api/training";
import { getApiConfig } from "@/api/config";
import { Card } from "@/components/Card";
import { ConfirmDestructive } from "@/components/ConfirmDestructive";
import { CopyButton } from "@/components/CopyButton";
import { ErrorState } from "@/components/ErrorState";
import { LossChart } from "@/components/LossChart";
import { PageHeader } from "@/components/PageHeader";
import { ShowSDKCode, trainJobSnippet } from "@/components/ShowSDKCode";
import { Skeleton } from "@/components/Skeleton";
import { Stat } from "@/components/Stat";
import { StatusBadge } from "@/components/StatusBadge";
import { formatDurationSeconds, formatNumber } from "@/lib/format";
import { setAlias } from "@/stores/aliases";
import { useAlias } from "@/stores/useAliases";

import { ConfigPanel } from "./ConfigPanel";
import { DatasetPanel } from "./DatasetPanel";
import { LogPane } from "./LogPane";
import { PublishToHFModal } from "./PublishToHFModal";

export function TrainDetail() {
  const { jobHash = "" } = useParams<{ jobHash: string }>();
  const navigate = useNavigate();
  const { data, error, refetch, isPending } = useTrainingJob(jobHash);
  const cancel = useCancelJob();
  const del = useDeleteJob();
  const restartJob = useRestartJob();
  const [askCancel, setAskCancel] = useState(false);
  const [askDelete, setAskDelete] = useState(false);
  const [publishOpen, setPublishOpen] = useState(false);
  const alias = useAlias(jobHash);
  const [editingAlias, setEditingAlias] = useState(false);
  const [aliasDraft, setAliasDraft] = useState("");

  const status = data?.job_status.status ?? "UNKNOWN";
  const history = data?.job_status.history ?? [];
  const maxSteps = data?.job_status.max_steps ?? data?.job_config.max_steps;
  const lastStep = history.length > 0 ? history[history.length - 1].step : 0;
  const lastLoss =
    history.length > 0 ? history[history.length - 1].loss : undefined;
  const elapsed =
    history.length > 0 ? history[history.length - 1].time : undefined;
  const prefix = jobHash.slice(0, 8);

  const cancellable = status === "QUEUED" || status === "TRAINING";
  const deletable = status !== "TRAINING";
  const restartable = status === "FAILED" || status === "CANCELLED";

  const onConfirmCancel = async () => {
    try {
      await cancel.mutateAsync(jobHash);
    } finally {
      setAskCancel(false);
    }
  };
  const onConfirmDelete = async () => {
    try {
      await del.mutateAsync(jobHash);
      setAskDelete(false);
      navigate("/train", { replace: true });
    } catch {
      setAskDelete(false);
    }
  };

  return (
    <>
      <PageHeader
        title={
          editingAlias ? (
            <input
              autoFocus
              value={aliasDraft}
              onChange={(e) => setAliasDraft(e.target.value)}
              onBlur={() => {
                setAlias(jobHash, aliasDraft);
                setEditingAlias(false);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setAlias(jobHash, aliasDraft);
                  setEditingAlias(false);
                } else if (e.key === "Escape") {
                  setEditingAlias(false);
                }
              }}
              placeholder="Add a nickname…"
              className="rounded-md border border-border bg-bg px-2 py-0.5 text-lg font-semibold text-fg focus:border-accent focus:outline-none"
            />
          ) : (
            <button
              type="button"
              onClick={() => {
                setAliasDraft(alias ?? "");
                setEditingAlias(true);
              }}
              className={`rounded-md px-1 -mx-1 text-left transition-colors hover:bg-bg-card ${
                alias ? "" : "font-mono"
              }`}
              title={
                alias
                  ? `${alias} · click to rename (local only)`
                  : "Click to add a nickname"
              }
            >
              {alias ?? (prefix || "(no hash)")}
            </button>
          )
        }
        subtitle={
          <span className="flex items-center gap-2 font-mono text-xs">
            <span>{jobHash}</span>
            {jobHash && (
              <CopyButton
                value={jobHash}
                label="Copy hash"
                size="sm"
              />
            )}
          </span>
        }
        actions={
          <div className="flex items-center gap-2">
            {data && <StatusBadge status={status} />}
            {data?.deployed && (
              <span className="rounded-md bg-success/15 px-2 py-1 text-[10px] uppercase tracking-wider text-success ring-1 ring-inset ring-success/30">
                deployed
              </span>
            )}
            <Link
              to="/train"
              className="rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg-muted hover:border-border hover:bg-bg-hover"
            >
              Back
            </Link>
          </div>
        }
      />
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6">
        {isPending && !data ? (
          <div className="flex flex-col gap-4">
            <Skeleton className="h-28" />
            <Skeleton className="h-48" />
            <Skeleton className="h-64" />
          </div>
        ) : error ? (
          <ErrorState error={error} onRetry={refetch} />
        ) : data ? (
          <>
            <Card title="Overview">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat
                  label="Step"
                  value={
                    maxSteps
                      ? `${lastStep} / ${maxSteps}`
                      : formatNumber(lastStep)
                  }
                />
                <Stat
                  label="Last loss"
                  value={lastLoss !== undefined ? formatNumber(lastLoss) : "—"}
                />
                <Stat
                  label="Elapsed"
                  value={
                    elapsed !== undefined
                      ? formatDurationSeconds(elapsed)
                      : "—"
                  }
                />
                <Stat
                  label="Adapter"
                  value={
                    <span className="font-mono text-sm">
                      {String(data.job_config.adapter_type ?? "—")}
                    </span>
                  }
                />
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                {data.deployed ? (
                  <Link
                    to={`/chat?model=${encodeURIComponent(jobHash)}`}
                    className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
                  >
                    Open in Chat
                  </Link>
                ) : (
                  <span
                    title="Not yet registered with vLLM"
                    className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg-subtle"
                  >
                    Open in Chat
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setPublishOpen(true)}
                  disabled={status !== "COMPLETED"}
                  title={
                    status === "COMPLETED"
                      ? "Publish this checkpoint to a HuggingFace repo"
                      : `Publish is enabled once the job is COMPLETED (current: ${status}).`
                  }
                  className="rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Publish to HF
                </button>
                <button
                  type="button"
                  onClick={() => setAskCancel(true)}
                  disabled={!cancellable || cancel.isPending}
                  className="rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Cancel
                </button>
                {restartable && (
                  <button
                    type="button"
                    onClick={() => restartJob.mutate(jobHash)}
                    disabled={restartJob.isPending}
                    title="Re-queue this job; training resumes from the last checkpoint"
                    className="rounded-md border border-border-subtle bg-bg-card px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {restartJob.isPending ? "Restarting…" : "Restart"}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setAskDelete(true)}
                  disabled={!deletable || del.isPending}
                  className="rounded-md border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger hover:border-danger hover:bg-danger/20 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Delete
                </button>
                {restartJob.isError && (
                  <span
                    className="ml-auto rounded-md border border-danger/40 bg-danger/5 px-2 py-1 font-mono text-xs text-danger"
                    title={String(restartJob.error)}
                  >
                    Restart failed: {String(restartJob.error).slice(0, 80)}
                  </span>
                )}
                {data.job_status.error && (
                  <span className="ml-auto rounded-md border border-danger/40 bg-danger/5 px-2 py-1 font-mono text-xs text-danger">
                    {String(data.job_status.error).slice(0, 80)}
                  </span>
                )}
              </div>
            </Card>

            <Card title="Loss">
              <LossChart
                history={history}
                jobHash={jobHash}
                maxSteps={maxSteps}
              />
            </Card>

            <DatasetPanel jobHash={jobHash} />

            <LogPane jobHash={jobHash} status={status} />

            <ConfigPanel
              config={data.job_config as Record<string, unknown>}
            />

            <ShowSDKCode
              code={trainJobSnippet(
                data.job_config as Record<string, unknown>,
                getApiConfig().api_base === "/v1"
                  ? window.location.origin
                  : getApiConfig().api_base.replace(/\/v1$/, ""),
              )}
              title="Run this job from the SDK"
            />
          </>
        ) : null}
      </div>

      <PublishToHFModal
        open={publishOpen}
        jobHash={jobHash}
        onClose={() => setPublishOpen(false)}
      />

      <ConfirmDestructive
        open={askCancel}
        title="Cancel training job"
        description={`This calls scancel on SLURM job ${prefix} and transitions the job to a non-running state.`}
        confirmationText={prefix}
        actionLabel="Cancel job"
        busy={cancel.isPending}
        onConfirm={onConfirmCancel}
        onClose={() => setAskCancel(false)}
      />
      <ConfirmDestructive
        open={askDelete}
        title="Delete training job"
        description={`Removes the job directory ${prefix}. Checkpoints and logs will be lost. This does not unregister an already-loaded adapter from vLLM.`}
        confirmationText={prefix}
        actionLabel="Delete job"
        busy={del.isPending}
        onConfirm={onConfirmDelete}
        onClose={() => setAskDelete(false)}
      />
    </>
  );
}
