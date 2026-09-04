import { Link } from "react-router-dom";
import clsx from "clsx";

import { useSqueue, type SqueueJob } from "@/api/squeue";
import { Badge, type BadgeTone } from "@/components/Badge";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { Stat } from "@/components/Stat";

const RUNNING_STATES = new Set(["R", "RUNNING", "COMPLETING"]);
const PENDING_STATES = new Set(["PD", "PENDING", "CONFIGURING", "REQUEUED"]);

function stateTone(state: string): BadgeTone {
  if (RUNNING_STATES.has(state)) return "accent";
  if (PENDING_STATES.has(state)) return "neutral";
  if (state === "COMPLETED" || state === "CD") return "success";
  if (state === "FAILED" || state === "F" || state === "NODE_FAIL") return "danger";
  return "warning";
}

export function QueueCard() {
  const { data, error, refetch, isPending } = useSqueue();

  const running = data?.jobs.filter((j) => RUNNING_STATES.has(j.state)).length ?? 0;
  const queued = data?.jobs.filter((j) => PENDING_STATES.has(j.state)).length ?? 0;

  return (
    <Card title="Training queue" subtitle="GET /v1/megatron/squeue">
      {isPending && !data ? (
        <div className="flex flex-col gap-4">
          <div className="flex gap-6">
            <Skeleton className="h-10 w-20" />
            <Skeleton className="h-10 w-20" />
          </div>
          <Skeleton className="h-32 w-full" />
        </div>
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : data ? (
        <div className="flex flex-col gap-4">
          <div className="flex gap-8">
            <Stat label="Running" value={running} />
            <Stat label="Queued" value={queued} />
            <Stat label="Total" value={data.jobs.length} />
          </div>
          {data.errored && data.errorMessage ? (
            <div className="rounded-md border border-warning/30 bg-warning/10 p-3 text-xs text-warning">
              squeue returned: {data.errorMessage}
            </div>
          ) : null}
          {data.jobs.length === 0 ? (
            <div className="rounded-md border border-border-subtle bg-bg/50 p-4 text-sm text-fg-muted">
              No jobs in the SLURM queue.
            </div>
          ) : (
            <QueueTable jobs={data.jobs} />
          )}
        </div>
      ) : null}
    </Card>
  );
}

function QueueTable({ jobs }: { jobs: SqueueJob[] }) {
  return (
    <div className="overflow-x-auto rounded-md border border-border-subtle">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-bg/50 text-xs uppercase tracking-wider text-fg-subtle">
          <tr>
            <Th>Job ID</Th>
            <Th>Name</Th>
            <Th>Partition</Th>
            <Th>User</Th>
            <Th>State</Th>
            <Th align="right">Time</Th>
            <Th align="right">Limit</Th>
            <Th align="right">Nodes</Th>
            <Th>Nodelist / reason</Th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr
              key={`${job.jobId}-${job.name}`}
              className="border-t border-border-subtle transition-colors hover:bg-bg-hover"
            >
              <Td className="font-mono text-xs">{job.jobId}</Td>
              <Td>
                {/* Training jobs set job-name = basename(job_directory) = hash. */}
                <Link
                  to={`/train/${job.name}`}
                  className="font-mono text-xs text-accent hover:underline"
                >
                  {job.name}
                </Link>
              </Td>
              <Td className="text-fg-muted">{job.partition}</Td>
              <Td className="text-fg-muted">{job.user}</Td>
              <Td>
                <Badge tone={stateTone(job.state)}>{job.state}</Badge>
              </Td>
              <Td align="right" className="font-mono tabular-nums text-xs">
                {job.time}
              </Td>
              <Td align="right" className="font-mono tabular-nums text-xs text-fg-muted">
                {job.timeLimit}
              </Td>
              <Td align="right" className="font-mono tabular-nums">
                {job.nodes}
              </Td>
              <Td className="max-w-[24ch] truncate text-fg-muted" title={job.nodelistReason}>
                {job.nodelistReason}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={clsx(
        "px-3 py-2 font-medium",
        align === "right" ? "text-right" : "text-left",
      )}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  className,
  title,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
  title?: string;
}) {
  return (
    <td
      className={clsx(
        "px-3 py-2",
        align === "right" ? "text-right" : "text-left",
        className,
      )}
      title={title}
    >
      {children}
    </td>
  );
}
