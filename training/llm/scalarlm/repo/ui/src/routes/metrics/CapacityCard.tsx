import { useGpuCount, useNodeCount } from "@/api/capacity";
import { useSqueue } from "@/api/squeue";
import { Card } from "@/components/Card";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { Stat } from "@/components/Stat";

/**
 * Capacity card today shows total GPUs/nodes from the dedicated megatron
 * endpoints plus a rough "nodes busy" count derived from running SLURM jobs.
 *
 * The design calls for inference-vs-training GPU gauges, but that requires
 * `/v1/vllm/stats` (KV cache utilization + GPU allocation) which the backend
 * does not yet expose. Per `docs/ui-design.md` §8.3 we gracefully degrade and
 * surface the missing dependency inline.
 */
const RUNNING_STATES = new Set(["R", "RUNNING", "COMPLETING"]);

export function CapacityCard() {
  const gpuCountQuery = useGpuCount();
  const nodeCountQuery = useNodeCount();
  const squeueQuery = useSqueue();

  const error = gpuCountQuery.error ?? nodeCountQuery.error ?? squeueQuery.error;
  const isPending =
    gpuCountQuery.isPending || nodeCountQuery.isPending || squeueQuery.isPending;

  const gpuCount = gpuCountQuery.data;
  const nodeCount = nodeCountQuery.data;
  const nodesBusy = squeueQuery.data?.jobs
    .filter((j) => RUNNING_STATES.has(j.state))
    .reduce((acc, j) => acc + (parseInt(j.nodes, 10) || 0), 0);

  const retryAll = () => {
    gpuCountQuery.refetch();
    nodeCountQuery.refetch();
    squeueQuery.refetch();
  };

  return (
    <Card
      title="Cluster capacity"
      subtitle="GPUs and MPI nodes available to SLURM"
    >
      {isPending && !gpuCount && !nodeCount ? (
        <div className="flex gap-8">
          <Skeleton className="h-10 w-24" />
          <Skeleton className="h-10 w-24" />
          <Skeleton className="h-10 w-24" />
        </div>
      ) : error ? (
        <ErrorState error={error} onRetry={retryAll} />
      ) : (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Stat
              label="Training GPUs"
              value={gpuCount ?? "—"}
              hint="Sum of Gres=gpu across slurmd-registered nodes"
            />
            <Stat
              label="MPI nodes"
              value={nodeCount ?? "—"}
              hint="One per running megatron pod"
            />
            <Stat
              label="Nodes busy"
              value={nodesBusy ?? "—"}
              hint="Sum of nodes allocated to running SLURM jobs"
            />
          </div>
          <div className="rounded-md border border-border-subtle bg-bg/50 p-3 text-xs leading-relaxed text-fg-muted">
            <span className="font-semibold text-fg-muted">
              Per-GPU inference/training breakdown not yet available.
            </span>{" "}
            Shows total capacity only. Populating the inference-GPU gauge needs
            a <code className="font-mono text-[11px]">/v1/vllm/stats</code>{" "}
            endpoint exposing KV-cache utilization (see{" "}
            <code className="font-mono text-[11px]">docs/ui-design.md</code> §13).
          </div>
        </div>
      )}
    </Card>
  );
}
