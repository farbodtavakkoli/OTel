import { useState } from "react";

import { useClearQueue, useGenerateMetrics } from "@/api/metrics";
import { Card } from "@/components/Card";
import { ConfirmDestructive } from "@/components/ConfirmDestructive";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/Skeleton";
import { Sparkline } from "@/components/Sparkline";
import { Stat } from "@/components/Stat";
import { formatNumber } from "@/lib/format";
import { useRollingBuffer } from "@/lib/useRollingBuffer";

interface ThroughputCardProps {
  /** Max samples retained for the sparkline. Governed by the page's time-range control. */
  sparklineCapacity: number;
}

const UTILIZATION_TOOLTIP =
  "Throughput while the queue was non-empty. Not wall-clock throughput. " +
  "See docs/inference-queue.md §6.2.";

export function ThroughputCard({ sparklineCapacity }: ThroughputCardProps) {
  const { data, error, refetch, isPending } = useGenerateMetrics();
  const tokenHistory = useRollingBuffer(data?.["token/s"] ?? null, sparklineCapacity);
  const clearQueue = useClearQueue();
  const [askClear, setAskClear] = useState(false);

  return (
    <Card
      title="Inference throughput"
      subtitle="GET /v1/generate/metrics"
      action={
        <button
          type="button"
          onClick={() => setAskClear(true)}
          disabled={clearQueue.isPending || !data || data.queue_depth === 0}
          className="rounded-md border border-danger/40 bg-danger/10 px-3 py-1 text-xs text-danger hover:border-danger hover:bg-danger/20 disabled:cursor-not-allowed disabled:opacity-40"
          title="Drop every pending request from the inference queue"
        >
          Clear queue
        </button>
      }
    >
      {isPending && !data ? (
        <ThroughputSkeleton />
      ) : error ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : data ? (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat
              label="Token/s"
              value={formatNumber(data["token/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Request/s"
              value={formatNumber(data["request/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Flop/s"
              value={formatNumber(data["flop/s"])}
              hint={UTILIZATION_TOOLTIP}
            />
            <Stat
              label="Queue depth"
              value={formatNumber(data.queue_depth)}
              hint="Outstanding inference requests"
            />
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between text-xs text-fg-muted">
              <span>token/s history</span>
              <span className="font-mono">{tokenHistory.length} samples</span>
            </div>
            <div className="w-full overflow-hidden rounded-md border border-border-subtle bg-bg/50 p-2">
              <Sparkline
                values={tokenHistory}
                width={640}
                height={56}
                label="token/s over time"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 border-t border-border-subtle pt-4 sm:grid-cols-3">
            <Stat label="Requests completed" value={formatNumber(data.requests)} />
            <Stat label="Tokens completed" value={formatNumber(data.tokens)} />
            <Stat
              label="Effective seconds"
              value={formatNumber(data.total_time)}
              hint="Accumulated time the queue was non-empty"
            />
          </div>
          <p className="text-xs leading-relaxed text-fg-subtle">
            <span className="font-semibold text-fg-muted">Note.</span>{" "}
            {UTILIZATION_TOOLTIP}
          </p>
        </div>
      ) : null}

      <ConfirmDestructive
        open={askClear}
        title="Clear inference queue"
        description="Drops every pending and unacked request from the inference work queue. In-flight requests already handed to vLLM continue to run but their results are discarded. New submissions after this call proceed normally."
        confirmationText="clear"
        actionLabel="Clear queue"
        busy={clearQueue.isPending}
        onConfirm={async () => {
          try {
            await clearQueue.mutateAsync();
          } finally {
            setAskClear(false);
          }
        }}
        onClose={() => setAskClear(false)}
      />
    </Card>
  );
}

function ThroughputSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="flex flex-col gap-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-7 w-20" />
          </div>
        ))}
      </div>
      <Skeleton className="h-16 w-full" />
    </div>
  );
}
