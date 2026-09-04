import { useState } from "react";

import { PageHeader } from "@/components/PageHeader";

import { CapacityCard } from "./CapacityCard";
import { HealthCard } from "./HealthCard";
import { QueueCard } from "./QueueCard";
import { ServiceLogsCard } from "./ServiceLogsCard";
import { ThroughputCard } from "./ThroughputCard";

/**
 * Sample capacity for the throughput sparkline. Samples are taken at the
 * throughput card's poll cadence (3s), so capacity × 3s = time span.
 */
const RANGE_OPTIONS = [
  { label: "Last 5 min", samples: 100 },
  { label: "Last 15 min", samples: 300 },
  { label: "Last 1 h", samples: 1200 },
] as const;

type RangeOption = (typeof RANGE_OPTIONS)[number];

export function MetricsPage() {
  const [range, setRange] = useState<RangeOption>(RANGE_OPTIONS[0]);

  return (
    <>
      <PageHeader
        title="Metrics"
        subtitle="Inference throughput, training queue, and cluster capacity"
        actions={<RangeSelector value={range} onChange={setRange} />}
      />
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6">
        <ThroughputCard sparklineCapacity={range.samples} />
        <QueueCard />
        <CapacityCard />
        <HealthCard />
        <ServiceLogsCard />
      </div>
    </>
  );
}

function RangeSelector({
  value,
  onChange,
}: {
  value: RangeOption;
  onChange: (next: RangeOption) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-fg-muted">
      Range
      <select
        value={value.label}
        onChange={(e) => {
          const next = RANGE_OPTIONS.find((o) => o.label === e.target.value);
          if (next) onChange(next);
        }}
        className="rounded-md border border-border-subtle bg-bg-card px-2 py-1 text-sm text-fg focus:border-border focus:outline-none"
      >
        {RANGE_OPTIONS.map((o) => (
          <option key={o.label} value={o.label}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
