import type { ReactNode } from "react";

interface StatProps {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string;
}

export function Stat({ label, value, unit, hint }: StatProps) {
  return (
    <div className="flex flex-col gap-1" title={hint}>
      <div className="text-xs uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="font-mono text-2xl font-semibold tabular-nums text-fg">
          {value}
        </span>
        {unit && <span className="text-xs text-fg-muted">{unit}</span>}
      </div>
    </div>
  );
}
