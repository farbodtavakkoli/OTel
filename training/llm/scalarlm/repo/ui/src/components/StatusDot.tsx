import clsx from "clsx";

import type { HealthState } from "@/api/health";

interface StatusDotProps {
  state: HealthState;
  label: string;
  onClick?: () => void;
  title?: string;
}

// Glow uses the same token; `rgb(var(--color-xxx) / 0.6)` gives the halo.
const STATE_STYLES: Record<HealthState, string> = {
  up: "bg-success shadow-[0_0_8px_rgb(var(--color-success)/0.6)]",
  down: "bg-danger shadow-[0_0_8px_rgb(var(--color-danger)/0.6)]",
  unknown: "bg-fg-subtle",
};

export function StatusDot({ state, label, onClick, title }: StatusDotProps) {
  const inner = (
    <>
      <span
        aria-hidden
        className={clsx("h-2 w-2 rounded-full", STATE_STYLES[state])}
      />
      <span className="text-sm text-fg">{label}</span>
      <span className="text-xs text-fg-muted">{state}</span>
    </>
  );
  const className =
    "flex items-center gap-2 rounded-md border border-border-subtle bg-bg px-3 py-2";

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        title={title}
        className={clsx(
          className,
          "text-left transition-colors hover:border-border hover:bg-bg-hover",
        )}
      >
        {inner}
      </button>
    );
  }
  return (
    <div className={className} title={title}>
      {inner}
    </div>
  );
}
