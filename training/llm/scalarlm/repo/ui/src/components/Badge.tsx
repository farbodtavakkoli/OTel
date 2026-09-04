import type { ReactNode } from "react";
import clsx from "clsx";

export type BadgeTone = "neutral" | "accent" | "success" | "warning" | "danger";

interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
}

const TONE_STYLES: Record<BadgeTone, string> = {
  neutral: "bg-bg-hover text-fg-muted ring-1 ring-inset ring-border-subtle",
  accent: "bg-accent/15 text-accent ring-1 ring-inset ring-accent/30",
  success: "bg-success/15 text-success ring-1 ring-inset ring-success/30",
  warning: "bg-warning/15 text-warning ring-1 ring-inset ring-warning/30",
  danger: "bg-danger/15 text-danger ring-1 ring-inset ring-danger/30",
};

export function Badge({ tone = "neutral", className, children }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-md px-1.5 py-0.5 font-mono text-[11px]",
        TONE_STYLES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
