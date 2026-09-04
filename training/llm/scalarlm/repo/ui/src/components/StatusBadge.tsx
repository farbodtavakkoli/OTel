import clsx from "clsx";

import { Badge, type BadgeTone } from "./Badge";
import type { TrainingJobStatus } from "@/api/training";

interface StatusBadgeProps {
  status: TrainingJobStatus;
  className?: string;
}

const TONE_BY_STATUS: Record<TrainingJobStatus, BadgeTone> = {
  QUEUED: "neutral",
  TRAINING: "accent",
  COMPLETED: "success",
  FAILED: "danger",
  CANCELLED: "neutral",
  UNKNOWN: "warning",
};

const PULSES_BY_STATUS: Partial<Record<TrainingJobStatus, boolean>> = {
  TRAINING: true,
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const tone = TONE_BY_STATUS[status] ?? "neutral";
  const pulse = PULSES_BY_STATUS[status];
  return (
    <Badge tone={tone} className={clsx(pulse && "animate-pulse", className)}>
      {status}
    </Badge>
  );
}
