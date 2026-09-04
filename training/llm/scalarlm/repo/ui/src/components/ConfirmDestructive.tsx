import { useEffect, useState } from "react";

interface ConfirmDestructiveProps {
  open: boolean;
  title: string;
  description: string;
  /** User must type this exact string to enable the confirm button. */
  confirmationText: string;
  /** Label on the destructive button. */
  actionLabel: string;
  onConfirm: () => void;
  onClose: () => void;
  busy?: boolean;
}

export function ConfirmDestructive({
  open,
  title,
  description,
  confirmationText,
  actionLabel,
  onConfirm,
  onClose,
  busy,
}: ConfirmDestructiveProps) {
  const [typed, setTyped] = useState("");

  useEffect(() => {
    if (!open) setTyped("");
  }, [open]);

  if (!open) return null;

  const ready = typed === confirmationText && !busy;

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="flex w-full max-w-md flex-col gap-4 rounded-lg border border-border bg-bg-card p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="text-sm font-semibold text-fg">{title}</h3>
          <p className="mt-1 text-sm text-fg-muted">{description}</p>
        </div>
        <label className="flex flex-col gap-1 text-xs text-fg-muted">
          Type{" "}
          <code className="inline rounded bg-bg px-1 py-0.5 font-mono text-[11px] text-fg">
            {confirmationText}
          </code>{" "}
          to confirm:
          <input
            type="text"
            autoFocus
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            className="mt-1 rounded-md border border-border-subtle bg-bg px-2 py-1 font-mono text-sm text-fg focus:border-accent focus:outline-none"
          />
        </label>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={!ready}
            className="rounded-md bg-danger px-3 py-1.5 text-sm font-medium text-white hover:bg-danger-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? "Working…" : actionLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
