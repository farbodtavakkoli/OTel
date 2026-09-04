import type { ReactNode } from "react";

interface PlaceholderProps {
  title: string;
  phase: string;
  children?: ReactNode;
}

export function Placeholder({ title, phase, children }: PlaceholderProps) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-start gap-3 px-6 py-12">
      <div className="rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs uppercase tracking-wider text-fg-muted">
        Phase {phase}
      </div>
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      <div className="text-sm leading-relaxed text-fg-muted">
        {children ?? (
          <p>
            Not implemented yet. See{" "}
            <code className="rounded bg-bg-card px-1 py-0.5 font-mono text-xs">
              docs/ui-design.md
            </code>{" "}
            for the design.
          </p>
        )}
      </div>
    </div>
  );
}
