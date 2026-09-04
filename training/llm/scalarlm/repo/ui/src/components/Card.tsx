import type { ReactNode } from "react";
import clsx from "clsx";

interface CardProps {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Card({ title, subtitle, action, className, children }: CardProps) {
  return (
    <section
      className={clsx(
        "rounded-lg border border-border-subtle bg-bg-card",
        className,
      )}
    >
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-border-subtle px-4 py-3">
          <div>
            {title && (
              <h2 className="text-sm font-semibold tracking-tight text-fg">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-fg-muted">{subtitle}</p>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}
