import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

interface MenuItem {
  label: string;
  onClick: () => void;
  /** Style the item as destructive. */
  danger?: boolean;
  /** Disable the item (e.g., no conversation to delete). */
  disabled?: boolean;
  /** Optional kbd label rendered on the right. */
  hint?: string;
}

interface ConversationMenuProps {
  items: MenuItem[];
}

/**
 * Lightweight "⋯" popover used in the conversation header. Close on outside
 * click, Escape, or item activation. No portal — the popover sits directly
 * below the trigger; at the header's z-index the small overflow is fine.
 */
export function ConversationMenu({ items }: ConversationMenuProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={clsx(
          "rounded-md px-2 py-1 text-sm text-fg-muted transition-colors",
          open ? "bg-bg-hover text-fg" : "hover:bg-bg-card hover:text-fg",
        )}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Conversation options"
      >
        ⋯
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-20 mt-1 min-w-[12rem] overflow-hidden rounded-md border border-border bg-bg-card py-1 text-sm shadow-lg"
        >
          {items.map((item) => (
            <button
              key={item.label}
              type="button"
              role="menuitem"
              onClick={() => {
                if (item.disabled) return;
                setOpen(false);
                item.onClick();
              }}
              disabled={item.disabled}
              className={clsx(
                "flex w-full items-center justify-between gap-4 px-3 py-1.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                item.danger
                  ? "text-danger hover:bg-danger/10"
                  : "text-fg hover:bg-bg-hover",
              )}
            >
              <span>{item.label}</span>
              {item.hint && (
                <span className="font-mono text-[10px] text-fg-subtle">
                  {item.hint}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
