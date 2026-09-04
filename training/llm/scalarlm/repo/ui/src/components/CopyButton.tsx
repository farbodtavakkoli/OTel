import { useEffect, useState } from "react";
import clsx from "clsx";

interface CopyButtonProps {
  /** Value to copy to the clipboard on click. */
  value: string;
  /** Label shown on the button; defaults to "Copy". */
  label?: string;
  /** Tailwind size variant; "sm" is default. */
  size?: "sm" | "md";
  className?: string;
  /** Optional callback after a successful copy. */
  onCopied?: () => void;
}

/**
 * Shared copy-to-clipboard button. Briefly swaps the label to "Copied ✓" for
 * 1.2s on success. Falls back to a 1-shot textarea-select when the async
 * Clipboard API is unavailable (http origins, old Safari).
 */
export function CopyButton({
  value,
  label = "Copy",
  size = "sm",
  className,
  onCopied,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1200);
    return () => clearTimeout(t);
  }, [copied]);

  const doCopy = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
      } else {
        legacyCopy(value);
      }
      setCopied(true);
      onCopied?.();
    } catch {
      // Silently drop — better than throwing a modal at the user.
    }
  };

  return (
    <button
      type="button"
      onClick={doCopy}
      className={clsx(
        "inline-flex items-center gap-1 rounded-md border border-border-subtle bg-bg-card text-fg transition-colors hover:border-border hover:bg-bg-hover",
        size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-1.5 text-sm",
        className,
      )}
      aria-label={copied ? "Copied" : `Copy ${label.toLowerCase()}`}
      title={copied ? "Copied" : `Copy ${label.toLowerCase()}`}
    >
      {copied ? "Copied ✓" : label}
    </button>
  );
}

function legacyCopy(value: string) {
  const ta = document.createElement("textarea");
  ta.value = value;
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(ta);
  }
}
