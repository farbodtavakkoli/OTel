import { ApiError } from "@/api/client";

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
}

export function ErrorState({ error, onRetry }: ErrorStateProps) {
  const status = error instanceof ApiError ? error.status : undefined;
  const url = error instanceof ApiError ? error.url : undefined;
  const message =
    error instanceof Error ? error.message : String(error ?? "Unknown error");

  return (
    <div className="flex flex-col gap-2 rounded-md border border-danger/30 bg-danger/5 p-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-danger">
          {status ?? "network"}
        </span>
        <span className="text-fg">{message}</span>
      </div>
      {url && (
        <code className="font-mono text-xs text-fg-muted">{url}</code>
      )}
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-1 w-fit rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover"
        >
          Retry
        </button>
      )}
    </div>
  );
}
