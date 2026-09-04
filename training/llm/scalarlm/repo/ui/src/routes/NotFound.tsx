import { Link, useLocation } from "react-router-dom";

export function NotFound() {
  const location = useLocation();
  return (
    <div className="mx-auto max-w-xl px-6 py-16 text-center">
      <h1 className="text-3xl font-semibold tracking-tight">Not found</h1>
      <p className="mt-3 text-sm text-fg-muted">
        No route matches{" "}
        <code className="rounded bg-bg-card px-1.5 py-0.5 font-mono text-xs">
          /app{location.pathname}
        </code>
        .
      </p>
      <Link
        to="/"
        className="mt-6 inline-block rounded-md border border-border-subtle bg-bg-card px-4 py-2 text-sm hover:border-border hover:bg-bg-hover"
      >
        Go home
      </Link>
    </div>
  );
}
