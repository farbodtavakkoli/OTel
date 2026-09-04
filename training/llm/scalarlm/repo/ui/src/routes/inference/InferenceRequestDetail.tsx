import { useInferenceRequestDetail } from "@/api/inference";

/**
 * Lazy detail panel for one inference request. Mounts only when the
 * row is expanded — `enabled` gates the network fetch — so collapsing
 * a row also stops paying for it.
 */
export function InferenceRequestDetail({
  requestId,
  expanded,
}: {
  requestId: string;
  expanded: boolean;
}) {
  const { data, error, isPending } = useInferenceRequestDetail(
    requestId,
    expanded,
  );

  if (!expanded) return null;

  if (isPending) {
    return (
      <div className="px-4 py-3 text-xs text-fg-subtle">Loading…</div>
    );
  }
  if (error) {
    return (
      <div className="px-4 py-3 text-xs text-danger">
        Failed to load: {error.message}
      </div>
    );
  }
  if (!data) return null;

  return (
    <div className="flex flex-col gap-3 border-t border-border-subtle bg-bg/40 px-4 py-3">
      <Section label="Request" payload={data.request} />
      <Section label="Response" payload={data.response} />
      <Section label="Status" payload={data.status} />
    </div>
  );
}

function Section({ label, payload }: { label: string; payload: unknown }) {
  const json =
    payload === null
      ? "null"
      : JSON.stringify(payload, null, 2);
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[10px] uppercase tracking-wider text-fg-subtle">
        {label}
      </div>
      <pre className="max-h-[400px] overflow-auto rounded-md border border-border-subtle bg-bg p-3 font-mono text-xs leading-relaxed text-fg">
        {json}
      </pre>
    </div>
  );
}
