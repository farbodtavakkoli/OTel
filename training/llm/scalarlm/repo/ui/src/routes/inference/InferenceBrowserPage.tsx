import { PageHeader } from "@/components/PageHeader";

import { InferenceRequestList } from "./InferenceRequestList";

export function InferenceBrowserPage() {
  return (
    <>
      <PageHeader
        title="Inference requests"
        subtitle="GET /v1/generate/list_requests · file-backed history, newest first"
      />
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-6">
        <InferenceRequestList />
      </div>
    </>
  );
}
