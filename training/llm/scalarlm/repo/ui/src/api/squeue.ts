import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

/**
 * Raw response from GET /v1/megatron/squeue. The backend shells out to
 * `squeue --format="%.18i %.9P %.12j %.8u %.8T %.10M %.9l %.6D %R"` and hands
 * back the literal text. We parse client-side so the UI can render a table.
 */
interface SqueueRawResponse {
  squeue_output: string | null;
  error_message: string | null;
}

/** One job row from squeue. */
export interface SqueueJob {
  jobId: string;
  partition: string;
  /** Job name. For ScalarLM training jobs this is the job hash. */
  name: string;
  user: string;
  state: string;
  time: string;
  timeLimit: string;
  nodes: string;
  nodelistReason: string;
}

export interface SqueueResult {
  jobs: SqueueJob[];
  /** True when the squeue command itself failed (not just no rows). */
  errored: boolean;
  errorMessage: string | null;
}

/**
 * Parse the first whitespace-separated columns from a squeue --format line.
 * Everything after the 8th column becomes NODELIST(REASON), which may contain
 * parentheses and spaces ("(Priority)", "node01,node02").
 */
function parseSqueueOutput(raw: string | null): SqueueJob[] {
  if (!raw) return [];
  const lines = raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length <= 1) return [];
  // Skip the header row.
  return lines.slice(1).map((line) => {
    // Split on runs of whitespace, keeping the 9th+ columns together.
    const parts = line.split(/\s+/);
    const [
      jobId = "",
      partition = "",
      name = "",
      user = "",
      state = "",
      time = "",
      timeLimit = "",
      nodes = "",
      ...reasonRest
    ] = parts;
    return {
      jobId,
      partition,
      name,
      user,
      state,
      time,
      timeLimit,
      nodes,
      nodelistReason: reasonRest.join(" "),
    };
  });
}

export function useSqueue() {
  return useQuery<SqueueResult>({
    queryKey: ["squeue"],
    queryFn: async () => {
      const resp = await apiFetch<SqueueRawResponse>("/megatron/squeue");
      return {
        jobs: parseSqueueOutput(resp.squeue_output),
        errored: Boolean(resp.error_message),
        errorMessage: resp.error_message,
      };
    },
    refetchInterval: 3_000,
    staleTime: 0,
  });
}
