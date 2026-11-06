import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

interface SqueueRawResponse {
  squeue_output: string | null;
  error_message: string | null;
}

export interface SqueueJob {
  jobId: string;
  partition: string;
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
  errored: boolean;
  errorMessage: string | null;
}

function parseSqueueOutput(raw: string | null): SqueueJob[] {
  if (!raw) return [];
  const lines = raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length <= 1) return [];
  return lines.slice(1).map((line) => {
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
