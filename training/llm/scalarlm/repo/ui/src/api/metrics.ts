import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "./client";

export interface GenerateMetrics {
  queue_depth: number;
  requests: number;
  tokens: number;
  total_time: number;
  "token/s": number;
  "request/s": number;
  "flop/s": number;
}

export function useGenerateMetrics() {
  return useQuery<GenerateMetrics>({
    queryKey: ["generate-metrics"],
    queryFn: () => apiFetch<GenerateMetrics>("/generate/metrics"),
    refetchInterval: 3_000,
    staleTime: 0,
  });
}

export function useClearQueue() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<unknown>("/generate/clear_queue", { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["generate-metrics"] });
    },
  });
}
