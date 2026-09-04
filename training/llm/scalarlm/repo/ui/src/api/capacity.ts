import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

interface GetGPUCountResponse {
  gpu_count: number;
}
interface GetNodeCountResponse {
  node_count: number;
}

export function useGpuCount() {
  return useQuery<number>({
    queryKey: ["gpu-count"],
    queryFn: async () =>
      (await apiFetch<GetGPUCountResponse>("/megatron/gpu_count")).gpu_count,
    // Topology rarely changes — cache for 5 minutes, no poll.
    staleTime: 5 * 60_000,
  });
}

export function useNodeCount() {
  return useQuery<number>({
    queryKey: ["node-count"],
    queryFn: async () =>
      (await apiFetch<GetNodeCountResponse>("/megatron/node_count")).node_count,
    staleTime: 5 * 60_000,
  });
}
