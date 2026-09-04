/**
 * Publish-to-HuggingFace API surface.
 *
 * Phase 1 covers: list checkpoints, submit a publish SLURM job, poll
 * its status, cancel it. Log tailing comes in Phase 2.
 *
 * See ui/docs/publish-to-hf.md for the full design.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { apiFetch } from "./client";
import { getApiConfig } from "./config";
import { streamNdjsonLogOnce, type LogLine } from "./training";

export interface CheckpointEntry {
  name: string;
  step: number;
  /** Float seconds since epoch — same convention as TrainingJobStatus.start_time. */
  mtime: number;
}

interface CheckpointsResponse {
  checkpoints: CheckpointEntry[];
}

/**
 * GET /v1/megatron/train/{job_hash}/checkpoints
 */
export function useCheckpoints(jobHash: string | undefined) {
  return useQuery<CheckpointEntry[]>({
    queryKey: ["checkpoints", jobHash],
    enabled: Boolean(jobHash),
    queryFn: async () =>
      (
        await apiFetch<CheckpointsResponse>(
          `/megatron/train/${jobHash}/checkpoints`,
        )
      ).checkpoints,
    staleTime: 30_000,
  });
}

// ---------------------------------------------------------------------------
// Publish lifecycle
// ---------------------------------------------------------------------------

export type PublishMode = "merged" | "adapter";

export type PublishPhase =
  | "queued"
  | "validating"
  | "loading_base"
  | "merging"
  | "saving"
  | "uploading"
  | "done"
  | "error";

export interface PublishStatus {
  publish_job_id: string | null;
  publish_dir?: string;
  mode: PublishMode;
  phase: PublishPhase;
  repo_id?: string;
  repo_url?: string | null;
  error?: string | null;
  started_at?: number | null;
  completed_at?: number | null;
  uploaded_bytes?: number;
  total_bytes?: number;
  base_model?: string;
}

export interface PublishRequestBody {
  mode: PublishMode;
  repo_id: string;
  private: boolean;
  hf_token: string;
  checkpoint?: string;
  lora_alpha?: number;
  commit_message?: string;
}

export interface PublishSubmissionResponse {
  publish_job_id: string;
  publish_dir: string;
  status: "QUEUED";
}

const TERMINAL_PHASES: ReadonlySet<PublishPhase> = new Set(["done", "error"]);

export function isTerminalPhase(phase: PublishPhase | undefined): boolean {
  return phase !== undefined && TERMINAL_PHASES.has(phase);
}

/**
 * POST /v1/megatron/train/{hash}/publish
 *
 * On success the modal should switch to polling /publish/status — the
 * mutation invalidates the status query so the next render starts a
 * poll without waiting for the staleTime.
 */
export function useSubmitPublish(jobHash: string) {
  const qc = useQueryClient();
  return useMutation<PublishSubmissionResponse, Error, PublishRequestBody>({
    mutationFn: (body) =>
      apiFetch<PublishSubmissionResponse>(
        `/megatron/train/${jobHash}/publish`,
        { method: "POST", json: body },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["publish-status", jobHash] });
    },
  });
}

/**
 * GET /v1/megatron/train/{hash}/publish/status
 *
 * Polls every 2s while non-terminal, otherwise drops to once a minute
 * so a successfully-pushed publish doesn't keep hammering the API.
 * `enabled` should be false until the user has submitted in this
 * modal session — the modal flips it after `useSubmitPublish` resolves.
 */
export function usePublishStatus(
  jobHash: string,
  enabled: boolean,
) {
  return useQuery<PublishStatus>({
    queryKey: ["publish-status", jobHash],
    enabled,
    queryFn: () =>
      apiFetch<PublishStatus>(
        `/megatron/train/${jobHash}/publish/status`,
      ),
    refetchInterval: (q) => {
      const phase = q.state.data?.phase;
      if (isTerminalPhase(phase)) return 60_000;
      return 2_000;
    },
    staleTime: 0,
    retry: (count, err) => {
      // 404 means no publish has been submitted yet for this job; don't
      // retry — the modal will flip `enabled` true once submit lands.
      if ((err as { status?: number })?.status === 404) return false;
      return count < 3;
    },
  });
}

/**
 * Tail the latest publish job's publish.log via the same NDJSON
 * stream protocol as service logs (`tail=N`, then byte-offset
 * resume). Reuses `streamNdjsonLogOnce` for the actual parsing so
 * we don't duplicate the per-line logic.
 */
export async function tailPublishLog({
  jobHash,
  signal,
  onLine,
  initialTail = 200,
  reopenDelayMs = 1500,
}: {
  jobHash: string;
  signal: AbortSignal;
  onLine: (line: LogLine) => void;
  initialTail?: number;
  reopenDelayMs?: number;
}): Promise<void> {
  const { api_base } = getApiConfig();

  let nextOffset: number | null = null;
  let firstPass = true;

  while (!signal.aborted) {
    try {
      const params = new URLSearchParams();
      if (firstPass && initialTail > 0) {
        params.set("tail", String(initialTail));
      } else if (nextOffset !== null) {
        params.set("starting_byte_offset", String(nextOffset));
      }
      const url =
        `${api_base}/megatron/train/${encodeURIComponent(jobHash)}/publish/logs` +
        `?${params}`;
      await streamNdjsonLogOnce(url, 0, signal, (ln) => {
        if (typeof ln.next_offset === "number") nextOffset = ln.next_offset;
        onLine(ln);
      });
      firstPass = false;
    } catch (err) {
      if (signal.aborted || (err as { name?: string }).name === "AbortError")
        break;
      // Most common: 404 because the publish.log hasn't been created yet
      // (SLURM hasn't started the job). Quietly retry after the delay.
    }
    await sleep(reopenDelayMs, signal);
  }
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * POST /v1/megatron/train/{hash}/publish/cancel
 */
export function useCancelPublish(jobHash: string) {
  const qc = useQueryClient();
  return useMutation<PublishStatus, Error, void>({
    mutationFn: () =>
      apiFetch<PublishStatus>(
        `/megatron/train/${jobHash}/publish/cancel`,
        { method: "POST" },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["publish-status", jobHash] });
    },
  });
}
