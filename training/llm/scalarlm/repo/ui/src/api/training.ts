import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { apiFetch, ApiError } from "./client";
import { getApiConfig } from "./config";
import { buildTar } from "@/lib/tar";
import type { TrainArgs } from "@/lib/trainArgsSchema";

export type TrainingJobStatus =
  | "QUEUED"
  | "TRAINING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED"
  | "UNKNOWN";

export interface TrainingHistoryPoint {
  step: number;
  loss: number;
  epoch?: number;
  time?: number;
}

export interface ModelListEntry {
  name: string;
  deployed: boolean;
  status: TrainingJobStatus;
  start_time: number;
  step: number;
  max_steps: number;
  train_time: number;
  loss: number;
}

interface ListModelsResponse {
  models: ModelListEntry[];
}

export function useModelList() {
  return useQuery<ModelListEntry[]>({
    queryKey: ["model-list"],
    queryFn: async () =>
      (await apiFetch<ListModelsResponse>("/megatron/list_models")).models,
    refetchInterval: 5_000,
    staleTime: 0,
  });
}

export interface TrainingJobDetail {
  job_status: {
    status: TrainingJobStatus;
    start_time?: number;
    max_steps?: number;
    history?: TrainingHistoryPoint[];
    job_id?: string;
    completed_at?: number;
    error?: string;
    output?: string;
    [key: string]: unknown;
  };
  job_config: {
    job_directory?: string;
    training_data_path?: string;
    dataset_hash?: string;
    llm_name?: string;
    max_steps?: number;
    learning_rate?: number;
    batch_size?: number;
    gpus?: number;
    nodes?: number;
    adapter_type?: string;
    timeout?: number;
    [key: string]: unknown;
  };
  deployed: boolean;
}

export function useTrainingJob(jobHash: string | undefined) {
  return useQuery<TrainingJobDetail>({
    queryKey: ["training-job", jobHash],
    enabled: Boolean(jobHash),
    queryFn: () => apiFetch<TrainingJobDetail>(`/megatron/train/${jobHash}`),
    refetchInterval: (query) => {
      const status = query.state.data?.job_status?.status;
      if (status === "QUEUED" || status === "TRAINING") return 2_000;
      return 30_000;
    },
    staleTime: 0,
  });
}

export interface DatasetExample {
  index: number;
  input?: string;
  output?: string;
  raw: Record<string, unknown>;
  truncated_fields?: string[];
}

export interface DatasetSlice {
  total: number;
  offset: number;
  limit: number;
  matched: number;
  truncated: boolean;
  examples: DatasetExample[];
}

export interface DatasetQuery {
  offset: number;
  limit: number;
  q: string;
}

export function useTrainingDataset(
  jobHash: string | undefined,
  { offset, limit, q }: DatasetQuery,
) {
  return useQuery<DatasetSlice>({
    queryKey: ["training-dataset", jobHash, offset, limit, q],
    enabled: Boolean(jobHash),
    queryFn: () => {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
      });
      if (q) params.set("q", q);
      return apiFetch<DatasetSlice>(
        `/megatron/train/${jobHash}/dataset?${params.toString()}`,
      );
    },
    placeholderData: keepPreviousData,
    staleTime: 30_000,
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobHash: string) =>
      apiFetch(`/megatron/cancel/${jobHash}`, { method: "POST" }),
    onSuccess: (_data, jobHash) => {
      qc.invalidateQueries({ queryKey: ["training-job", jobHash] });
      qc.invalidateQueries({ queryKey: ["model-list"] });
      qc.invalidateQueries({ queryKey: ["squeue"] });
    },
  });
}

export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobHash: string) =>
      apiFetch(`/megatron/delete/${jobHash}`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["model-list"] });
      qc.invalidateQueries({ queryKey: ["squeue"] });
    },
  });
}

export function useRestartJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobHash: string) =>
      apiFetch(`/megatron/restart/${jobHash}`, { method: "POST" }),
    onSuccess: (_data, jobHash) => {
      qc.invalidateQueries({ queryKey: ["training-job", jobHash] });
      qc.invalidateQueries({ queryKey: ["model-list"] });
      qc.invalidateQueries({ queryKey: ["squeue"] });
    },
  });
}

export interface LogLine {
  line: string;
  line_number: number;
  byte_offset?: number;
  next_offset?: number;
}

export interface TailOptions {
  modelName: string;
  signal: AbortSignal;
  isStillRunning: () => boolean;
  onLine: (line: LogLine) => void;
  onStatus?: (status: "connecting" | "streaming" | "reconnecting" | "closed") => void;
  tailReopenDelayMs?: number;
}

export async function tailTrainingLogs({
  modelName,
  signal,
  isStillRunning,
  onLine,
  onStatus,
  tailReopenDelayMs = 2_000,
}: TailOptions): Promise<void> {
  let lineNumber = 0;

  while (!signal.aborted) {
    onStatus?.(lineNumber === 0 ? "connecting" : "reconnecting");
    try {
      lineNumber = await streamOnce(modelName, lineNumber, signal, (ln) => {
        onLine(ln);
      });
    } catch (err) {
      if (signal.aborted || (err as { name?: string }).name === "AbortError") break;
    }

    if (!isStillRunning()) {
      onStatus?.("closed");
      return;
    }

    onStatus?.("reconnecting");
    await sleep(tailReopenDelayMs, signal);
  }
  onStatus?.("closed");
}

async function streamOnce(
  modelName: string,
  startingLine: number,
  signal: AbortSignal,
  onLine: (line: LogLine) => void,
): Promise<number> {
  const { api_base } = getApiConfig();
  const url =
    `${api_base}/megatron/train/logs/${encodeURIComponent(modelName)}` +
    `?starting_line_number=${startingLine}`;
  return streamNdjsonLogOnce(url, startingLine, signal, onLine);
}

export async function fetchFullTrainingLog(
  modelName: string,
  signal: AbortSignal,
): Promise<string> {
  const { api_base } = getApiConfig();
  const url =
    `${api_base}/megatron/train/logs/${encodeURIComponent(modelName)}` +
    `?starting_line_number=0`;
  const parts: string[] = [];
  await streamNdjsonLogOnce(url, 0, signal, (l) => parts.push(l.line));
  return parts.join("\n");
}

export async function streamNdjsonLogOnce(
  url: string,
  startingLine: number,
  signal: AbortSignal,
  onLine: (line: LogLine) => void,
): Promise<number> {
  const resp = await fetch(url, { signal });
  if (!resp.ok || !resp.body) {
    throw new Error(`Log stream HTTP ${resp.status}`);
  }

  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let lastLine = startingLine;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const chunk = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!chunk) continue;
      try {
        const parsed = JSON.parse(chunk) as LogLine;
        if (typeof parsed.line_number === "number") {
          lastLine = parsed.line_number + 1;
        }
        onLine(parsed);
      } catch {
      }
    }
  }
  return lastLine;
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

export interface SubmitProgress {
  loaded: number;
  total: number;
  fraction: number | null;
}

export interface SubmitTrainingJobInput {
  datasetJsonl: string;
  trainArgs: TrainArgs;
  onProgress?: (p: SubmitProgress) => void;
  signal?: AbortSignal;
}

export interface SubmitTrainingJobResponse {
  job_status: TrainingJobDetail["job_status"] & { model_name?: string };
  job_config: TrainingJobDetail["job_config"];
  deployed: boolean;
}

const MULTIPART_BOUNDARY = "----scalarlmFormBoundary7MA4YWxkTrZu0gW";

function buildMultipartBody(
  tarBytes: Uint8Array,
  params: unknown,
): Uint8Array {
  const enc = new TextEncoder();
  const CRLF = "\r\n";
  const head1 =
    `--${MULTIPART_BOUNDARY}${CRLF}` +
    `Content-Disposition: form-data; name="file"; filename="dataset.tar"${CRLF}` +
    `Content-Type: application/x-tar${CRLF}${CRLF}`;
  const head2 =
    `${CRLF}--${MULTIPART_BOUNDARY}${CRLF}` +
    `Content-Disposition: form-data; name="params"${CRLF}` +
    `Content-Type: application/json${CRLF}${CRLF}`;
  const tail = `${CRLF}--${MULTIPART_BOUNDARY}--${CRLF}`;

  const head1Bytes = enc.encode(head1);
  const head2Bytes = enc.encode(head2);
  const paramsBytes = enc.encode(JSON.stringify(params));
  const tailBytes = enc.encode(tail);

  const total =
    head1Bytes.byteLength +
    tarBytes.byteLength +
    head2Bytes.byteLength +
    paramsBytes.byteLength +
    tailBytes.byteLength;
  const body = new Uint8Array(total);
  let off = 0;
  body.set(head1Bytes, off); off += head1Bytes.byteLength;
  body.set(tarBytes, off);   off += tarBytes.byteLength;
  body.set(head2Bytes, off); off += head2Bytes.byteLength;
  body.set(paramsBytes, off); off += paramsBytes.byteLength;
  body.set(tailBytes, off);
  return body;
}

export async function submitTrainingJob({
  datasetJsonl,
  trainArgs,
  onProgress,
  signal,
}: SubmitTrainingJobInput): Promise<SubmitTrainingJobResponse> {
  const { api_base } = getApiConfig();
  const url = `${api_base}/megatron/train`;

  const datasetBytes = new TextEncoder().encode(
    datasetJsonl.endsWith("\n") ? datasetJsonl : datasetJsonl + "\n",
  );
  const tar = buildTar([{ name: "dataset.jsonlines", data: datasetBytes }]);
  const body = buildMultipartBody(tar, trainArgs);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader(
      "Content-Type",
      `multipart/form-data; boundary=${MULTIPART_BOUNDARY}`,
    );
    xhr.responseType = "text";

    if (onProgress) {
      xhr.upload.addEventListener("progress", (e) => {
        onProgress({
          loaded: e.loaded,
          total: e.lengthComputable ? e.total : body.byteLength,
          fraction: e.lengthComputable ? e.loaded / e.total : null,
        });
      });
    }

    xhr.addEventListener("load", () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        let detail: unknown = xhr.responseText;
        try {
          const parsed = JSON.parse(xhr.responseText) as { detail?: string };
          if (parsed?.detail) detail = parsed.detail;
        } catch {
        }
        reject(
          new ApiError(
            xhr.status,
            url,
            `${xhr.status} ${String(detail).slice(0, 200)}`,
            detail,
          ),
        );
        return;
      }
      try {
        resolve(JSON.parse(xhr.responseText) as SubmitTrainingJobResponse);
      } catch (e) {
        reject(e);
      }
    });
    xhr.addEventListener("error", () =>
      reject(new ApiError(0, url, "Network error during upload")),
    );
    xhr.addEventListener("abort", () =>
      reject(new DOMException("Upload aborted", "AbortError")),
    );

    if (signal) {
      if (signal.aborted) {
        xhr.abort();
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }

    xhr.send(new Blob([body as BlobPart]));
  });
}

export function useSubmitTrainingJob() {
  const qc = useQueryClient();
  return useMutation<SubmitTrainingJobResponse, Error, SubmitTrainingJobInput>({
    mutationFn: submitTrainingJob,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["model-list"] });
      qc.invalidateQueries({ queryKey: ["squeue"] });
    },
  });
}
