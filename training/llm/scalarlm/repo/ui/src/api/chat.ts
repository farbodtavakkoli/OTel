import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";
import { getApiConfig } from "./config";

export interface OpenAIModel {
  id: string;
  object: "model";
  created: number;
  owned_by?: string;
  [extra: string]: unknown;
}

interface ModelsListResponse {
  object: "list";
  data: OpenAIModel[];
}

export function useModels() {
  return useQuery<OpenAIModel[]>({
    queryKey: ["models"],
    queryFn: async () =>
      (await apiFetch<ModelsListResponse>("/models")).data ?? [],
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}

export interface ChatMessageInput {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatStreamOptions {
  model: string;
  messages: ChatMessageInput[];
  signal: AbortSignal;
  onDelta: (delta: string) => void;
  onFinal?: (usage: Record<string, unknown> | null) => void;
  temperature?: number;
  max_tokens?: number;
}

export interface ChatStreamError {
  status: number;
  message: string;
  detail?: unknown;
}

export async function streamChatCompletion({
  model,
  messages,
  signal,
  onDelta,
  onFinal,
  temperature,
  max_tokens,
}: ChatStreamOptions): Promise<void> {
  const { api_base } = getApiConfig();
  const url = `${api_base}/chat/completions`;

  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
  };
  if (temperature !== undefined) body.temperature = temperature;
  if (max_tokens !== undefined) body.max_tokens = max_tokens;

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok || !resp.body) {
    let detail: unknown = await resp.text().catch(() => undefined);
    try {
      detail = JSON.parse(detail as string);
    } catch {
    }
    const err: ChatStreamError = {
      status: resp.status,
      message:
        typeof detail === "object" && detail !== null && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : `${resp.status} ${resp.statusText}`,
      detail,
    };
    throw err;
  }

  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  let lastUsage: Record<string, unknown> | null = null;

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;

      let sep: number;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const event = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trimStart();
          if (!payload) continue;
          if (payload === "[DONE]") {
            onFinal?.(lastUsage);
            return;
          }
          try {
            const parsed = JSON.parse(payload);
            if (parsed?.error) {
              throw {
                status: 0,
                message: String(parsed.error),
                detail: parsed,
              } satisfies ChatStreamError;
            }
            const delta: string | undefined =
              parsed?.choices?.[0]?.delta?.content;
            if (typeof delta === "string" && delta.length > 0) {
              onDelta(delta);
            }
            if (parsed?.usage) lastUsage = parsed.usage;
          } catch (err) {
            if (
              typeof err === "object" &&
              err !== null &&
              "status" in err &&
              "message" in err
            ) {
              throw err as ChatStreamError;
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  onFinal?.(lastUsage);
}
