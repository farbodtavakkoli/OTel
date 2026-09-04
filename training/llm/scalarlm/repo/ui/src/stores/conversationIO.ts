/**
 * Round-trippable JSON export/import for conversations. The on-disk shape is
 * versioned so future UI releases can migrate or reject old files cleanly.
 */

import {
  appendMessage,
  listMessages,
  newConversationId,
  newMessageId,
  putConversation,
  type Conversation,
  type Message,
} from "./conversations";

export const EXPORT_VERSION = 1 as const;

export interface ConversationExport {
  version: typeof EXPORT_VERSION;
  /** ISO string for human inspection; not used on import. */
  exportedAt: string;
  conversation: Conversation;
  messages: Message[];
}

export async function exportConversation(
  conversation: Conversation,
): Promise<ConversationExport> {
  const messages = await listMessages(conversation.id);
  return {
    version: EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    conversation,
    messages,
  };
}

export function downloadConversationJson(
  conversation: Conversation,
  payload: ConversationExport,
): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = sanitizeFilename(
    `${conversation.title || conversation.id}.scalarlm-chat.json`,
  );
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Parse and validate an import file. The imported conversation receives a
 * fresh id (so it can't clash with one we already have), and every message
 * row is re-stamped with a new id + the new conversationId. createdAt is
 * preserved so the imported thread keeps its original chronology.
 */
export async function importConversationJson(
  file: File,
): Promise<Conversation> {
  const text = await file.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(
      `Not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  const payload = validateExport(parsed);

  const newConvId = newConversationId();
  const now = Date.now();
  const imported: Conversation = {
    ...payload.conversation,
    id: newConvId,
    title: payload.conversation.title || "(imported)",
    updatedAt: now,
    // createdAt preserved from the export so "recent" buckets still make sense.
  };
  await putConversation(imported);

  for (const m of payload.messages) {
    const remapped: Message = {
      ...m,
      id: newMessageId(),
      conversationId: newConvId,
    };
    await appendMessage(remapped);
  }
  return imported;
}

function validateExport(raw: unknown): ConversationExport {
  if (!isObject(raw)) throw new Error("Root must be an object");
  if (raw.version !== EXPORT_VERSION) {
    throw new Error(`Unsupported export version: ${String(raw.version)}`);
  }
  const conversation = validateConversation(raw.conversation);
  const messagesRaw = raw.messages;
  if (!Array.isArray(messagesRaw)) {
    throw new Error("Missing messages[]");
  }
  const messages = messagesRaw.map(validateMessage);
  return {
    version: EXPORT_VERSION,
    exportedAt:
      typeof raw.exportedAt === "string"
        ? raw.exportedAt
        : new Date().toISOString(),
    conversation,
    messages,
  };
}

function validateConversation(raw: unknown): Conversation {
  if (!isObject(raw)) throw new Error("Missing conversation");
  const id = requireString(raw, "conversation.id");
  const title = requireString(raw, "conversation.title");
  const model = requireString(raw, "conversation.model");
  const createdAt = requireNumber(raw, "conversation.createdAt");
  const updatedAt = requireNumber(raw, "conversation.updatedAt");
  const systemPrompt =
    typeof raw.systemPrompt === "string" ? raw.systemPrompt : undefined;
  const maxTokens =
    typeof raw.maxTokens === "number" && Number.isFinite(raw.maxTokens) && raw.maxTokens > 0
      ? Math.floor(raw.maxTokens)
      : undefined;
  return { id, title, model, createdAt, updatedAt, systemPrompt, maxTokens };
}

function validateMessage(raw: unknown): Message {
  if (!isObject(raw)) throw new Error("Malformed message row");
  const id = requireString(raw, "message.id");
  const conversationId = requireString(raw, "message.conversationId");
  const content = requireString(raw, "message.content");
  const createdAt = requireNumber(raw, "message.createdAt");
  const role = raw.role;
  if (role !== "user" && role !== "assistant" && role !== "system") {
    throw new Error(`Unknown role: ${String(role)}`);
  }
  const completedAt =
    raw.completedAt === null
      ? null
      : typeof raw.completedAt === "number"
      ? raw.completedAt
      : null;
  const modelAtSend =
    typeof raw.modelAtSend === "string" ? raw.modelAtSend : undefined;
  const tokenCount =
    typeof raw.tokenCount === "number" ? raw.tokenCount : undefined;
  return {
    id,
    conversationId,
    role,
    content,
    createdAt,
    completedAt,
    modelAtSend,
    tokenCount,
  };
}

function requireString(obj: Record<string, unknown>, path: string): string {
  const v = obj[path.split(".").pop() as string];
  if (typeof v !== "string") throw new Error(`${path} must be string`);
  return v;
}

function requireNumber(obj: Record<string, unknown>, path: string): number {
  const v = obj[path.split(".").pop() as string];
  if (typeof v !== "number" || !Number.isFinite(v))
    throw new Error(`${path} must be number`);
  return v;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function sanitizeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]+/g, "_").slice(0, 120);
}
