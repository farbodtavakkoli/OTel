import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { streamChatCompletion, type ChatStreamError } from "@/api/chat";
import {
  appendMessage,
  deleteConversation,
  deleteMessage,
  newMessageId,
  putConversation,
  updateMessage,
  type Conversation,
  type Message,
} from "@/stores/conversations";
import {
  downloadConversationJson,
  exportConversation,
} from "@/stores/conversationIO";
import { useMessages } from "@/stores/useConversationStore";
import { getTemperatureForRequest } from "@/stores/sampling";

import { ConversationMenu } from "./ConversationMenu";
import { MarkdownMessage } from "./MarkdownMessage";
import { ModelPicker } from "./ModelPicker";
import { SystemPromptModal } from "./SystemPromptEditor";

interface ConversationViewProps {
  conversation: Conversation;
  /** Notifies the parent when the first user turn creates a conversation. */
  onFirstTurn?: (id: string) => void;
  /** Mobile only: open the conversation-list drawer. */
  onOpenSidebar?: () => void;
}

type OpenAIMessage = { role: "system" | "user" | "assistant"; content: string };

export function ConversationView({
  conversation,
  onFirstTurn,
  onOpenSidebar,
}: ConversationViewProps) {
  const navigate = useNavigate();
  const messages = useMessages(conversation.id);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [systemPromptOpen, setSystemPromptOpen] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages, unless the user scrolled up.
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (!follow) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, follow]);

  const handleModelChange = useCallback(
    (model: string) => {
      putConversation({ ...conversation, model, updatedAt: Date.now() });
    },
    [conversation],
  );

  const handleSystemPromptChange = useCallback(
    (systemPrompt: string) => {
      putConversation({
        ...conversation,
        systemPrompt: systemPrompt || undefined,
        updatedAt: Date.now(),
      });
    },
    [conversation],
  );

  const handleMaxTokensChange = useCallback(
    (next: number | undefined) => {
      putConversation({
        ...conversation,
        maxTokens: next,
        updatedAt: Date.now(),
      });
    },
    [conversation],
  );

  const handleExport = useCallback(async () => {
    const payload = await exportConversation(conversation);
    downloadConversationJson(conversation, payload);
  }, [conversation]);

  const handleDelete = useCallback(async () => {
    await deleteConversation(conversation.id);
    navigate("/chat", { replace: true });
  }, [conversation.id, navigate]);

  /**
   * Shared streaming driver. Takes the prompt history to send to the model
   * and the assistant-message row we want to populate. Keeps Send and
   * Regenerate in lockstep around abort + error handling.
   */
  const runAssistantTurn = useCallback(
    async (openAIMessages: OpenAIMessage[], assistantMsg: Message) => {
      setError(null);
      setStreaming(true);
      const controller = new AbortController();
      controllerRef.current = controller;

      let buffered = "";
      let flushPending = false;
      const flush = async () => {
        flushPending = false;
        await updateMessage({ ...assistantMsg, content: buffered });
      };

      try {
        await streamChatCompletion({
          model: assistantMsg.modelAtSend ?? conversation.model,
          messages: openAIMessages,
          signal: controller.signal,
          max_tokens: conversation.maxTokens,
          temperature: getTemperatureForRequest(),
          onDelta: (delta) => {
            buffered += delta;
            if (!flushPending) {
              flushPending = true;
              requestAnimationFrame(flush);
            }
          },
          onFinal: async (usage) => {
            await updateMessage({
              ...assistantMsg,
              content: buffered,
              completedAt: Date.now(),
              tokenCount:
                typeof usage?.total_tokens === "number"
                  ? (usage.total_tokens as number)
                  : undefined,
            });
          },
        });
      } catch (err) {
        if ((err as DOMException)?.name === "AbortError") {
          await updateMessage({
            ...assistantMsg,
            content: buffered + (buffered ? "\n" : "") + "_[stopped]_",
            completedAt: Date.now(),
          });
        } else {
          const e = err as ChatStreamError;
          setError(e.message ?? String(err));
          await updateMessage({
            ...assistantMsg,
            content:
              buffered +
              (buffered ? "\n\n" : "") +
              `_[error: ${e.message ?? String(err)}]_`,
            completedAt: Date.now(),
          });
        }
      } finally {
        setStreaming(false);
        controllerRef.current = null;
      }
    },
    [conversation.model, conversation.maxTokens],
  );

  const handleSend = useCallback(async () => {
    const trimmed = draft.trim();
    if (!trimmed || streaming) return;

    const now = Date.now();
    const existingMessages = messages ?? [];
    const firstTurn = existingMessages.length === 0;

    const userMsg: Message = {
      id: newMessageId(),
      conversationId: conversation.id,
      role: "user",
      content: trimmed,
      createdAt: now,
      completedAt: now,
    };
    await appendMessage(userMsg);

    const titleUpdate =
      firstTurn && conversation.title === "(new chat)"
        ? deriveTitleShort(trimmed)
        : conversation.title;
    await putConversation({ ...conversation, title: titleUpdate, updatedAt: now });

    setDraft("");
    onFirstTurn?.(conversation.id);

    const openAIMessages = buildOpenAIMessages(
      conversation.systemPrompt,
      existingMessages,
      { role: "user", content: trimmed },
    );

    const assistantMsg: Message = {
      id: newMessageId(),
      conversationId: conversation.id,
      role: "assistant",
      content: "",
      modelAtSend: conversation.model,
      createdAt: now + 1,
      completedAt: null,
    };
    await appendMessage(assistantMsg);

    await runAssistantTurn(openAIMessages, assistantMsg);

    await putConversation({
      ...conversation,
      title: titleUpdate,
      updatedAt: Date.now(),
    });
  }, [conversation, draft, messages, onFirstTurn, runAssistantTurn, streaming]);

  const handleRegenerate = useCallback(async () => {
    if (streaming || !messages || messages.length === 0) return;
    const lastAssistantIndex = messages.map((m) => m.role).lastIndexOf("assistant");
    if (lastAssistantIndex < 0) return;
    const lastAssistant = messages[lastAssistantIndex];

    const prior = messages.slice(0, lastAssistantIndex);
    const lastUserIndex = prior.map((m) => m.role).lastIndexOf("user");
    if (lastUserIndex < 0) return;

    const historyBeforeUser = prior.slice(0, lastUserIndex);
    const lastUserMsg = prior[lastUserIndex];

    const openAIMessages = buildOpenAIMessages(
      conversation.systemPrompt,
      historyBeforeUser,
      { role: "user", content: lastUserMsg.content },
    );

    await deleteMessage(
      lastAssistant.conversationId,
      lastAssistant.createdAt,
      lastAssistant.id,
    );

    const assistantMsg: Message = {
      id: newMessageId(),
      conversationId: conversation.id,
      role: "assistant",
      content: "",
      modelAtSend: conversation.model,
      createdAt: Date.now(),
      completedAt: null,
    };
    await appendMessage(assistantMsg);

    await runAssistantTurn(openAIMessages, assistantMsg);

    await putConversation({ ...conversation, updatedAt: Date.now() });
  }, [conversation, messages, runAssistantTurn, streaming]);

  const handleStop = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const onComposerKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSend();
    }
  };

  const canRegenerate = useMemo(() => {
    if (streaming || !messages || messages.length === 0) return false;
    const hasAssistant = messages.some((m) => m.role === "assistant");
    const hasUser = messages.some((m) => m.role === "user");
    return hasAssistant && hasUser;
  }, [messages, streaming]);

  const lastAssistantId = useMemo(() => {
    if (!messages) return null;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return messages[i].id;
    }
    return null;
  }, [messages]);

  // Sparse "model:" ribbon under a bubble only when the model changes between
  // turns. The composer footer already shows the current model, so we don't
  // need to re-announce it on every assistant reply.
  const modelLabelAtIndex = useMemo(() => {
    if (!messages) return new Map<string, boolean>();
    const show = new Map<string, boolean>();
    let last: string | undefined;
    for (const m of messages) {
      if (m.role !== "assistant") continue;
      const mm = m.modelAtSend;
      if (mm && mm !== last) {
        show.set(m.id, true);
        last = mm;
      }
    }
    return show;
  }, [messages]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border-subtle px-3 sm:gap-3 sm:px-6">
        {onOpenSidebar && (
          <button
            type="button"
            onClick={onOpenSidebar}
            aria-label="Open conversation list"
            className="rounded-md p-1 text-fg-muted hover:bg-bg-card hover:text-fg md:hidden"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        )}
        <input
          type="text"
          value={conversation.title}
          onChange={(e) =>
            putConversation({
              ...conversation,
              title: e.target.value || "(untitled)",
              updatedAt: Date.now(),
            })
          }
          className="min-w-0 flex-1 rounded-md bg-transparent px-2 py-1 text-sm text-fg hover:bg-bg-card focus:bg-bg-card focus:outline-none"
          aria-label="Conversation title"
        />
        {conversation.systemPrompt && (
          <button
            type="button"
            onClick={() => setSystemPromptOpen(true)}
            className="rounded-md border border-border-subtle bg-bg-card px-2 py-1 text-[10px] uppercase tracking-wider text-fg-muted hover:border-border hover:bg-bg-hover"
            title={conversation.systemPrompt}
          >
            System prompt
          </button>
        )}
        <ConversationMenu
          items={[
            {
              label: conversation.systemPrompt
                ? "Edit system prompt"
                : "Set system prompt",
              onClick: () => setSystemPromptOpen(true),
            },
            { label: "Export as JSON", onClick: handleExport },
            {
              label: "Delete conversation",
              onClick: handleDelete,
              danger: true,
            },
          ]}
        />
      </header>

      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          const near = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          setFollow(near);
        }}
        className="min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto flex max-w-4xl flex-col gap-5 px-3 py-4 sm:px-6 sm:py-6">
          {messages === null ? (
            <div className="text-sm text-fg-subtle">Loading…</div>
          ) : messages.length === 0 ? (
            <EmptyConversation
              hasSystemPrompt={Boolean(conversation.systemPrompt)}
              onSetSystemPrompt={() => setSystemPromptOpen(true)}
            />
          ) : (
            messages.map((m) => (
              <MessageTurn
                key={m.id}
                message={m}
                showModelLabel={modelLabelAtIndex.get(m.id) === true}
                showRegenerate={m.id === lastAssistantId && canRegenerate}
                onRegenerate={handleRegenerate}
              />
            ))
          )}
          {error && (
            <div className="rounded-md border border-danger/30 bg-danger/5 p-3 text-xs text-danger">
              {error}
            </div>
          )}
        </div>
      </div>

      <footer className="border-t border-border-subtle bg-bg px-3 py-3 sm:px-6">
        <div className="mx-auto flex max-w-4xl flex-col gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onComposerKey}
            rows={3}
            placeholder="Message… (⌘/ctrl + enter)"
            className="w-full resize-none rounded-md border border-border-subtle bg-bg-card px-3 py-2 text-sm text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-2 gap-y-2">
            <ModelPicker
              value={conversation.model}
              onChange={handleModelChange}
            />
            <MaxTokensInput
              value={conversation.maxTokens}
              onChange={handleMaxTokensChange}
            />
            <div className="ml-auto flex items-center gap-2">
              {streaming && (
                <button
                  type="button"
                  onClick={handleStop}
                  className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-danger hover:text-danger"
                >
                  Stop
                </button>
              )}
              <button
                type="button"
                onClick={handleSend}
                disabled={!draft.trim() || streaming}
                className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </footer>

      <SystemPromptModal
        open={systemPromptOpen}
        value={conversation.systemPrompt ?? ""}
        onChange={handleSystemPromptChange}
        onClose={() => setSystemPromptOpen(false)}
      />
    </div>
  );
}

function MessageTurn({
  message,
  showModelLabel,
  showRegenerate,
  onRegenerate,
}: {
  message: Message;
  showModelLabel: boolean;
  showRegenerate: boolean;
  onRegenerate: () => void;
}) {
  if (message.role === "system") return null;
  const streaming = message.role === "assistant" && message.completedAt === null;

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-accent/15 px-4 py-2 text-sm text-fg ring-1 ring-inset ring-accent/30">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant: no bubble — full-column flow so markdown / code blocks breathe.
  return (
    <div className="flex flex-col gap-1.5">
      {showModelLabel && message.modelAtSend && (
        <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
          {message.modelAtSend}
        </div>
      )}
      <div className="text-sm leading-relaxed text-fg">
        <MarkdownMessage content={message.content || (streaming ? " " : "")} />
        {streaming && (
          <span
            aria-hidden
            className="ml-0.5 inline-block h-3 w-2 animate-pulse bg-fg align-middle"
          />
        )}
      </div>
      <div className="flex items-center gap-3 text-[10px] text-fg-subtle">
        {message.tokenCount !== undefined && (
          <span className="font-mono">{message.tokenCount} tok</span>
        )}
        {showRegenerate && (
          <button
            type="button"
            onClick={onRegenerate}
            className="rounded-md border border-border-subtle bg-bg-card px-2 py-0.5 text-[11px] font-normal text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
            title="Regenerate last turn with the current model"
          >
            Regenerate
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Compact "max tokens" override for the composer. Empty input → undefined →
 * field omitted from the OpenAI request (server default applies). Any positive
 * integer is sent verbatim. Negatives / zero / NaN snap back to undefined.
 */
function MaxTokensInput({
  value,
  onChange,
}: {
  value: number | undefined;
  onChange: (next: number | undefined) => void;
}) {
  const [draft, setDraft] = useState(value === undefined ? "" : String(value));

  // Re-sync if the conversation prop swaps (e.g., navigation between chats).
  useEffect(() => {
    setDraft(value === undefined ? "" : String(value));
  }, [value]);

  const commit = (raw: string) => {
    const trimmed = raw.trim();
    if (trimmed === "") {
      if (value !== undefined) onChange(undefined);
      return;
    }
    const n = Math.floor(Number(trimmed));
    if (!Number.isFinite(n) || n <= 0) {
      if (value !== undefined) onChange(undefined);
      setDraft(value === undefined ? "" : String(value));
      return;
    }
    if (n !== value) onChange(n);
  };

  return (
    <label
      className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-fg-subtle"
      title="Max tokens per response. Leave blank to use the server default."
    >
      <span>Tokens</span>
      <input
        type="number"
        inputMode="numeric"
        min={1}
        step={1}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit((e.target as HTMLInputElement).value);
            (e.target as HTMLInputElement).blur();
          }
        }}
        placeholder="auto"
        className="w-16 rounded-md border border-border-subtle bg-bg-card px-2 py-1 font-mono text-xs text-fg placeholder-fg-subtle hover:border-border focus:border-border focus:outline-none"
      />
    </label>
  );
}

function EmptyConversation({
  hasSystemPrompt,
  onSetSystemPrompt,
}: {
  hasSystemPrompt: boolean;
  onSetSystemPrompt: () => void;
}) {
  return (
    <div className="mx-auto max-w-md py-16 text-center">
      <p className="text-sm text-fg-muted">
        Type a message below to begin.
      </p>
      {!hasSystemPrompt && (
        <button
          type="button"
          onClick={onSetSystemPrompt}
          className="mt-3 text-xs text-fg-subtle underline-offset-2 hover:text-fg hover:underline"
        >
          Set a system prompt first
        </button>
      )}
    </div>
  );
}

function buildOpenAIMessages(
  systemPrompt: string | undefined,
  history: Message[],
  next: OpenAIMessage,
): OpenAIMessage[] {
  const out: OpenAIMessage[] = [];
  if (systemPrompt) {
    out.push({ role: "system", content: systemPrompt });
  }
  for (const m of history) {
    if (m.role === "system") continue;
    out.push({ role: m.role, content: m.content });
  }
  out.push(next);
  return out;
}

function deriveTitleShort(content: string): string {
  const cleaned = content.trim().replace(/\s+/g, " ");
  return cleaned.length <= 60 ? cleaned || "(untitled)" : cleaned.slice(0, 59) + "…";
}
