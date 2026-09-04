import { useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import clsx from "clsx";

import {
  deleteConversation,
  type Conversation,
} from "@/stores/conversations";
import { importConversationJson } from "@/stores/conversationIO";
import { useConversations } from "@/stores/useConversationStore";

interface ConversationListProps {
  onNew: () => void;
  /** Mobile drawer state. Ignored at md+ where the sidebar is always visible. */
  open?: boolean;
  /** Mobile-only: close the drawer after picking / creating / importing. */
  onClose?: () => void;
}

/**
 * Sidebar: groups existing conversations by recency bucket, highlights the
 * currently-viewed one, and offers delete with no confirmation — the design
 * has no server-side record so restoring requires the JSON export (future
 * feature). Matches the pattern in docs/ui-design.md §6.1.
 *
 * On mobile (<md) the sidebar becomes a slide-in drawer driven by `open`.
 * At md+ it is always visible as a 256px column.
 */
export function ConversationList({ onNew, open = false, onClose }: ConversationListProps) {
  const { conversationId } = useParams<{ conversationId: string }>();
  const navigate = useNavigate();
  const list = useConversations();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const groups = useMemo(() => groupByBucket(list ?? []), [list]);

  const handleDelete = async (id: string) => {
    await deleteConversation(id);
    if (id === conversationId) navigate("/chat");
  };

  const handleImport = async (file: File) => {
    setImportError(null);
    try {
      const imported = await importConversationJson(file);
      navigate(`/chat/${imported.id}`);
      onClose?.();
    } catch (err) {
      setImportError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <aside
      className={clsx(
        "fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-border-subtle bg-bg shadow-2xl transition-transform duration-200",
        "md:static md:z-auto md:w-64 md:shrink-0 md:translate-x-0 md:shadow-none",
        open ? "translate-x-0" : "-translate-x-full",
      )}
    >
      <div className="flex items-center justify-end px-3 pt-2 md:hidden">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close sidebar"
          className="rounded-md p-1 text-fg-subtle hover:bg-bg-card hover:text-fg"
        >
          ✕
        </button>
      </div>
      <div className="flex flex-col gap-2 border-b border-border-subtle p-3">
        <button
          type="button"
          onClick={() => {
            onNew();
            onClose?.();
          }}
          className="w-full rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
        >
          + New chat
        </button>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleImport(file);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full rounded-md border border-border-subtle bg-bg-card px-3 py-1 text-xs text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
            title="Import from JSON"
          >
            Import…
          </button>
          {importError && (
            <div className="mt-1 break-words text-[11px] text-danger" role="alert">
              {importError}
            </div>
          )}
        </div>
      </div>
      <nav className="min-h-0 flex-1 overflow-y-auto">
        {list === null ? (
          <div className="p-3 text-xs text-fg-subtle">Loading…</div>
        ) : list.length === 0 ? (
          <div className="p-3 text-xs text-fg-subtle">No conversations yet.</div>
        ) : (
          groups.map((g) => (
            <section key={g.label}>
              <h3 className="px-3 pb-1 pt-3 text-[10px] uppercase tracking-wider text-fg-subtle">
                {g.label}
              </h3>
              {g.items.map((c) => (
                <ConversationRow
                  key={c.id}
                  conversation={c}
                  active={c.id === conversationId}
                  onSelect={() => {
                    navigate(`/chat/${c.id}`);
                    onClose?.();
                  }}
                  onDelete={() => handleDelete(c.id)}
                />
              ))}
            </section>
          ))
        )}
      </nav>
    </aside>
  );
}

function ConversationRow({
  conversation,
  active,
  onSelect,
  onDelete,
}: {
  conversation: Conversation;
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  return (
    <div
      className={clsx(
        "group flex items-start gap-2 px-3 py-1.5 hover:bg-bg-card",
        active && "bg-bg-card",
      )}
    >
      <button
        type="button"
        onClick={onSelect}
        className="min-w-0 flex-1 text-left"
      >
        <div className="truncate text-sm text-fg">{conversation.title}</div>
        <div className="truncate font-mono text-[10px] text-fg-subtle">
          {conversation.model.length > 20
            ? `${conversation.model.slice(0, 14)}…`
            : conversation.model}
        </div>
      </button>
      <button
        type="button"
        onClick={onDelete}
        aria-label="Delete conversation"
        className="rounded px-1 text-fg-subtle opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger group-hover:opacity-100"
        title="Delete"
      >
        ✕
      </button>
    </div>
  );
}

interface Group {
  label: string;
  items: Conversation[];
}

function groupByBucket(list: Conversation[]): Group[] {
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);

  const today: Conversation[] = [];
  const yesterday: Conversation[] = [];
  const lastWeek: Conversation[] = [];
  const older: Conversation[] = [];

  for (const c of list) {
    const age = c.updatedAt;
    if (age >= midnight.getTime()) today.push(c);
    else if (age >= midnight.getTime() - dayMs) yesterday.push(c);
    else if (age >= now - 7 * dayMs) lastWeek.push(c);
    else older.push(c);
  }

  const result: Group[] = [];
  if (today.length) result.push({ label: "Today", items: today });
  if (yesterday.length) result.push({ label: "Yesterday", items: yesterday });
  if (lastWeek.length) result.push({ label: "Last 7 days", items: lastWeek });
  if (older.length) result.push({ label: "Older", items: older });
  return result;
}
