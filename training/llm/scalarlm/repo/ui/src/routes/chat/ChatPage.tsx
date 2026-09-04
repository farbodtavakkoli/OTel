import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { getApiConfig } from "@/api/config";
import {
  getConversation,
  newConversationId,
  type Conversation,
} from "@/stores/conversations";
import { useConversations } from "@/stores/useConversationStore";

import { ConversationList } from "./ConversationList";
import { ConversationView } from "./ConversationView";

/**
 * Top-level chat route. Sidebar + single-pane layout.
 *
 * Routing:
 *   /chat                    → landing (create a new conversation in memory)
 *   /chat/:conversationId    → specific conversation
 *   ?model=<hash>            → deep link: pre-select a model for the new chat.
 *                              Used by "Open in Chat" from TrainDetail.
 *
 * IndexedDB is the source of truth. Conversations only reach storage once
 * the first user message is sent — this keeps rapid-fire "click New chat"
 * navigations from littering the sidebar.
 */
export function ChatPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const [search] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const conversations = useConversations();

  const [active, setActive] = useState<Conversation | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Resolve the active conversation.
  useEffect(() => {
    let cancelled = false;

    async function resolve() {
      setLoadError(null);
      if (conversationId) {
        const existing = await getConversation(conversationId);
        if (cancelled) return;
        if (!existing) {
          setLoadError(`No conversation ${conversationId}`);
          setActive(null);
          return;
        }
        setActive(existing);
        return;
      }
      // No id in URL — spawn an in-memory draft, not persisted until first send.
      const modelFromQuery = search.get("model");
      const { default_model } = getApiConfig();
      setActive(buildDraft(modelFromQuery ?? default_model));
    }

    resolve();
    return () => {
      cancelled = true;
    };
  }, [conversationId, search]);

  const handleNew = useCallback(() => {
    // Replace `active` with a fresh draft BEFORE navigating. Without this,
    // when you click "+ New chat" from /chat/:id, there's a render where
    // `conversationId` is undefined but `active` still points at the old
    // conversation — and the persist-draft effect below then fires
    // navigate(`/chat/${active.id}`, {replace: true}), bouncing the URL
    // back. Setting active first means the subsequent effect finds an
    // active.id that isn't in the conversations list (a new draft), so
    // it stays quiet.
    const modelFromQuery = search.get("model");
    const { default_model } = getApiConfig();
    setActive(buildDraft(modelFromQuery ?? default_model));
    setLoadError(null);
    navigate("/chat");
  }, [navigate, search]);

  const handleFirstTurn = useCallback(
    async (id: string) => {
      // Persist the draft conversation the first time a user sends a message.
      if (!conversationId) {
        navigate(`/chat/${id}`, { replace: true });
      }
    },
    [conversationId, navigate],
  );

  // Persist draft conversations implicitly when the user edits them via the
  // header title or system-prompt editor BEFORE sending: ConversationView
  // writes through putConversation, which creates the record. Our live list
  // hook then picks it up.
  useEffect(() => {
    if (!conversationId && active && conversations) {
      const stored = conversations.find((c) => c.id === active.id);
      if (stored) navigate(`/chat/${active.id}`, { replace: true });
    }
  }, [active, conversationId, conversations, navigate]);

  return (
    <div className="relative flex h-full min-h-0">
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
        />
      )}
      <ConversationList
        onNew={handleNew}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="min-h-0 min-w-0 flex-1">
        {loadError ? (
          <div className="flex h-full items-center justify-center px-3 text-sm text-danger sm:px-6">
            {loadError}
          </div>
        ) : active ? (
          <ConversationView
            key={active.id}
            conversation={active}
            onFirstTurn={handleFirstTurn}
            onOpenSidebar={() => setSidebarOpen(true)}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-fg-subtle">
            Loading…
          </div>
        )}
      </main>
    </div>
  );
}

function buildDraft(model: string): Conversation {
  const now = Date.now();
  return {
    id: newConversationId(),
    title: "(new chat)",
    model,
    createdAt: now,
    updatedAt: now,
  };
}
