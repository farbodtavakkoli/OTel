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
    const modelFromQuery = search.get("model");
    const { default_model } = getApiConfig();
    setActive(buildDraft(modelFromQuery ?? default_model));
    setLoadError(null);
    navigate("/chat");
  }, [navigate, search]);

  const handleFirstTurn = useCallback(
    async (id: string) => {
      if (!conversationId) {
        navigate(`/chat/${id}`, { replace: true });
      }
    },
    [conversationId, navigate],
  );

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
