import { useEffect, useState } from "react";

import {
  listConversations,
  listMessages,
  subscribe,
  type Conversation,
  type Message,
} from "./conversations";

/** Hook: live list of all conversations, most recent first. */
export function useConversations() {
  const [data, setData] = useState<Conversation[] | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      listConversations()
        .then((rows) => {
          if (active) setData(rows);
        })
        .catch(() => {
          if (active) setData([]);
        });
    };
    refresh();
    return subscribe(refresh);
  }, []);

  return data;
}

/** Hook: live message list for a conversation, oldest first. */
export function useMessages(conversationId: string | undefined) {
  const [data, setData] = useState<Message[] | null>(null);

  useEffect(() => {
    if (!conversationId) {
      setData([]);
      return;
    }
    let active = true;
    const refresh = () => {
      listMessages(conversationId)
        .then((rows) => {
          if (active) setData(rows);
        })
        .catch(() => {
          if (active) setData([]);
        });
    };
    refresh();
    return subscribe(refresh);
  }, [conversationId]);

  return data;
}
