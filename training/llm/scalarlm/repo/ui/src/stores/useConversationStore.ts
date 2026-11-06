import { useEffect, useState } from "react";

import {
  listConversations,
  listMessages,
  subscribe,
  type Conversation,
  type Message,
} from "./conversations";

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
