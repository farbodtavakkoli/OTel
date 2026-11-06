const DB_NAME = "scalarlm-chat";
const DB_VERSION = 1;
const STORE_CONV = "conversations";
const STORE_MSG = "messages";

export interface Conversation {
  id: string;
  title: string;
  model: string;
  systemPrompt?: string;
  maxTokens?: number;
  createdAt: number;
  updatedAt: number;
}

export interface Message {
  id: string;
  conversationId: string;
  role: "user" | "assistant" | "system";
  content: string;
  modelAtSend?: string;
  createdAt: number;
  completedAt: number | null;
  tokenCount?: number;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_CONV)) {
        const s = db.createObjectStore(STORE_CONV, { keyPath: "id" });
        s.createIndex("updatedAt", "updatedAt", { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_MSG)) {
        const s = db.createObjectStore(STORE_MSG, {
          keyPath: ["conversationId", "createdAt", "id"],
        });
        s.createIndex("conversationId", "conversationId", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IDB open failed"));
  });
  return dbPromise;
}

function promisify<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IDB op failed"));
  });
}

function txDone(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IDB transaction failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IDB transaction aborted"));
  });
}

type Listener = () => void;
const listeners = new Set<Listener>();

function notify() {
  for (const l of listeners) {
    try {
      l();
    } catch {
    }
  }
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export async function listConversations(): Promise<Conversation[]> {
  const db = await openDB();
  const tx = db.transaction(STORE_CONV, "readonly");
  const all = await promisify(tx.objectStore(STORE_CONV).getAll() as IDBRequest<Conversation[]>);
  all.sort((a, b) => b.updatedAt - a.updatedAt);
  return all;
}

export async function getConversation(
  id: string,
): Promise<Conversation | null> {
  const db = await openDB();
  const tx = db.transaction(STORE_CONV, "readonly");
  const value = await promisify(
    tx.objectStore(STORE_CONV).get(id) as IDBRequest<Conversation | undefined>,
  );
  return value ?? null;
}

export async function putConversation(conv: Conversation): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_CONV, "readwrite");
  tx.objectStore(STORE_CONV).put(conv);
  await txDone(tx);
  notify();
}

export async function deleteConversation(id: string): Promise<void> {
  const db = await openDB();
  const tx = db.transaction([STORE_CONV, STORE_MSG], "readwrite");
  tx.objectStore(STORE_CONV).delete(id);
  const msgStore = tx.objectStore(STORE_MSG);
  const range = IDBKeyRange.bound([id, -Infinity, ""], [id, Infinity, "\uffff"]);
  await new Promise<void>((resolve, reject) => {
    const cursor = msgStore.openCursor(range);
    cursor.onsuccess = () => {
      const c = cursor.result;
      if (c) {
        c.delete();
        c.continue();
      } else {
        resolve();
      }
    };
    cursor.onerror = () => reject(cursor.error);
  });
  notify();
}

export async function listMessages(
  conversationId: string,
): Promise<Message[]> {
  const db = await openDB();
  const tx = db.transaction(STORE_MSG, "readonly");
  const index = tx.objectStore(STORE_MSG).index("conversationId");
  const messages = await promisify(
    index.getAll(IDBKeyRange.only(conversationId)) as IDBRequest<Message[]>,
  );
  messages.sort((a, b) => a.createdAt - b.createdAt);
  return messages;
}

export async function appendMessage(message: Message): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_MSG, "readwrite");
  await promisify(tx.objectStore(STORE_MSG).put(message));
  notify();
}

export async function updateMessage(
  message: Message,
): Promise<void> {
  return appendMessage(message);
}

export async function deleteMessage(
  conversationId: string,
  createdAt: number,
  id: string,
): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_MSG, "readwrite");
  await promisify(
    tx.objectStore(STORE_MSG).delete([conversationId, createdAt, id]),
  );
  notify();
}

export function newConversationId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function newMessageId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `m-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function deriveTitle(content: string, limit = 60): string {
  const cleaned = content.trim().replace(/\s+/g, " ");
  if (cleaned.length <= limit) return cleaned || "(empty)";
  return cleaned.slice(0, limit - 1) + "…";
}
