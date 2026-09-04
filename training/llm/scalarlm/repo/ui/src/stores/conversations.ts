/**
 * IndexedDB-backed conversation + message store.
 *
 * Two object stores:
 *   - "conversations" keyed by id; secondary index on updatedAt (desc in UI via sort)
 *   - "messages" keyed by [conversationId, createdAt, id] (array keyPath) with
 *      a secondary index on conversationId for cursor-range reads
 *
 * Design notes:
 *   - Metadata is small; putting everything in IDB (not splitting conversation
 *     titles into localStorage) keeps the storage story simple.
 *   - Listener pattern: every write fires a subscriber callback so React state
 *     can resync without polling. Keeps the component layer free of
 *     per-operation cache invalidation logic.
 *   - Wraps each IDBRequest in a Promise. No library.
 */

const DB_NAME = "scalarlm-chat";
const DB_VERSION = 1;
const STORE_CONV = "conversations";
const STORE_MSG = "messages";

export interface Conversation {
  id: string;
  title: string;
  model: string;
  systemPrompt?: string;
  /**
   * Per-conversation max_tokens for completion requests. When undefined the
   * field is omitted from the OpenAI-shape request body and the server's
   * default applies.
   */
  maxTokens?: number;
  createdAt: number;
  updatedAt: number;
}

export interface Message {
  id: string;
  conversationId: string;
  role: "user" | "assistant" | "system";
  content: string;
  /** Model that generated this assistant turn; undefined for user/system rows. */
  modelAtSend?: string;
  createdAt: number;
  /** Set when the message is fully streamed (or failed); in-flight messages have null. */
  completedAt: number | null;
  /** Populated from the terminal usage block of the stream when available. */
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

/**
 * Wait for an IDBTransaction to commit. Transactions emit `complete`, not
 * `success` — do NOT feed one into promisify().
 */
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
      // subscriber threw — don't let it take out others
    }
  }
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

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
  // Delete every message row within this conversation via the cursor range.
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

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

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
  // The compound keyPath means put() replaces the exact row.
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

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

export function newConversationId(): string {
  // crypto.randomUUID is available in all evergreen browsers and secure
  // contexts (including localhost). Fall back to a low-entropy id otherwise.
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

/** Derive a short title from the first user message. */
export function deriveTitle(content: string, limit = 60): string {
  const cleaned = content.trim().replace(/\s+/g, " ");
  if (cleaned.length <= limit) return cleaned || "(empty)";
  return cleaned.slice(0, limit - 1) + "…";
}
