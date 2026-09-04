# UI Design

## Status

Shipped. The React SPA lives in `ui/` and is mounted at `/app/*` by FastAPI via `infra/cray_infra/api/fastapi/setup_ui.py::add_ui()`. The `chat-ui/` directory and its SvelteKit-based HF chat-ui fork have been removed from the repo. Chat, training, metrics, and inference surfaces are all served from the React app under one origin.

This document is kept as the design record for *why* the UI looks the way it does, not as a migration plan.

## Goals

- **One process, one origin, one build.** A single `./scalarlm up` produces a container that serves chat, training, metrics, and API traffic from the same Uvicorn process. Nothing spawns a subprocess for the UI.
- **Drop the Node runtime.** The final image contains no Node.js. The UI bundle is a directory of hashed static assets. An operator who pulls a ScalarLM image doesn't need to know what SvelteKit is.
- **Match the closed-loop thesis.** Submitting a training job and trying the resulting model happens in the same browser tab against the same origin, with no iframe or cross-app redirect.
- **Cluster-operator grade metrics.** The metrics page replaces `scalarlm squeue` + `/v1/generate/metrics` + `scalarlm plot` for the common case. Debuggable from a browser without SSH.
- **Stay light.** Dependency budget: < 5 MB gzipped JS, < 500 KB CSS, across all routes. React + Vite + TanStack Query is the target stack; anything heavier needs justification.
- **First-paint under 300 ms on a warm CDN.** Static assets are content-hashed and cacheable forever; the HTML shell is ~3 KB.

## Non-Goals

- Authentication and multi-tenant authorization. ScalarLM is operator-deployed; the UI inherits whatever network-level controls sit in front of the API pod. A future layer can add OAuth.
- A Hub browser. Model discovery happens on huggingface.co; the UI only lists models already known to this deployment.
- A prompt-library or agent-evaluation product.
- A mobile-optimized layout. Touch targets should work; the primary form factor is a laptop.
- Server-side rendering. Everything is a client-rendered SPA. The first request returns `index.html`; the app hydrates from API calls after that.

---

## 1. Architecture

### 1.1 Shape

```
        browser
           │
           ▼
  ┌────────────────────────────────────┐
  │  FastAPI :8000  (single origin,    │
  │                  single process)   │
  │                                    │
  │  GET /                 → 302 /app/ │
  │  GET /app/*            → static    │
  │                          (React    │
  │                          SPA bundle│
  │                          on disk)  │
  │  GET/POST/... /v1/*    → API       │
  │                          handlers  │
  └────────────────────────────────────┘
                │
                ▼
     /app/ui-bundle/       (built at image build time)
       index.html
       assets/
         index-{hash}.js
         index-{hash}.css
         ...
```

Everything the pod needs to serve the UI is a directory of files on disk, mounted by FastAPI's `StaticFiles` middleware. No subprocess, no port 3000, no port 3100. The pod exposes only 8000 (and 8001 internally for vLLM); a browser pointed at the API URL gets the UI and the API from the same place.

### 1.2 Serving the bundle

`infra/cray_infra/api/fastapi/main.py` gains a static-file mount. The important properties:

- **Exact-asset requests** (hashed JS/CSS) — serve with `Cache-Control: public, max-age=31536000, immutable`. Safe because the filename contains the content hash.
- **`/app/index.html`** — serve with `Cache-Control: no-cache`. Always fresh so the latest asset manifest is visible.
- **Unmatched `/app/*` paths** — SPA history-mode routing, so `/app/train/abc123` must return `index.html` (200) rather than 404. The React router then renders the correct route client-side.
- **`/app/api-config.json`** — a tiny dynamic endpoint rendered by FastAPI (not a static file) that the SPA fetches on boot to learn `{api_base, version, features}`. Lets the same bundle run against any deployment; nothing about the API URL gets baked into JS at build time.

Conceptual wiring (in `main.py`):

```python
app.mount("/app/assets",
          StaticFiles(directory="/app/ui-bundle/assets", max_age=31536000),
          name="ui-assets")

@app.get("/app/api-config.json")
async def api_config():
    return {"api_base": "/v1", "version": VERSION, "features": {...}}

@app.get("/app/{full_path:path}")
async def ui_spa_fallback(full_path: str):
    # Serve index.html for any unmatched /app/* path (SPA history routing)
    return FileResponse("/app/ui-bundle/index.html",
                        headers={"Cache-Control": "no-cache"})

@app.get("/")
async def root():
    return RedirectResponse("/app/")
```

### 1.3 Routing reference

| URL | Served by | Handler |
|---|---|---|
| `/` | FastAPI | 302 to `/app/` |
| `/app/` | FastAPI | `FileResponse(index.html)` |
| `/app/assets/*` | FastAPI `StaticFiles` | Hashed immutable assets |
| `/app/api-config.json` | FastAPI | Dynamic small JSON |
| `/app/chat` | FastAPI | `FileResponse(index.html)` → React route |
| `/app/chat/{conversation_id}` | FastAPI | Same — React route |
| `/app/train` | FastAPI | Same |
| `/app/train/{job_hash}` | FastAPI | Same |
| `/app/metrics` | FastAPI | Same |
| `/app/models` | FastAPI | Same |
| `/app/settings` | FastAPI | Same |
| `/v1/*` | FastAPI | API handlers |

The `/chat/*` prefix that the old HF chat-ui claimed is no longer served — chat lives at `/app/chat`. Deep links into the old chat-ui return 404.

---

## 2. Chat Client Decisions

### 2.1 Conversation persistence

No server-side store. Three tiers:

- **`sessionStorage`** — current draft message only. Survives a page reload, not a tab close.
- **`localStorage`** — conversation metadata (id, title, model, last-updated, message count). Small, synchronous, quota-friendly.
- **IndexedDB** — full message bodies. Larger quota, async. Wrapped in a tiny helper that makes read/write feel synchronous-ish via React Suspense.

Backups are out of scope for v1. Users concerned about losing chat history can export a conversation to JSON via a download button; import reverses it.

Decision to *not* add a backend conversation store: it would require a migration plan, a schema, authorization, and a database. The backend ScalarLM already runs is the inference engine; conversations are a UI concern.

### 2.2 Streaming

The HF chat-ui did its own SSE parsing over a WebSocket-flavored protocol. The React app uses the browser's native `fetch` + `ReadableStream` API to read the standard OpenAI `/v1/chat/completions` streaming response:

```ts
const resp = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true }),
});
const reader = resp.body!.pipeThrough(new TextDecoderStream()).getReader();

for await (const chunk of streamSSEChunks(reader)) {
    const delta = JSON.parse(chunk).choices[0].delta.content ?? "";
    appendToCurrentMessage(delta);
}
```

`streamSSEChunks` is a small utility (~30 lines) that parses `data: ...\n\n` framing out of a text stream. No library. Cancellation is via `AbortController` — clicking Stop in the UI aborts the fetch.

### 2.3 What users lose

Honest list of features that don't carry over:

- Assistants / custom-persona library → users save a system-prompt string per conversation instead.
- Multi-user with profiles → no users.
- Web search tool → out of scope.
- Shared conversation links → conversations are client-local; sharing means exporting JSON.
- MongoDB-backed persistence → local browser storage.

If any of these turn out to matter in practice, they're individually addressable in later phases. None is load-bearing for the closed-loop training flow.

---

## 3. Technology Choices

| Concern | Choice | Why |
|---|---|---|
| Framework | **React 18** + TypeScript | Largest ecosystem for the component patterns we need (virtualized lists, forms, charts). Team familiarity. |
| Bundler | **Vite** | Fast dev server, small build output, first-class TS/React support. Same toolchain the rest of the JS world has converged on. |
| Routing | **React Router v6** (data-router API) | History-mode client routing matches the SPA fallback on the server side. The loader/action API gives us route-scoped data fetching without Redux. |
| Data fetching | **TanStack Query (React Query v5)** | Built-in caching, polling, retry, focus refetch, request deduplication. The whole UI's refresh strategy (§5) is configured declaratively through it. |
| State (local UI) | Zustand | Tiny (~1 KB), no boilerplate, no Context ceremony. Used only for ephemeral UI state (drawer open?, current chat draft). All server state is in React Query. |
| Styling | Tailwind CSS | Atomic CSS, no component-library weight. A small `~8 KB` custom component set built on top (Button, Badge, Card). |
| Charts | [uPlot](https://github.com/leeoniya/uPlot) for timeseries; small hand-rolled SVG for bar/gauge | uPlot renders 100k-point loss curves without dropping frames. Plotly/Chart.js are 10× heavier than we need. |
| Forms | React Hook Form + Zod | Schema-validated `train_args` form. The Zod schema mirrors `JobConfig` exactly. |
| Markdown/code rendering | `react-markdown` + `shiki` (pre-compiled themes) | Syntax highlighting in chat without a WASM build. |
| Build size target | < 5 MB gzipped total, < 500 KB first-load | CI-enforced via `vite-bundle-visualizer` + size budget. |

Not chosen and rejected:

- **Next.js** — we don't want server-rendering or a Node runtime. Pure SPA is the right fit.
- **Redux** — React Query covers server-state needs; Zustand covers the rest.
- **Material UI / Chakra / Ant** — bundle weight, design lock-in, and we'd override enough of the defaults to not get value.
- **SvelteKit** — the HF chat-ui uses it, but we're explicitly migrating off SvelteKit; React's larger ecosystem for the specific components we need (uPlot bindings, virtualization, form libraries) wins.

---

## 4. Repo Layout

New top-level directory, replacing `chat-ui/` and `frontend/`:

```
scalarlm/
├── ui/                           # NEW — the React SPA
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── index.html                # Single HTML shell
│   └── src/
│       ├── main.tsx              # Entrypoint
│       ├── App.tsx               # Router + layout shell
│       ├── api/
│       │   ├── client.ts         # fetch wrapper, api-config loader
│       │   ├── openai.ts         # /v1/chat/completions streaming
│       │   ├── training.ts       # /v1/megatron/* wrappers
│       │   ├── metrics.ts        # /v1/generate/metrics
│       │   └── health.ts         # /v1/health
│       ├── routes/
│       │   ├── chat/
│       │   │   ├── ChatLayout.tsx
│       │   │   ├── ConversationList.tsx
│       │   │   ├── ConversationView.tsx
│       │   │   └── MessageStream.tsx
│       │   ├── train/
│       │   │   ├── TrainIndex.tsx
│       │   │   ├── TrainDetail.tsx
│       │   │   ├── SubmitModal.tsx
│       │   │   ├── LossChart.tsx
│       │   │   └── LogPane.tsx
│       │   ├── metrics/
│       │   │   ├── MetricsIndex.tsx
│       │   │   ├── ThroughputCard.tsx
│       │   │   ├── QueueCard.tsx
│       │   │   ├── CapacityCard.tsx
│       │   │   └── HealthCard.tsx
│       │   └── models/
│       │       └── ModelsIndex.tsx
│       ├── components/           # Shared UI primitives
│       │   ├── Badge.tsx
│       │   ├── Button.tsx
│       │   ├── Card.tsx
│       │   ├── StatusBadge.tsx
│       │   ├── CopyButton.tsx
│       │   ├── ShowSDKCode.tsx
│       │   └── ErrorState.tsx
│       ├── stores/               # Local (non-server) state
│       │   ├── conversations.ts  # IndexedDB wrapper
│       │   └── preferences.ts    # Theme, polling cadence
│       └── lib/
│           ├── sse.ts            # SSE parser
│           └── schema.ts         # Zod schemas mirroring JobConfig
│
└── (chat-ui/ and frontend/ directories removed)
```

No MongoDB, no `entrypoint.sh` for the UI, no `.env.local`. Configuration comes from `/app/api-config.json` at runtime.

---

## 5. Data Layer

### 5.1 Query client setup

One `QueryClient` at the app root. Global defaults:

```ts
new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 10_000,
            gcTime: 5 * 60_000,
            retry: (failureCount, error) =>
                failureCount < 3 && error.status !== 404,
            refetchOnWindowFocus: true,
        },
    },
});
```

Per-query overrides express the refresh strategy:

| Query | `staleTime` | `refetchInterval` | Notes |
|---|---|---|---|
| `GET /v1/models` | 30 s | 60 s | Revalidates when a new training job completes. |
| `GET /v1/megatron/squeue` | 0 | 3 s | Queue view is "live". |
| `GET /v1/generate/metrics` | 0 | 3 s | Throughput card. |
| `GET /v1/megatron/train/{hash}` | 0 | 2 s while `QUEUED\|TRAINING`, 30 s otherwise | Adapts via the query's `refetchInterval` function. |
| `GET /v1/megatron/gpu_count` | 5 min | — | Rarely changes. |
| `GET /v1/megatron/node_count` | 5 min | — | Rarely changes. |
| `GET /v1/health` | 10 s | 10 s | Health dots. |
| `GET /app/api-config.json` | Infinity | — | Loaded once at boot. |

Tab-in-background: TanStack Query's `refetchIntervalInBackground: false` (default) already pauses polling when the tab isn't visible.

### 5.2 Streaming reads

Two streams, both using `fetch` + `ReadableStream`:

- **Chat completions** — `POST /v1/chat/completions` with `stream: true`. Custom hook `useChatCompletion(messages, model)` returns an `AsyncIterable<string>` of content deltas. The UI appends to the current assistant message as deltas arrive.
- **Training logs** — `GET /v1/megatron/train/logs/{model}`. Currently returns SSE; the UI uses `EventSource` for this (no auth headers needed, native reconnection). When auth lands, swap to `fetch` + body-stream reader — an abstraction layer in `lib/sse.ts` hides the choice.

### 5.3 Errors

Every query has three rendering paths, encoded once in a shared `<QueryBoundary>` component:

```tsx
<QueryBoundary
    query={query}
    loading={<Skeleton />}
    error={(err) => <ErrorState error={err} onRetry={query.refetch} />}
>
    {(data) => <Actual view={data} />}
</QueryBoundary>
```

- 404 → friendly empty state + link to list view.
- 5xx → last-known data kept rendered, banner "Stale data since {time}", retry button.
- Network failure → full-page error with copy-to-clipboard for the failing URL.

### 5.4 Mutations

Mutations (submit training job, cancel, delete, set alias) go through `useMutation` with optimistic updates where safe:

- Cancel → immediate status badge flip to `CANCELLING`, reconciled on next poll.
- Delete → immediate removal from list, rolled back on error.
- Submit → optimistic entry in list with `QUEUED` badge while the request is in flight.

---

## 6. Chat UI

The chat UI is a React surface at `/app/chat` and `/app/chat/{conversation_id}`.

### 6.1 Layout

```
┌──────────────────────────────────────────────────────────────┐
│ ScalarLM   Chat  Train  Metrics  Models              ⚙       │
├──────────────┬───────────────────────────────────────────────┤
│              │                                                │
│ + New chat   │      Conversation title (click to rename)     │
│              │      Model: gemma-3-4b-it ▾                   │
│ ─────────    │      ────────────────────────────────────    │
│              │                                                │
│ Recent       │      [user]      What is 2 + 2?              │
│  • Today     │      [asst]      4                            │
│    - chat 1  │      [user]      Again, but explain.          │
│    - chat 2  │      [asst]      (streaming response...)      │
│  • Yesterday │                                                │
│    ...       │                                                │
│              │                                                │
│              │      ──────────────────────────────────────   │
│              │      [ Type a message...              ] [Send]│
└──────────────┴───────────────────────────────────────────────┘
```

Left sidebar: conversation list from IndexedDB, grouped by "Today / Yesterday / Last 7 days / Older". Each entry shows the first user message as preview. Right main area: current conversation.

### 6.2 Conversation model

```ts
type Conversation = {
    id: string;                    // uuid
    title: string;                 // user-edited or auto-derived from first message
    model: string;                 // HF model ID or training job hash
    systemPrompt?: string;
    createdAt: number;
    updatedAt: number;
};
type Message = {
    conversationId: string;
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    createdAt: number;
    tokenCount?: number;
};
```

`Conversation` records live in localStorage keyed by `conv:{id}`. `Message` records live in IndexedDB in an object store with a compound index on `(conversationId, createdAt)`. Retrieval for a conversation is a cursor range.

### 6.3 Model picker

Sources from `GET /v1/models`. The picker shows:

- The base model at the top (from `/app/api-config.json:default_model`), labeled "(default)".
- Trained adapters below, sorted by registration time (newest first), shown as "{nickname or hash[:8]} · trained {relative_time}".
- A "latest" special entry that maps to `model: "latest"` on the API side.

Switching models on an existing conversation starts a new turn with the new model; prior messages keep the model they were generated with (annotated in the UI).

### 6.4 System prompt

Above the first message, a collapsed "System prompt: (none)" bar that expands to an editable textarea. Change triggers a new message in the conversation; old messages retain the prior prompt. Simple, no template library.

### 6.5 Streaming display

Uses the `useChatCompletion` hook from §5.2. As tokens stream in:

- Markdown is re-parsed incrementally (react-markdown re-renders on each chunk — fine for chat-sized messages).
- Code blocks are highlighted once the triple-backtick close fence is seen (don't highlight mid-stream).
- A subtle caret `▌` at the end of the current assistant message until streaming completes.
- `Stop generating` button wired to `AbortController`.

### 6.6 Tool use / function calling

Out of scope for v1 as a first-class UI. The `POST /v1/chat/completions` endpoint already supports `tools` and `tool_choice`; a future iteration can add a tool editor. Today, the chat UI doesn't surface tools.

### 6.7 Deep link from Training

`/app/train/{hash}` detail view has "Open in Chat" — creates a new conversation with `model: hash`, navigates to `/app/chat/{new_conversation_id}`. Zero-click path from "training completed" to "try it out."

---

## 7. Training UI

Same information architecture as the prior revision — what changes is that it's React, in the same app as chat, sharing the chrome and auth-less origin.

### 7.1 List — `/app/train`

Cards sourced by joining two queries:

- `GET /v1/megatron/list_models` — every completed training job directory with a `.pt` file.
- `GET /v1/megatron/squeue` — running and queued SLURM jobs.

Merged, de-duped, sorted by `start_time` desc, rendered with the shared `StatusBadge` component. Filter controls for status, text search on nickname / hash.

Top-right: `+ New training job` opens `SubmitModal`.

### 7.2 Submit — `SubmitModal.tsx`

Two input modes:

- Drag-drop JSONL file → multipart POST with upload progress (`XMLHttpRequest` used once specifically for `progress` events, since `fetch` lacks upload progress in most browsers).
- Paste key/value pairs → UI builds JSONL client-side and submits.

`train_args` form derives from a Zod schema that mirrors `JobConfig`. Defaults come from the previous successful submission (in localStorage). JSON-edit escape hatch under an "Advanced" disclosure.

On success, navigate to `/app/train/{returned_hash}` and begin polling.

### 7.3 Detail — `/app/train/{job_hash}`

- **Header**: nickname (inline edit) → `PUT /v1/megatron/train/{hash}/alias`, status badge, runtime, actions (Cancel, Delete, Open in Chat for COMPLETED).
- **Config panel**: collapsible YAML view of `config.yaml` from `GET /v1/megatron/train/{hash}`.
- **Loss chart**: `LossChart.tsx` wraps uPlot; feeds it the `history` array from the status endpoint. Step vs. loss, log-scale toggle, per-epoch vertical markers. Re-fed on every poll while `TRAINING`.
- **Live logs**: `LogPane.tsx` — `EventSource` on `/v1/megatron/train/logs/{hash}`, virtualized list (react-virtuoso or a hand-rolled windowing), pause/resume auto-scroll, find-in-page with regex, download-as-text, "jump to last error".
- **Adapter info**: `adapter_type`, `registered-with-vllm` (from `VLLMModelManager`), `Open in Chat` CTA.

### 7.4 Actions

- Cancel: `POST /v1/megatron/cancel/{hash}` + optimistic UI → `CANCELLING` badge.
- Delete: confirm by typing hash prefix → `POST /v1/megatron/delete/{hash}` → remove from list.
- Download checkpoint: link to a future `GET /v1/megatron/train/{hash}/checkpoint/{step}` endpoint.

---

## 8. Metrics UI

Single page at `/app/metrics`, four stacked cards. Same structure as the prior revision:

### 8.1 Throughput card

Reads `GET /v1/generate/metrics` on a 3 s poll. Renders:

- `token/s`, `request/s`, `flop/s`, `queue_depth` as large numbers.
- A sparkline of `token/s` over the last N samples (client-side rolling buffer, N defaults to 100 ≈ 5 min).

Tooltip on the `token/s` number explicitly says: "Throughput while the queue was non-empty. Not wall-clock throughput." This prevents the classic misreading of the metric (see `docs/inference-queue.md` §6.2).

### 8.2 Queue card

`GET /v1/megatron/squeue`. Big numbers for running/queued counts. Sortable table below with links to `/app/train/{hash}`.

### 8.3 Capacity card

Two horizontal gauges:

- Inference GPUs used (from a new `GET /v1/vllm/stats` — not yet implemented; the vLLM fork already exposes `get_current_kv_cache_size`, this endpoint wraps it).
- Training GPUs used (parse `--gres=gpu:N` summed from `squeue` rows).

Both over `GET /v1/megatron/gpu_count`, with `node_count` as a secondary stat.

### 8.4 Health card

Three indicator dots (api, vllm, slurm) from a structured `GET /v1/health` response. Red dot click opens a modal with the raw JSON + copy button.

### 8.5 Time range

Page-header dropdown ("Last 5 min / 15 min / 1 h") adjusts the sparkline buffer depth. Client-side only — no server-side history store in v1.

---

## 9. Shared Design System

### 9.1 Chrome

```
┌─────────────────────────────────────────────────────────────┐
│ ScalarLM    Chat   Train   Metrics   Models         ⚙       │
└─────────────────────────────────────────────────────────────┘
```

Fixed top bar on every `/app/*` route. The "ScalarLM" wordmark links to `/app/` (the original design referenced `frontend/assets/logo.svg` but the shipped chrome uses text). Settings gear opens a drawer with: server health summary, deployment version, default model, an `SCALARLM_API_URL` copy-button for external SDK use, a theme toggle.

### 9.2 Tokens

- **Typography**: `Inter` for UI, `JetBrains Mono` for code and hash-like identifiers. Self-hosted from `/app/assets/fonts/`.
- **Colors**: dark mode default. Two accent colors — blue for in-progress/primary, red for destructive. Everything else grayscale.
- **Radii**: 6px default. Rounded-full only for status dots and avatars.
- **Spacing**: 4/8/12/16/24/32 px scale via Tailwind.

### 9.3 Components with cross-page reuse

| Component | Usage |
|---|---|
| `StatusBadge` | `TrainingJobStatus` everywhere; API health dots |
| `CopyButton` | Job hashes, model names, SDK snippets, config YAML lines |
| `ShowSDKCode` | On every mutation/object view — "Show equivalent Python SDK call" |
| `ErrorState` | The standard error render surface for every route-level query |
| `ConfirmDestructive` | Cancel/delete modals requiring prefix-typed confirmation |
| `CodeBlock` | Read-only syntax-highlighted code (JSON, YAML, Python, bash) |

### 9.4 Keyboard

- `g c` → chat
- `g t` → train
- `g m` → metrics
- `?` → show keyboard cheatsheet
- `/` → focus the search box on list pages
- `cmd/ctrl+enter` → submit in forms
- `y` → copy current URL

---

## 10. Real-Time Updates

Three patterns, same as before, chosen intentionally:

| Pattern | Where | Transport |
|---|---|---|
| Event stream | Training logs | `EventSource` on `/v1/megatron/train/logs/{model}` |
| Token stream | Chat response | `fetch` body-stream reader on `/v1/chat/completions` |
| Short-poll | Everything else | TanStack Query `refetchInterval` |

No WebSockets. No long-poll. Everything traverses standard HTTP and Cloudflare tunnels without accommodation. `EventSource` for logs is chosen for native auto-reconnect; when auth arrives, the abstraction in `lib/sse.ts` swaps it for a fetch-based reader (which supports custom headers) without touching callers.

---

## 11. Deployment

### 11.1 Dockerfile

Replace the existing `ui_base`/`ui_builder` stages (for HF chat-ui) with a single React build stage:

```Dockerfile
FROM node:24.2.0 AS ui_builder
WORKDIR /build
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ .
RUN npm run build       # Vite outputs to /build/dist

# In the final stage:
COPY --from=ui_builder /build/dist /app/ui-bundle
```

No Node, `npx`, `mongod`, or `dotenv-cli` in the final image. The final image's delta vs. today: **removed** ~400 MB of Node + chat-ui dependencies, **added** ~3 MB of hashed static assets.

### 11.2 FastAPI wiring

One-time addition in `infra/cray_infra/api/fastapi/main.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

UI_BUNDLE = "/app/ui-bundle"

app.mount("/app/assets",
          StaticFiles(directory=f"{UI_BUNDLE}/assets", max_age=31536000),
          name="ui-assets")

@app.get("/")
async def root():
    return RedirectResponse("/app/", status_code=302)

@app.get("/app/api-config.json")
async def api_config():
    return {"api_base": "/v1",
            "version": VERSION,
            "default_model": get_config()["model"],
            "features": {}}

@app.get("/app/{full_path:path}")
async def ui_spa_fallback(full_path: str):
    return FileResponse(f"{UI_BUNDLE}/index.html",
                        headers={"Cache-Control": "no-cache"})
```

Because the SPA fallback is a catch-all, it must come **after** any more-specific routes (api-config, assets mount). FastAPI's routing order is declaration order, so the `/app/assets` mount and `/app/api-config.json` must register before the `/app/{full_path:path}` handler.

### 11.3 docker-compose.yaml

The dev compose service exposes only ports 8000 (API + UI) and 8001 (vLLM). Source directories (`infra/`, `ml/`, `scripts/`, `test/`, `models/`) are bind-mounted for hot-edit.

### 11.4 Helm

No changes. The `api_deployment.yaml` already exposes port 8000; that's the only port externally visible.

### 11.5 Dev workflow

```
cd ui
SCALARLM_API_URL=http://localhost:8000 npm run dev    # Vite dev server on :5173
```

`vite.config.ts` proxies `/v1/*` and `/app/api-config.json` to the configured API URL. Hot module reload, React DevTools, etc. The production build is exercised in CI via `npm run build && npm run preview`.

Running the full stack locally is still `./scalarlm up` — the built UI is baked into the image.

---

## 12. Rollout

Shipped. Metrics, Train (list + detail + submit), Chat, Inference Request Browser, and Settings are all live in `ui/` and served from `/app/*`. Ongoing work is incremental — adding routes / cards / polishing — and does not need a phased plan.

---

## 13. Backend Dependencies

UI surfaces that depend on backend additions:

| Endpoint | Purpose | Status |
|---|---|---|
| `GET /v1/health` returning `{api, vllm, slurm}` dict | Health card granularity | Live |
| `GET /v1/vllm/stats` returning `{gpus_used, kv_free, kv_total}` | Capacity card | Capacity card degrades gracefully if missing |
| `GET /v1/megatron/train/{hash}/checkpoint/{step}` | Checkpoint download | Not yet implemented; UI hides the action when absent |

Aliases are client-local (per-browser localStorage) rather than a server endpoint — see §5.4 / §9.3.

---

## 14. Open Questions

- **Persistent metrics history.** Throughput sparkline is client-local; refreshing loses history. A later revision could add a server-side ring buffer or a Prometheus sidecar. Out of scope for v1.
- **Multi-pod Helm deployments with more than 1 API replica.** Static-file serving from every replica is fine, but the SPA fallback needs each replica to have the bundle on disk at the same path. Today the image bakes it in, so this is uncomplicated; worth double-checking if anyone ever splits the image.
- **Source maps in production.** Vite ships source maps by default. Do we want to include them in the production bundle (for debuggability) or strip them (for size)? Default: include but serve with `access denied` if an auth layer ever exists.
- **Clearing local data.** A "Clear all local data" button in settings wipes IndexedDB + localStorage. Needed for users sharing a browser or debugging stuck state.
- **Partial-response recovery on chat.** If the browser tab closes mid-stream, the in-flight assistant message is orphaned. Do we resume on return? Simpler answer: drop it and show a "Regenerate" button. Yes.
- **Alias namespace collisions.** Two users might set the same nickname on different jobs. Since there are no users, the last write wins. If auth lands, aliases become user-scoped.
