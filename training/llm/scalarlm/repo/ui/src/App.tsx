import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";

import { AppLayout } from "./components/AppLayout";
import { HomePage } from "./routes/home/HomePage";
import { InferenceBrowserPage } from "./routes/inference/InferenceBrowserPage";
import { TrainIndex } from "./routes/train/TrainIndex";
import { TrainDetail } from "./routes/train/TrainDetail";
import { MetricsPage } from "./routes/metrics/MetricsPage";
import { ModelsPage } from "./routes/models/ModelsPage";
import { SettingsPage } from "./routes/settings/SettingsPage";
import { NotFound } from "./routes/NotFound";

// The chat surface pulls in react-markdown + remark-gfm (~50 KB gzipped),
// and it's not needed for any other route. Split it so the first paint on
// /train, /metrics, etc. doesn't pay that cost.
const ChatPage = lazy(() =>
  import("./routes/chat/ChatPage").then((m) => ({ default: m.ChatPage })),
);

export function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route
          path="chat"
          element={
            <Suspense fallback={<RouteLoading />}>
              <ChatPage />
            </Suspense>
          }
        />
        <Route
          path="chat/:conversationId"
          element={
            <Suspense fallback={<RouteLoading />}>
              <ChatPage />
            </Suspense>
          }
        />
        <Route path="train" element={<TrainIndex />} />
        <Route path="train/:jobHash" element={<TrainDetail />} />
        <Route path="inference" element={<InferenceBrowserPage />} />
        <Route path="metrics" element={<MetricsPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

function RouteLoading() {
  return (
    <div className="flex h-full items-center justify-center text-sm text-fg-subtle">
      Loading…
    </div>
  );
}
