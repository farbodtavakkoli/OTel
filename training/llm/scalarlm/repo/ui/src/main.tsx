import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import "./styles.css";
import { App } from "./App";
import { loadApiConfig } from "./api/config";
import { ApiError } from "./api/client";
import { initTheme } from "./stores/theme";

// Set up OS-preference listener + re-apply in case the index.html inline
// script raced (or the saved preference changed since). Idempotent.
initTheme();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      gcTime: 5 * 60_000,
      retry: (failureCount, error: unknown) => {
        const status = error instanceof ApiError ? error.status : undefined;
        return failureCount < 3 && status !== 404;
      },
      refetchOnWindowFocus: true,
      refetchIntervalInBackground: false,
    },
  },
});

// Block first paint on the runtime config so components can read it
// synchronously via getApiConfig(). On failure we fall back to defaults
// (see api/config.ts) and render anyway.
loadApiConfig().finally(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter basename="/app">
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </React.StrictMode>,
  );
});
