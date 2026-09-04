import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

import { useKeyboardShortcuts } from "@/lib/useKeyboardShortcuts";

import { KeyboardCheatsheet } from "./KeyboardCheatsheet";

const navItems = [
  { to: "/chat", label: "Chat" },
  { to: "/train", label: "Train" },
  { to: "/inference", label: "Inference" },
  { to: "/metrics", label: "Metrics" },
  { to: "/models", label: "Models" },
];

export function AppLayout() {
  const [cheatsheetOpen, setCheatsheetOpen] = useState(false);
  useKeyboardShortcuts({ onShowCheatsheet: () => setCheatsheetOpen(true) });

  return (
    <div className="flex h-full flex-col">
      <TopBar onShowCheatsheet={() => setCheatsheetOpen(true)} />
      <main className="min-h-0 flex-1 overflow-auto">
        <Outlet />
      </main>
      <KeyboardCheatsheet
        open={cheatsheetOpen}
        onClose={() => setCheatsheetOpen(false)}
      />
    </div>
  );
}

function GitHubIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="14"
      height="14"
      fill="currentColor"
      aria-hidden
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function TopBar({ onShowCheatsheet }: { onShowCheatsheet: () => void }) {
  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-border-subtle bg-bg px-4">
      <div className="flex items-center gap-6">
        <NavLink to="/" className="text-sm font-semibold tracking-tight">
          ScalarLM
        </NavLink>
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  "rounded-md px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "bg-bg-hover text-fg"
                    : "text-fg-muted hover:bg-bg-card hover:text-fg",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-1">
        <a
          href="https://scalarlm.com"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md px-2 py-1.5 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
          title="Documentation"
        >
          Docs
        </a>
        <a
          href="https://github.com/supermassive-intelligence/scalarlm"
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-md px-2 py-1.5 text-fg-muted hover:bg-bg-card hover:text-fg"
          aria-label="GitHub repository"
          title="GitHub repository"
        >
          <GitHubIcon />
        </a>
        <button
          type="button"
          onClick={onShowCheatsheet}
          className="rounded-md px-2 py-1.5 text-xs text-fg-muted hover:bg-bg-card hover:text-fg"
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          ?
        </button>
        <NavLink
          to="/settings"
          className="rounded-md px-2 py-1.5 text-sm text-fg-muted hover:bg-bg-card hover:text-fg"
          aria-label="Settings"
          title="Settings"
        >
          ⚙
        </NavLink>
      </div>
    </header>
  );
}
