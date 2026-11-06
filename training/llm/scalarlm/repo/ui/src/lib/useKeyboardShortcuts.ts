import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export function useKeyboardShortcuts(options: {
  onShowCheatsheet: () => void;
}) {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    let gPending = false;
    let gTimer: ReturnType<typeof setTimeout> | null = null;

    const expireG = () => {
      gPending = false;
      gTimer = null;
    };

    const isEditable = (el: EventTarget | null) => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
      if (el.isContentEditable) return true;
      return false;
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditable(e.target)) return;

      if (gPending) {
        const target = keyToRoute(e.key);
        if (target) {
          e.preventDefault();
          navigate(target);
        }
        if (gTimer) clearTimeout(gTimer);
        expireG();
        return;
      }

      switch (e.key) {
        case "g":
          e.preventDefault();
          gPending = true;
          gTimer = setTimeout(expireG, 1500);
          return;
        case "?": {
          e.preventDefault();
          options.onShowCheatsheet();
          return;
        }
        case "y": {
          e.preventDefault();
          navigator.clipboard?.writeText(window.location.href).catch(() => {
          });
          return;
        }
        case "/": {
          const input = document.querySelector<HTMLInputElement>(
            'input[type="search"], [data-search="true"]',
          );
          if (input) {
            e.preventDefault();
            input.focus();
          }
          return;
        }
        default:
          return;
      }
    };

    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      if (gTimer) clearTimeout(gTimer);
    };
  }, [navigate, location.pathname, options]);
}

function keyToRoute(key: string): string | null {
  switch (key) {
    case "c":
      return "/chat";
    case "t":
      return "/train";
    case "m":
      return "/metrics";
    case "o":
      return "/models";
    case "s":
      return "/settings";
    case "h":
      return "/";
    default:
      return null;
  }
}
