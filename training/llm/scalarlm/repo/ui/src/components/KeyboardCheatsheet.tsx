import { Modal } from "./Modal";

interface KeyboardCheatsheetProps {
  open: boolean;
  onClose: () => void;
}

const SHORTCUTS: { keys: string[]; description: string }[] = [
  { keys: ["g", "c"], description: "Go to Chat" },
  { keys: ["g", "t"], description: "Go to Train" },
  { keys: ["g", "m"], description: "Go to Metrics" },
  { keys: ["g", "o"], description: "Go to Models" },
  { keys: ["g", "s"], description: "Go to Settings" },
  { keys: ["g", "h"], description: "Go home" },
  { keys: ["/"], description: "Focus search input on current page" },
  { keys: ["y"], description: "Copy current URL" },
  { keys: ["?"], description: "Show this cheatsheet" },
  { keys: ["⌘/Ctrl", "↵"], description: "Send (in chat composer, submit modal)" },
  { keys: ["Esc"], description: "Close modals" },
];

export function KeyboardCheatsheet({ open, onClose }: KeyboardCheatsheetProps) {
  return (
    <Modal open={open} title="Keyboard shortcuts" onClose={onClose} maxWidth="max-w-lg">
      <ul className="flex flex-col gap-1.5 text-sm">
        {SHORTCUTS.map((s) => (
          <li key={s.keys.join(" ")} className="flex items-center justify-between gap-4">
            <span className="text-fg-muted">{s.description}</span>
            <span className="flex items-center gap-1">
              {s.keys.map((k, i) => (
                <kbd
                  key={i}
                  className="rounded-md border border-border-subtle bg-bg px-1.5 py-0.5 font-mono text-[11px] text-fg"
                >
                  {k}
                </kbd>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </Modal>
  );
}
