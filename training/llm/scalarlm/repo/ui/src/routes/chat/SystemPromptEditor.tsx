import { useEffect, useState } from "react";

import { Modal } from "@/components/Modal";

interface SystemPromptModalProps {
  open: boolean;
  value: string;
  onChange: (value: string) => void;
  onClose: () => void;
}

/**
 * Modal system-prompt editor, opened from the conversation overflow menu.
 * Keeps the main chat surface free of editorial chrome; the prompt still
 * applies per-conversation (docs/ui-design.md §6.4).
 */
export function SystemPromptModal({
  open,
  value,
  onChange,
  onClose,
}: SystemPromptModalProps) {
  const [draft, setDraft] = useState(value);

  // Sync when opened so an earlier cancel doesn't leak stale edits.
  useEffect(() => {
    if (open) setDraft(value);
  }, [open, value]);

  const save = () => {
    onChange(draft.trim());
    onClose();
  };

  return (
    <Modal
      open={open}
      title="System prompt"
      onClose={onClose}
      maxWidth="max-w-2xl"
      footer={
        <>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover"
          >
            Cancel
          </button>
          {draft && (
            <button
              type="button"
              onClick={() => {
                onChange("");
                onClose();
              }}
              className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={save}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover"
          >
            Save
          </button>
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <p className="text-xs text-fg-muted">
          Applied as the first message on every turn. Leave blank to omit it
          from the completion request entirely.
        </p>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          autoFocus
          rows={8}
          placeholder="You are a helpful assistant…"
          className="min-h-[160px] w-full resize-y rounded-md border border-border-subtle bg-bg px-3 py-2 text-sm text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
        />
      </div>
    </Modal>
  );
}
