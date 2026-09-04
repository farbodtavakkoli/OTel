/**
 * Publish-to-HuggingFace modal.
 *
 * Phase 1: submit a SLURM publish job, poll its status.json, render
 * phase progress, allow cancellation. Log tailing comes in Phase 2.
 *
 * See ui/docs/publish-to-hf.md for the full design.
 */

import { useEffect, useId, useRef, useState } from "react";

import type { LogLine } from "@/api/training";
import {
  isTerminalPhase,
  tailPublishLog,
  useCancelPublish,
  useCheckpoints,
  usePublishStatus,
  useSubmitPublish,
  type CheckpointEntry,
  type PublishMode,
  type PublishPhase,
  type PublishStatus,
} from "@/api/publish";

interface PublishToHFModalProps {
  open: boolean;
  jobHash: string;
  onClose: () => void;
}

// Loose client-side guard: must contain exactly one `/`, both halves
// non-empty, and only word-ish characters. The server forwards to HF
// which is the authority on naming, so we don't try to reproduce
// every quirk of org-vs-repo allowed-character lists here.
const REPO_ID_RE = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

export function PublishToHFModal({
  open,
  jobHash,
  onClose,
}: PublishToHFModalProps) {
  // Default to adapter — smaller, faster, more commonly useful. Users
  // who specifically need a self-contained model flip to merged.
  const [mode, setMode] = useState<PublishMode>("adapter");
  const [repoId, setRepoId] = useState("");
  const [isPrivate, setIsPrivate] = useState(false);
  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loraAlphaOverride, setLoraAlphaOverride] = useState<string>("");
  const [commitMessage, setCommitMessage] = useState<string>("");

  // True once the user has submitted in *this* modal session. Polling
  // status.json before submit would 404 (or return a stale prior run).
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const submit = useSubmitPublish(jobHash);
  const cancel = useCancelPublish(jobHash);
  const status = usePublishStatus(jobHash, open && hasSubmitted);

  const { data: checkpoints, isPending: ckptPending, error: ckptError } =
    useCheckpoints(open ? jobHash : undefined);

  // Default to the latest checkpoint once the list arrives.
  useEffect(() => {
    if (!checkpoints || checkpoints.length === 0) return;
    if (selectedCheckpoint === "") {
      setSelectedCheckpoint(checkpoints[0].name);
    }
  }, [checkpoints, selectedCheckpoint]);

  // Wipe sensitive state and per-session flags when closing.
  useEffect(() => {
    if (!open) {
      setToken("");
      setShowToken(false);
      setHasSubmitted(false);
      setAdvancedOpen(false);
      setLoraAlphaOverride("");
      setCommitMessage("");
      submit.reset();
      cancel.reset();
    }
    // submit/cancel intentionally not in deps — calling .reset() inside
    // a deps array on those would re-fire on every mutation update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Esc to close (but only when there's no in-flight job — closing
  // mid-submit would lose the SSE handle; the user must explicitly
  // cancel or wait).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        const phase = status.data?.phase;
        if (hasSubmitted && phase && !isTerminalPhase(phase)) return;
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, hasSubmitted, status.data?.phase]);

  const repoIdValid = repoId === "" || REPO_ID_RE.test(repoId);

  const parsedLoraAlphaForValidation =
    loraAlphaOverride.trim() === ""
      ? undefined
      : Number.parseInt(loraAlphaOverride, 10);
  const loraAlphaIsInvalid =
    loraAlphaOverride.trim() !== "" &&
    (Number.isNaN(parsedLoraAlphaForValidation) ||
      (parsedLoraAlphaForValidation as number) < 1);

  const formReady =
    repoId !== "" &&
    repoIdValid &&
    token !== "" &&
    selectedCheckpoint !== "" &&
    !loraAlphaIsInvalid &&
    !submit.isPending;

  const tokenInputId = useId();
  const repoInputId = useId();
  const ckptInputId = useId();

  const inFlightPhase = status.data?.phase;
  const isLive = hasSubmitted && !isTerminalPhase(inFlightPhase);

  const handleSubmit = async () => {
    try {
      await submit.mutateAsync({
        mode,
        repo_id: repoId,
        private: isPrivate,
        hf_token: token,
        checkpoint: selectedCheckpoint || undefined,
        // Advanced overrides — only sent when populated. Server defaults
        // (latest checkpoint metadata for lora_alpha, generated commit
        // message) apply otherwise.
        lora_alpha:
          parsedLoraAlphaForValidation !== undefined && !loraAlphaIsInvalid
            ? parsedLoraAlphaForValidation
            : undefined,
        commit_message:
          commitMessage.trim() === "" ? undefined : commitMessage,
      });
      // Once submitted, drop the token from React state so it can't
      // leak into a console.log of the component tree.
      setToken("");
      setHasSubmitted(true);
    } catch {
      // Error surfaces via submit.error in the UI.
    }
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="publish-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={() => {
        if (!isLive) onClose();
      }}
    >
      <div
        className="flex w-full max-w-lg flex-col gap-4 rounded-lg border border-border bg-bg-card p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4">
          <div>
            <h2
              id="publish-modal-title"
              className="text-sm font-semibold text-fg"
            >
              Publish to HuggingFace
            </h2>
            <p className="mt-0.5 text-xs text-fg-muted">
              Job <code className="font-mono">{jobHash.slice(0, 8)}</code>
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isLive}
            title={isLive ? "Cancel the publish first" : "Close"}
            aria-label="Close"
            className="rounded-md border border-border-subtle bg-bg px-2 py-1 text-xs text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg disabled:cursor-not-allowed disabled:opacity-40"
          >
            Close
          </button>
        </header>

        {hasSubmitted ? (
          <PublishProgress
            jobHash={jobHash}
            status={status.data}
            error={status.error as Error | null}
            cancelPending={cancel.isPending}
            onCancel={() => cancel.mutate()}
            onClose={onClose}
          />
        ) : (
          <PublishForm
            mode={mode}
            setMode={setMode}
            repoId={repoId}
            setRepoId={setRepoId}
            repoIdValid={repoIdValid}
            isPrivate={isPrivate}
            setIsPrivate={setIsPrivate}
            token={token}
            setToken={setToken}
            showToken={showToken}
            setShowToken={setShowToken}
            tokenInputId={tokenInputId}
            repoInputId={repoInputId}
            ckptInputId={ckptInputId}
            checkpoints={checkpoints}
            ckptPending={ckptPending}
            ckptError={ckptError as Error | null}
            selectedCheckpoint={selectedCheckpoint}
            setSelectedCheckpoint={setSelectedCheckpoint}
            advancedOpen={advancedOpen}
            setAdvancedOpen={setAdvancedOpen}
            loraAlphaOverride={loraAlphaOverride}
            setLoraAlphaOverride={setLoraAlphaOverride}
            loraAlphaInvalid={loraAlphaIsInvalid}
            commitMessage={commitMessage}
            setCommitMessage={setCommitMessage}
            submitError={submit.error as Error | null}
            formReady={formReady}
            submitPending={submit.isPending}
            onCancel={onClose}
            onSubmit={handleSubmit}
          />
        )}

        {/* Visually-hidden status for screen readers + e2e selectors. */}
        <span className="sr-only" aria-live="polite">
          {hasSubmitted
            ? `Publish phase ${inFlightPhase ?? "loading"}`
            : formReady
            ? "Form complete"
            : "Form incomplete"}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pre-submit form
// ---------------------------------------------------------------------------

interface PublishFormProps {
  mode: PublishMode;
  setMode: (m: PublishMode) => void;
  repoId: string;
  setRepoId: (s: string) => void;
  repoIdValid: boolean;
  isPrivate: boolean;
  setIsPrivate: (v: boolean) => void;
  token: string;
  setToken: (s: string) => void;
  showToken: boolean;
  setShowToken: (v: boolean) => void;
  tokenInputId: string;
  repoInputId: string;
  ckptInputId: string;
  checkpoints: CheckpointEntry[] | undefined;
  ckptPending: boolean;
  ckptError: Error | null;
  selectedCheckpoint: string;
  setSelectedCheckpoint: (s: string) => void;
  advancedOpen: boolean;
  setAdvancedOpen: (v: boolean) => void;
  loraAlphaOverride: string;
  setLoraAlphaOverride: (s: string) => void;
  loraAlphaInvalid: boolean;
  commitMessage: string;
  setCommitMessage: (s: string) => void;
  submitError: Error | null;
  formReady: boolean;
  submitPending: boolean;
  onCancel: () => void;
  onSubmit: () => void;
}

function PublishForm(p: PublishFormProps) {
  return (
    <>
      <ModeSelector mode={p.mode} setMode={p.setMode} />

      <div className="flex flex-col gap-1.5">
        <label htmlFor={p.repoInputId} className="text-xs text-fg-muted">
          Repository
        </label>
        <input
          id={p.repoInputId}
          type="text"
          placeholder="myorg/my-adapter"
          value={p.repoId}
          onChange={(e) => p.setRepoId(e.target.value)}
          className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 font-mono text-sm text-fg focus:border-accent focus:outline-none"
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
        />
        {!p.repoIdValid && (
          <p className="text-[11px] text-danger">
            Use the form <code className="font-mono">org/repo-name</code>.
            Letters, digits, <code>.</code>, <code>-</code>, <code>_</code>.
          </p>
        )}
        <label className="flex items-center gap-2 text-xs text-fg-muted">
          <input
            type="checkbox"
            checked={p.isPrivate}
            onChange={(e) => p.setIsPrivate(e.target.checked)}
            className="h-3 w-3 accent-accent"
          />
          Create as private (existing repo visibility wins)
        </label>
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor={p.tokenInputId} className="text-xs text-fg-muted">
          HuggingFace token{" "}
          <span className="text-fg-subtle">(write access)</span>
        </label>
        <div className="flex items-center gap-1.5">
          <input
            id={p.tokenInputId}
            type={p.showToken ? "text" : "password"}
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
            data-1p-ignore
            data-lpignore="true"
            placeholder="hf_***"
            value={p.token}
            onChange={(e) => p.setToken(e.target.value)}
            className="flex-1 rounded-md border border-border-subtle bg-bg px-3 py-1.5 font-mono text-sm text-fg focus:border-accent focus:outline-none"
          />
          <button
            type="button"
            onClick={() => p.setShowToken(!p.showToken)}
            className="rounded-md border border-border-subtle bg-bg px-2 py-1.5 text-[11px] text-fg-muted hover:border-border hover:text-fg"
          >
            {p.showToken ? "Hide" : "Show"}
          </button>
        </div>
        <p className="text-[11px] text-fg-subtle">
          Generate at{" "}
          <a
            href="https://huggingface.co/settings/tokens"
            target="_blank"
            rel="noopener noreferrer"
            className="underline-offset-2 hover:underline"
          >
            huggingface.co/settings/tokens
          </a>
          . Held only in this tab; not saved.
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <label htmlFor={p.ckptInputId} className="text-xs text-fg-muted">
          Checkpoint
        </label>
        <select
          id={p.ckptInputId}
          value={p.selectedCheckpoint}
          onChange={(e) => p.setSelectedCheckpoint(e.target.value)}
          disabled={!p.checkpoints || p.checkpoints.length === 0}
          className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg focus:border-accent focus:outline-none disabled:opacity-50"
        >
          {p.ckptPending && <option>Loading…</option>}
          {p.checkpoints &&
            p.checkpoints.map((c, i) => (
              <option key={c.name} value={c.name}>
                {formatCheckpointLabel(c, i === 0)}
              </option>
            ))}
          {p.checkpoints && p.checkpoints.length === 0 && (
            <option value="">No checkpoints written yet</option>
          )}
        </select>
        {p.ckptError && (
          <p className="text-[11px] text-danger">
            Couldn't load checkpoints: {String(p.ckptError.message)}
          </p>
        )}
      </div>

      <AdvancedOptions
        mode={p.mode}
        open={p.advancedOpen}
        setOpen={p.setAdvancedOpen}
        loraAlphaOverride={p.loraAlphaOverride}
        setLoraAlphaOverride={p.setLoraAlphaOverride}
        loraAlphaInvalid={p.loraAlphaInvalid}
        commitMessage={p.commitMessage}
        setCommitMessage={p.setCommitMessage}
      />

      {p.submitError && (
        <p
          role="alert"
          className="rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-[11px] text-danger"
        >
          {String(p.submitError.message)}
        </p>
      )}

      <footer className="flex justify-end gap-2 pt-2">
        <button
          type="button"
          onClick={p.onCancel}
          className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={p.onSubmit}
          disabled={!p.formReady}
          className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
        >
          {p.submitPending ? "Submitting…" : "Publish"}
        </button>
      </footer>
    </>
  );
}

// ---------------------------------------------------------------------------
// Post-submit progress view
// ---------------------------------------------------------------------------

const PHASE_LABEL: Record<PublishPhase, string> = {
  queued: "Waiting for SLURM…",
  validating: "Reading checkpoint…",
  loading_base: "Loading base model…",
  merging: "Merging LoRA into base…",
  saving: "Saving model files…",
  uploading: "Uploading to HuggingFace…",
  done: "Done.",
  error: "Failed.",
};

const PHASE_ORDER: PublishPhase[] = [
  "queued",
  "validating",
  "loading_base",
  "merging",
  "saving",
  "uploading",
  "done",
];

interface PublishProgressProps {
  jobHash: string;
  status: PublishStatus | undefined;
  error: Error | null;
  cancelPending: boolean;
  onCancel: () => void;
  onClose: () => void;
}

function PublishProgress({
  jobHash,
  status,
  error,
  cancelPending,
  onCancel,
  onClose,
}: PublishProgressProps) {
  if (error && !status) {
    return (
      <div
        role="alert"
        className="rounded-md border border-danger/30 bg-danger/5 px-3 py-3 text-sm text-danger"
      >
        Couldn't load publish status: {String(error.message)}
      </div>
    );
  }

  if (!status) {
    return (
      <div className="rounded-md border border-border-subtle bg-bg/50 px-3 py-3 text-sm text-fg-muted">
        Submitting…
      </div>
    );
  }

  const phase = status.phase;
  const isError = phase === "error";
  const isDone = phase === "done";
  const inFlight = !isTerminalPhase(phase);

  const currentIdx = isError
    ? -1
    : PHASE_ORDER.indexOf(phase);

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-md border border-border-subtle bg-bg/50 p-3">
        <div className="flex items-center gap-2">
          <PhaseDot phase={phase} />
          <span className="text-sm text-fg">{PHASE_LABEL[phase]}</span>
        </div>
        <ol className="mt-3 flex flex-col gap-1 text-[11px] font-mono text-fg-muted">
          {PHASE_ORDER.filter((p) => p !== "done").map((p, i) => {
            const reached = !isError && i <= currentIdx;
            const active = !isError && p === phase;
            return (
              <li
                key={p}
                className={
                  active
                    ? "text-accent"
                    : reached
                    ? "text-fg"
                    : "text-fg-subtle"
                }
              >
                {reached ? "✓" : active ? "▸" : "·"} {p.replace(/_/g, " ")}
              </li>
            );
          })}
        </ol>
      </div>

      {status.repo_id && (
        <p className="text-xs text-fg-muted">
          Repository:{" "}
          <code className="font-mono">{status.repo_id}</code>
        </p>
      )}

      {isDone && status.repo_url && (
        <p className="rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success">
          Pushed to{" "}
          <a
            href={status.repo_url}
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono underline-offset-2 hover:underline"
          >
            {status.repo_url.replace(/^https?:\/\//, "")}
          </a>
        </p>
      )}

      {isError && (
        <p
          role="alert"
          className="rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-[11px] text-danger"
        >
          {status.error || "Publish failed."}
        </p>
      )}

      <PublishLogTail jobHash={jobHash} forceOpen={isError} />

      <footer className="flex justify-end gap-2 pt-1">
        {inFlight && (
          <button
            type="button"
            onClick={onCancel}
            disabled={cancelPending}
            title="Cancels the SLURM job. A partial upload may remain on huggingface.co."
            className="rounded-md border border-danger/40 bg-danger/10 px-3 py-1.5 text-sm text-danger hover:border-danger hover:bg-danger/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {cancelPending ? "Cancelling…" : "Cancel"}
          </button>
        )}
        {!inFlight && (
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover"
          >
            Close
          </button>
        )}
      </footer>
    </div>
  );
}

/**
 * Live tail of the publish job's publish.log. Collapsed by default;
 * the modal opens it automatically on phase=error so the user lands
 * on the actual stderr without an extra click.
 */
function PublishLogTail({
  jobHash,
  forceOpen,
}: {
  jobHash: string;
  forceOpen: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [lines, setLines] = useState<LogLine[]>([]);
  const scrollRef = useRef<HTMLPreElement>(null);
  // Auto-open on error.
  useEffect(() => {
    if (forceOpen) setOpen(true);
  }, [forceOpen]);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLines([]);
    void tailPublishLog({
      jobHash,
      signal: controller.signal,
      onLine: (ln) =>
        setLines((prev) => {
          const next =
            prev.length >= 4_000 ? prev.slice(prev.length - 4_000 + 1) : prev.slice();
          next.push(ln);
          return next;
        }),
    });
    return () => controller.abort();
  }, [jobHash, open]);

  // Auto-scroll to bottom as new lines arrive.
  useEffect(() => {
    if (!open) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines, open]);

  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="rounded-md border border-border-subtle bg-bg/50"
    >
      <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-fg-muted hover:text-fg">
        publish.log{lines.length > 0 ? ` (${lines.length} lines)` : ""}
      </summary>
      <pre
        ref={scrollRef}
        className="m-0 max-h-64 overflow-auto whitespace-pre-wrap break-all border-t border-border-subtle px-2 py-1 font-mono text-[11px] leading-snug text-fg"
      >
        {lines.length === 0
          ? "Waiting for log output…"
          : lines.map((l) => l.line).join("\n")}
      </pre>
    </details>
  );
}

function PhaseDot({ phase }: { phase: PublishPhase }) {
  const cls =
    phase === "done"
      ? "bg-success"
      : phase === "error"
      ? "bg-danger"
      : "bg-accent animate-pulse";
  return <span className={`h-2 w-2 rounded-full ${cls}`} aria-hidden />;
}

// ---------------------------------------------------------------------------
// Mode selector — unchanged from Phase 0 except labels match the new
// Mode 1 (PEFT adapter, vs the old "raw checkpoint" framing).
// ---------------------------------------------------------------------------

interface AdvancedOptionsProps {
  mode: PublishMode;
  open: boolean;
  setOpen: (v: boolean) => void;
  loraAlphaOverride: string;
  setLoraAlphaOverride: (s: string) => void;
  loraAlphaInvalid: boolean;
  commitMessage: string;
  setCommitMessage: (s: string) => void;
}

/**
 * Phase 3 advanced-options pane. lora_alpha override is only
 * meaningful for merged mode — adapter-mode publishes the alpha that
 * was actually trained with (encoded in the adapter_config), so an
 * override there would silently be ignored. Hide it when adapter
 * mode is selected to avoid the confusing-no-op.
 */
function AdvancedOptions({
  mode,
  open,
  setOpen,
  loraAlphaOverride,
  setLoraAlphaOverride,
  loraAlphaInvalid,
  commitMessage,
  setCommitMessage,
}: AdvancedOptionsProps) {
  return (
    <details
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
      className="rounded-md border border-border-subtle bg-bg/50"
    >
      <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-fg-muted hover:text-fg">
        Advanced
      </summary>
      <div className="flex flex-col gap-3 border-t border-border-subtle px-3 py-3">
        {mode === "merged" ? (
          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-fg-muted">
              lora_alpha override{" "}
              <span className="text-fg-subtle">
                (defaults to checkpoint metadata, then 2×rank)
              </span>
            </label>
            <input
              type="number"
              min={1}
              step={1}
              placeholder="e.g. 32"
              value={loraAlphaOverride}
              onChange={(e) => setLoraAlphaOverride(e.target.value)}
              className="w-32 rounded-md border border-border-subtle bg-bg px-3 py-1.5 font-mono text-sm text-fg focus:border-accent focus:outline-none"
              spellCheck={false}
            />
            {loraAlphaInvalid && (
              <p className="text-[11px] text-danger">
                Must be a positive integer.
              </p>
            )}
          </div>
        ) : (
          <p className="text-[11px] text-fg-subtle">
            <span className="font-semibold">lora_alpha</span> is only
            adjustable in <span className="font-semibold">merged</span>{" "}
            mode. Adapter-mode exports preserve the value the trainer
            wrote.
          </p>
        )}

        <div className="flex flex-col gap-1.5">
          <label className="text-xs text-fg-muted">
            Commit message{" "}
            <span className="text-fg-subtle">(optional)</span>
          </label>
          <input
            type="text"
            placeholder="Auto-generated from job + checkpoint when blank"
            value={commitMessage}
            onChange={(e) => setCommitMessage(e.target.value)}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg focus:border-accent focus:outline-none"
            maxLength={500}
          />
        </div>
      </div>
    </details>
  );
}

function ModeSelector({
  mode,
  setMode,
}: {
  mode: PublishMode;
  setMode: (m: PublishMode) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2 rounded-md border border-border-subtle bg-bg/50 p-3">
      <legend className="px-1 text-xs text-fg-muted">Mode</legend>
      <ModeRadio
        value="adapter"
        current={mode}
        onSelect={setMode}
        title="PEFT adapter"
        body="Small HF-standard adapter repo (adapter_config.json + adapter_model.safetensors). Loadable via PeftModel.from_pretrained(base, repo)."
      />
      <ModeRadio
        value="merged"
        current={mode}
        onSelect={setMode}
        title="Merged full model"
        body="Single self-contained model repo. Larger upload (~30–60 GB for a 31B base). Required for off-platform inference."
      />
    </fieldset>
  );
}

function ModeRadio({
  value,
  current,
  onSelect,
  title,
  body,
  disabled,
}: {
  value: PublishMode;
  current: PublishMode;
  onSelect: (m: PublishMode) => void;
  title: string;
  body: string;
  disabled?: boolean;
}) {
  const checked = current === value;
  return (
    <label
      className={
        "flex items-start gap-2 rounded-md border px-3 py-2 text-sm transition-colors " +
        (disabled
          ? "cursor-not-allowed border-border-subtle bg-bg opacity-60"
          : "cursor-pointer ") +
        (checked && !disabled
          ? "border-accent/40 bg-accent/10"
          : !disabled
          ? "border-border-subtle bg-bg hover:border-border"
          : "")
      }
    >
      <input
        type="radio"
        name="publish-mode"
        value={value}
        checked={checked}
        disabled={disabled}
        onChange={() => onSelect(value)}
        className="mt-0.5 h-3 w-3 accent-accent"
      />
      <span className="flex flex-col">
        <span className="text-fg">{title}</span>
        <span className="text-[11px] text-fg-muted">{body}</span>
      </span>
    </label>
  );
}

function formatCheckpointLabel(c: CheckpointEntry, isLatest: boolean): string {
  const ts = c.mtime ? new Date(c.mtime * 1000) : null;
  const ago = ts ? formatAgo(Date.now() - ts.getTime()) : "";
  const head = `step ${c.step}`;
  const suffix = isLatest ? "  (latest)" : "";
  return ago ? `${head} · ${ago} ago${suffix}` : `${head}${suffix}`;
}

function formatAgo(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}

// Exported only so vitest can verify the regex without a full DOM mount.
export const __test = { REPO_ID_RE, formatCheckpointLabel };
