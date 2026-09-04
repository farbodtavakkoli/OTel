import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";

import {
  useSubmitTrainingJob,
  type SubmitProgress,
} from "@/api/training";
import { useGpuCount } from "@/api/capacity";
import { getApiConfig } from "@/api/config";
import { ErrorState } from "@/components/ErrorState";
import { Modal } from "@/components/Modal";
import { readJson, writeJson } from "@/lib/preferences";
import {
  ADAPTER_TYPES,
  DISTRIBUTION_STRATEGIES,
  TRAINING_MODES,
  TRAIN_ARGS_FIELD_META,
  defaultTrainArgs,
  trainArgsSchema,
  validateTrainArgs,
  type TrainArgs,
} from "@/lib/trainArgsSchema";

const LAST_ARGS_KEY = "submit.lastTrainArgs";
const MAX_DATASET_BYTES = 10 * 1024 * 1024 * 1024; // 10 GB, matches config.max_upload_file_size

// Tiny canned dataset for the "Paste example" affordance. Ten arithmetic
// pairs — small enough to overfit on a tiny model in seconds, useful enough
// to sanity-check the round trip from submit to /chat without staring at a
// blank textarea wondering what the file format is.
const EXAMPLE_JSONL = [
  { input: "What is 1 + 1?", output: "2" },
  { input: "What is 2 + 2?", output: "4" },
  { input: "What is 3 + 5?", output: "8" },
  { input: "What is 10 - 7?", output: "3" },
  { input: "What is 6 * 7?", output: "42" },
  { input: "What is 9 * 9?", output: "81" },
  { input: "What is 100 / 4?", output: "25" },
  { input: "What is 12 - 8?", output: "4" },
  { input: "What is 2 ^ 10?", output: "1024" },
  { input: "What is 15 + 27?", output: "42" },
]
  .map((row) => JSON.stringify(row))
  .join("\n");

type Mode = "upload" | "paste";

interface SubmitModalProps {
  open: boolean;
  onClose: () => void;
}

export function SubmitModal({ open, onClose }: SubmitModalProps) {
  const navigate = useNavigate();

  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [pasted, setPasted] = useState("");
  const [dragging, setDragging] = useState(false);

  const [args, setArgs] = useState<TrainArgs>(() => {
    // First-time defaults: seed llm_name from the running vLLM model rather
    // than the schema's hardcoded HF id. Saved args from a previous successful
    // submission win over both, so a user who explicitly picked something else
    // last time keeps that preference.
    const saved = readJson<unknown>(LAST_ARGS_KEY, null);
    const parsed = trainArgsSchema.safeParse(saved);
    if (parsed.success) return parsed.data;
    const fresh = defaultTrainArgs();
    const deploymentModel = getApiConfig().default_model;
    if (deploymentModel) fresh.llm_name = deploymentModel;
    return fresh;
  });
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [rawJson, setRawJson] = useState(() => JSON.stringify(args, null, 2));
  const [rawJsonError, setRawJsonError] = useState<string | null>(null);

  const [progress, setProgress] = useState<SubmitProgress | null>(null);
  const submit = useSubmitTrainingJob();

  // On CPU-only deployments (gpu_count === 0) the server's validate_gpu_request
  // rejects gpus≥1 with HTTP 400. Auto-flip the default to 0 once, when we
  // learn the cluster has no GPUs, but only if the user hasn't customized
  // gpus themselves. The ref makes the flip a one-shot per modal lifetime.
  const gpuCountQuery = useGpuCount();
  const gpuAutoFlipped = useRef(false);
  useEffect(() => {
    if (gpuAutoFlipped.current) return;
    if (gpuCountQuery.data === 0 && args.gpus === 1) {
      gpuAutoFlipped.current = true;
      setArgs((prev) => ({ ...prev, gpus: 0 }));
    }
  }, [gpuCountQuery.data, args.gpus]);

  const datasetSize = useMemo(() => {
    if (mode === "upload") return file?.size ?? 0;
    return new TextEncoder().encode(pasted).byteLength;
  }, [mode, file, pasted]);

  const datasetReady = mode === "upload" ? file !== null : pasted.trim().length > 0;
  const datasetTooBig = datasetSize > MAX_DATASET_BYTES;

  const validation = useMemo(() => validateTrainArgs(args), [args]);

  // Advanced JSON view takes precedence if edited: parse on commit.
  const commitRaw = () => {
    try {
      const parsed = JSON.parse(rawJson);
      const result = validateTrainArgs(parsed);
      if (!result.ok) {
        setRawJsonError(
          Object.entries(result.errors)
            .map(([k, v]) => `${k}: ${v}`)
            .join("; "),
        );
        return false;
      }
      setArgs(result.value);
      setRawJsonError(null);
      return true;
    } catch (e) {
      setRawJsonError(e instanceof Error ? e.message : "invalid JSON");
      return false;
    }
  };

  const canSubmit =
    datasetReady &&
    !datasetTooBig &&
    validation.ok &&
    !submit.isPending &&
    (!advancedOpen || rawJsonError === null);

  const onSubmit = async () => {
    if (advancedOpen && !commitRaw()) return;
    if (!validation.ok) return;
    const datasetJsonl =
      mode === "upload"
        ? await file!.text()
        : pasted;

    setProgress({ loaded: 0, total: datasetSize, fraction: 0 });

    try {
      const resp = await submit.mutateAsync({
        datasetJsonl,
        trainArgs: validation.value,
        onProgress: setProgress,
      });
      writeJson(LAST_ARGS_KEY, validation.value);
      const hash =
        resp.job_status.model_name ??
        (typeof resp.job_config.job_directory === "string"
          ? resp.job_config.job_directory.split("/").pop()
          : undefined);
      onClose();
      if (hash) navigate(`/train/${hash}`);
    } catch {
      // Error surfaced via submit.error
    } finally {
      setProgress(null);
    }
  };

  return (
    <Modal
      open={open}
      title="New training job"
      onClose={() => {
        if (submit.isPending) return;
        onClose();
      }}
      maxWidth="max-w-3xl"
      footer={
        <>
          <button
            type="button"
            onClick={onClose}
            disabled={submit.isPending}
            className="rounded-md border border-border-subtle bg-bg px-3 py-1.5 text-sm text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSubmit}
            disabled={!canSubmit}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submit.isPending ? "Submitting…" : "Submit"}
          </button>
        </>
      }
    >
      <div className="flex flex-col gap-5">
        {submit.error && <ErrorState error={submit.error} />}

        <section>
          <h4 className="mb-2 text-sm font-semibold">1. Dataset</h4>
          <div className="mb-2 flex gap-1 text-xs">
            <ModeTab active={mode === "upload"} onClick={() => setMode("upload")}>
              Upload JSONL file
            </ModeTab>
            <ModeTab active={mode === "paste"} onClick={() => setMode("paste")}>
              Paste JSONL
            </ModeTab>
          </div>
          {mode === "upload" ? (
            <DropZone
              file={file}
              dragging={dragging}
              onDragChange={setDragging}
              onPick={setFile}
            />
          ) : (
            <div className="flex flex-col gap-1">
              <textarea
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
                placeholder={`{"input": "What is 2+2?", "output": "4"}\n{"input": "...", "output": "..."}`}
                className="min-h-[160px] w-full rounded-md border border-border-subtle bg-bg px-3 py-2 font-mono text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none"
              />
              <div>
                <button
                  type="button"
                  onClick={() => setPasted(EXAMPLE_JSONL)}
                  className="text-[11px] text-fg-muted underline-offset-2 hover:text-fg hover:underline"
                >
                  Paste 10-sample example
                </button>
              </div>
            </div>
          )}
          <div className="mt-1 flex items-center justify-between text-xs text-fg-subtle">
            <span>
              Will be packed as{" "}
              <code className="font-mono">dataset.jsonlines</code> inside an
              uncompressed tar.
            </span>
            <span className={clsx(datasetTooBig && "text-danger")}>
              {datasetReady ? formatBytes(datasetSize) : "no data"}
              {datasetTooBig && " · exceeds 10 GB cap"}
            </span>
          </div>
        </section>

        <section>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-sm font-semibold">2. Training args</h4>
            <button
              type="button"
              onClick={() => {
                if (!advancedOpen) setRawJson(JSON.stringify(args, null, 2));
                else commitRaw();
                setAdvancedOpen(!advancedOpen);
              }}
              className="text-xs text-fg-muted underline-offset-2 hover:text-fg hover:underline"
            >
              {advancedOpen ? "Hide raw JSON" : "Edit raw JSON"}
            </button>
          </div>

          {advancedOpen ? (
            <div className="flex flex-col gap-2">
              <textarea
                value={rawJson}
                onChange={(e) => setRawJson(e.target.value)}
                spellCheck={false}
                className="min-h-[260px] w-full rounded-md border border-border-subtle bg-bg px-3 py-2 font-mono text-xs text-fg focus:border-border focus:outline-none"
              />
              {rawJsonError && (
                <div className="text-xs text-danger">{rawJsonError}</div>
              )}
            </div>
          ) : (
            <TrainArgsForm
              value={args}
              onChange={setArgs}
              errors={validation.ok ? {} : validation.errors}
            />
          )}
        </section>

        {progress && (
          <section>
            <div className="mb-1 flex items-center justify-between text-xs text-fg-muted">
              <span>Uploading</span>
              <span>
                {formatBytes(progress.loaded)} / {formatBytes(progress.total)}
                {progress.fraction !== null &&
                  ` · ${Math.round(progress.fraction * 100)}%`}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg">
              <div
                className="h-full bg-accent transition-[width] duration-100"
                style={{
                  width: `${Math.round((progress.fraction ?? 0) * 100)}%`,
                }}
              />
            </div>
          </section>
        )}
      </div>
    </Modal>
  );
}

function ModeTab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "rounded-md px-3 py-1",
        active
          ? "bg-bg-hover text-fg ring-1 ring-border"
          : "text-fg-muted hover:bg-bg-card hover:text-fg",
      )}
    >
      {children}
    </button>
  );
}

function DropZone({
  file,
  dragging,
  onDragChange,
  onPick,
}: {
  file: File | null;
  dragging: boolean;
  onDragChange: (v: boolean) => void;
  onPick: (f: File | null) => void;
}) {
  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        onDragChange(true);
      }}
      onDragLeave={() => onDragChange(false)}
      onDrop={(e) => {
        e.preventDefault();
        onDragChange(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onPick(f);
      }}
      className={clsx(
        "flex cursor-pointer flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed px-4 py-8 text-sm",
        dragging
          ? "border-accent bg-accent/5 text-accent"
          : "border-border-subtle bg-bg/50 text-fg-muted hover:border-border hover:bg-bg-hover",
      )}
    >
      <input
        type="file"
        accept=".jsonl,.json,.ndjson,application/json,application/x-ndjson,text/plain"
        className="hidden"
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <>
          <span className="font-mono text-xs text-fg">{file.name}</span>
          <span className="text-xs text-fg-subtle">
            {formatBytes(file.size)} · click to replace
          </span>
        </>
      ) : (
        <>
          <span>Drop a .jsonl file, or click to pick</span>
          <span className="text-xs text-fg-subtle">
            One JSON object per line
          </span>
        </>
      )}
    </label>
  );
}

function TrainArgsForm({
  value,
  onChange,
  errors,
}: {
  value: TrainArgs;
  onChange: (next: TrainArgs) => void;
  errors: Record<string, string>;
}) {
  const update = <K extends keyof TrainArgs>(key: K, v: TrainArgs[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <TextField
        label={TRAIN_ARGS_FIELD_META.llm_name.label}
        hint={TRAIN_ARGS_FIELD_META.llm_name.hint}
        value={value.llm_name}
        onChange={(v) => update("llm_name", v)}
        error={errors.llm_name}
        className="sm:col-span-2"
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.max_steps.label}
        value={value.max_steps}
        onChange={(v) => update("max_steps", v)}
        error={errors.max_steps}
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.learning_rate.label}
        value={value.learning_rate}
        onChange={(v) => update("learning_rate", v)}
        error={errors.learning_rate}
        step="any"
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.batch_size.label}
        hint={TRAIN_ARGS_FIELD_META.batch_size.hint}
        value={value.batch_size}
        onChange={(v) => update("batch_size", v)}
        error={errors.batch_size}
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.gradient_accumulation_steps.label}
        hint={TRAIN_ARGS_FIELD_META.gradient_accumulation_steps.hint}
        value={value.gradient_accumulation_steps}
        onChange={(v) => update("gradient_accumulation_steps", v)}
        error={errors.gradient_accumulation_steps}
      />
      {/* GPUs is exposed because CPU-only dev clusters need gpus=0 to bypass
          the server-side validate_gpu_request check; production users with GPU
          nodes normally leave the default. */}
      <NumberField
        label={TRAIN_ARGS_FIELD_META.gpus.label}
        hint={TRAIN_ARGS_FIELD_META.gpus.hint}
        value={value.gpus}
        onChange={(v) => update("gpus", v)}
        error={errors.gpus}
        min={0}
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.nodes.label}
        value={value.nodes}
        onChange={(v) => update("nodes", v)}
        error={errors.nodes}
      />
      <SelectField
        label={TRAIN_ARGS_FIELD_META.adapter_type.label}
        value={value.adapter_type}
        options={ADAPTER_TYPES}
        onChange={(v) => update("adapter_type", v)}
      />
      <SelectField
        label={TRAIN_ARGS_FIELD_META.training_mode.label}
        value={value.training_mode}
        options={TRAINING_MODES}
        onChange={(v) => update("training_mode", v)}
      />
      <SelectField
        label={TRAIN_ARGS_FIELD_META.distribution_strategy.label}
        value={value.distribution_strategy}
        options={DISTRIBUTION_STRATEGIES}
        onChange={(v) => update("distribution_strategy", v)}
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.timeout.label}
        hint={TRAIN_ARGS_FIELD_META.timeout.hint}
        value={value.timeout}
        onChange={(v) => update("timeout", v)}
        error={errors.timeout}
      />
      <NumberField
        label={TRAIN_ARGS_FIELD_META.steps_per_checkpoint.label}
        value={value.steps_per_checkpoint}
        onChange={(v) => update("steps_per_checkpoint", v)}
        error={errors.steps_per_checkpoint}
      />
    </div>
  );
}

function TextField({
  label,
  hint,
  value,
  onChange,
  error,
  className,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
  className?: string;
}) {
  return (
    <label className={clsx("flex flex-col gap-1 text-xs text-fg-muted", className)}>
      <span>
        {label}
        {hint && <span className="ml-1 text-fg-subtle">· {hint}</span>}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={clsx(
          "rounded-md border bg-bg px-2 py-1 font-mono text-sm text-fg focus:outline-none",
          error ? "border-danger" : "border-border-subtle focus:border-border",
        )}
      />
      {error && <span className="text-[11px] text-danger">{error}</span>}
    </label>
  );
}

function NumberField({
  label,
  hint,
  value,
  onChange,
  error,
  step,
  min,
}: {
  label: string;
  hint?: string;
  value: number;
  onChange: (v: number) => void;
  error?: string;
  step?: "any" | number;
  min?: number;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-fg-muted">
      <span>
        {label}
        {hint && <span className="ml-1 text-fg-subtle">· {hint}</span>}
      </span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        step={step ?? 1}
        min={min}
        onChange={(e) => {
          const v = e.target.value === "" ? NaN : Number(e.target.value);
          onChange(v);
        }}
        className={clsx(
          "rounded-md border bg-bg px-2 py-1 font-mono text-sm text-fg focus:outline-none",
          error ? "border-danger" : "border-border-subtle focus:border-border",
        )}
      />
      {error && <span className="text-[11px] text-danger">{error}</span>}
    </label>
  );
}

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-fg-muted">
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded-md border border-border-subtle bg-bg px-2 py-1 font-mono text-sm text-fg focus:border-border focus:outline-none"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
}
