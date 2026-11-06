import { z } from "zod";

export const ADAPTER_TYPES = ["tokenformer", "lora", "none"] as const;
export const TRAINING_MODES = ["language_model", "embedding"] as const;
export const DISTRIBUTION_STRATEGIES = ["fsdp", "ddp", "none"] as const;

export const loraConfigSchema = z.object({
  r: z.number().int().positive().default(32),
  lora_alpha: z.number().int().positive().default(32),
  lora_dropout: z.number().min(0).max(1).default(0.1),
  target_modules: z.union([z.string(), z.array(z.string())]).default("all-linear"),
});

export const trainArgsSchema = z.object({
  llm_name: z.string().min(1).default("meta-llama/Llama-3.2-1B-Instruct"),

  max_steps: z.number().int().positive().default(100),
  learning_rate: z.number().positive().default(3e-3),
  batch_size: z.number().int().positive().default(1),
  gradient_clip_value: z.number().positive().default(1.0),
  gradient_accumulation_steps: z.number().int().positive().default(4),
  max_token_block_size: z.number().int().positive().default(16_777_216),

  training_mode: z.enum(TRAINING_MODES).default("language_model"),
  distribution_strategy: z.enum(DISTRIBUTION_STRATEGIES).default("fsdp"),

  steps_per_checkpoint: z.number().int().positive().default(100),
  max_checkpoints_to_keep: z.number().int().positive().default(3),

  gpus: z.number().int().nonnegative().default(1),
  nodes: z.number().int().positive().default(1),

  adapter_type: z.enum(ADAPTER_TYPES).default("tokenformer"),
  lora_config: loraConfigSchema.optional(),

  timeout: z.number().int().positive().default(4 * 60 * 60),
  training_history_length: z.number().int().positive().default(1024),
});

export type TrainArgs = z.infer<typeof trainArgsSchema>;
export type LoraConfig = z.infer<typeof loraConfigSchema>;

export function defaultTrainArgs(): TrainArgs {
  return trainArgsSchema.parse({});
}

export interface FieldMeta {
  label: string;
  hint?: string;
  group: "model" | "optimizer" | "distribution" | "checkpointing" | "adapter" | "runtime";
}

export const TRAIN_ARGS_FIELD_META: Record<
  Exclude<keyof TrainArgs, "lora_config">,
  FieldMeta
> = {
  llm_name: { label: "Base model", hint: "HF Hub ID", group: "model" },
  max_steps: { label: "Max steps", group: "optimizer" },
  learning_rate: { label: "Learning rate", group: "optimizer" },
  batch_size: { label: "Batch size", hint: "per-rank micro-batch", group: "optimizer" },
  gradient_accumulation_steps: {
    label: "Grad accum steps",
    hint: "Effective batch = batch × accum × world_size",
    group: "optimizer",
  },
  gradient_clip_value: { label: "Grad clip", group: "optimizer" },
  max_token_block_size: {
    label: "Token block size",
    hint: "Dataset chunking limit (tokens)",
    group: "optimizer",
  },
  training_mode: { label: "Training mode", group: "model" },
  distribution_strategy: { label: "Distribution", group: "distribution" },
  gpus: { label: "GPUs", hint: "tasks-per-node · 0 = CPU", group: "distribution" },
  nodes: { label: "Nodes", group: "distribution" },
  steps_per_checkpoint: { label: "Steps per checkpoint", group: "checkpointing" },
  max_checkpoints_to_keep: { label: "Keep latest N ckpts", group: "checkpointing" },
  adapter_type: { label: "Adapter", group: "adapter" },
  timeout: { label: "Timeout (s)", hint: "clamped by max_train_time", group: "runtime" },
  training_history_length: {
    label: "History length",
    hint: "entries in status.json::history",
    group: "runtime",
  },
};

export function validateTrainArgs(
  raw: unknown,
): { ok: true; value: TrainArgs } | { ok: false; errors: Record<string, string> } {
  const result = trainArgsSchema.safeParse(raw);
  if (result.success) return { ok: true, value: result.data };
  const errors: Record<string, string> = {};
  for (const issue of result.error.issues) {
    const key = issue.path.join(".");
    if (!errors[key]) errors[key] = issue.message;
  }
  return { ok: false, errors };
}
