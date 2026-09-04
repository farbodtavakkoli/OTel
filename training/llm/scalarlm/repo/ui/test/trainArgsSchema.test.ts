/**
 * Unit tests for ui/src/lib/trainArgsSchema.ts.
 *
 * Contract under test: docs/test-plan.md §5.14 — the Zod schema mirrors
 * infra/cray_infra/util/default_job_config.py::JobConfig field-for-field.
 * When a Python default changes, the TS default must change with it or the
 * SubmitModal silently sends the wrong body.
 */

import { describe, expect, it } from "vitest";
import {
  ADAPTER_TYPES,
  DISTRIBUTION_STRATEGIES,
  TRAINING_MODES,
  defaultTrainArgs,
  trainArgsSchema,
  validateTrainArgs,
} from "../src/lib/trainArgsSchema";

describe("defaultTrainArgs", () => {
  it("matches Python JobConfig defaults byte-for-byte", () => {
    const d = defaultTrainArgs();

    // Mirror of default_job_config.py:
    expect(d.llm_name).toBe("meta-llama/Llama-3.2-1B-Instruct");
    expect(d.max_steps).toBe(100);
    expect(d.learning_rate).toBe(3e-3);
    expect(d.batch_size).toBe(1);
    expect(d.gradient_clip_value).toBe(1.0);
    expect(d.gradient_accumulation_steps).toBe(4);
    expect(d.max_token_block_size).toBe(16_777_216);
    expect(d.training_mode).toBe("language_model");
    expect(d.distribution_strategy).toBe("fsdp");
    expect(d.steps_per_checkpoint).toBe(100);
    expect(d.max_checkpoints_to_keep).toBe(3);
    expect(d.gpus).toBe(1);
    expect(d.nodes).toBe(1);
    expect(d.adapter_type).toBe("tokenformer");
    expect(d.timeout).toBe(4 * 60 * 60);
    expect(d.training_history_length).toBe(1024);
  });

  it("exposes allowed enum values matching JobConfig", () => {
    expect(ADAPTER_TYPES).toEqual(["tokenformer", "lora", "none"]);
    expect(TRAINING_MODES).toEqual(["language_model", "embedding"]);
    expect(DISTRIBUTION_STRATEGIES).toEqual(["fsdp", "ddp", "none"]);
  });
});

describe("validateTrainArgs", () => {
  it("accepts empty input and fills all defaults", () => {
    const result = validateTrainArgs({});
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.max_steps).toBe(100);
    }
  });

  it("rejects max_steps <= 0", () => {
    const result = validateTrainArgs({ max_steps: 0 });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.max_steps).toBeDefined();
    }

    const neg = validateTrainArgs({ max_steps: -1 });
    expect(neg.ok).toBe(false);
  });

  it("rejects learning_rate <= 0", () => {
    expect(validateTrainArgs({ learning_rate: 0 }).ok).toBe(false);
    expect(validateTrainArgs({ learning_rate: -1e-4 }).ok).toBe(false);
  });

  it("rejects non-integer step counts", () => {
    const result = validateTrainArgs({ max_steps: 1.5 });
    expect(result.ok).toBe(false);
  });

  it("rejects unknown adapter_type", () => {
    const result = validateTrainArgs({ adapter_type: "bogus" });
    expect(result.ok).toBe(false);
  });

  it("accepts lora_config and preserves nested defaults", () => {
    const result = validateTrainArgs({
      adapter_type: "lora",
      lora_config: { r: 64 },
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.lora_config?.r).toBe(64);
      // Unspecified nested fields fall back to Zod defaults.
      expect(result.value.lora_config?.lora_alpha).toBe(32);
      expect(result.value.lora_config?.lora_dropout).toBe(0.1);
      expect(result.value.lora_config?.target_modules).toBe("all-linear");
    }
  });

  it("accepts lora_config.target_modules as a list", () => {
    const result = validateTrainArgs({
      adapter_type: "lora",
      lora_config: { target_modules: ["q_proj", "v_proj"] },
    });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.lora_config?.target_modules).toEqual([
        "q_proj",
        "v_proj",
      ]);
    }
  });

  it("reports field-level errors mapping to a flat record", () => {
    const result = validateTrainArgs({
      max_steps: -1,
      learning_rate: -1,
      adapter_type: "nope",
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      // Each invalid field surfaces its error message, keyed by field name
      // so SubmitModal can highlight cells directly.
      expect(Object.keys(result.errors)).toEqual(
        expect.arrayContaining(["max_steps", "learning_rate", "adapter_type"]),
      );
    }
  });
});

describe("trainArgsSchema round-trip", () => {
  it("defaultTrainArgs parses cleanly through the schema", () => {
    const d = defaultTrainArgs();
    // Stripping away the parsing should still leave a valid shape.
    const reparsed = trainArgsSchema.parse(d);
    expect(reparsed).toEqual(d);
  });
});
