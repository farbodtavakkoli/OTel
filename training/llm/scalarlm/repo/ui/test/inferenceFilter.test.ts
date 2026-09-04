/**
 * Unit tests for the regex helper backing the Inference Requests
 * page filter input. See docs/inference-request-browser.md §5.
 */

import { describe, expect, it } from "vitest";

import {
  compileRegex,
  rowMatches,
  type InferenceListRow,
} from "../src/api/inference";

function row(overrides: Partial<InferenceListRow> = {}): InferenceListRow {
  return {
    request_id: "a".repeat(64),
    mtime: 0,
    size_bytes: 0,
    request_count: 1,
    status: "completed",
    completed_at: null,
    model: "meta-llama/Llama-3-8B",
    request_type: "generate",
    prompt_preview: "hello world",
    has_response: true,
    ...overrides,
  };
}

describe("compileRegex", () => {
  it("returns null for empty input so the filter is a no-op", () => {
    expect(compileRegex("")).toBeNull();
    expect(compileRegex("   ")).toBeNull();
  });

  it("compiles valid patterns case-insensitively", () => {
    const re = compileRegex("HELLO");
    expect(re).toBeInstanceOf(RegExp);
    expect((re as RegExp).test("hello")).toBe(true);
  });

  it("returns undefined for invalid patterns rather than throwing", () => {
    expect(compileRegex("(unclosed")).toBeUndefined();
    expect(compileRegex("[")).toBeUndefined();
  });
});

describe("rowMatches", () => {
  it("matches against the prompt preview", () => {
    expect(rowMatches(row({ prompt_preview: "tell me a joke" }), /joke/i)).toBe(
      true,
    );
  });

  it("matches against the model name", () => {
    expect(rowMatches(row({ model: "Qwen-2.5-7B" }), /qwen/i)).toBe(true);
  });

  it("matches against the request_id prefix", () => {
    expect(rowMatches(row({ request_id: "deadbeef" + "0".repeat(56) }), /^deadbeef/)).toBe(
      true,
    );
  });

  it("returns false when nothing matches", () => {
    expect(rowMatches(row({ prompt_preview: "hello" }), /goodbye/)).toBe(false);
  });
});
