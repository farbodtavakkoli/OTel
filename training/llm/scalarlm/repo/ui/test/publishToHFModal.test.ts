/**
 * Unit tests for the pure helpers exported from PublishToHFModal.
 * The full modal component is exercised by hand for now; these
 * pin the validation rules called out in ui/docs/publish-to-hf.md.
 */

import { describe, expect, it } from "vitest";
import { __test } from "../src/routes/train/PublishToHFModal";

const { REPO_ID_RE, formatCheckpointLabel } = __test;

describe("REPO_ID_RE", () => {
  it.each([
    "myorg/my-adapter",
    "ScalarLM/Gemma-4-31B-Tuned",
    "a/b",
    "a-b-c/d.e_f",
  ])("accepts %s", (id) => {
    expect(REPO_ID_RE.test(id)).toBe(true);
  });

  it.each([
    "",
    "noslash",
    "/missingorg",
    "missingrepo/",
    "Two/slashes/here",
    "has spaces/repo",
    "owner/has spaces",
  ])("rejects %s", (id) => {
    expect(REPO_ID_RE.test(id)).toBe(false);
  });
});

describe("formatCheckpointLabel", () => {
  it("marks the latest entry", () => {
    const now = Math.floor(Date.now() / 1000);
    const label = formatCheckpointLabel(
      { name: "checkpoint_100.pt", step: 100, mtime: now - 60 },
      true,
    );
    expect(label).toContain("step 100");
    expect(label).toContain("(latest)");
  });

  it("shows a relative age", () => {
    const now = Math.floor(Date.now() / 1000);
    const label = formatCheckpointLabel(
      { name: "checkpoint_50.pt", step: 50, mtime: now - 90 },
      false,
    );
    // 90 seconds ago → 1m ago
    expect(label).toMatch(/step 50 · 1m ago/);
    expect(label).not.toContain("(latest)");
  });

  it("falls back when mtime is missing", () => {
    const label = formatCheckpointLabel(
      { name: "checkpoint_1.pt", step: 1, mtime: 0 },
      false,
    );
    expect(label).toBe("step 1");
  });
});
