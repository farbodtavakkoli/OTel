/**
 * Unit tests for the DatasetPanel download helpers. The Download link is
 * shown even while the panel is collapsed (row count unknown), so the URL
 * builder must work without a limit and the label must degrade to a generic
 * "Download" when total is undefined.
 */

import { describe, expect, it } from "vitest";

import { datasetDownloadUrl, downloadAllLabel } from "../src/routes/train/DatasetPanel";

describe("datasetDownloadUrl", () => {
  it("builds the full-dataset URL with no limit", () => {
    expect(datasetDownloadUrl("/v1", "abc123")).toBe(
      "/v1/megatron/train/abc123/dataset/download",
    );
  });

  it("appends a limit for a sample download", () => {
    expect(datasetDownloadUrl("/v1", "abc123", 100)).toBe(
      "/v1/megatron/train/abc123/dataset/download?limit=100",
    );
  });

  it("url-encodes the job hash", () => {
    expect(datasetDownloadUrl("/v1", "a b/c")).toContain(
      encodeURIComponent("a b/c"),
    );
  });
});

describe("downloadAllLabel", () => {
  it("is generic when the row count is unknown (collapsed)", () => {
    expect(downloadAllLabel(undefined)).toBe("Download");
  });

  it("shows an exact count under 1K", () => {
    expect(downloadAllLabel(42)).toBe("Download all (42 rows)");
  });

  it("abbreviates thousands and millions", () => {
    expect(downloadAllLabel(2_500)).toBe("Download all (3K rows)");
    expect(downloadAllLabel(3_000_000)).toBe("Download all (3M rows)");
  });
});
