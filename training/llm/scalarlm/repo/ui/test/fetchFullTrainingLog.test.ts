import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchFullTrainingLog } from "../src/api/training";

function ndjsonResponse(lines: string[]): Response {
  const body = lines
    .map((line, i) => JSON.stringify({ line, line_number: i }) + "\n")
    .join("");
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const bytes = new TextEncoder().encode(body);
      const mid = Math.floor(bytes.length / 2);
      controller.enqueue(bytes.slice(0, mid));
      controller.enqueue(bytes.slice(mid));
      controller.close();
    },
  });
  return { ok: true, status: 200, body: stream } as unknown as Response;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("fetchFullTrainingLog", () => {
  it("returns the full stitched log joined by newlines", async () => {
    const logLines = [
      "slice-1 line A",
      "slice-1 line B",
      "slice-2 line C",
      "slice-2 line D",
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(ndjsonResponse(logLines)),
    );

    const text = await fetchFullTrainingLog("abc123", new AbortController().signal);

    expect(text).toBe(logLines.join("\n"));
  });

  it("requests the log from line 0 (the whole stitched log, not a tail)", async () => {
    const fetchMock = vi.fn().mockResolvedValue(ndjsonResponse(["x"]));
    vi.stubGlobal("fetch", fetchMock);

    await fetchFullTrainingLog("my model/hash", new AbortController().signal);

    const calledUrl = fetchMock.mock.calls[0][0] as string;
    expect(calledUrl).toContain("/megatron/train/logs/");
    expect(calledUrl).toContain("starting_line_number=0");
    expect(calledUrl).toContain(encodeURIComponent("my model/hash"));
  });

  it("throws on a non-OK response so the UI can surface a retry", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, body: null } as Response),
    );

    await expect(
      fetchFullTrainingLog("abc", new AbortController().signal),
    ).rejects.toThrow(/HTTP 500/);
  });
});
