/**
 * Unit tests for ui/src/lib/tar.ts.
 *
 * Contract under test: docs/test-plan.md §5.14 — `buildTar` must emit a
 * valid USTAR archive whose bytes are deterministic for a given input, so
 * the server-side dataset_hash stays stable across retries of the same
 * browser submission.
 */

import { describe, expect, it } from "vitest";
import { buildTar } from "../src/lib/tar";

function decodeField(buf: Uint8Array, offset: number, length: number): string {
  // USTAR fields are NUL-terminated ASCII; trim anything after first NUL.
  // Decoder label is "utf-8" rather than "ascii" because Node 23.x has a bug
  // where `TextDecoder("ascii").decode(...)` returns a Buffer instead of a
  // string. UTF-8 decodes pure-ASCII bytes to an identical string.
  const slice = buf.subarray(offset, offset + length);
  let end = slice.indexOf(0);
  if (end === -1) end = slice.length;
  return new TextDecoder("utf-8").decode(slice.subarray(0, end));
}

describe("buildTar", () => {
  it("produces a well-formed USTAR header for a single entry", () => {
    const data = new TextEncoder().encode("hello world\n");
    const archive = buildTar([{ name: "greeting.txt", data }]);

    // One data block minimum plus two zero end-blocks.
    expect(archive.byteLength).toBeGreaterThanOrEqual(512 * 3);

    // name field (0..99)
    expect(decodeField(archive, 0, 100)).toBe("greeting.txt");

    // size field (124..135) is 12 bytes: 11 octal digits + NUL terminator,
    // per the USTAR convention (tar.ts pads to length-1, then writes NUL).
    // "hello world\n" is 12 bytes → octal 14 → "00000000014".
    expect(decodeField(archive, 124, 12)).toBe("00000000014");

    // magic + version: "ustar\0" at 257, "00" at 263.
    expect(decodeField(archive, 257, 6)).toBe("ustar");
    expect(decodeField(archive, 263, 2)).toBe("00");

    // typeflag '0' at offset 156 marks a regular file.
    expect(archive[156]).toBe(0x30);
  });

  it("writes the payload immediately after the header, padded to 512 bytes", () => {
    const text = "hello world\n";
    const data = new TextEncoder().encode(text);
    const archive = buildTar([{ name: "g.txt", data }]);

    // Data starts at byte 512 (after the header).
    const payload = archive.subarray(512, 512 + data.byteLength);
    expect(new TextDecoder().decode(payload)).toBe(text);

    // Padding up to the next 512-byte boundary is zero-filled.
    const padStart = 512 + data.byteLength;
    const padEnd = 512 * 2;
    for (let i = padStart; i < padEnd; i++) {
      expect(archive[i]).toBe(0);
    }
  });

  it("ends with two zero 512-byte blocks", () => {
    const archive = buildTar([
      { name: "a.txt", data: new Uint8Array([0x61]) }, // "a"
    ]);

    const last1024 = archive.subarray(archive.byteLength - 1024);
    for (const byte of last1024) {
      expect(byte).toBe(0);
    }
  });

  it("is byte-deterministic for identical inputs", () => {
    const mk = () =>
      buildTar([
        { name: "a.txt", data: new TextEncoder().encode("abc") },
        { name: "b.txt", data: new TextEncoder().encode("xyz") },
      ]);

    const first = mk();
    const second = mk();

    expect(first.byteLength).toBe(second.byteLength);
    for (let i = 0; i < first.byteLength; i++) {
      expect(first[i]).toBe(second[i]);
    }
  });

  it("rejects entry names longer than 100 bytes", () => {
    const longName = "x".repeat(101);
    expect(() =>
      buildTar([{ name: longName, data: new Uint8Array([0]) }]),
    ).toThrow(/exceeds 100 bytes/);
  });

  it("packs multiple entries back-to-back with correct sizes", () => {
    const entries = [
      { name: "one.txt", data: new TextEncoder().encode("AAA") },
      { name: "two.txt", data: new TextEncoder().encode("BBBBB") },
    ];
    const archive = buildTar(entries);

    // Entry layout: [hdr1(512)][data1 padded to 512][hdr2(512)][data2 padded to 512][zero(1024)]
    expect(archive.byteLength).toBe(512 + 512 + 512 + 512 + 1024);

    // Second entry name is in the second header, at byte 1024.
    expect(decodeField(archive, 1024, 100)).toBe("two.txt");
    // Size 5 → octal 5 → "00000000005" + NUL (11-digit convention).
    expect(decodeField(archive, 1024 + 124, 12)).toBe("00000000005");
  });

  it("computes a valid USTAR checksum", () => {
    const data = new TextEncoder().encode("ok");
    const archive = buildTar([{ name: "c.txt", data }]);
    const header = archive.subarray(0, 512);

    // Replicate the checksum algorithm: sum all bytes, treating the
    // checksum field itself as spaces.
    let expected = 0;
    for (let i = 0; i < 512; i++) {
      if (i >= 148 && i < 156) {
        expected += 0x20; // space
      } else {
        expected += header[i];
      }
    }

    const chkField = decodeField(header, 148, 6);
    expect(parseInt(chkField, 8)).toBe(expected);
  });
});
