/**
 * Minimal USTAR tar writer — just enough to pack a single file for upload to
 * /v1/megatron/train. The server calls tarfile.open(..., "r") which
 * auto-detects compression and extracts; a plain uncompressed tar is fine.
 *
 * Format reference: POSIX.1-1988 USTAR. Each file is one 512-byte header
 * followed by content padded to a 512-byte multiple. End-of-archive is two
 * zero-filled 512-byte blocks.
 *
 * We pack each entry as uid/gid 0, mode 0644, mtime 0 — matches the SDK's
 * tar_info_strip_file_info so archive bytes are deterministic per content.
 */

const BLOCK_SIZE = 512;

function encodeAsciiFixed(value: string, length: number): Uint8Array {
  const out = new Uint8Array(length);
  const bytes = new TextEncoder().encode(value);
  out.set(bytes.subarray(0, Math.min(length, bytes.length)));
  return out;
}

/** Write an octal number padded with leading zeros, terminated by NUL+space. */
function octalField(value: number, length: number): Uint8Array {
  const out = new Uint8Array(length);
  const str = value.toString(8).padStart(length - 1, "0");
  out.set(new TextEncoder().encode(str));
  out[length - 1] = 0x00; // NUL terminator
  return out;
}

function writeAt(buf: Uint8Array, offset: number, bytes: Uint8Array) {
  buf.set(bytes, offset);
}

function ustarHeader(name: string, size: number): Uint8Array {
  if (new TextEncoder().encode(name).byteLength > 100) {
    throw new Error(`tar: entry name exceeds 100 bytes: ${name}`);
  }

  const header = new Uint8Array(BLOCK_SIZE);

  // Initialise checksum field to spaces before computing the checksum.
  for (let i = 148; i < 156; i++) header[i] = 0x20;

  writeAt(header, 0, encodeAsciiFixed(name, 100));
  writeAt(header, 100, octalField(0o0644, 8));       // mode
  writeAt(header, 108, octalField(0, 8));            // uid
  writeAt(header, 116, octalField(0, 8));            // gid
  writeAt(header, 124, octalField(size, 12));        // size
  writeAt(header, 136, octalField(0, 12));           // mtime
  // chksum (148..155) — computed below
  header[156] = 0x30;                                 // typeflag '0' = regular file
  // linkname (157..256) — zeros
  writeAt(header, 257, encodeAsciiFixed("ustar", 6)); // magic
  writeAt(header, 263, encodeAsciiFixed("00", 2));    // version
  writeAt(header, 265, encodeAsciiFixed("root", 32)); // uname
  writeAt(header, 297, encodeAsciiFixed("root", 32)); // gname
  writeAt(header, 329, octalField(0, 8));            // devmajor
  writeAt(header, 337, octalField(0, 8));            // devminor
  // prefix (345..499) — zeros, we cap names at 100 bytes

  // Checksum: 6 octal digits + NUL + space
  let checksum = 0;
  for (let i = 0; i < BLOCK_SIZE; i++) checksum += header[i];
  const chkStr = checksum.toString(8).padStart(6, "0");
  writeAt(header, 148, new TextEncoder().encode(chkStr));
  header[154] = 0x00;
  header[155] = 0x20;

  return header;
}

/** Build a USTAR tar archive from an in-memory payload. */
export function buildTar(
  entries: { name: string; data: Uint8Array }[],
): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const { name, data } of entries) {
    parts.push(ustarHeader(name, data.byteLength));
    parts.push(data);
    const pad = (BLOCK_SIZE - (data.byteLength % BLOCK_SIZE)) % BLOCK_SIZE;
    if (pad > 0) parts.push(new Uint8Array(pad));
  }
  // Two zero blocks signal end of archive.
  parts.push(new Uint8Array(BLOCK_SIZE * 2));

  let total = 0;
  for (const p of parts) total += p.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}
