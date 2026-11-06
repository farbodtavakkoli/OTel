const BLOCK_SIZE = 512;

function encodeAsciiFixed(value: string, length: number): Uint8Array {
  const out = new Uint8Array(length);
  const bytes = new TextEncoder().encode(value);
  out.set(bytes.subarray(0, Math.min(length, bytes.length)));
  return out;
}

function octalField(value: number, length: number): Uint8Array {
  const out = new Uint8Array(length);
  const str = value.toString(8).padStart(length - 1, "0");
  out.set(new TextEncoder().encode(str));
  out[length - 1] = 0x00;
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

  for (let i = 148; i < 156; i++) header[i] = 0x20;

  writeAt(header, 0, encodeAsciiFixed(name, 100));
  writeAt(header, 100, octalField(0o0644, 8));
  writeAt(header, 108, octalField(0, 8));
  writeAt(header, 116, octalField(0, 8));
  writeAt(header, 124, octalField(size, 12));
  writeAt(header, 136, octalField(0, 12));
  header[156] = 0x30;
  writeAt(header, 257, encodeAsciiFixed("ustar", 6));
  writeAt(header, 263, encodeAsciiFixed("00", 2));
  writeAt(header, 265, encodeAsciiFixed("root", 32));
  writeAt(header, 297, encodeAsciiFixed("root", 32));
  writeAt(header, 329, octalField(0, 8));
  writeAt(header, 337, octalField(0, 8));

  let checksum = 0;
  for (let i = 0; i < BLOCK_SIZE; i++) checksum += header[i];
  const chkStr = checksum.toString(8).padStart(6, "0");
  writeAt(header, 148, new TextEncoder().encode(chkStr));
  header[154] = 0x00;
  header[155] = 0x20;

  return header;
}

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
