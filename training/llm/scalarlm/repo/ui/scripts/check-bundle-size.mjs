#!/usr/bin/env node
/**
 * Fail-fast guard on bundle growth. Runs against ui/dist/ produced by
 * `npm run build`. Two JS budgets:
 *
 *   firstPaintJs — the main entry chunk the browser must download before any
 *                  route can render. Tight budget; this is what first-paint
 *                  latency is bottlenecked on.
 *   totalJs      — every chunk summed, including route-lazy splits. Loose
 *                  budget; this is our overall footprint.
 *
 * CSS currently ships as a single file, so no split-vs-total distinction.
 *
 * Usage:   node scripts/check-bundle-size.mjs
 * Or:      npm run size      (after npm run build)
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const DIST = join(HERE, "..", "dist");

const LIMITS = {
  firstPaintJs: 150 * 1024,
  totalJs: 300 * 1024,
  css: 20 * 1024,
};

// Vite names the main entry `index-{hash}.js` (see rollup's defaults) — any
// other .js file in dist/assets/ is a split chunk loaded by React.lazy /
// dynamic import.
const ENTRY_PATTERN = /^assets[\\/]index-[^/\\]+\.js$/;

function gzipLenOfFile(path) {
  const buf = readFileSync(path);
  return gzipSync(buf).byteLength;
}

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...walk(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

try {
  statSync(DIST);
} catch {
  console.error(`No dist directory at ${DIST}. Run \`npm run build\` first.`);
  process.exit(2);
}

const files = walk(DIST);

let firstPaintJs = 0;
let totalJs = 0;
let cssGz = 0;
const report = [];

for (const f of files) {
  const ext = extname(f).toLowerCase();
  if (ext !== ".js" && ext !== ".css") continue;
  const relative = f.slice(DIST.length + 1);
  const gz = gzipLenOfFile(f);
  const raw = statSync(f).size;
  const isEntry = ext === ".js" && ENTRY_PATTERN.test(relative);
  report.push({ file: relative, raw, gz, ext, isEntry });
  if (ext === ".js") {
    totalJs += gz;
    if (isEntry) firstPaintJs += gz;
  }
  if (ext === ".css") cssGz += gz;
}

const fmt = (n) => `${(n / 1024).toFixed(1)} KB`;
const pad = (s, n) => s.padEnd(n, " ");
const padR = (s, n) => s.padStart(n, " ");

const nameWidth = Math.max(18, ...report.map((r) => r.file.length));
console.log(
  pad("file", nameWidth),
  padR("raw", 12),
  padR("gzip", 12),
  padR("role", 10),
);
console.log("-".repeat(nameWidth + 38));
for (const r of report.sort((a, b) => b.gz - a.gz)) {
  const role =
    r.ext === ".css" ? "first-paint" : r.isEntry ? "first-paint" : "lazy";
  console.log(
    pad(r.file, nameWidth),
    padR(fmt(r.raw), 12),
    padR(fmt(r.gz), 12),
    padR(role, 10),
  );
}
console.log("-".repeat(nameWidth + 38));
console.log(
  pad("first-paint js", nameWidth),
  padR("", 12),
  padR(fmt(firstPaintJs), 12),
  padR(`/${fmt(LIMITS.firstPaintJs)}`, 10),
);
console.log(
  pad("total js", nameWidth),
  padR("", 12),
  padR(fmt(totalJs), 12),
  padR(`/${fmt(LIMITS.totalJs)}`, 10),
);
console.log(
  pad("css", nameWidth),
  padR("", 12),
  padR(fmt(cssGz), 12),
  padR(`/${fmt(LIMITS.css)}`, 10),
);

let failed = false;
if (firstPaintJs > LIMITS.firstPaintJs) {
  console.error(
    `FAIL: first-paint js gzip ${fmt(firstPaintJs)} exceeds ${fmt(
      LIMITS.firstPaintJs,
    )}`,
  );
  failed = true;
}
if (totalJs > LIMITS.totalJs) {
  console.error(
    `FAIL: total js gzip ${fmt(totalJs)} exceeds ${fmt(LIMITS.totalJs)}`,
  );
  failed = true;
}
if (cssGz > LIMITS.css) {
  console.error(
    `FAIL: css gzip ${fmt(cssGz)} exceeds ${fmt(LIMITS.css)}`,
  );
  failed = true;
}
if (failed) process.exit(1);
console.log("ok");
