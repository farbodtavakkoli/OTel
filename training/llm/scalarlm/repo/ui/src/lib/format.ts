/**
 * Format a numeric value with SI suffixes for display in stat cards.
 * Keeps exactly 3 significant digits for readability at a glance.
 *   1234         → "1.23k"
 *   4.56e9       → "4.56G"
 *   0.0123       → "12.3m"
 *   NaN | null   → "—"
 */
export function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs === 0) return "0";

  const units: { suffix: string; divisor: number }[] = [
    { suffix: "T", divisor: 1e12 },
    { suffix: "G", divisor: 1e9 },
    { suffix: "M", divisor: 1e6 },
    { suffix: "k", divisor: 1e3 },
    { suffix: "", divisor: 1 },
    { suffix: "m", divisor: 1e-3 },
    { suffix: "µ", divisor: 1e-6 },
  ];
  for (const { suffix, divisor } of units) {
    if (abs >= divisor) {
      const scaled = value / divisor;
      const absScaled = Math.abs(scaled);
      const digits = absScaled >= 100 ? 0 : absScaled >= 10 ? 1 : 2;
      return `${scaled.toFixed(digits)}${suffix}`;
    }
  }
  return value.toExponential(2);
}

/** Format a duration in seconds to "HH:MM:SS" for SLURM-like columns. */
export function formatDurationSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}
