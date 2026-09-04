import { useEffect, useRef, useState } from "react";

/**
 * Append `value` to a rolling buffer capped at `capacity` samples. Each call
 * with a new, non-null value records a sample; identity-stable output is a
 * reactive array so React re-renders on push. When `capacity` shrinks, the
 * oldest samples are dropped.
 *
 * Intended for client-side metric histories — the throughput sparkline keeps
 * the last N polls worth of token/s in memory only. Refreshing the page
 * clears the buffer (by design; no server-side history store in v1).
 */
export function useRollingBuffer<T>(value: T | null | undefined, capacity: number) {
  const [buffer, setBuffer] = useState<T[]>([]);
  const lastValueRef = useRef<T | null | undefined>(undefined);

  useEffect(() => {
    if (value === null || value === undefined) return;
    if (lastValueRef.current === value) return;
    lastValueRef.current = value;
    setBuffer((prev) => {
      const next = prev.length >= capacity ? prev.slice(prev.length - capacity + 1) : prev.slice();
      next.push(value);
      return next;
    });
  }, [value, capacity]);

  useEffect(() => {
    setBuffer((prev) => (prev.length > capacity ? prev.slice(prev.length - capacity) : prev));
  }, [capacity]);

  return buffer;
}
