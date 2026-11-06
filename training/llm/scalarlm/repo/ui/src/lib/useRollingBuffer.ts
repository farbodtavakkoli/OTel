import { useEffect, useRef, useState } from "react";

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
