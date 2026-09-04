import { useEffect, useState } from "react";

import { getAlias, subscribe } from "./aliases";

/**
 * Re-renders whenever any alias changes. Returns just the alias for one hash.
 * Callers that need a map should subscribe to the store directly.
 */
export function useAlias(hash: string | undefined): string | undefined {
  const [value, setValue] = useState<string | undefined>(() =>
    hash ? getAlias(hash) : undefined,
  );

  useEffect(() => {
    if (!hash) {
      setValue(undefined);
      return;
    }
    setValue(getAlias(hash));
    return subscribe(() => setValue(getAlias(hash)));
  }, [hash]);

  return value;
}
