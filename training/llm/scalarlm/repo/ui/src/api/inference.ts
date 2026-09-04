import { useInfiniteQuery, useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";

/**
 * Backend contract: see docs/inference-request-browser.md §4.
 *
 * The list endpoint paginates by mtime cursor (newest first); the
 * detail endpoint returns the request batch + response + status JSONs
 * for one group_request_id. Both are read-only views over files
 * already on disk in `upload_base_path`.
 */

export interface InferenceListRow {
  request_id: string;
  mtime: number;
  size_bytes: number;
  request_count: number;
  status: string;
  completed_at: number | null;
  model: string;
  request_type: string;
  prompt_preview: string;
  has_response: boolean;
}

export interface InferenceListPage {
  rows: InferenceListRow[];
  next_cursor: number | null;
  has_more: boolean;
}

export interface InferenceRequestDetail {
  request_id: string;
  /** Full request batch JSON, or a {error, ...} placeholder if too large or unreadable. */
  request: unknown;
  /** Response file or null if not yet written. May also be a placeholder. */
  response: unknown;
  /** Status file or null if missing. */
  status: unknown;
  request_mtime: number | null;
  response_mtime: number | null;
}

const PAGE_SIZE = 50;

/**
 * Infinite query that scrolls back through `inference_requests/`.
 * The first page polls every 5 s so freshly-completed batches show
 * up without a manual refresh; further pages only refetch on
 * explicit user action.
 */
export function useInferenceRequestList() {
  return useInfiniteQuery<InferenceListPage, Error>({
    queryKey: ["inference-list"],
    queryFn: async ({ pageParam }) => {
      const params = new URLSearchParams();
      params.set("limit", String(PAGE_SIZE));
      if (pageParam !== undefined && pageParam !== null) {
        params.set("cursor", String(pageParam));
      }
      return apiFetch<InferenceListPage>(`/generate/list_requests?${params}`);
    },
    initialPageParam: null as number | null,
    getNextPageParam: (last) => (last.has_more ? last.next_cursor : undefined),
    refetchInterval: (query) => {
      // Only poll while the user is on page 1; loading more pauses
      // the poll so we don't lose their scroll position to a refetch.
      const pages = query.state.data?.pages.length ?? 0;
      return pages <= 1 ? 5_000 : false;
    },
    staleTime: 0,
  });
}

export function useInferenceRequestDetail(
  requestId: string | null,
  enabled: boolean,
) {
  return useQuery<InferenceRequestDetail, Error>({
    queryKey: ["inference-detail", requestId],
    queryFn: async () => {
      if (!requestId) throw new Error("requestId required");
      return apiFetch<InferenceRequestDetail>(
        `/generate/request/${encodeURIComponent(requestId)}`,
      );
    },
    enabled: enabled && Boolean(requestId),
    staleTime: 60_000,
  });
}

/**
 * Compile a regex from a free-form pattern. Returns null on empty
 * input and undefined on invalid input — callers branch on the two
 * to render an "Invalid regex" hint without crashing the page.
 */
export function compileRegex(pattern: string): RegExp | null | undefined {
  const trimmed = pattern.trim();
  if (!trimmed) return null;
  try {
    return new RegExp(trimmed, "i");
  } catch {
    return undefined;
  }
}

export function rowMatches(row: InferenceListRow, regex: RegExp): boolean {
  const haystack = `${row.request_id} ${row.model} ${row.request_type} ${row.prompt_preview}`;
  return regex.test(haystack);
}
