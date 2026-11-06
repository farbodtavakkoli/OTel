import { useEffect, useId, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { useQueries } from "@tanstack/react-query";

import { apiFetch } from "@/api/client";
import type {
  TrainingHistoryPoint,
  TrainingJobDetail,
} from "@/api/training";
import { useModelList } from "@/api/training";
import { readJson, writeJson } from "@/lib/preferences";

interface LossChartProps {
  history: TrainingHistoryPoint[];
  jobHash?: string;
  maxSteps?: number;
}

const SMOOTH_KEY = "loss-chart.smoothing";
const SMOOTH_MAX = 0.99;

const COMPARE_PALETTE = [
  "#22c55e",
  "#f59e0b",
  "#a855f7",
  "#ec4899",
  "#06b6d4",
  "#ef4444",
];

const HASH_RE = /^[0-9a-f]+$/i;

interface Series {
  key: string;
  label: string;
  color: string;
  history: TrainingHistoryPoint[];
  loading: boolean;
  error: string | null;
  isPrimary: boolean;
}

interface ViewDomain {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

interface DragRect {
  startX: number;
  startY: number;
  curX: number;
  curY: number;
}

const MIN_DRAG_PX = 6;

export function LossChart({ history, jobHash, maxSteps }: LossChartProps) {
  const [logY, setLogY] = useState(false);
  const [hoverFrac, setHoverFrac] = useState<number | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [compareHashes, setCompareHashes] = useState<string[]>([]);
  const [compareInput, setCompareInput] = useState("");
  const [compareError, setCompareError] = useState<string | null>(null);
  const [viewDomain, setViewDomain] = useState<ViewDomain | null>(null);
  const [drag, setDrag] = useState<DragRect | null>(null);
  const [smoothing, setSmoothing] = useState<number>(() =>
    clamp(readJson<number>(SMOOTH_KEY, 0), 0, SMOOTH_MAX),
  );
  const clipId = `loss-clip-${useId().replace(/:/g, "")}`;

  useEffect(() => {
    writeJson(SMOOTH_KEY, smoothing);
  }, [smoothing]);

  useEffect(() => {
    if (!fullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fullscreen]);

  const modelList = useModelList();

  const compareQueries = useQueries({
    queries: compareHashes.map((hash) => ({
      queryKey: ["training-job", hash],
      queryFn: () => apiFetch<TrainingJobDetail>(`/megatron/train/${hash}`),
      staleTime: 0,
      refetchInterval: (query: { state: { data?: TrainingJobDetail } }) => {
        const s = query.state.data?.job_status?.status;
        return s === "QUEUED" || s === "TRAINING" ? 10_000 : 60_000;
      },
    })),
  });

  const series = useMemo<Series[]>(() => {
    const primary: Series = {
      key: jobHash ?? "primary",
      label: jobHash ? jobHash.slice(0, 8) : "this job",
      color: "rgb(var(--color-accent))",
      history,
      loading: false,
      error: null,
      isPrimary: true,
    };
    const compares = compareHashes.map<Series>((hash, idx) => {
      const q = compareQueries[idx];
      const detail = q?.data as TrainingJobDetail | undefined;
      const compareHistory = detail?.job_status?.history ?? [];
      return {
        key: hash,
        label: hash.slice(0, 8),
        color: COMPARE_PALETTE[idx % COMPARE_PALETTE.length],
        history: compareHistory,
        loading: q?.isPending ?? false,
        error: q?.error
          ? (q.error as Error).message ?? String(q.error)
          : null,
        isPrimary: false,
      };
    });
    return [primary, ...compares];
  }, [history, jobHash, compareHashes, compareQueries]);

  const dims = fullscreen
    ? { width: 1600, height: 800, padL: 60, padR: 24, padT: 16, padB: 40 }
    : { width: 800, height: 240, padL: 40, padR: 8, padT: 8, padB: 24 };
  const plotW = dims.width - dims.padL - dims.padR;
  const plotH = dims.height - dims.padT - dims.padB;

  const computation = useMemo(
    () => computeChart(series, logY, smoothing, dims, viewDomain),
    [series, logY, smoothing, dims, viewDomain],
  );

  const {
    perSeriesPaths,
    perSeriesSmoothedPaths,
    perSeriesSmoothed,
    epochMarkers,
    loMin,
    loMax,
    xMin,
    xMax,
  } = computation;

  const xForStep = (step: number) =>
    dims.padL +
    (xMax === xMin ? 0 : ((step - xMin) / (xMax - xMin)) * plotW);

  const xFromViewbox = (vx: number) =>
    xMin + clamp((vx - dims.padL) / plotW, 0, 1) * (xMax - xMin);
  const yFromViewbox = (vy: number) => {
    const frac = 1 - clamp((vy - dims.padT) / plotH, 0, 1);
    if (logY) {
      const lo = Math.log10(Math.max(loMin, 1e-12));
      const hi = Math.log10(Math.max(loMax, 1e-12));
      return Math.pow(10, lo + frac * (hi - lo));
    }
    return loMin + frac * (loMax - loMin);
  };

  const eventToViewbox = (
    e:
      | React.MouseEvent<SVGElement>
      | React.PointerEvent<SVGElement>,
  ): { vx: number; vy: number } => {
    const svg = e.currentTarget.ownerSVGElement ?? (e.currentTarget as SVGSVGElement);
    const rect = svg.getBoundingClientRect();
    return {
      vx: ((e.clientX - rect.left) / rect.width) * dims.width,
      vy: ((e.clientY - rect.top) / rect.height) * dims.height,
    };
  };

  const insidePlot = (vx: number, vy: number) =>
    vx >= dims.padL &&
    vx <= dims.padL + plotW &&
    vy >= dims.padT &&
    vy <= dims.padT + plotH;

  const handleResetZoom = () => {
    setViewDomain(null);
    setDrag(null);
  };

  const resolveHash = (raw: string): string | null => {
    const cleaned = raw.trim().toLowerCase();
    if (!cleaned) return null;
    if (!HASH_RE.test(cleaned)) {
      setCompareError("Hash must be hexadecimal");
      return null;
    }
    if (cleaned.length >= 64) {
      return cleaned.slice(0, 64);
    }
    const matches = (modelList.data ?? []).filter((m) =>
      m.name.toLowerCase().startsWith(cleaned),
    );
    if (matches.length === 0) {
      setCompareError(`No job hash starts with "${cleaned}"`);
      return null;
    }
    if (matches.length > 1) {
      setCompareError(
        `Prefix is ambiguous (${matches.length} matches). Type more characters.`,
      );
      return null;
    }
    return matches[0].name;
  };

  const handleAddCompare = () => {
    setCompareError(null);
    const resolved = resolveHash(compareInput);
    if (!resolved) return;
    if (jobHash && resolved === jobHash) {
      setCompareError("That's the current job.");
      return;
    }
    if (compareHashes.includes(resolved)) {
      setCompareError("Already added.");
      return;
    }
    setCompareHashes((prev) => [...prev, resolved]);
    setCompareInput("");
  };

  const handleRemoveCompare = (hash: string) => {
    setCompareHashes((prev) => prev.filter((h) => h !== hash));
  };

  const totalSamples = series.reduce((n, s) => n + s.history.length, 0);
  if (history.length < 2 && compareHashes.length === 0) {
    return (
      <div className="flex flex-col gap-2">
        <Toolbar
          fullscreen={false}
          onFullscreen={() => setFullscreen(true)}
          smoothing={smoothing}
          onSmoothingChange={setSmoothing}
          logY={logY}
          onLogYChange={setLogY}
          compareInput={compareInput}
          onCompareInputChange={(v) => {
            setCompareInput(v);
            setCompareError(null);
          }}
          onAddCompare={handleAddCompare}
          compareError={compareError}
          zoomed={false}
          onResetZoom={handleResetZoom}
          countText={
            history.length === 0
              ? "no samples yet"
              : `${history.length} sample · waiting for more`
          }
          stepRange=""
        />
        <div className="flex h-48 items-center justify-center rounded-md border border-border-subtle bg-bg/50 text-sm text-fg-muted">
          Waiting for training samples…
        </div>
      </div>
    );
  }

  const hoveredStep =
    hoverFrac !== null
      ? Math.round(xMin + hoverFrac * (xMax - xMin))
      : null;

  const hoverReadout =
    hoveredStep !== null
      ? series
          .map((s, sIdx) => {
            if (s.history.length === 0) return null;
            const nearest = findNearestByStep(s.history, hoveredStep);
            if (!nearest) return null;
            const smoothed = perSeriesSmoothed[sIdx]?.[nearest.index] ?? null;
            return {
              series: s,
              point: nearest.point,
              smoothed,
            };
          })
          .filter((r): r is NonNullable<typeof r> => r !== null)
      : [];

  const chartBody = (
    <>
      <Toolbar
        fullscreen={fullscreen}
        onFullscreen={() => setFullscreen((f) => !f)}
        smoothing={smoothing}
        onSmoothingChange={setSmoothing}
        logY={logY}
        onLogYChange={setLogY}
        compareInput={compareInput}
        onCompareInputChange={(v) => {
          setCompareInput(v);
          setCompareError(null);
        }}
        onAddCompare={handleAddCompare}
        compareError={compareError}
        zoomed={viewDomain !== null}
        onResetZoom={handleResetZoom}
        countText={`${totalSamples} samples across ${series.length} job${
          series.length === 1 ? "" : "s"
        }`}
        stepRange={
          Number.isFinite(xMin) && Number.isFinite(xMax)
            ? `step ${formatStep(xMin)} → ${formatStep(xMax)}${
                maxSteps ? ` / ${maxSteps}` : ""
              }`
            : ""
        }
      />
      {series.length > 1 && (
        <Legend
          series={series}
          onRemove={handleRemoveCompare}
        />
      )}
      <div
        className={clsx(
          "relative rounded-md border border-border-subtle bg-bg/50 p-2",
          fullscreen && "flex-1 min-h-0",
        )}
      >
        <svg
          viewBox={`0 0 ${dims.width} ${dims.height}`}
          className={clsx(
            "w-full select-none",
            fullscreen ? "h-full" : "",
            drag ? "cursor-crosshair" : "cursor-crosshair",
          )}
          preserveAspectRatio={fullscreen ? "none" : "xMidYMid meet"}
          onPointerDown={(e) => {
            if (e.button !== 0) return;
            const { vx, vy } = eventToViewbox(e);
            if (!insidePlot(vx, vy)) return;
            e.preventDefault();
            try {
              e.currentTarget.setPointerCapture(e.pointerId);
            } catch {
            }
            setDrag({ startX: vx, startY: vy, curX: vx, curY: vy });
            setHoverFrac(null);
          }}
          onPointerMove={(e) => {
            const { vx, vy } = eventToViewbox(e);
            if (drag) {
              setDrag({ ...drag, curX: vx, curY: vy });
              return;
            }
            const frac = (vx - dims.padL) / plotW;
            if (frac < 0 || frac > 1) {
              setHoverFrac(null);
              return;
            }
            setHoverFrac(frac);
          }}
          onPointerUp={(e) => {
            try {
              e.currentTarget.releasePointerCapture(e.pointerId);
            } catch {
            }
            if (!drag) return;
            const { vx, vy } = eventToViewbox(e);
            const cvx = clamp(vx, dims.padL, dims.padL + plotW);
            const cvy = clamp(vy, dims.padT, dims.padT + plotH);
            const dx = Math.abs(cvx - drag.startX);
            const dy = Math.abs(cvy - drag.startY);
            if (dx < MIN_DRAG_PX || dy < MIN_DRAG_PX) {
              setDrag(null);
              return;
            }
            const x1 = xFromViewbox(Math.min(drag.startX, cvx));
            const x2 = xFromViewbox(Math.max(drag.startX, cvx));
            const yTop = yFromViewbox(Math.min(drag.startY, cvy));
            const yBot = yFromViewbox(Math.max(drag.startY, cvy));
            setViewDomain({
              xMin: Math.min(x1, x2),
              xMax: Math.max(x1, x2),
              yMin: Math.min(yTop, yBot),
              yMax: Math.max(yTop, yBot),
            });
            setDrag(null);
          }}
          onPointerCancel={() => setDrag(null)}
          onPointerLeave={() => {
            if (!drag) setHoverFrac(null);
          }}
          onDoubleClick={handleResetZoom}
        >
          <defs>
            <clipPath id={clipId}>
              <rect
                x={dims.padL}
                y={dims.padT}
                width={plotW}
                height={plotH}
              />
            </clipPath>
          </defs>
          <rect
            x={dims.padL}
            y={dims.padT}
            width={plotW}
            height={plotH}
            fill="transparent"
          />
          <g clipPath={`url(#${clipId})`}>
            {                                          }
            {epochMarkers.map(({ step, epoch }) => (
              <line
                key={`ep-${step}`}
                x1={xForStep(step)}
                x2={xForStep(step)}
                y1={dims.padT}
                y2={dims.padT + plotH}
                stroke="rgb(var(--color-border))"
                strokeWidth={1}
                strokeDasharray="3 3"
              >
                <title>epoch {epoch}</title>
              </line>
            ))}
            {
                                                           }
            {perSeriesPaths.map((paths, sIdx) =>
              paths.map((d, segIdx) => (
                <path
                  key={`raw-${series[sIdx].key}-${segIdx}`}
                  d={d}
                  fill="none"
                  stroke={series[sIdx].color}
                  strokeWidth={smoothing > 0 ? 1 : 1.5}
                  strokeOpacity={smoothing > 0 ? 0.35 : 1}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )),
            )}
            {smoothing > 0 &&
              perSeriesSmoothedPaths.map((paths, sIdx) =>
                paths.map((d, segIdx) => (
                  <path
                    key={`sm-${series[sIdx].key}-${segIdx}`}
                    d={d}
                    fill="none"
                    stroke={series[sIdx].color}
                    strokeWidth={1.75}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )),
              )}
            {                                                             }
            {!drag && hoveredStep !== null && (
              <line
                x1={xForStep(hoveredStep)}
                x2={xForStep(hoveredStep)}
                y1={dims.padT}
                y2={dims.padT + plotH}
                stroke="rgb(var(--color-fg-muted))"
                strokeWidth={1}
              />
            )}
            {                    }
            {drag && (
              <rect
                x={Math.min(drag.startX, drag.curX)}
                y={Math.min(drag.startY, drag.curY)}
                width={Math.abs(drag.curX - drag.startX)}
                height={Math.abs(drag.curY - drag.startY)}
                fill="rgb(var(--color-accent))"
                fillOpacity={0.12}
                stroke="rgb(var(--color-accent))"
                strokeOpacity={0.6}
                strokeWidth={1}
                strokeDasharray="4 3"
              />
            )}
          </g>
          {                   }
          <text
            x={dims.padL - 4}
            y={dims.padT + 10}
            textAnchor="end"
            fontSize={fullscreen ? 14 : 10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatLoss(loMax)}
          </text>
          <text
            x={dims.padL - 4}
            y={dims.padT + plotH - 2}
            textAnchor="end"
            fontSize={fullscreen ? 14 : 10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatLoss(loMin)}
          </text>
          {                   }
          <text
            x={dims.padL}
            y={dims.height - 6}
            fontSize={fullscreen ? 14 : 10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatStep(xMin)}
          </text>
          <text
            x={dims.width - dims.padR}
            y={dims.height - 6}
            textAnchor="end"
            fontSize={fullscreen ? 14 : 10}
            fill="rgb(var(--color-fg-subtle))"
            className="font-mono"
          >
            {formatStep(xMax)}
          </text>
        </svg>
        {!drag && hoveredStep !== null && hoverReadout.length > 0 && (
          <div
            className="pointer-events-none absolute top-2 rounded-md border border-border-subtle bg-bg-card px-2 py-1 font-mono text-[11px] shadow-lg"
            style={{
              left: `${(xForStep(hoveredStep) / dims.width) * 100}%`,
              transform: "translateX(8px)",
              maxWidth: "260px",
            }}
          >
            <div className="text-fg-subtle">step {hoveredStep}</div>
            {hoverReadout.map(({ series: s, point, smoothed }) => (
              <div key={s.key} className="mt-0.5 flex items-baseline gap-1">
                <span
                  aria-hidden
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: s.color }}
                />
                <span className="text-fg-subtle">{s.label}</span>
                <span className="text-fg">{formatLoss(point.loss)}</span>
                {smoothing > 0 && smoothed !== null && (
                  <span className="text-fg-subtle">
                    (ema {formatLoss(smoothed)})
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );

  if (fullscreen) {
    return (
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Loss chart, full screen"
        className="fixed inset-0 z-50 flex flex-col gap-2 bg-bg p-4 sm:p-6"
      >
        {chartBody}
      </div>
    );
  }

  return <div className="flex flex-col gap-2">{chartBody}</div>;
}

interface ToolbarProps {
  fullscreen: boolean;
  onFullscreen: () => void;
  smoothing: number;
  onSmoothingChange: (v: number) => void;
  logY: boolean;
  onLogYChange: (v: boolean) => void;
  compareInput: string;
  onCompareInputChange: (v: string) => void;
  onAddCompare: () => void;
  compareError: string | null;
  zoomed: boolean;
  onResetZoom: () => void;
  countText: string;
  stepRange: string;
}

function Toolbar({
  fullscreen,
  onFullscreen,
  smoothing,
  onSmoothingChange,
  logY,
  onLogYChange,
  compareInput,
  onCompareInputChange,
  onAddCompare,
  compareError,
  zoomed,
  onResetZoom,
  countText,
  stepRange,
}: ToolbarProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="text-xs text-fg-muted">
          {countText}
          {stepRange ? ` · ${stepRange}` : ""}
        </div>
        <div className="flex items-center gap-1.5">
          <label
            htmlFor="loss-compare-input"
            className="text-xs text-fg-muted"
          >
            compare
          </label>
          <input
            id="loss-compare-input"
            ref={inputRef}
            type="text"
            value={compareInput}
            onChange={(e) => onCompareInputChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onAddCompare();
              }
            }}
            placeholder="job hash or prefix"
            spellCheck={false}
            className="w-44 rounded-md border border-border-subtle bg-bg-card px-2 py-1 font-mono text-xs text-fg placeholder-fg-subtle focus:border-border focus:outline-none sm:w-56"
          />
          <button
            type="button"
            onClick={onAddCompare}
            disabled={compareInput.trim().length === 0}
            className="rounded-md border border-border-subtle bg-bg-card px-2 py-1 text-xs text-fg hover:border-border hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            Add
          </button>
        </div>
        {
                                                                    }
        <div className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-2">
          <label
            className="flex items-center gap-2 text-xs text-fg-muted"
            title="Exponential moving average. 0 = raw, 0.99 = heavily smoothed."
          >
            smoothing
            <input
              type="range"
              min={0}
              max={SMOOTH_MAX}
              step={0.01}
              value={smoothing}
              onChange={(e) => onSmoothingChange(Number(e.target.value))}
              aria-label="Loss smoothing"
              className="w-32 accent-accent"
            />
            <span className="w-10 text-right font-mono">
              {smoothing.toFixed(2)}
            </span>
          </label>
          <label className="flex items-center gap-2 text-xs text-fg-muted">
            <input
              type="checkbox"
              checked={logY}
              onChange={(e) => onLogYChange(e.target.checked)}
              className="h-3 w-3 accent-accent"
            />
            log y
          </label>
          {zoomed && (
            <button
              type="button"
              onClick={onResetZoom}
              title="Reset zoom (or double-click the chart)"
              className="rounded-md border border-accent/40 bg-accent/10 px-2 py-1 text-xs text-accent hover:border-accent hover:bg-accent/20"
            >
              Reset zoom
            </button>
          )}
          <button
            type="button"
            onClick={onFullscreen}
            aria-label={fullscreen ? "Exit full screen" : "Full screen"}
            title={fullscreen ? "Exit full screen (Esc)" : "Full screen"}
            className="rounded-md border border-border-subtle bg-bg-card px-2 py-1 text-xs text-fg-muted hover:border-border hover:bg-bg-hover hover:text-fg"
          >
            {fullscreen ? "⛶ Exit" : "⛶ Fullscreen"}
          </button>
        </div>
      </div>
      {compareError && (
        <div role="alert" className="text-[11px] text-danger">
          {compareError}
        </div>
      )}
    </div>
  );
}

function Legend({
  series,
  onRemove,
}: {
  series: Series[];
  onRemove: (hash: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
      {series.map((s) => (
        <div
          key={s.key}
          className="flex items-center gap-1.5 rounded-md border border-border-subtle bg-bg-card px-2 py-0.5"
        >
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ background: s.color }}
          />
          <span className="font-mono text-fg">{s.label}</span>
          {s.loading && (
            <span className="text-fg-subtle">loading…</span>
          )}
          {s.error && (
            <span className="text-danger" title={s.error}>
              error
            </span>
          )}
          {!s.isPrimary && (
            <button
              type="button"
              onClick={() => onRemove(s.key)}
              aria-label={`Remove ${s.label}`}
              className="rounded px-1 text-fg-subtle hover:bg-bg-hover hover:text-danger"
            >
              ✕
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

interface ChartComputation {
  perSeriesPaths: string[][];
  perSeriesSmoothedPaths: string[][];
  perSeriesSmoothed: (number | null)[][];
  epochMarkers: { step: number; epoch: number }[];
  loMin: number;
  loMax: number;
  xMin: number;
  xMax: number;
}

interface ChartDims {
  width: number;
  height: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
}

function computeChart(
  series: Series[],
  logY: boolean,
  smoothing: number,
  dims: ChartDims,
  viewDomain: ViewDomain | null,
): ChartComputation {
  const empty: ChartComputation = {
    perSeriesPaths: [],
    perSeriesSmoothedPaths: [],
    perSeriesSmoothed: [],
    epochMarkers: [],
    loMin: 0,
    loMax: 1,
    xMin: 0,
    xMax: 1,
  };
  const allPoints = series.flatMap((s) => s.history);
  if (allPoints.length === 0) return empty;

  const plotW = dims.width - dims.padL - dims.padR;
  const plotH = dims.height - dims.padT - dims.padB;

  const transform = (loss: number) =>
    logY ? (loss > 0 ? Math.log10(loss) : null) : loss;

  const perSeriesSmoothed: (number | null)[][] = series.map((s) =>
    applyEma(
      s.history.map((h) => h.loss),
      smoothing,
    ),
  );

  const ys: number[] = [];
  series.forEach((s, idx) => {
    s.history.forEach((h) => {
      const t = transform(h.loss);
      if (t !== null && Number.isFinite(t)) ys.push(t);
    });
    perSeriesSmoothed[idx].forEach((v) => {
      if (v === null) return;
      const t = transform(v);
      if (t !== null && Number.isFinite(t)) ys.push(t);
    });
  });
  if (ys.length === 0) return empty;
  const autoLoMinT = Math.min(...ys);
  const autoLoMaxT = Math.max(...ys);

  const steps = allPoints.map((p) => p.step);
  const autoXMin = Math.min(...steps);
  const autoXMax = Math.max(...steps);

  let xMin = autoXMin;
  let xMax = autoXMax;
  let loMinT = autoLoMinT;
  let loMaxT = autoLoMaxT;
  if (viewDomain) {
    xMin = viewDomain.xMin;
    xMax = viewDomain.xMax;
    const yMinT = transform(viewDomain.yMin);
    const yMaxT = transform(viewDomain.yMax);
    if (yMinT !== null && yMaxT !== null && Number.isFinite(yMinT) && Number.isFinite(yMaxT)) {
      loMinT = Math.min(yMinT, yMaxT);
      loMaxT = Math.max(yMinT, yMaxT);
    }
  }

  const xRange = xMax - xMin || 1;
  const yRange = loMaxT - loMinT || 1;

  const xForStep = (step: number) =>
    dims.padL + ((step - xMin) / xRange) * plotW;
  const yForLoss = (loss: number | null): number | null => {
    if (loss === null) return null;
    const t = transform(loss);
    if (t === null) return null;
    return dims.padT + plotH - ((t - loMinT) / yRange) * plotH;
  };

  const buildSeriesPaths = (
    history: TrainingHistoryPoint[],
    rawValues: number[] | null,
    smoothedValues: (number | null)[] | null,
  ): string[] => {
    const out: string[] = [];
    let current: string[] = [];
    const flush = () => {
      if (current.length >= 2) out.push(current.join(" "));
      current = [];
    };
    history.forEach((h, i) => {
      const v = rawValues ? rawValues[i] : smoothedValues![i];
      const y = yForLoss(v);
      if (y === null) {
        flush();
        return;
      }
      const x = xForStep(h.step);
      current.push(
        `${current.length === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`,
      );
    });
    flush();
    return out;
  };

  const perSeriesPaths = series.map((s) =>
    buildSeriesPaths(
      s.history,
      s.history.map((h) => h.loss),
      null,
    ),
  );
  const perSeriesSmoothedPaths =
    smoothing > 0
      ? series.map((s, idx) =>
          buildSeriesPaths(s.history, null, perSeriesSmoothed[idx]),
        )
      : [];

  const epochMarkers: { step: number; epoch: number }[] = [];
  const primary = series[0];
  let lastEpoch: number | undefined;
  primary.history.forEach((h) => {
    if (
      h.epoch !== undefined &&
      lastEpoch !== undefined &&
      h.epoch !== lastEpoch
    ) {
      epochMarkers.push({ step: h.step, epoch: h.epoch });
    }
    lastEpoch = h.epoch;
  });

  return {
    perSeriesPaths,
    perSeriesSmoothedPaths,
    perSeriesSmoothed,
    epochMarkers,
    loMin: logY ? Math.pow(10, loMinT) : loMinT,
    loMax: logY ? Math.pow(10, loMaxT) : loMaxT,
    xMin,
    xMax,
  };
}

export function applyEma(
  values: number[],
  weight: number,
): (number | null)[] {
  if (weight <= 0) {
    return values.map((v) => (Number.isFinite(v) ? v : null));
  }
  const out: (number | null)[] = new Array(values.length).fill(null);
  let acc = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      out[i] = null;
      continue;
    }
    acc = weight * acc + (1 - weight) * v;
    n += 1;
    const debias = 1 - Math.pow(weight, n);
    out[i] = debias > 0 ? acc / debias : v;
  }
  return out;
}

function findNearestByStep(
  history: TrainingHistoryPoint[],
  step: number,
): { point: TrainingHistoryPoint; index: number } | null {
  if (history.length === 0) return null;
  let best = 0;
  let bestDist = Math.abs(history[0].step - step);
  for (let i = 1; i < history.length; i++) {
    const d = Math.abs(history[i].step - step);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return { point: history[best], index: best };
}

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(3);
  return v.toFixed(4);
}

function formatStep(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return String(Math.round(v));
}
