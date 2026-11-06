import { useEffect, useLayoutEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useDisplay } from "../state/display";
import { formatValue } from "../format";

export interface SpectrumSeries {
  label: string;
  data: Float32Array;
  color: string;
}

interface Props {
  /** Channel centres along the spectral axis. */
  x: Float64Array;
  series: SpectrumSeries[];
  xLabel: string;
  xUnits: string;
  loading?: string | null;
  error?: string | null;
}

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Spectrum plot (uPlot): drag to zoom energy, wheel to zoom, double-click to reset. */
export function SpectrumPlot({ x, series, xLabel, xUnits, loading, error }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const log = useDisplay((s) => s.spectrumLog);
  const setLog = useDisplay((s) => s.setSpectrumLog);
  const [hover, setHover] = useState<{ x: number; values: (number | null)[] } | null>(null);

  const toData = (): uPlot.AlignedData => [
    x,
    ...series.map((s) => (log ? Array.from(s.data, (v) => (v > 0 ? v : null)) : s.data)),
  ];

  // (Re)build when the series layout or scale type changes; otherwise setData.
  const layoutKey = `${log}|${series.map((s) => `${s.label}:${s.color}`).join(",")}|${xLabel}`;

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const text = cssVar("--muted");
    const grid = cssVar("--border");
    const axis = (label?: string): uPlot.Axis => ({
      stroke: text,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      label,
      labelSize: label ? 18 : 0,
      font: "11px -apple-system, sans-serif",
      labelFont: "11px -apple-system, sans-serif",
    });
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height: el.clientHeight,
      scales: { x: { time: false }, y: log ? { distr: 3, log: 10 } : {} },
      axes: [
        axis(xUnits ? `${xLabel} (${xUnits})` : xLabel),
        { ...axis(), size: 56, values: (_u, vals) => vals.map((v) => (v == null ? "" : formatValue(v))) },
      ],
      series: [
        {},
        ...series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: 1.25,
          fill: `${s.color}22`,
          points: { show: false },
        })),
      ],
      cursor: { drag: { x: true, y: false, setScale: true }, points: { show: false } },
      legend: { show: false },
      hooks: {
        setCursor: [
          (u) => {
            const i = u.cursor.idx;
            setHover(
              i == null
                ? null
                : { x: x[i], values: series.map((s) => (s.data[i] !== undefined ? s.data[i] : null)) },
            );
          },
        ],
      },
    };
    const plot = new uPlot(opts, toData(), el);
    plotRef.current = plot;

    // Wheel zooms the energy axis around the cursor.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { min, max } = plot.scales.x;
      if (min == null || max == null) return;
      const rect = plot.over.getBoundingClientRect();
      const at = plot.posToVal(e.clientX - rect.left, "x");
      const f = Math.exp(e.deltaY * 0.0015);
      const lo = Math.max(x[0], at - (at - min) * f);
      const hi = Math.min(x[x.length - 1], at + (max - at) * f);
      if (hi - lo > (x[1] - x[0]) * 4) plot.setScale("x", { min: lo, max: hi });
    };
    plot.over.addEventListener("wheel", onWheel, { passive: false });

    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [layoutKey, x]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Keep the current energy zoom when only the data changes.
    plotRef.current?.setData(toData(), false);
    const p = plotRef.current;
    if (p) {
      const { min, max } = p.scales.x;
      p.setScale("x", { min: min ?? x[0], max: max ?? x[x.length - 1] });
    }
  }, [series]); // eslint-disable-line react-hooks/exhaustive-deps

  const resetZoom = () => plotRef.current?.setScale("x", { min: x[0], max: x[x.length - 1] });

  return (
    <div className="spectrum-view">
      <div className="spectrum-toolbar">
        <div className="segmented" role="group" aria-label="Intensity scale">
          <button className={!log ? "active" : undefined} aria-pressed={!log} onClick={() => setLog(false)}>
            Linear
          </button>
          <button className={log ? "active" : undefined} aria-pressed={log} onClick={() => setLog(true)}>
            Log
          </button>
        </div>
        <button className="secondary small-button" onClick={resetZoom} title="Show the full range (double-click plot)">
          Reset zoom
        </button>
        <span className="muted small">Drag to zoom · wheel to zoom · double-click to reset</span>
      </div>
      <div className="spectrum-plot-wrap">
        <div ref={wrapRef} className="spectrum-plot" />
        {(loading || error) && (
          <div className={`image-message${error ? " error" : ""}`}>
            {loading && !error && <span className="spinner" aria-hidden />}
            {error ?? loading}
          </div>
        )}
      </div>
      <div className="readout">
        <span>
          {hover ? (
            <>
              {formatValue(hover.x)} {xUnits}
              {hover.values.map((v, i) => (
                <span key={i}>
                  {" · "}
                  <span style={{ color: series[i]?.color }}>{series[i]?.label}</span>{" "}
                  {v === null ? "–" : formatValue(v)}
                </span>
              ))}
            </>
          ) : (
            <span className="muted">
              {x.length} channels · {formatValue(x[0] ?? 0)}–{formatValue(x[x.length - 1] ?? 0)} {xUnits}
            </span>
          )}
        </span>
      </div>
    </div>
  );
}
