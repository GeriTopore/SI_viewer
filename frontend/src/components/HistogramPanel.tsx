import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Frame } from "../api/arrays";
import { useUi } from "../state/ui";
import { formatValue } from "../format";
import { percentile, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";

const BIN_CHOICES = [32, 64, 128, 256, 512];
type Range = "data" | "contrast";

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

interface Stats {
  count: number;
  nan: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  median: number;
}

/** Count, min, max, mean and standard deviation of the finite values (one
 * pass), the median from a sample. */
function statsOf(data: Float32Array): Stats {
  let count = 0, min = Infinity, max = -Infinity, sum = 0, sum2 = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    count++;
    sum += v;
    sum2 += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const mean = count ? sum / count : 0;
  const std = count ? Math.sqrt(Math.max(0, sum2 / count - mean * mean)) : 0;
  return { count, nan: data.length - count, min, max, mean, std, median: percentile(sortedSample(data), 50) };
}

/** Bin centres and counts of the finite values in [lo, hi] (values outside
 * are left out). */
function binsOf(data: Float32Array, lo: number, hi: number, n: number): { x: number[]; y: number[]; width: number } {
  const width = hi > lo ? (hi - lo) / n : 1;
  const y = new Array<number>(n).fill(0);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!(v >= lo && v <= hi)) continue;
    y[Math.min(n - 1, Math.floor((v - lo) / width))]++;
  }
  return { x: y.map((_, k) => lo + (k + 0.5) * width), y, width };
}

/** Bottom-pane tab: the distribution of the values of the image being edited
 * (the toolbar's layer), over its whole range or its contrast limits, which
 * are marked. Drag to zoom, double-click resets. */
export function HistogramPanel({ frame, label, display }: {
  frame: Frame | null;
  label: string;
  display?: DisplayParams;
}) {
  const [bins, setBins] = useState(256);
  const [range, setRange] = useState<Range>("data");
  const [log, setLog] = useState(false);
  const theme = useUi((s) => s.theme);
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [hover, setHover] = useState<{ lo: number; hi: number; count: number } | null>(null);

  const stats = useMemo(() => (frame ? statsOf(frame.data) : null), [frame]);
  const [lo, hi] =
    range === "contrast" && display ? [display.lo, display.hi] : stats && stats.count ? [stats.min, stats.max] : [0, 1];
  const hist = useMemo(() => (frame ? binsOf(frame.data, lo, hi, bins) : null), [frame, lo, hi, bins]);
  const live = useRef({ display, hist });
  live.current = { display, hist };

  const data = (): uPlot.AlignedData =>
    hist ? [hist.x, hist.y.map((c) => (log && c <= 0 ? null : c))] : [[], []];

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const text = cssVar("--muted");
    const grid = cssVar("--border");
    const accent = cssVar("--accent");
    const axis = (name?: string): uPlot.Axis => ({
      stroke: text,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      label: name,
      labelSize: name ? 16 : 0,
      font: "11px -apple-system, sans-serif",
      labelFont: "11px -apple-system, sans-serif",
    });
    const values = (_u: uPlot, vals: number[]) => vals.map((v) => (v == null ? "" : formatValue(v)));
    const plot = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: {
          x: { time: false },
          y: log ? { distr: 3 } : { range: (_u, _lo, hiY) => [0, hiY > 0 ? hiY * 1.05 : 1] },
        },
        axes: [axis("Value"), { ...axis(), size: 52, values }],
        series: [
          {},
          {
            label: "Count",
            stroke: accent,
            fill: `${accent}66`,
            width: 1,
            paths: uPlot.paths.bars!({ size: [1, Infinity], gap: 0 }),
            points: { show: false },
          },
        ],
        cursor: { drag: { x: true, y: false }, points: { show: false } },
        legend: { show: false },
        hooks: {
          setCursor: [
            (u) => {
              const i = u.cursor.idx;
              const h = live.current.hist;
              if (i == null || !h) return setHover(null);
              const x = h.x[i];
              setHover({ lo: x - h.width / 2, hi: x + h.width / 2, count: h.y[i] });
            },
          ],
          // The contrast limits, dashed.
          draw: [
            (u) => {
              const d = live.current.display;
              if (!d) return;
              const ctx = u.ctx;
              const dpr = devicePixelRatio || 1;
              ctx.save();
              ctx.strokeStyle = cssVar("--text");
              ctx.lineWidth = dpr;
              ctx.setLineDash([4 * dpr, 3 * dpr]);
              for (const v of [d.lo, d.hi]) {
                const p = Math.round(u.valToPos(v, "x", true)) + 0.5;
                if (p < u.bbox.left || p > u.bbox.left + u.bbox.width) continue;
                ctx.beginPath();
                ctx.moveTo(p, u.bbox.top);
                ctx.lineTo(p, u.bbox.top + u.bbox.height);
                ctx.stroke();
              }
              ctx.restore();
            },
          ],
        },
      },
      data(),
      el,
    );
    plotRef.current = plot;
    const onLeave = () => setHover(null);
    const onDblClick = () => plot.setData(plot.data);
    plot.over.addEventListener("mouseleave", onLeave);
    plot.over.addEventListener("dblclick", onDblClick);
    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [theme, log]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    plotRef.current?.setData(data());
  }, [hist]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    plotRef.current?.redraw(false);
  }, [display?.lo, display?.hi]);

  return (
    <div className="profile-panel histogram-panel">
      <div className="profile-header">
        <span className="profile-line-name" title="The layer being edited">
          {label}
        </span>
        <span className="profile-spacer" />
        <div className="segmented" role="group" aria-label="Histogram range">
          {(["data", "contrast"] as Range[]).map((r) => (
            <button
              key={r}
              className={range === r ? "active" : undefined}
              aria-pressed={range === r}
              onClick={() => setRange(r)}
              title={r === "data" ? "From the smallest to the largest value" : "Between the contrast limits"}
            >
              {r === "data" ? "Data" : "Contrast"}
            </button>
          ))}
        </div>
        <label className="profile-width" title="Number of bins">
          Bins
          <select value={bins} onChange={(e) => setBins(Number(e.target.value))}>
            {BIN_CHOICES.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
        <div className="segmented" role="group" aria-label="Count scale">
          <button className={!log ? "active" : undefined} aria-pressed={!log} onClick={() => setLog(false)}>
            Linear
          </button>
          <button className={log ? "active" : undefined} aria-pressed={log} onClick={() => setLog(true)}>
            Log
          </button>
        </div>
      </div>
      <div className="profile-plot">
        <div className="profile-readout">
          {hover ? (
            <>
              {formatValue(hover.lo)} – {formatValue(hover.hi)}: {hover.count} px
            </>
          ) : stats && stats.count ? (
            <span className="muted">
              {stats.count} px{stats.nan ? ` (+${stats.nan} not a number)` : ""} · mean {formatValue(stats.mean)} · σ{" "}
              {formatValue(stats.std)} · median {formatValue(stats.median)} · {formatValue(stats.min)} –{" "}
              {formatValue(stats.max)} · drag to zoom · double-click resets
            </span>
          ) : (
            <span className="muted">{frame ? "No values" : "No image"}</span>
          )}
        </div>
        <div ref={wrapRef} className="profile-canvas" />
      </div>
    </div>
  );
}
