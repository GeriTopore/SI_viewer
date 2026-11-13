import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Frame } from "../api/arrays";
import { useUi } from "../state/ui";
import { formatValue } from "../format";
import { percentile, sortedSample } from "../viewer/contrast";
import type { DisplayParams } from "../viewer/renderer";
import { containsPoint, shapeBounds, type Roi } from "../state/si";

const WHOLE = "whole";
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

/** The values of ``frame`` whose pixel centres are inside ``roi`` (region
 * coordinates are ``scale`` frame pixels per unit, after the frame's own binning). */
function regionValues(frame: Frame, roi: Roi, scale: number): Float32Array {
  const per = (frame.bin || 1) / scale; // region units per frame pixel
  const [x0, y0, x1, y1] = shapeBounds(roi.shape);
  const i0 = Math.max(0, Math.floor(x0 / per)), i1 = Math.min(frame.width - 1, Math.ceil(x1 / per));
  const j0 = Math.max(0, Math.floor(y0 / per)), j1 = Math.min(frame.height - 1, Math.ceil(y1 / per));
  const out: number[] = [];
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      if (containsPoint(roi.shape, (i + 0.5) * per, (j + 0.5) * per)) out.push(frame.data[j * frame.width + i]);
    }
  }
  return Float32Array.from(out);
}

/** Bottom-pane tab: the distribution of the values of the image being edited
 * (the toolbar's layer) and of the pixels in its regions, each ticked in the
 * legend and drawn in its colour, over the whole image's range or its
 * contrast limits, which are marked. Drag to zoom, double-click resets. */
export function HistogramPanel({ frame: whole, label, display, regions }: {
  frame: Frame | null;
  label: string;
  display?: DisplayParams;
  /** The view's regions (``scale``: frame pixels per region unit). */
  regions?: { rois: Roi[]; scale: number };
}) {
  const [bins, setBins] = useState(128);
  // Ticked in the legend: "whole" and region ids.
  const [shown, setShown] = useState<string[]>([WHOLE]);
  const rois = regions?.rois ?? [];
  const toggle = (id: string, on: boolean) => setShown((s) => (on ? [...s.filter((x) => x !== id), id] : s.filter((x) => x !== id)));
  const accent = cssVar("--accent") || "#4c8dff";
  const series = useMemo(() => {
    if (!whole) return [];
    const out: { id: string; name: string; color: string; data: Float32Array }[] = [];
    const ticked = rois.length ? shown : [WHOLE]; // no regions: the whole image
    if (ticked.includes(WHOLE)) out.push({ id: WHOLE, name: "Whole image", color: accent, data: whole.data });
    for (const r of rois) {
      if (ticked.includes(r.id)) out.push({ id: r.id, name: r.name, color: r.color, data: regionValues(whole, r, regions!.scale) });
    }
    return out;
  }, [whole, shown, rois, regions?.scale, accent]); // eslint-disable-line react-hooks/exhaustive-deps
  const frame = whole;
  const [range, setRange] = useState<Range>("data");
  const [log, setLog] = useState(false);
  const theme = useUi((s) => s.theme);
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [hover, setHover] = useState<{ lo: number; hi: number; counts: number[] } | null>(null);

  // Bins over the whole image's range (shared by every series).
  const wholeStats = useMemo(() => (frame ? statsOf(frame.data) : null), [frame]);
  const stats = useMemo(() => (series.length ? statsOf(series[0].data) : null), [series]);
  const [lo, hi] =
    range === "contrast" && display
      ? [display.lo, display.hi]
      : wholeStats && wholeStats.count ? [wholeStats.min, wholeStats.max] : [0, 1];
  const hists = useMemo(() => series.map((s) => binsOf(s.data, lo, hi, bins)), [series, lo, hi, bins]);
  const hist = hists[0] ?? null;
  const live = useRef({ display, hists });
  live.current = { display, hists };
  const seriesKey = series.map((s) => `${s.id}:${s.color}`).join("|");

  const data = (): uPlot.AlignedData =>
    hist ? [hist.x, ...hists.map((h) => h.y.map((c) => (log && c <= 0 ? null : c)))] : [[]];

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const text = cssVar("--muted");
    const grid = cssVar("--border");
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
          ...series.map((sr) => ({
            label: sr.name,
            stroke: sr.color,
            fill: `${sr.color}${series.length > 1 ? "44" : "66"}`,
            width: 1,
            paths: uPlot.paths.bars!({ size: [1, Infinity], gap: 0 }),
            points: { show: false },
          })),
        ],
        cursor: { drag: { x: true, y: false }, points: { show: false } },
        legend: { show: false },
        hooks: {
          setCursor: [
            (u) => {
              const i = u.cursor.idx;
              const hs = live.current.hists;
              if (i == null || !hs.length) return setHover(null);
              const x = hs[0].x[i];
              setHover({ lo: x - hs[0].width / 2, hi: x + hs[0].width / 2, counts: hs.map((h) => h.y[i]) });
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
  }, [theme, log, seriesKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    plotRef.current?.setData(data());
  }, [hists]); // eslint-disable-line react-hooks/exhaustive-deps
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
      {rois.length > 0 && (
        <div className="series-legend">
          {[{ id: WHOLE, name: "Whole image", color: accent }, ...rois].map((r) => (
            <label key={r.id} className={`series-chip${shown.includes(r.id) ? "" : " off"}`}>
              <input type="checkbox" checked={shown.includes(r.id)} onChange={(e) => toggle(r.id, e.target.checked)} />
              <span className="series-swatch" style={{ background: r.color }} />
              {r.name}
            </label>
          ))}
        </div>
      )}
      <div className="profile-plot">
        <div className="profile-readout">
          {hover ? (
            <>
              {formatValue(hover.lo)} – {formatValue(hover.hi)}:{" "}
              {series.length > 1
                ? series.map((sr, k) => `${sr.name} ${hover.counts[k]}`).join(" · ")
                : `${hover.counts[0] ?? 0} px`}
            </>
          ) : stats && stats.count ? (
            <span className="muted">
              {series.length > 1 ? `${series[0].name}: ` : ""}
              {stats.count} px{stats.nan ? ` (+${stats.nan} not a number)` : ""} · mean {formatValue(stats.mean)} · σ{" "}
              {formatValue(stats.std)} · median {formatValue(stats.median)} · {formatValue(stats.min)} –{" "}
              {formatValue(stats.max)} · drag to zoom · double-click resets
            </span>
          ) : (
            <span className="muted">{!frame ? "No image" : series.length ? "No values" : "Nothing ticked"}</span>
          )}
        </div>
        <div ref={wrapRef} className="profile-canvas" />
      </div>
    </div>
  );
}
