import { useEffect, useLayoutEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useUi } from "../state/ui";
import { formatValue } from "../format";

export interface ProfileSeries {
  label: string;
  color: string;
  dash?: number[];
  /** Values at the distances of its table. */
  values: ArrayLike<number>;
  /** "y2": the right-hand axis (e.g. a HAADF profile beside element maps). */
  scale: "y" | "y2";
}

/** Series sampled at the same distances (one profile line). */
export interface ProfileTable {
  x: ArrayLike<number>;
  series: ProfileSeries[];
}

/** A distance range picked on the plot (d0 = d1 for a click). */
export interface DistanceRange {
  d0: number;
  d1: number;
}

interface Props {
  tables: ProfileTable[];
  xUnits: string;
  /** Picking: a click picks a distance and a drag a range (else drag zooms). */
  onPick?: (range: DistanceRange) => void;
  /** Picked range, drawn as a band. */
  picked?: DistanceRange | null;
  onHover?: (d: number | null) => void;
}

const cssVar = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Line-profile plot (uPlot): profiles of several lines (joined on their
 * distances), with a second axis on the right when used. */
export function ProfilePlot({ tables, xUnits, onPick, picked, onHover }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const theme = useUi((s) => s.theme);
  const series = tables.flatMap((t) => t.series);
  const [hover, setHover] = useState<{ x: number; values: (number | null)[] } | null>(null);
  const live = useRef({ onPick, picked, onHover, series });
  live.current = { onPick, picked, onHover, series };

  const data = (): uPlot.AlignedData => {
    if (!tables.length) return [[]];
    const aligned = tables.map(
      (t) =>
        [
          Array.from(t.x),
          ...t.series.map((s) => Array.from(s.values, (v) => (Number.isFinite(v) ? v : null))),
        ] as uPlot.AlignedData,
    );
    return aligned.length === 1 ? aligned[0] : uPlot.join(aligned);
  };

  const hasY2 = series.some((s) => s.scale === "y2");
  const layoutKey = `${theme}|${xUnits}|${hasY2}|${series.map((s) => `${s.label}:${s.color}:${s.scale}:${s.dash ?? ""}`).join(",")}`;

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const text = cssVar("--muted");
    const grid = cssVar("--border");
    const axis = (label?: string): uPlot.Axis => ({
      stroke: text,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
      label,
      labelSize: label ? 16 : 0,
      font: "11px -apple-system, sans-serif",
      labelFont: "11px -apple-system, sans-serif",
    });
    const values = (_u: uPlot, vals: number[]) => vals.map((v) => (v == null ? "" : formatValue(v)));
    const auto: uPlot.Range.Function = (_u, lo, hi) =>
      lo == null || hi == null ? [0, 1] : uPlot.rangeNum(lo, hi, 0.1, true);
    const opts: uPlot.Options = {
      width: el.clientWidth,
      height: el.clientHeight,
      scales: { x: { time: false }, y: { range: auto }, y2: { range: auto } },
      axes: [
        axis(`Distance (${xUnits})`),
        { ...axis(), scale: "y", size: 52, values },
        ...(hasY2 ? [{ ...axis(), scale: "y2", side: 1, size: 52, values, grid: { show: false } }] : []),
      ],
      series: [
        {},
        ...live.current.series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: 1.5,
          dash: s.dash,
          scale: s.scale,
          spanGaps: true,
          points: { show: false },
        })),
      ],
      cursor: { drag: { x: true, y: false, setScale: false }, points: { show: false } },
      legend: { show: false },
      hooks: {
        setCursor: [
          (u) => {
            const i = u.cursor.idx;
            const x = i == null ? null : (u.data[0][i] as number);
            setHover(x == null ? null : { x, values: u.data.slice(1).map((d) => (d[i!] as number | null) ?? null) });
            live.current.onHover?.(x);
          },
        ],
        // Release: pick the range (or the clicked point), or zoom to it.
        setSelect: [
          (u) => {
            const d0 = u.posToVal(u.select.left, "x");
            const d1 = u.posToVal(u.select.left + u.select.width, "x");
            if (u.select.width >= 2) {
              if (live.current.onPick) live.current.onPick({ d0, d1 });
              else u.setScale("x", { min: d0, max: d1 });
            }
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          },
        ],
        draw: [
          (u) => {
            const p = live.current.picked;
            if (!p) return;
            const ctx = u.ctx;
            const a = u.valToPos(p.d0, "x", true);
            const b = u.valToPos(p.d1, "x", true);
            ctx.save();
            ctx.fillStyle = `${cssVar("--accent")}33`;
            ctx.strokeStyle = cssVar("--accent");
            ctx.lineWidth = devicePixelRatio || 1;
            if (Math.abs(b - a) < 2) {
              ctx.beginPath();
              ctx.moveTo(a, u.bbox.top);
              ctx.lineTo(a, u.bbox.top + u.bbox.height);
              ctx.stroke();
            } else ctx.fillRect(Math.min(a, b), u.bbox.top, Math.abs(b - a), u.bbox.height);
            ctx.restore();
          },
        ],
      },
    };
    const plot = new uPlot(opts, data(), el);
    plotRef.current = plot;

    // A click (no drag) picks a point.
    let downX = 0;
    const onDown = (e: MouseEvent) => (downX = e.clientX);
    const onUp = (e: MouseEvent) => {
      if (Math.abs(e.clientX - downX) > 2 || !live.current.onPick) return;
      const rect = plot.over.getBoundingClientRect();
      const d = plot.posToVal(e.clientX - rect.left, "x");
      live.current.onPick({ d0: d, d1: d });
    };
    const onLeave = () => live.current.onHover?.(null);
    const onDblClick = () => plot.setData(plot.data);
    plot.over.addEventListener("mousedown", onDown);
    plot.over.addEventListener("mouseup", onUp);
    plot.over.addEventListener("mouseleave", onLeave);
    plot.over.addEventListener("dblclick", onDblClick);

    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
  }, [layoutKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // New values (e.g. a line being dragged): redraw at full range.
  useEffect(() => {
    plotRef.current?.setData(data());
  }, [tables]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    plotRef.current?.redraw(false);
  }, [picked]);

  return (
    <div className="profile-plot">
      <div className="profile-readout">
        {hover ? (
          <>
            {formatValue(hover.x)} {xUnits}
            {series.map((s, i) =>
              hover.values[i] == null ? null : (
                <span key={i}>
                  {" · "}
                  <span style={{ color: s.color }}>{s.label}</span> {formatValue(hover.values[i]!)}
                </span>
              ),
            )}
          </>
        ) : (
          <span className="muted">
            {onPick ? "Click or drag on the profile to see the spectrum there · " : "Drag to zoom · "}
            double-click resets
          </span>
        )}
      </div>
      <div ref={wrapRef} className="profile-canvas" />
    </div>
  );
}
