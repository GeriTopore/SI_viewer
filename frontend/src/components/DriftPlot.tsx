import { useLayoutEffect, useRef, useState } from "react";
import type { Drift } from "../api/client";
import { formatValue, inAngstrom } from "../format";
import { useUi } from "../state/ui";
import type { AxisCal } from "./ImageView";

interface Props {
  /** Each frame's measured position (px; image x right, y down). */
  drift: Drift;
  /** Pixel size (calibrated units, shown in Å for lengths). */
  calX: AxisCal;
  /** Frames to ring (e.g. the pair being inspected). */
  highlight?: number[];
}

const MARGIN = { left: 44, right: 12, top: 10, bottom: 34 };
// Points shade from the first frame to the last: light to dark on a light
// background, dim to bright on a dark one.
const SHADES = {
  light: { first: [172, 208, 245], last: [24, 86, 168] },
  dark: { first: [43, 76, 122], last: [150, 205, 255] },
};

/** A round step for about ``n`` ticks over ``range``. */
function niceStep(range: number, n = 3): number {
  const raw = range / n;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return ([1, 2, 5, 10].find((m) => m * pow >= raw) ?? 10) * pow;
}

/** The measured sample drift: every frame's position (relative to their
 * mean), joined in time and shaded from first (light) to last (dark), with
 * an arrow for the net drift. Equal scales on both axes; y points up as in
 * the image. */
export function DriftPlot({ drift, calX, highlight = [] }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<number | null>(null);
  const theme = useUi((s) => s.theme);
  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const cal = calX.units ? inAngstrom(calX.scale, calX.units) : { size: 1, units: "px" };
  const xs = drift.x.map((v) => v * cal.size);
  const ys = drift.y.map((v) => -v * cal.size); // up in the plot = up in the image
  const n = xs.length;
  const extent = Math.max(1e-9, ...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.2;
  const side = Math.max(0, Math.min(size.w - MARGIN.left - MARGIN.right, size.h - MARGIN.top - MARGIN.bottom));
  const ox = MARGIN.left + (size.w - MARGIN.left - MARGIN.right - side) / 2;
  const oy = MARGIN.top;
  const sx = (v: number) => ox + ((v + extent) / (2 * extent)) * side;
  const sy = (v: number) => oy + ((extent - v) / (2 * extent)) * side;
  const step = niceStep(extent);
  const ticks: number[] = [];
  for (let t = -Math.floor(extent / step) * step; t <= extent + 1e-9; t += step) ticks.push(Number(t.toPrecision(6)));

  const colour = (k: number) => {
    const f = n > 1 ? k / (n - 1) : 1;
    const { first, last } = SHADES[theme];
    const c = first.map((a, i) => Math.round(a + (last[i] - a) * f));
    return `rgb(${c.join(",")})`;
  };

  // Net drift: an arrow from the first frame's position to the last's,
  // set off to one side of the path.
  let arrow = null;
  if (n > 1 && side > 0) {
    const [x0, y0, x1, y1] = [sx(xs[0]), sy(ys[0]), sx(xs[n - 1]), sy(ys[n - 1])];
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len > 12) {
      const [ux, uy] = [(x1 - x0) / len, (y1 - y0) / len];
      const off = 16;
      const [px, py] = [-uy * off, ux * off];
      const angle = (Math.atan2(uy, ux) * 180) / Math.PI;
      const upright = angle > 90 || angle < -90 ? angle + 180 : angle;
      arrow = (
        <g className="drift-arrow">
          <line x1={x0 + px} y1={y0 + py} x2={x1 + px - ux * 6} y2={y1 + py - uy * 6} markerEnd="url(#drift-head)" />
          <text
            x={(x0 + x1) / 2 + px * 1.9}
            y={(y0 + y1) / 2 + py * 1.9}
            transform={`rotate(${upright} ${(x0 + x1) / 2 + px * 1.9} ${(y0 + y1) / 2 + py * 1.9})`}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            drift
          </text>
        </g>
      );
    }
  }

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const [mx, my] = [e.clientX - r.left, e.clientY - r.top];
    let best: number | null = null;
    let bestD = 12;
    for (let k = 0; k < n; k++) {
      const d = Math.hypot(sx(xs[k]) - mx, sy(ys[k]) - my);
      if (d < bestD) [best, bestD] = [k, d];
    }
    setHover(best);
  };

  return (
    <div ref={wrapRef} className="drift-plot">
      {side > 0 && (
        <svg width={size.w} height={size.h} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          <defs>
            <marker id="drift-head" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" className="drift-head" />
            </marker>
          </defs>
          <rect x={ox} y={oy} width={side} height={side} className="drift-frame" />
          <line x1={sx(0)} y1={oy} x2={sx(0)} y2={oy + side} className="drift-zero" />
          <line x1={ox} y1={sy(0)} x2={ox + side} y2={sy(0)} className="drift-zero" />
          {ticks.map((t) => (
            <g key={t} className="drift-tick">
              <text x={sx(t)} y={oy + side + 14} textAnchor="middle">
                {formatValue(t)}
              </text>
              <text x={ox - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle">
                {formatValue(t)}
              </text>
            </g>
          ))}
          <text x={ox + side / 2} y={oy + side + 29} textAnchor="middle" className="drift-label">
            x ({cal.units})
          </text>
          <text
            x={12}
            y={oy + side / 2}
            textAnchor="middle"
            transform={`rotate(-90 12 ${oy + side / 2})`}
            className="drift-label"
          >
            y ({cal.units})
          </text>
          {arrow}
          <polyline points={xs.map((x, k) => `${sx(x)},${sy(ys[k])}`).join(" ")} className="drift-path" />
          {xs.map((x, k) => (
            <circle key={k} cx={sx(x)} cy={sy(ys[k])} r={4} fill={colour(k)} className="drift-point" />
          ))}
          {/* Each frame once: a diagonal cell (i, i) names it twice, and duplicate keys leave stale circles. */}
          {[...new Set(highlight)].map((f) => {
            const k = drift.frames.indexOf(f);
            return k < 0 ? null : <circle key={`h${f}`} cx={sx(xs[k])} cy={sy(ys[k])} r={7} className="drift-highlight" />;
          })}
          {hover !== null && (
            <text x={ox + 6} y={oy + 14} className="drift-hover">
              frame {drift.frames[hover]}: x {formatValue(xs[hover])}, y {formatValue(ys[hover])} {cal.units}
            </text>
          )}
        </svg>
      )}
    </div>
  );
}
