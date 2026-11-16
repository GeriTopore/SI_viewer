import { useMemo, useRef } from "react";
import { formatValue } from "../format";
import { sublatticeOf } from "../state/columns";

const W = 260;
const H = 64;
const BINS = 64;

/** A histogram of ``values`` with draggable thresholds (largest first) that
 * split it into groups, each bar in its group's colour. */
export function ThresholdHistogram({ values, thresholds, colours, onChange }: {
  values: Float32Array;
  thresholds: number[];
  colours: string[];
  onChange: (thresholds: number[]) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef<number | null>(null);
  // The bulk of the values (0.5–99.5 %), widened to the thresholds.
  const sorted = useMemo(() => Float32Array.from(values).sort(), [values]);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
  let lo = sorted.length ? at(0.005) : 0;
  let hi = sorted.length ? at(0.995) : 1;
  for (const t of thresholds) [lo, hi] = [Math.min(lo, t), Math.max(hi, t)];
  if (hi <= lo) hi = lo + 1;
  const counts = useMemo(() => {
    const c = new Array(BINS).fill(0);
    for (const v of sorted) {
      const b = Math.floor(((v - lo) / (hi - lo)) * BINS);
      if (b >= 0 && b < BINS) c[b]++;
    }
    return c;
  }, [sorted, lo, hi]);
  const top = Math.max(1, ...counts);
  const xOf = (v: number) => ((v - lo) / (hi - lo)) * W;
  const valueAt = (clientX: number) => {
    const r = svgRef.current!.getBoundingClientRect();
    return lo + ((clientX - r.left) / r.width) * (hi - lo);
  };
  const move = (clientX: number) => {
    const k = dragging.current;
    if (k === null) return;
    // Kept between their neighbours (largest first).
    const upper = k > 0 ? thresholds[k - 1] : Infinity;
    const lower = k < thresholds.length - 1 ? thresholds[k + 1] : -Infinity;
    const v = Math.min(upper, Math.max(lower, valueAt(clientX)));
    onChange(thresholds.map((t, i) => (i === k ? v : t)));
  };
  const sizes = useMemo(() => {
    const n = new Array(thresholds.length + 1).fill(0);
    for (const v of values) {
      const g = sublatticeOf(v, thresholds);
      if (g !== null) n[g]++;
    }
    return n;
  }, [values, thresholds]);

  return (
    <div className="threshold-hist">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        onPointerMove={(e) => move(e.clientX)}
        onPointerUp={() => (dragging.current = null)}
        onPointerLeave={() => (dragging.current = null)}
      >
        {counts.map((c, b) => {
          const v = lo + ((b + 0.5) / BINS) * (hi - lo);
          const h = (c / top) * (H - 2);
          return (
            <rect key={b} x={(b / BINS) * W} y={H - h} width={W / BINS - 0.5} height={h} fill={colours[sublatticeOf(v, thresholds) ?? 0]} />
          );
        })}
        {thresholds.map((t, k) => (
          <g
            key={k}
            className="threshold-line"
            onPointerDown={(e) => {
              dragging.current = k;
              (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture(e.pointerId);
            }}
          >
            <rect x={xOf(t) - 5} y={0} width={10} height={H} fill="transparent" />
            <line x1={xOf(t)} x2={xOf(t)} y1={0} y2={H} />
          </g>
        ))}
      </svg>
      <div className="threshold-axis small muted">
        <span>{formatValue(lo)}</span>
        <span>{thresholds.map((t) => formatValue(t)).join(" · ")}</span>
        <span>{formatValue(hi)}</span>
      </div>
      <div className="small">
        {sizes.map((n, k) => (
          <span key={k} className="threshold-count">
            <span className="threshold-swatch" style={{ background: colours[k] }} />
            {String.fromCharCode(65 + k)} {n}
          </span>
        ))}
      </div>
    </div>
  );
}
