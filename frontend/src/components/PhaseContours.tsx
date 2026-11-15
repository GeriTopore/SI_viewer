import { useMemo } from "react";
import type { Frame } from "../api/arrays";
import type { Transform } from "./ImageView";

/** kemstem's plot_phase contour angles (its defaults). */
const ANGLES = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];

/** Line segments (x1, y1, x2, y2 in full-resolution pixels, flat) where
 * ``value`` crosses 0 between pixel centres, by marching squares. A cell with
 * any NaN corner is skipped, as matplotlib's contour skips masked cells. */
function zeroCrossings(value: Float32Array, w: number, h: number, bin: number): number[] {
  const out: number[] = [];
  // Point on an edge between corners (pa at a, pb at b), in pixel-centre coordinates.
  const lerp = (a: number, b: number) => a / (a - b);
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const i = y * w + x;
      const v0 = value[i]; // top-left
      const v1 = value[i + 1]; // top-right
      const v2 = value[i + w + 1]; // bottom-right
      const v3 = value[i + w]; // bottom-left
      if (Number.isNaN(v0) || Number.isNaN(v1) || Number.isNaN(v2) || Number.isNaN(v3)) continue;
      const c = (v0 > 0 ? 1 : 0) | (v1 > 0 ? 2 : 0) | (v2 > 0 ? 4 : 0) | (v3 > 0 ? 8 : 0);
      if (c === 0 || c === 15) continue;
      // Crossings on each edge (top, right, bottom, left), in cell coordinates.
      const top = (): [number, number] => [x + lerp(v0, v1), y];
      const right = (): [number, number] => [x + 1, y + lerp(v1, v2)];
      const bottom = (): [number, number] => [x + lerp(v3, v2), y + 1];
      const left = (): [number, number] => [x, y + lerp(v0, v3)];
      const seg = (a: [number, number], b: [number, number]) =>
        out.push((a[0] + 0.5) * bin, (a[1] + 0.5) * bin, (b[0] + 0.5) * bin, (b[1] + 0.5) * bin);
      switch (c) {
        case 1: case 14: seg(left(), top()); break;
        case 2: case 13: seg(top(), right()); break;
        case 3: case 12: seg(left(), right()); break;
        case 4: case 11: seg(right(), bottom()); break;
        case 6: case 9: seg(top(), bottom()); break;
        case 7: case 8: seg(left(), bottom()); break;
        case 5: case 10: {
          // Saddle: decided by the cell's mean, as matplotlib does.
          const centre = (v0 + v1 + v2 + v3) / 4 > 0;
          if ((c === 5) === centre) {
            seg(left(), bottom());
            seg(top(), right());
          } else {
            seg(left(), top());
            seg(right(), bottom());
          }
          break;
        }
      }
    }
  }
  return out;
}

/** kemstem's phase contours: for each angle a, the zero of sin(φ − a) where
 * cos(φ − a) < 0, i.e. the lines φ = a + π (plot_phase's defaults). */
export function phaseContours(frame: Frame): number[] {
  const { data, width: w, height: h, bin } = frame;
  const value = new Float32Array(data.length);
  const out: number[] = [];
  for (const a of ANGLES) {
    for (let i = 0; i < data.length; i++) {
      const d = data[i] - a;
      value[i] = Math.cos(d) > 0 ? NaN : Math.sin(d);
    }
    const segs = zeroCrossings(value, w, h, bin);
    for (let i = 0; i < segs.length; i++) out.push(segs[i]);
  }
  return out;
}

/** The contours of a phase map, drawn over the image (black, 1 px). */
export function PhaseContours({ t, frame }: { t: Transform; frame: Frame }) {
  const path = useMemo(() => {
    const s = phaseContours(frame);
    const parts: string[] = [];
    for (let i = 0; i < s.length; i += 4) {
      parts.push(`M${s[i].toFixed(2)} ${s[i + 1].toFixed(2)}L${s[i + 2].toFixed(2)} ${s[i + 3].toFixed(2)}`);
    }
    return parts.join("");
  }, [frame]);
  const [x0, y0] = t.toScreen(0, 0);
  return (
    <g className="phase-contours" transform={`translate(${x0} ${y0}) scale(${t.zoom})`}>
      <path d={path} />
    </g>
  );
}

/** kemstem's coarsening_marker: a white circle edged in black, of the
 * coarsening radius (image px), centred at 95 % of the image's width and
 * height (its bottom right, as matplotlib places it on matshow's axes). */
export function CoarseningMarker({ t, width, height, radius }: {
  t: Transform;
  width: number;
  height: number;
  radius: number;
}) {
  // matshow's axes run from -0.5 to N - 0.5 (pixel centres at integers);
  // here pixel i spans [i, i + 1].
  const [cx, cy] = t.toScreen(0.95 * (width - 0.5) + 0.5, 0.95 * (height - 0.5) + 0.5);
  return <circle className="coarsening-marker" cx={cx} cy={cy} r={radius * t.zoom} />;
}
