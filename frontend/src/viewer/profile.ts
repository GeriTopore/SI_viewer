import type { Frame } from "../api/arrays";
import type { AxisCal } from "../components/ImageView";
import type { LineGeometry, ProfileLine } from "../state/profiles";

/** Length of a line in pixels. */
export const pixelLength = (l: LineGeometry) => Math.hypot(l.x1 - l.x0, l.y1 - l.y0);

/** Length of a line in calibrated units (pixels may be non-square). */
export const calibratedLength = (l: LineGeometry, calX: AxisCal, calY: AxisCal) =>
  Math.hypot((l.x1 - l.x0) * calX.scale, (l.y1 - l.y0) * calY.scale);

/** Point at fraction ``t`` along a line. */
export const pointAt = (l: LineGeometry, t: number): [number, number] => [
  l.x0 + (l.x1 - l.x0) * t,
  l.y0 + (l.y1 - l.y0) * t,
];

/** Unit vector perpendicular to a line (zero for a zero-length line). */
function normal(l: LineGeometry): [number, number] {
  const len = pixelLength(l);
  return len ? [-(l.y1 - l.y0) / len, (l.x1 - l.x0) / len] : [0, 0];
}

/** Corners of the band from t0 to t1 along a line, ``width`` pixels across. */
export function bandPolygon(l: LineGeometry, width: number, t0 = 0, t1 = 1): [number, number][] {
  const [nx, ny] = normal(l);
  const h = width / 2;
  const [ax, ay] = pointAt(l, t0);
  const [bx, by] = pointAt(l, t1);
  return [
    [ax + nx * h, ay + ny * h],
    [bx + nx * h, by + ny * h],
    [bx - nx * h, by - ny * h],
    [ax - nx * h, ay - ny * h],
  ];
}

/** Value of ``frame`` at a full-resolution data point, bilinear between
 * pixel centres (NaN outside the image). */
function bilinear(frame: Frame, x: number, y: number): number {
  const fx = x / frame.bin - 0.5;
  const fy = y / frame.bin - 0.5;
  const { width: w, height: h, data } = frame;
  if (fx < -0.5 || fy < -0.5 || fx > w - 0.5 || fy > h - 0.5) return NaN;
  const ix = Math.min(w - 1, Math.max(0, Math.floor(fx)));
  const iy = Math.min(h - 1, Math.max(0, Math.floor(fy)));
  const jx = Math.min(w - 1, ix + 1);
  const jy = Math.min(h - 1, iy + 1);
  const ax = Math.min(1, Math.max(0, fx - ix));
  const ay = Math.min(1, Math.max(0, fy - iy));
  const top = data[iy * w + ix] * (1 - ax) + data[iy * w + jx] * ax;
  const bottom = data[jy * w + ix] * (1 - ax) + data[jy * w + jx] * ax;
  return top * (1 - ay) + bottom * ay;
}

/** Number of samples along a line: about one per pixel, at least two. */
export const sampleCount = (l: LineGeometry) => Math.max(2, Math.floor(pixelLength(l)) + 1);

/** Profile of ``frame`` along ``line``: about one sample per pixel, each the
 * mean across the line's width (samples 1 pixel apart). NaN where the line
 * leaves the image or the data is masked. */
export function sampleProfile(frame: Frame, line: ProfileLine): Float32Array {
  const n = sampleCount(line);
  const [nx, ny] = normal(line);
  const m = Math.max(1, Math.round(line.width));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const [px, py] = pointAt(line, i / (n - 1));
    let sum = 0;
    let count = 0;
    for (let j = 0; j < m; j++) {
      const o = j - (m - 1) / 2;
      const v = bilinear(frame, px + nx * o, py + ny * o);
      if (Number.isFinite(v)) {
        sum += v;
        count++;
      }
    }
    out[i] = count ? sum / count : NaN;
  }
  return out;
}

/** Distance from point (x, y) to the segment of a line, in pixels. */
export function distanceToLine(l: LineGeometry, x: number, y: number): number {
  const dx = l.x1 - l.x0;
  const dy = l.y1 - l.y0;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.min(1, Math.max(0, ((x - l.x0) * dx + (y - l.y0) * dy) / len2)) : 0;
  return Math.hypot(x - (l.x0 + dx * t), y - (l.y0 + dy * t));
}
