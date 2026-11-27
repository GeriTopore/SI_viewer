// Zone axes: parsing, the real-space direction of [uvw], and the nearest one.

import type { Vec3 } from "./structure";

/** Low-index directions searched for the nearest zone axis go up to this. */
const MAX_INDEX = 3;
/** Within this angle (degrees) the view is on the zone axis. */
export const ON_AXIS = 0.01;
const LENGTH_TOLERANCE = 1e-3;
const ANGLE_TOLERANCE = 0.01;

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : Math.abs(a));
const reduce = (indices: number[]): number[] => {
  const d = indices.reduce(gcd, 0) || 1;
  return indices.map((x) => x / d + 0);
};

const dot = (p: number[], q: number[]) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
const normalised = (p: number[]): Vec3 => {
  const n = Math.sqrt(dot(p, p));
  return p.map((x) => x / n) as Vec3;
};

/** Whether a, b, c, α, β, γ describe a hexagonal cell (γ = 120°). */
export const isHexagonal = ([a, b, , alpha, beta, gamma]: number[]): boolean =>
  Math.abs(a - b) < LENGTH_TOLERANCE * a &&
  Math.abs(alpha - 90) < ANGLE_TOLERANCE &&
  Math.abs(beta - 90) < ANGLE_TOLERANCE &&
  Math.abs(gamma - 120) < ANGLE_TOLERANCE;

/** a, b, c (Å), α, β, γ (degrees) of a cell given as three Cartesian vectors. */
export function cellParameters(cell: Vec3[]): number[] {
  const [a, b, c] = cell;
  const length = (p: Vec3) => Math.sqrt(dot(p, p));
  const angle = (p: Vec3, q: Vec3) => (Math.acos(Math.max(-1, Math.min(1, dot(p, q) / (length(p) * length(q))))) * 180) / Math.PI;
  return [length(a), length(b), length(c), angle(b, c), angle(a, c), angle(a, b)];
}

/** [uvw] from text such as "110", "1 -1 0", "[1,-1,0]" or, for a hexagonal
 * cell, "2-1-10". Throws with a short message if it isn't one. */
export function parseZone(text: string, hexagonal = false): number[] {
  const body = text.trim().replace(/[[\](){}<>]/g, "").trim();
  let tokens: string[];
  if (/[\s,]/.test(body)) tokens = body.split(/[\s,]+/).filter(Boolean);
  else tokens = /^(-?\d)+$/.test(body) ? (body.match(/-?\d/g) ?? []) : [];
  if (!tokens.length || !tokens.every((t) => /^-?\d+$/.test(t))) throw new Error(hexagonal ? "Use [uvw] or [uvtw]" : "Use [uvw]");
  const n = tokens.map(Number);
  if (n.length === 4) {
    if (!hexagonal) throw new Error("[uvtw] needs a hexagonal cell");
    const [u, v, t, w] = n;
    if (u + v + t !== 0) throw new Error("u + v + t must be 0");
    if (!u && !v && !w) throw new Error("Not a direction");
    return reduce([u - t, v - t, w]);
  }
  if (n.length !== 3) throw new Error(hexagonal ? "Use [uvw] or [uvtw]" : "Use [uvw]");
  if (n.every((x) => x === 0)) throw new Error("Not a direction");
  return reduce(n);
}

/** The Cartesian direction of [uvw] in ``cell`` (a, b, c as rows). */
export const zoneVector = (cell: Vec3[], [u, v, w]: number[]): Vec3 =>
  [0, 1, 2].map((i) => u * cell[0][i] + v * cell[1][i] + w * cell[2][i]) as Vec3;

/** [uvw] as [uvtw] (the three-index direction of a hexagonal cell). */
export function toFourIndex([U, V, W]: number[]): number[] {
  const u = 2 * U - V;
  const v = 2 * V - U;
  return reduce([u, v, -(u + v), 3 * W]);
}

/** Indices in brackets, negatives with a bar over them. */
export function formatZone(indices: number[]): string {
  const wide = indices.some((x) => Math.abs(x) > 9);
  const text = indices.map((x) => (x < 0 ? [...String(-x)].map((c) => `${c}̅`).join("") : String(x)));
  return `[${text.join(wide ? " " : "")}]`;
}

/** A function from a Cartesian view direction to the nearest low-index zone
 * axis of ``cell``: {uvw, angle (degrees)}; the lowest index wins a tie. */
export function zoneFinder(cell: Vec3[]): (view: number[]) => { uvw: number[]; angle: number } {
  const candidates: { uvw: number[]; direction: Vec3 }[] = [];
  for (let u = -MAX_INDEX; u <= MAX_INDEX; u++) {
    for (let v = -MAX_INDEX; v <= MAX_INDEX; v++) {
      for (let w = -MAX_INDEX; w <= MAX_INDEX; w++) {
        if ([u, v, w].reduce(gcd, 0) !== 1) continue;
        candidates.push({ uvw: [u + 0, v + 0, w + 0], direction: normalised(zoneVector(cell, [u, v, w])) });
      }
    }
  }
  const size = ({ uvw }: { uvw: number[] }) => dot(uvw, uvw);
  candidates.sort((p, q) => size(p) - size(q));
  return (view) => {
    const d = normalised(view);
    let best = candidates[0];
    let bestDot = -2;
    for (const c of candidates) {
      const x = dot(c.direction, d);
      if (x > bestDot + 1e-10) {
        best = c;
        bestDot = x;
      }
    }
    // acos loses precision near 1: the cross product's length is exact there.
    const p = best.direction;
    const cross = [p[1] * d[2] - p[2] * d[1], p[2] * d[0] - p[0] * d[2], p[0] * d[1] - p[1] * d[0]];
    const angle = (Math.atan2(Math.sqrt(dot(cross, cross)), bestDot) * 180) / Math.PI;
    return { uvw: best.uvw, angle };
  };
}
