import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { SceneOrientation } from "../components/StructureScene";
import { useOrientation } from "./orientation";

/** The template nearest a beam direction, its reflections on that zone (the
 * backend's ``template_toward``). */
type Toward = Awaited<ReturnType<typeof api.templateToward>>;

/** A template drawn as the crystal is seen: reflections at ``x`` (right),
 * ``y`` (up), in 1/Å; the template's ``index`` and ``angle`` off the view. */
export interface ViewPattern {
  index: number;
  angle: number;
  x: number[];
  y: number[];
  intensity: number[];
  hkl: number[][];
  /** The beam direction seen down, [u v w] (largest 1). */
  uvw: number[];
}

function inv3(m: number[][]): number[][] {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
const mul = (m: number[][], v: number[]) => m.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const transpose = (m: number[][]) => m[0].map((_, j) => m.map((r) => r[j]));
const matmul = (a: number[][], b: number[][]) => a.map((r) => b[0].map((_, j) => r[0] * b[0][j] + r[1] * b[1][j] + r[2] * b[2][j]));

/** The beam direction of a view as [u v w] of the cell (largest 1). */
export function uvwOf(o: SceneOrientation): number[] {
  const uvw = mul(inv3(transpose(o.cell)), o.view);
  const top = Math.max(...uvw.map(Math.abs));
  return uvw.map((x) => Math.round((x / top) * 1e4) / 1e4);
}

/** Reflections ``hkl`` (1/Å, no 2π) in view coordinates: x right, y up. */
export function project(o: SceneOrientation, hkl: number[][]): { x: number[]; y: number[] } {
  // Cartesian → view: the cell's unit directions map to the axes reported.
  const unit = o.cell.map((r) => {
    const n = Math.hypot(...r);
    return r.map((x) => x / n);
  });
  const rot = matmul(transpose(o.axes), inv3(transpose(unit)));
  const recip = inv3(o.cell); // g = C⁻¹ · hkl (C: rows a, b, c)
  const x: number[] = [], y: number[] = [];
  for (const h of hkl) {
    const v = mul(rot, mul(recip, h));
    x.push(v[0]);
    y.push(v[1]);
  }
  return { x, y };
}

/** The template of Templates object ``templatesId`` nearest the way its
 * Structure (``structureId``) is seen now, drawn turned as the crystal is
 * (``null`` until the Structure has been shown, or with ``enabled`` off).
 * Requests follow the view: one at a time, the newest next. */
export function useViewTemplate(templatesId: string | null, structureId: string | null, enabled = true): ViewPattern | null {
  const view = useOrientation((s) => (structureId ? s.views[structureId] : undefined));
  const [toward, setToward] = useState<{ key: string; t: Toward } | null>(null);
  const busy = useRef(false);
  const wanted = useRef<{ key: string; uvw: number[] } | null>(null);
  const uvw = enabled && view && templatesId ? uvwOf(view) : null;
  const key = uvw ? `${templatesId}|${uvw.map((x) => x.toFixed(3)).join()}` : "";

  useEffect(() => {
    if (!key || !uvw || toward?.key === key) return;
    wanted.current = { key, uvw };
    const next = () => {
      const w = wanted.current;
      if (!w || busy.current) return;
      wanted.current = null;
      busy.current = true;
      api.templateToward(templatesId!, w.uvw)
        .then((t) => setToward({ key: w.key, t }), () => undefined)
        .finally(() => {
          busy.current = false;
          next();
        });
    };
    next();
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!view || !toward || !uvw || !toward.key.startsWith(`${templatesId}|`)) return null;
  const { x, y } = project(view, toward.t.hkl);
  return { index: toward.t.index, angle: toward.t.angle, x, y, intensity: toward.t.intensity, hkl: toward.t.hkl, uvw };
}
