import type { PlotHandle } from "../components/SpectrumPlot";
import { PEAK_PARAMS, STEP_PARAMS, valueAt, type ModelComponent } from "../state/curveFit";

/** What a drag of a component's handles works from: its parameters and the
 * handles' places when the drag began. */
export interface DragStart {
  params: ModelComponent["params"];
  handles: Record<string, { x: number; y: number }>;
}

/** Where a component's handles go, from its drawn curve (``xs``, ``curve``
 * as plotted) over the fit range [lo, hi]. Ids are "<part>" (top, left,
 * right, level, slope, p1, p2, pos, amp). */
export function handlesOf(c: ModelComponent, xs: number[], curve: number[], range: [number, number]): PlotHandle[] {
  if (!xs.length || !curve.length) return [];
  const [lo, hi] = range;
  const at = (x: number) => valueAt(xs, curve, x);
  const h = (part: string, x: number, y: number, axis: PlotHandle["axis"], title: string): PlotHandle => ({
    id: part, x, y, color: c.color, axis, title: `${c.name}: ${title}`,
  });
  const peak = PEAK_PARAMS[c.kind];
  if (peak) {
    let i = 0;
    for (let j = 1; j < curve.length; j++) if (curve[j] > curve[i]) i = j;
    const top = curve[i];
    if (!(top > 0)) return [];
    // Where the curve falls to half its height on each side (interpolated).
    const half = top / 2;
    let l = i;
    while (l > 0 && curve[l] > half) l--;
    let r = i;
    while (r < curve.length - 1 && curve[r] > half) r++;
    const cross = (a: number, b: number) =>
      curve[a] === curve[b] ? xs[a] : xs[a] + ((half - curve[a]) / (curve[b] - curve[a])) * (xs[b] - xs[a]);
    const xl = l < i ? cross(l, l + 1) : xs[0];
    const xr = r > i ? cross(r - 1, r) : xs[xs.length - 1];
    return [
      h("top", xs[i], top, "xy", "drag to move the peak"),
      h("left", xl, half, "x", "drag to change the width"),
      h("right", xr, half, "x", "drag to change the width"),
    ];
  }
  const xm = (lo + hi) / 2;
  if (c.kind === "Offset") return [h("level", xm, at(xm), "y", "drag to change the level")];
  if (c.kind === "Polynomial") {
    const xs2 = lo + (hi - lo) * 0.85;
    const out = [h("level", xm, at(xm), "y", "drag to move it up or down")];
    if ((c.order ?? 2) >= 1) out.push(h("slope", xs2, at(xs2), "y", "drag to change the slope"));
    return out;
  }
  if (c.kind === "PowerLaw" || c.kind === "Exponential") {
    const x1 = lo + (hi - lo) * 0.15;
    const x2 = lo + (hi - lo) * 0.85;
    return [h("p1", x1, at(x1), "y", "drag to change it here"), h("p2", x2, at(x2), "y", "drag to change it here")];
  }
  const step = STEP_PARAMS[c.kind];
  if (step) {
    const xp = c.params[step.pos]?.value ?? xm;
    const xa = lo + (hi - lo) * 0.9;
    return [h("pos", xp, at(xp), "x", "drag to move the step"), h("amp", xa, at(xa), "y", "drag to change its height")];
  }
  return [];
}

/** The parameters after dragging handle ``part`` to (x, y) (as plotted;
 * ``factor``: plotted counts per unit of the components' functions). */
export function dragParams(c: ModelComponent, part: string, x: number, y: number, s: DragStart, factor: number): ModelComponent["params"] {
  const p = structuredClone(s.params);
  const set = (name: string, value: number) => {
    if (p[name] && Number.isFinite(value)) p[name] = { ...p[name], value };
  };
  const v = (name: string) => s.params[name]?.value ?? 0;
  const at = (id: string) => s.handles[id] ?? { x: 0, y: 0 };
  const peak = PEAK_PARAMS[c.kind];
  if (peak) {
    const top = at("top");
    if (part === "top") {
      set(peak.centre, v(peak.centre) + (x - top.x));
      if (top.y > 0 && y > 0) set(peak.amp, v(peak.amp) * (y / top.y));
      return p;
    }
    const side = part === "left" ? at("left") : at("right");
    const was = Math.abs(top.x - side.x);
    if (!(was > 0)) return p;
    const f = Math.max(0.05, (part === "left" ? top.x - x : x - top.x) / was);
    if (peak.sides) {
      const w = peak.sides[part === "left" ? 0 : 1];
      set(w, v(w) * f);
      const [a, b] = peak.sides.map(v);
      if (a + b > 0) set(peak.amp, v(peak.amp) * (part === "left" ? (a * f + b) / (a + b) : (a + b * f) / (a + b)));
      return p;
    }
    for (const w of peak.widths) set(w, v(w) * f);
    if (peak.area) set(peak.amp, v(peak.amp) * f);
    return p;
  }
  if (!(factor > 0)) return p;
  if (c.kind === "Offset") {
    set("offset", y / factor);
    return p;
  }
  if (c.kind === "Polynomial") {
    const level = at("level");
    if (part === "level") set("a0", v("a0") + (y - level.y) / factor);
    else {
      const slope = at("slope");
      const da1 = (y - slope.y) / factor / (slope.x - level.x || 1);
      set("a1", v("a1") + da1);
      set("a0", v("a0") - da1 * level.x); // the level handle stays put
    }
    return p;
  }
  if (c.kind === "PowerLaw" || c.kind === "Exponential") {
    const [h1, h2] = [at("p1"), at("p2")];
    const y1 = (part === "p1" ? y : h1.y) / factor;
    const y2 = (part === "p2" ? y : h2.y) / factor;
    if (!(y1 > 0 && y2 > 0) || y1 === y2) return p;
    // The shape parameter within its limits; then the amplitude so the curve
    // passes through the point being dragged.
    const within = (name: string, value: number) => {
      const q = p[name];
      if (!q || !q.free) return value;
      return Math.min(q.bmax ?? Infinity, Math.max(q.bmin ?? -Infinity, value));
    };
    const [xd, yd] = part === "p1" ? [h1.x, y1] : [h2.x, y2];
    if (c.kind === "PowerLaw") {
      const o = v("origin");
      const [d1, d2, dd] = [h1.x - o, h2.x - o, xd - o];
      if (!(d1 > 0 && d2 > 0)) return p;
      const r = within("r", Math.log(y1 / y2) / Math.log(d2 / d1));
      set("r", r);
      set("A", yd * dd ** r);
    } else {
      const tau = within("tau", (h2.x - h1.x) / Math.log(y1 / y2));
      set("tau", tau);
      set("A", yd * Math.exp(xd / tau));
    }
    return p;
  }
  const step = STEP_PARAMS[c.kind];
  if (step) {
    if (part === "pos") set(step.pos, v(step.pos) + (x - at("pos").x));
    else if (at("amp").y !== 0) set(step.amp, v(step.amp) * (y / at("amp").y));
  }
  return p;
}
