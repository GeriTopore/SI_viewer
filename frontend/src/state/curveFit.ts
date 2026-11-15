import { create } from "zustand";
import {
  api, type CurveCatalogue, type CurveComponent, type CurveFitResult, type CurveModel, type CurveParam,
  type DatasetSummary, type FitSource, type JobSummary,
} from "../api/client";
import { useDatasets } from "./datasets";
import { energyRange } from "./background";

/** Colours of a model's components, in the order they are added. */
export const COLOURS = ["#22d3ee", "#a3e635", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c", "#34d399", "#60a5fa"];

/** A component of the model being built, with its colour. */
export interface ModelComponent extends CurveComponent {
  color: string;
}

/** The curve-fitting tool of one dataset: its model, which spectrum it is
 * fitted to, and the last result (a fit, or the model drawn with its values). */
export interface CurveFitState {
  components: ModelComponent[];
  /** "all", "cursor" or a region's id. */
  source: string;
  /** ``key``: the model and source it was made from (see ``keyOf``). */
  result: (CurveFitResult & { source: string; key: string }) | null;
  error: string | null;
  busy: boolean;
  /** The component selected (its table open, its handles on the plot). */
  selected: string | null;
  /** The parameter of the selected component whose limits are shown on the plot. */
  selectedParam: string | null;
  /** A peak kind waiting to be placed by a click on the spectrum. */
  placing: string | null;
  /** Background window (eV): the background components (PowerLaw, Exponential,
   * Polynomial, Offset) are fitted there alone, then fixed while the rest are
   * fitted. null: everything fitted together. */
  bgLo: number | null;
  bgHi: number | null;
  /** The background window is used (switched off, it keeps its place). */
  bgOn: boolean;
  /** Fitting every pixel: the job. */
  job: JobSummary | null;
  /** The cursor spectrum shows the fit of every pixel (when there is one)
   * instead of the model being edited. On after fitting every pixel; off
   * when a value is edited; the model's other settings leave it as it is. */
  showPixel: boolean;
  /** The fit of every pixel at the cursor (``at``: dataset, data and block). */
  pixelFit: { at: string; result: CurveFitResult } | null;
}

interface CurveFitStore {
  catalogue: CurveCatalogue | null;
  byId: Record<string, CurveFitState>;
  /** The dataset whose Curve fitting section is open: its model, fit range
   * and handles are drawn on the spectrum. The whole spectrum is fitted (crop
   * the data to fit part of it). */
  shown: string | null;
  loadCatalogue: () => Promise<CurveCatalogue>;
  patch: (d: DatasetSummary, patch: Partial<CurveFitState>) => void;
  setShown: (id: string | null) => void;
}

export function defaultCurveFit(_d: DatasetSummary): CurveFitState {
  return {
    components: [], source: "all", result: null, error: null, busy: false,
    selected: null, selectedParam: null, placing: null, bgLo: null, bgHi: null, bgOn: false,
    job: null, showPixel: false, pixelFit: null,
  };
}

// Each dataset's model is kept in this browser, so a reload keeps it.
const STORE_KEY = "si-viewer.curveFit";

/** ``byId`` as kept or restored: no fit in progress, nothing being placed. */
export function cleanCurveFits(byId: Record<string, CurveFitState>): Record<string, CurveFitState> {
  return Object.fromEntries(
    Object.entries(byId)
      .filter(([, st]) => st?.components?.length)
      .map(([id, st]) => [id, {
        ...st, busy: false, placing: null, error: null, job: null, pixelFit: null,
        // Kept before this switch: the key of the model fitted at every pixel.
        showPixel: st.showPixel ?? !!(st as { pixelKey?: string }).pixelKey,
        bgOn: st.bgOn ?? (st.bgLo !== null && st.bgHi !== null),
      }]),
  );
}

function loadKept(): Record<string, CurveFitState> {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? cleanCurveFits(JSON.parse(raw)) : {};
  } catch {
    return {};
  }
}

function keep(byId: Record<string, CurveFitState>) {
  const kept = cleanCurveFits(byId);
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(kept));
  } catch {
    // Too big with the results: the models alone (redrawn when shown).
    try {
      const models = Object.fromEntries(Object.entries(kept).map(([id, st]) => [id, { ...st, result: null }]));
      localStorage.setItem(STORE_KEY, JSON.stringify(models));
    } catch {
      /* storage unavailable */
    }
  }
}

export const useCurveFit = create<CurveFitStore>((set, get) => ({
  catalogue: null,
  byId: loadKept(),
  shown: null,
  loadCatalogue: async () => {
    const have = get().catalogue;
    if (have) return have;
    const catalogue = await api.curveComponents();
    set({ catalogue });
    return catalogue;
  },
  patch: (d, patch) =>
    set((s) => ({ byId: { ...s.byId, [d.id]: { ...(s.byId[d.id] ?? defaultCurveFit(d)), ...patch } } })),
  setShown: (shown) => set({ shown }),
}));

let keepTimer: ReturnType<typeof setTimeout> | null = null;
useCurveFit.subscribe((s, prev) => {
  if (s.byId === prev.byId) return;
  if (keepTimer) clearTimeout(keepTimer);
  keepTimer = setTimeout(() => keep(useCurveFit.getState().byId), 500);
});

export const useCurveFitState = (d: DatasetSummary) => useCurveFit((s) => s.byId[d.id]) ?? defaultCurveFit(d);
const stateOf = (d: DatasetSummary) => useCurveFit.getState().byId[d.id] ?? defaultCurveFit(d);

/** The spec sent to the backend. */
export function modelOf(st: Pick<CurveFitState, "components" | "bgLo" | "bgHi" | "bgOn">): CurveModel {
  const window = st.bgOn && st.bgLo !== null && st.bgHi !== null;
  return {
    lo: null,
    hi: null,
    bg_lo: window ? st.bgLo : null,
    bg_hi: window ? st.bgHi : null,
    components: st.components.map(({ name, kind, active, order, params }) => ({ name, kind, active, order, params })),
  };
}

/** Whether the cursor spectrum shows the fit of every pixel: there is one
 * of the data as it is, it is switched on, and the model is drawn on the
 * cursor's spectrum. */
export const showsPixelFit = (d: DatasetSummary, st: CurveFitState | undefined, source: FitSource | null) =>
  !!st && !!d.curve_pixel_fit && st.showPixel && source?.type === "pixel";

/** Fit the model at every pixel (a job; ``source``: the spectrum it was made on). */
export async function startPixelFits(d: DatasetSummary, source: FitSource) {
  const st = stateOf(d);
  const { patch } = useCurveFit.getState();
  try {
    const job = await api.startCurveFitMaps(d.id, source, modelOf(st));
    patch(d, { job, error: null });
  } catch (e) {
    patch(d, { error: (e as Error).message });
  }
}

/** Follow the job: on completion bring in the maps and show the fit at the cursor. */
export async function pollPixelFits(d: DatasetSummary) {
  const st = stateOf(d);
  if (!st.job) return;
  const job = await api.getJob(st.job.id);
  const { patch } = useCurveFit.getState();
  if (job.status === "done" && job.result) {
    useDatasets.getState().replaceFile(job.result.file);
    patch(d, { job, showPixel: true, pixelFit: null, source: "cursor" });
  } else patch(d, { job });
}

const pixelFetches = new Set<string>();
/** Fetch the fit of every pixel at the cursor's block ``at`` once (kept in ``pixelFit``). */
export async function fetchPixelFit(d: DatasetSummary, at: string, x: number, y: number, size: number) {
  if (stateOf(d).pixelFit?.at === at || pixelFetches.has(at)) return;
  pixelFetches.add(at);
  try {
    const result = await api.curvePixelFit(d.id, x, y, size);
    useCurveFit.getState().patch(d, { pixelFit: { at, result } });
  } catch {
    useCurveFit.getState().patch(d, { pixelFit: null }); // e.g. the fit failed at that pixel
  } finally {
    pixelFetches.delete(at);
  }
}

/** What a result was made from: when it differs, the model is drawn again. */
export const keyOf = (st: Pick<CurveFitState, "components" | "source" | "bgLo" | "bgHi" | "bgOn">) =>
  JSON.stringify([modelOf(st), st.source]);

const paramsFrom = (ps: CurveParam[]): ModelComponent["params"] =>
  Object.fromEntries(ps.map((p) => [p.name, { value: p.value, free: p.free, bmin: p.bmin, bmax: p.bmax }]));

/** ``params`` with the values from ``ps`` (a fit or an estimate); how each
 * is fitted (free, limits, twin) stays as set. */
const withValues = (params: ModelComponent["params"], ps: CurveParam[]): ModelComponent["params"] =>
  Object.fromEntries(
    Object.entries(params).map(([name, p]) => {
      const got = ps.find((q) => q.name === name);
      return [name, got ? { ...p, value: got.value } : p];
    }),
  );

/** A name not used yet: "Gaussian 1", "Gaussian 2", … */
function freshName(kind: string, used: string[]): string {
  for (let i = 1; ; i++) if (!used.includes(`${kind} ${i}`)) return `${kind} ${i}`;
}

/** Add a component of ``kind`` (HyperSpy's defaults; then its own estimate
 * over the spectrum when it has one). */
export async function addComponent(d: DatasetSummary, kind: string, source: FitSource | null) {
  const cat = await useCurveFit.getState().loadCatalogue();
  const k = cat.components.find((c) => c.kind === kind);
  if (!k) return;
  const st = stateOf(d);
  const used = st.components.map((c) => c.color);
  const comp: ModelComponent = {
    name: freshName(kind, st.components.map((c) => c.name)),
    kind,
    active: true,
    ...(kind === "Polynomial" && { order: 2 }),
    params: paramsFrom(k.params),
    color: COLOURS.find((c) => !used.includes(c)) ?? COLOURS[st.components.length % COLOURS.length],
  };
  useCurveFit.getState().patch(d, { components: [...st.components, comp], error: null, selected: comp.name, placing: null });
  if (k.estimate && source) await estimateComponent(d, comp.name, source);
  else if (source) await drawModel(d, source);
}

/** The kinds placed by a click on the spectrum (peaks). */
export const PLACED = new Set(["Gaussian", "GaussianHF", "Lorentzian", "Voigt", "SplitVoigt", "SkewNormal", "Doniach"]);

/** Peak parameters: its centre, amplitude, widths (scaled together; the
 * amplitude with them when it is an area, so the height stays), and for
 * SplitVoigt a width per side. */
export const PEAK_PARAMS: Record<string, { centre: string; amp: string; widths: string[]; area: boolean; sides?: [string, string] }> = {
  Gaussian: { centre: "centre", amp: "A", widths: ["sigma"], area: true },
  GaussianHF: { centre: "centre", amp: "height", widths: ["fwhm"], area: false },
  Lorentzian: { centre: "centre", amp: "A", widths: ["gamma"], area: true },
  Voigt: { centre: "centre", amp: "area", widths: ["sigma", "gamma"], area: true },
  SplitVoigt: { centre: "centre", amp: "A", widths: ["sigma1", "sigma2"], area: true, sides: ["sigma1", "sigma2"] },
  SkewNormal: { centre: "x0", amp: "A", widths: ["scale"], area: true },
  Doniach: { centre: "centre", amp: "A", widths: ["sigma"], area: true },
};

/** Step components: the parameter that places the step and its amplitude. */
export const STEP_PARAMS: Record<string, { pos: string; amp: string }> = {
  Arctan: { pos: "x0", amp: "A" },
  Erf: { pos: "origin", amp: "A" },
  HeavisideStep: { pos: "n", amp: "A" },
  Logistic: { pos: "origin", amp: "a" },
};

/** What a parameter is, for drawing its limits on the spectrum: a position
 * (energy), a peak's width, an amplitude (height), or none of these. */
export type ParamRole = "position" | "width" | "amplitude" | "other";
export function paramRole(kind: string, param: string): ParamRole {
  const peak = PEAK_PARAMS[kind];
  if (peak) {
    if (param === peak.centre) return "position";
    if (peak.widths.includes(param)) return "width";
    if (param === peak.amp) return "amplitude";
    return "other";
  }
  const step = STEP_PARAMS[kind];
  if (step) return param === step.pos ? "position" : param === step.amp ? "amplitude" : "other";
  if (kind === "Offset" && param === "offset") return "amplitude";
  return "other";
}

/** How a parameter is fitted: free, free within limits, fixed, or following
 * another (twin). */
export type ParamMode = "free" | "limited" | "fixed" | "twin";
export const modeOf = (p: { free: boolean; bmin: number | null; bmax: number | null; twin?: unknown }): ParamMode =>
  p.twin ? "twin" : !p.free ? "fixed" : p.bmin !== null || p.bmax !== null ? "limited" : "free";

/** Set how a parameter is fitted. Limiting one without limits gives it some
 * around its value (a position: ± 5 % of the spectrum; others: half to twice
 * the value, from 0 for an amplitude). */
export function setParamMode(d: DatasetSummary, name: string, param: string, mode: ParamMode) {
  const c = stateOf(d).components.find((x) => x.name === name);
  const p = c?.params[param];
  if (!c || !p) return;
  let { bmin, bmax } = p;
  if (mode === "twin") {
    // Another component's parameter of the same name (one of the same kind
    // first), else its first parameter.
    const others = stateOf(d).components.filter((x) => x.name !== name);
    const same = others.find((x) => x.kind === c.kind && param in x.params) ?? others.find((x) => param in x.params);
    const target = same ?? others[0];
    if (!target) return;
    const tp = param in target.params ? param : Object.keys(target.params)[0];
    updateComponent(d, name, { params: { ...c.params, [param]: { ...p, twin: { component: target.name, param: tp, expr: "x" } } } });
    useCurveFit.getState().patch(d, { selected: name, selectedParam: param });
    return;
  }
  if (mode === "free") [bmin, bmax] = [null, null];
  if (mode === "limited" && bmin === null && bmax === null) {
    const v = p.value;
    const role = paramRole(c.kind, param);
    if (role === "position") {
      const r = energyRange(d);
      const half = r ? (r.e1 - r.e0) * 0.05 : Math.abs(v) * 0.05 || 1;
      [bmin, bmax] = [v - half, v + half];
    } else if (v === 0) [bmin, bmax] = [-1, 1];
    else if (role === "amplitude") [bmin, bmax] = v > 0 ? [0, 2 * v] : [2 * v, 0];
    else [bmin, bmax] = v > 0 ? [v / 2, 2 * v] : [2 * v, v / 2];
  }
  updateComponent(d, name, { params: { ...c.params, [param]: { ...p, free: mode !== "fixed", bmin, bmax, twin: null } } });
  useCurveFit.getState().patch(d, { selected: name, selectedParam: param });
}

/** Change what a twinned parameter follows, or its formula. */
export function setTwin(d: DatasetSummary, name: string, param: string, patch: Partial<NonNullable<ModelComponent["params"][string]["twin"]>>) {
  const c = stateOf(d).components.find((x) => x.name === name);
  const p = c?.params[param];
  if (!c || !p?.twin) return;
  updateComponent(d, name, { params: { ...c.params, [param]: { ...p, twin: { ...p.twin, ...patch } } } });
}

/** Every component's twins after ``fn`` maps their targets (null: unlinked). */
function retarget(components: ModelComponent[], fn: (t: { component: string; param: string }) => string | null): ModelComponent[] {
  return components.map((c) => ({
    ...c,
    params: Object.fromEntries(
      Object.entries(c.params).map(([k, p]) => {
        if (!p.twin) return [k, p];
        const to = fn(p.twin);
        return [k, to === null ? { ...p, twin: null } : { ...p, twin: { ...p.twin, component: to } }];
      }),
    ),
  }));
}

/** Set a parameter's limits (null: none on that side); its value is kept inside them. */
export function setParamLimits(d: DatasetSummary, name: string, param: string, bmin: number | null, bmax: number | null) {
  const c = stateOf(d).components.find((x) => x.name === name);
  const p = c?.params[param];
  if (!c || !p) return;
  if (bmin !== null && bmax !== null && bmin > bmax) [bmin, bmax] = [bmax, bmin];
  let value = p.value;
  if (bmin !== null) value = Math.max(value, bmin);
  if (bmax !== null) value = Math.min(value, bmax);
  updateComponent(d, name, { params: { ...c.params, [param]: { ...p, bmin, bmax, value } } });
}

/** Place a peak of ``kind`` with its top at (x, y) on the plot (y as
 * plotted; ``perPixel``: the plot shows means per pixel): the centre at x, a
 * width of 1.5 % of the spectrum's range, and its amplitude set so the model
 * reaches y there. */
export async function placePeak(d: DatasetSummary, kind: string, at: { x: number; y: number }, source: FitSource, perPixel: boolean) {
  const cat = await useCurveFit.getState().loadCatalogue();
  const k = cat.components.find((c) => c.kind === kind);
  const pp = PEAK_PARAMS[kind];
  if (!k || !pp) return;
  const st = stateOf(d);
  const r0 = energyRange(d);
  const span = r0 ? r0.e1 - r0.e0 : 10;
  const width = Math.max(span * 0.015, 1e-6);
  const params = paramsFrom(k.params);
  params[pp.centre] = { ...params[pp.centre], value: at.x };
  for (const w of pp.widths) params[w] = { ...params[w], value: kind === "Voigt" ? width / 2 : width };
  params[pp.amp] = { ...params[pp.amp], value: 1 };
  const used = st.components.map((c) => c.color);
  const comp: ModelComponent = {
    name: freshName(kind, st.components.map((c) => c.name)),
    kind, active: true, params,
    color: COLOURS.find((c) => !used.includes(c)) ?? COLOURS[st.components.length % COLOURS.length],
  };
  useCurveFit.getState().patch(d, { components: [...st.components, comp], selected: comp.name, placing: null, error: null });
  await drawModel(d, source, true);
  const r = stateOf(d).result;
  const curve = r?.components.find((c) => c.name === comp.name)?.curve;
  if (!r || !curve) return;
  const scale = perPixel ? 1 / Math.max(1, r.pixels) : 1;
  // The peak makes up what the other components leave at x.
  const others = (valueAt(r.x, r.model, at.x) - valueAt(r.x, curve, at.x)) * scale;
  const top = Math.max(...curve) * scale;
  const want = at.y - others;
  if (top > 0 && want > 0) {
    updateComponent(d, comp.name, { params: { ...params, [pp.amp]: { ...params[pp.amp], value: want / top } } });
    await drawModel(d, source, true);
  }
}

/** The value of ``curve`` (on ``xs``) at x, by linear interpolation; 0 outside. */
export function valueAt(xs: number[] | undefined, curve: number[] | null | undefined, x: number): number {
  if (!xs || !curve || xs.length < 2) return 0;
  const step = xs[1] - xs[0];
  const f = (x - xs[0]) / step;
  const i = Math.floor(f);
  if (i < 0 || i >= xs.length - 1) return 0;
  return curve[i] + (curve[i + 1] - curve[i]) * (f - i);
}

/** A copy of a component (named "… copy"), selected. */
export function duplicateComponent(d: DatasetSummary, name: string, source: FitSource | null) {
  const st = stateOf(d);
  const c = st.components.find((x) => x.name === name);
  if (!c) return;
  const used = st.components.map((x) => x.color);
  const copy: ModelComponent = {
    ...c,
    name: freshName(c.kind, st.components.map((x) => x.name)),
    params: structuredClone(c.params),
    color: COLOURS.find((x) => !used.includes(x)) ?? COLOURS[st.components.length % COLOURS.length],
  };
  const i = st.components.indexOf(c);
  useCurveFit.getState().patch(d, { components: [...st.components.slice(0, i + 1), copy, ...st.components.slice(i + 1)], selected: copy.name });
  if (source) void drawModel(d, source);
}

/** Rename a component; false (with the reason in ``error``) when the name is taken or empty. */
export function renameComponent(d: DatasetSummary, name: string, to: string): boolean {
  const st = stateOf(d);
  const next = to.trim();
  if (!next || next === name) return !!next;
  if (st.components.some((c) => c.name === next)) {
    useCurveFit.getState().patch(d, { error: `There is already a component named ${next}` });
    return false;
  }
  useCurveFit.getState().patch(d, {
    components: retarget(
      st.components.map((c) => (c.name === name ? { ...c, name: next } : c)),
      (t) => (t.component === name ? next : t.component),
    ),
    selected: st.selected === name ? next : st.selected,
    error: null,
  });
  return true;
}

/** Every parameter of a component free (fitted) or fixed. */
export function setAllFree(d: DatasetSummary, name: string, free: boolean) {
  const c = stateOf(d).components.find((x) => x.name === name);
  if (!c) return;
  updateComponent(d, name, { params: Object.fromEntries(Object.entries(c.params).map(([k, p]) => [k, { ...p, free }])) });
}

/** Set a component's parameters from its own estimate over the spectrum. */
export async function estimateComponent(d: DatasetSummary, name: string, source: FitSource) {
  const st = stateOf(d);
  try {
    const r = await api.curveEstimate(d.id, source, modelOf(st), name);
    const c = stateOf(d).components.find((x) => x.name === name);
    if (c) updateComponent(d, name, { params: withValues(c.params, r.params) });
    await drawModel(d, source);
  } catch (e) {
    useCurveFit.getState().patch(d, { error: (e as Error).message });
  }
}

export function updateComponent(d: DatasetSummary, name: string, patch: Partial<ModelComponent>) {
  const st = stateOf(d);
  useCurveFit.getState().patch(d, { components: st.components.map((c) => (c.name === name ? { ...c, ...patch } : c)) });
}

export function removeComponent(d: DatasetSummary, name: string) {
  const st = stateOf(d);
  const components = retarget(st.components.filter((c) => c.name !== name), (t) => (t.component === name ? null : t.component));
  useCurveFit.getState().patch(d, {
    components,
    selected: st.selected === name ? null : st.selected,
    selectedParam: st.selected === name ? null : st.selectedParam,
    ...(components.length ? {} : { result: null }),
  });
}

let drawAbort: AbortController | null = null;
// One drawing request at a time while dragging: the latest wanted is sent
// when the one in flight comes back.
let inFlight: Promise<void> | null = null;
let queued: { d: DatasetSummary; source: FitSource } | null = null;

/** The model drawn on the spectrum with its current values (no fit).
 * ``now``: wait for it (and cancel any drawing in flight). */
export async function drawModel(d: DatasetSummary, source: FitSource, now = false): Promise<void> {
  if (!now && inFlight) {
    queued = { d, source };
    return;
  }
  const st = stateOf(d);
  if (!st.components.length) return useCurveFit.getState().patch(d, { result: null });
  drawAbort?.abort();
  const ac = (drawAbort = new AbortController());
  const run = (async () => {
    try {
      const r = await api.curveFit(d.id, source, modelOf(st), false, ac.signal);
      useCurveFit.getState().patch(d, { result: { ...r, source: st.source, key: keyOf(st) }, error: null });
    } catch (e) {
      if (!ac.signal.aborted) useCurveFit.getState().patch(d, { error: (e as Error).message });
    }
  })();
  inFlight = run;
  await run;
  if (inFlight === run) inFlight = null;
  const next = queued;
  queued = null;
  if (next) await drawModel(next.d, next.source);
}

/** Fit the model; the fitted values become the components' values. */
export async function fitModel(d: DatasetSummary, source: FitSource) {
  const st = stateOf(d);
  const { patch } = useCurveFit.getState();
  drawAbort?.abort();
  patch(d, { busy: true, error: null });
  try {
    const r = await api.curveFit(d.id, source, modelOf(st), true);
    const now = stateOf(d);
    const components = now.components.map((c) => {
      const fitted = r.components.find((f) => f.name === c.name);
      return fitted ? { ...c, params: withValues(c.params, fitted.params) } : c;
    });
    patch(d, { busy: false, components, result: { ...r, source: st.source, key: keyOf({ ...now, components }) } });
  } catch (e) {
    patch(d, { busy: false, error: (e as Error).message });
  }
}
