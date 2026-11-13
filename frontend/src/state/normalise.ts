import { create } from "zustand";
import type { DatasetSummary } from "../api/client";
import type { SpectrumSeries } from "../components/SpectrumPlot";
import { formatValue } from "../format";
import { energyRange } from "./background";

/** How a spectrum is normalised: scaled to 0–1 (its minimum to its
 * maximum), divided by its value at an energy, or by its mean over a window
 * (as the exploded view of a LineScan does). */
export type NormMode = "range" | "value" | "mean";

export interface NormSettings {
  /** The spectrum plot shows the spectra normalised. */
  on: boolean;
  mode: NormMode;
  /** The energy ("value": lo) or window (lo–hi). */
  lo: number | null;
  hi: number | null;
  /** The energy or window is drawn on the spectrum (while the Process section is open). */
  shown: boolean;
}

interface NormStore {
  byId: Record<string, NormSettings>;
  set: (d: DatasetSummary, patch: Partial<NormSettings>) => void;
}

/** Off, 0–1; the energy and window in the middle of the spectrum. */
export function defaultNorm(d: DatasetSummary): NormSettings {
  const r = energyRange(d);
  const mid = r ? r.e0 + Math.round((r.e1 - r.e0) / 2 / r.step) * r.step : null;
  const span = r ? Math.max(1, Math.round((r.e1 - r.e0) / 20 / r.step)) * r.step : 0;
  return { on: false, mode: "range", lo: mid, hi: mid === null ? null : mid + span, shown: false };
}

export const useNorm = create<NormStore>((set, get) => ({
  byId: {},
  set: (d, patch) => set({ byId: { ...get().byId, [d.id]: { ...(get().byId[d.id] ?? defaultNorm(d)), ...patch } } }),
}));

export const useNormSettings = (d: DatasetSummary) => useNorm((s) => s.byId[d.id]) ?? defaultNorm(d);

/** Spectra and spectrum images (and their linked members) can be normalised. */
export const canNormalise = (d: DatasetSummary) => (d.kind === "spectrum_image" || d.kind === "spectrum") && !!energyRange(d);

/** Channels c0…c1 (inclusive) of energies lo…hi on the axis ``x``. */
function channels(x: ArrayLike<number>, lo: number, hi: number): [number, number] {
  const n = x.length;
  const step = n > 1 ? x[1] - x[0] : 1;
  const at = (e: number) => Math.min(n - 1, Math.max(0, Math.round((e - x[0]) / step)));
  const [a, b] = [at(lo), at(hi)];
  return a <= b ? [a, b] : [b, a];
}

/** ``data`` normalised as ``s`` says (energies on the axis ``x``). A spectrum
 * whose reference is zero or less (or flat, for 0–1) is left as it is. */
export function normaliseData(data: ArrayLike<number | null>, x: ArrayLike<number>, s: NormSettings): (number | null)[] {
  const out = Array.from(data);
  if (s.mode === "range") {
    let lo = Infinity, hi = -Infinity;
    for (const v of out) if (v !== null && Number.isFinite(v)) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
    if (!(hi > lo)) return out;
    return out.map((v) => (v === null ? null : (v - lo) / (hi - lo)));
  }
  if (s.lo === null) return out;
  const [c0, c1] = channels(x, s.lo, s.mode === "value" ? s.lo : (s.hi ?? s.lo));
  let sum = 0, n = 0;
  for (let c = c0; c <= c1; c++) {
    const v = out[c];
    if (v !== null && Number.isFinite(v)) [sum, n] = [sum + v, n + 1];
  }
  const ref = n ? sum / n : 0;
  return ref > 0 ? out.map((v) => (v === null ? null : v / ref)) : out;
}

/** The plot's spectra normalised when it is on (fits and previews, which
 * belong to a spectrum's own scale, are left out). */
export function normaliseSeries(series: SpectrumSeries[], x: ArrayLike<number>, s: NormSettings): SpectrumSeries[] {
  if (!s.on) return series;
  return series.map((sr) => ({ ...sr, data: normaliseData(sr.data, x, s) }));
}

/** A short description of the normalisation, e.g. "0–1" or "÷ mean 700–720 eV". */
export function normText(s: NormSettings, units = ""): string {
  const u = units ? ` ${units}` : "";
  if (s.mode === "range") return "0–1";
  if (s.mode === "value") return `÷ value at ${formatValue(s.lo ?? 0)}${u}`;
  return `÷ mean over ${formatValue(s.lo ?? 0)}–${formatValue(s.hi ?? 0)}${u}`;
}
