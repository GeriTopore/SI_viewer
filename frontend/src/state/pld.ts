import { create } from "zustand";
import { api, type DatasetSummary, type FileSummary, type JobSummary, type PldFamily } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";

/** Periodic lattice displacements of an image (kemstem, M22f): its FFT
 * peaks labelled (found and grouped automatically, or picked), the
 * superlattice ones damped into a reference image, and each column's
 * displacement measured against it. */

/** An FFT peak of the PLD process: where (cycles/px), its (h, k) in the
 * lattice, its family (0 Bragg, n off-lattice family n, null none) and
 * whether it is damped. */
export interface PldPeak {
  fy: number;
  fx: number;
  height?: number;
  hk?: [number, number];
  family: number | null;
  damp: boolean;
}

export interface PldSettings {
  /** Detection: how many peaks at most, within how far of the centre (cycles/px). */
  count: number;
  within: number;
  /** Damping radius (FFT px). */
  radius: number;
  /** How far (px) a column may have moved. */
  threshold: number;
  /** Picks add their conjugate too. */
  conjugates: boolean;
}

export interface PldState {
  peaks: PldPeak[];
  basis: [[number, number], [number, number]] | null;
  families: PldFamily[];
  settings: PldSettings;
  /** Clicks on the FFT add peaks (and toggle them). */
  picking: boolean;
  busy: string | null;
  job: JobSummary | null;
  error: string | null;
}

// Notebook 3's settings: mask radius 8, threshold 0.4 px.
export const defaultPld = (): PldState => ({
  peaks: [],
  basis: null,
  families: [],
  settings: { count: 120, within: 0.25, radius: 8, threshold: 0.4, conjugates: true },
  picking: true,
  busy: null,
  job: null,
  error: null,
});

interface Store {
  byId: Record<string, PldState>;
  /** The image whose PLD process is open (its peaks are drawn on the FFT). */
  shown: string | null;
  setShown: (id: string | null) => void;
  patch: (id: string, p: Partial<PldState>) => void;
}

export const usePld = create<Store>((set) => ({
  byId: {},
  shown: null,
  setShown: (shown) => set({ shown }),
  patch: (id, p) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? defaultPld()), ...p } } })),
}));

export const usePldState = (id: string) => usePld((s) => s.byId[id]) ?? defaultPld();
const stateOf = (id: string) => usePld.getState().byId[id] ?? defaultPld();

/** (h, k) of a frequency in the lattice basis, if known. */
function indexOf(basis: PldState["basis"], fy: number, fx: number): [number, number] | undefined {
  if (!basis) return undefined;
  const [[ay, ax], [by, bx]] = basis;
  const det = ay * bx - by * ax;
  if (Math.abs(det) < 1e-12) return undefined;
  return [(fy * bx - by * fx) / det, (ay * fx - fy * ax) / det];
}

/** Find the FFT's peaks, index them in the lattice and group the off-lattice
 * ones into families; the strongest family is damped to start. */
export async function detectPeaks(d: DatasetSummary) {
  const { patch } = usePld.getState();
  const { settings } = stateOf(d.id);
  patch(d.id, { busy: "Detecting", error: null });
  try {
    const r = await api.pldDetect(d.id, { count: settings.count, within: settings.within });
    const first = r.families[0]?.id;
    patch(d.id, {
      busy: null,
      basis: r.basis,
      families: r.families,
      peaks: r.peaks.map((p) => ({ fy: p.fy, fx: p.fx, height: p.height, hk: p.hk, family: p.family, damp: p.family === first && first !== undefined })),
    });
  } catch (e) {
    patch(d.id, { busy: null, error: (e as Error).message });
  }
}

/** Damp (or not) every peak of family ``id``. */
export function setFamilyDamp(image: string, id: number, damp: boolean) {
  const st = stateOf(image);
  usePld.getState().patch(image, { peaks: st.peaks.map((p) => (p.family === id ? { ...p, damp } : p)) });
}

/** Toggle whether peak ``i`` (and its conjugate, if listed) is damped. */
export function togglePeak(image: string, i: number) {
  const st = stateOf(image);
  const p = st.peaks[i];
  if (!p) return;
  const conj = st.peaks.findIndex((q) => Math.hypot(q.fy + p.fy, q.fx + p.fx) < 1e-3);
  const damp = !p.damp;
  usePld.getState().patch(image, { peaks: st.peaks.map((q, j) => (j === i || j === conj ? { ...q, damp } : q)) });
}

/** A peak picked on the FFT (and its conjugate): damped. */
export function addPldPeak(image: string, fy: number, fx: number) {
  const st = stateOf(image);
  const make = (y: number, x: number): PldPeak => ({ fy: y, fx: x, hk: indexOf(st.basis, y, x), family: null, damp: true });
  const added = [make(fy, fx), ...(st.settings.conjugates ? [make(-fy, -fx)] : [])];
  usePld.getState().patch(image, { peaks: [...st.peaks, ...added], error: null });
}

/** Remove peak ``i`` (and its conjugate, if listed). */
export function deletePldPeak(image: string, i: number) {
  const st = stateOf(image);
  const p = st.peaks[i];
  if (!p) return;
  const conj = st.peaks.findIndex((q) => Math.hypot(q.fy + p.fy, q.fx + p.fx) < 1e-3);
  usePld.getState().patch(image, { peaks: st.peaks.filter((_, j) => j !== i && j !== conj) });
}

/** The image with its damped peaks taken out: "PLD reference" and "PLD difference" under it. */
export async function makeReference(d: DatasetSummary) {
  const st = stateOf(d.id);
  const { patch } = usePld.getState();
  const damped = st.peaks.filter((p) => p.damp).map((p): [number, number] => [p.fy, p.fx]);
  patch(d.id, { busy: "Damping", error: null });
  try {
    const r = await api.pldReference(d.id, damped, st.settings.radius);
    useDatasets.getState().replaceFile(r.file);
    patch(d.id, { busy: null });
  } catch (e) {
    patch(d.id, { busy: null, error: (e as Error).message });
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Each column's displacement from its place in the reference (a job): "Displacements" under the Columns. */
export async function measureDisplacements(d: DatasetSummary, columns: DatasetSummary) {
  const { patch } = usePld.getState();
  patch(d.id, { error: null });
  try {
    let job = await api.pldDisplacements(columns.id, stateOf(d.id).settings.threshold);
    patch(d.id, { job });
    while (job.status === "queued" || job.status === "running") {
      await sleep(400);
      job = await api.getJob(job.id);
      patch(d.id, { job });
    }
    if (job.status === "done" && job.result) useDatasets.getState().replaceFile((job.result as { file: FileSummary }).file);
    patch(d.id, { job: null, error: job.status === "failed" ? (job.error ?? "Failed") : null });
  } catch (e) {
    patch(d.id, { job: null, error: (e as Error).message });
  }
}

type Files = Parameters<typeof allDatasets>[0];
/** The image's PLD reference and difference, and its Columns' Displacements, if made. */
export function pldResults(files: Files, image: string) {
  const all = allDatasets(files);
  const columns = all.find((x) => x.parent_id === image && x.atomic?.output === "columns");
  return {
    columns,
    reference: all.find((x) => x.parent_id === image && x.atomic?.output === "pld-reference"),
    difference: all.find((x) => x.parent_id === image && x.atomic?.output === "pld-difference"),
    displacements: columns && all.find((x) => x.parent_id === columns.id && x.atomic?.output === "displacements"),
  };
}
