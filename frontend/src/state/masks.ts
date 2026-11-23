import { create } from "zustand";
import { api, type DatasetSummary, type MaskOptions } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";

/** Masks made from images (any image: a map, a Bragg-filtered grating, …),
 * each a dataset under its image; used by other processes (so far:
 * sublattices of atomic columns). Per image: the settings being worked on,
 * the mask they update, and a sample of the image's values for the
 * histogram. */

export interface MaskValues {
  smooth: number;
  values: Float32Array;
  otsu: number;
}

export interface MaskState {
  draft: MaskOptions;
  /** The mask the settings came from (Update makes it again). */
  selected: string | null;
  values: MaskValues | null;
  busy: string | null;
  error: string | null;
}

export const defaultMaskOptions = (): MaskOptions => ({
  smooth: 0, keep: "above", lo: null, hi: null, invert: false, clean: 0, min_area: 0,
});

const defaultState = (): MaskState => ({ draft: defaultMaskOptions(), selected: null, values: null, busy: null, error: null });

interface Store {
  byImage: Record<string, MaskState>;
  patch: (image: string, p: Partial<MaskState>) => void;
}

export const useMasks = create<Store>((set) => ({
  byImage: {},
  patch: (image, p) => set((s) => ({ byImage: { ...s.byImage, [image]: { ...(s.byImage[image] ?? defaultState()), ...p } } })),
}));

export const useMaskState = (image: string) => useMasks((s) => s.byImage[image]) ?? defaultState();
const stateOf = (image: string) => useMasks.getState().byImage[image] ?? defaultState();

type Files = Parameters<typeof allDatasets>[0];
/** The masks made from an image, in order. */
export const masksOf = (files: Files, image: string) =>
  allDatasets(files).filter((d) => d.origin === "mask" && d.parent_id === image);

/** The images a mask of ``image`` can be made from: the image (up to the
 * first image it was made from) and every image made from it (a filtered
 * grating, an amplitude, a PLD difference, …), masks aside. */
export function maskSources(files: Files, image: DatasetSummary): DatasetSummary[] {
  const all = allDatasets(files);
  const byId = new Map(all.map((d) => [d.id, d]));
  let root = image;
  for (let up = byId.get(root.parent_id ?? ""); up?.kind === "image"; up = byId.get(up.parent_id ?? "")) root = up;
  const out = [root];
  for (let i = 0; i < out.length; i++) {
    for (const d of all) if (d.parent_id === out[i].id && d.kind === "image" && d.origin !== "mask" && d.axes.length === 2) out.push(d);
  }
  return out;
}

/** Every mask of the file the size of ``image`` (made from it or another image). */
export function masksFor(files: Files, image: DatasetSummary) {
  const size = image.axes.map((a) => a.size).join("×");
  return allDatasets(files).filter((d) => d.origin === "mask" && d.file_id === image.file_id && d.axes.map((a) => a.size).join("×") === size);
}

/** The thresholds the draft needs, from the Otsu threshold where unset. */
function withThresholds(o: MaskOptions, otsu: number): MaskOptions {
  return { ...o, lo: o.lo ?? otsu, hi: o.hi ?? otsu };
}

/** Fetch a sample of the (smoothed) image's values for the histogram; unset
 * thresholds start at Otsu's. */
export async function loadValues(image: string) {
  const { patch } = useMasks.getState();
  const { draft } = stateOf(image);
  try {
    const r = await api.maskValues(image, draft.smooth);
    const now = stateOf(image);
    if (now.draft.smooth !== draft.smooth) return; // changed since
    patch(image, { values: { smooth: draft.smooth, values: Float32Array.from(r.values), otsu: r.otsu }, draft: withThresholds(now.draft, r.otsu) });
  } catch (e) {
    patch(image, { error: (e as Error).message });
  }
}

/** Thresholds at Otsu's (one threshold; "between" keeps the upper one). */
export function setOtsu(image: string) {
  const st = stateOf(image);
  if (!st.values) return;
  const t = st.values.otsu;
  useMasks.getState().patch(image, { draft: { ...st.draft, lo: t, hi: st.draft.keep === "between" ? Math.max(t, st.draft.hi ?? t) : t } });
}

/** Make a mask with the draft (or make ``replace`` again). */
export async function makeMask(image: string, replace?: DatasetSummary) {
  const { patch } = useMasks.getState();
  const st = stateOf(image);
  patch(image, { busy: replace ? "Updating" : "Adding", error: null });
  try {
    const r = await api.makeMask(image, st.draft, replace?.id);
    useDatasets.getState().replaceFile(r.file);
    patch(image, { busy: null, selected: r.id });
  } catch (e) {
    patch(image, { busy: null, error: (e as Error).message });
  }
}

/** Work on a mask: its settings into the draft. */
export function selectMask(image: string, m: DatasetSummary) {
  const st = stateOf(image);
  const { source_id: _s, fraction: _f, ...opts } = m.mask ?? { ...defaultMaskOptions(), source_id: "", fraction: 0 };
  useMasks.getState().patch(image, { selected: m.id, draft: { ...defaultMaskOptions(), ...opts }, values: st.values?.smooth === opts.smooth ? st.values : null });
}
