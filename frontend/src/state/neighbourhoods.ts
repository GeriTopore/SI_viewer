import { create } from "zustand";
import { api, type DatasetSummary, type FileSummary } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useUi } from "./ui";

/** Neighbourhoods of an image's atomic columns (kemstem, M22e): the vectors
 * from each column of a set (the origins: all columns or a sublattice) to
 * its k nearest of another (the neighbours). An image can have several,
 * each a dataset (named, renamed) with its own clusters, lattice vectors
 * and bonds; one is selected, in the process and in its plot. */

/** A lattice vector to measure bonds along: picked from a cluster (or typed),
 * [dy, dx] px. */
export interface LatticeVector {
  name: string;
  dy: number | null;
  dx: number | null;
}

/** The settings a neighbourhood is made with (origins and neighbours:
 * dataset ids, Columns or a sublattice; "" = the Columns). */
export interface NeighbourhoodDraft {
  origins: string;
  neighbours: string;
  k: number;
}

/** What is done with one neighbourhood: clusters, vectors, bonds. */
export interface NeighbourhoodAnalysis {
  /** Clusters to find. */
  n: number;
  /** How far (px) a neighbour may be from origin + vector. */
  threshold: number;
  vectors: LatticeVector[];
  /** The vector a click on a cluster sets (index), or none. */
  pick: number | null;
}

export interface NeighbourhoodState {
  draft: NeighbourhoodDraft;
  /** The neighbourhood worked on (and plotted), if any. */
  selected: string | null;
  busy: string | null;
  error: string | null;
}

// Notebook 2's settings: 20 neighbours, 30 clusters, 3 px.
export const defaultNeighbourhood = (): NeighbourhoodState => ({
  draft: { origins: "", neighbours: "", k: 20 },
  selected: null,
  busy: null,
  error: null,
});
export const defaultAnalysis = (): NeighbourhoodAnalysis => ({
  n: 30,
  threshold: 3,
  vectors: [
    { name: "v₁", dy: null, dx: null },
    { name: "v₂", dy: null, dx: null },
  ],
  pick: 0,
});

/** A neighbourhood's vectors for plotting, fetched once per version. */
export interface NeighbourhoodData {
  key: string;
  dy: Float32Array;
  dx: Float32Array;
}

interface Store {
  byImage: Record<string, NeighbourhoodState>;
  byNb: Record<string, NeighbourhoodAnalysis>;
  data: Record<string, NeighbourhoodData>;
  patch: (image: string, p: Partial<NeighbourhoodState>) => void;
  patchNb: (nb: string, p: Partial<NeighbourhoodAnalysis>) => void;
}

export const useNeighbourhoods = create<Store>((set) => ({
  byImage: {},
  byNb: {},
  data: {},
  patch: (image, p) =>
    set((s) => ({ byImage: { ...s.byImage, [image]: { ...(s.byImage[image] ?? defaultNeighbourhood()), ...p } } })),
  patchNb: (nb, p) => set((s) => ({ byNb: { ...s.byNb, [nb]: { ...(s.byNb[nb] ?? defaultAnalysis()), ...p } } })),
}));

export const useNeighbourhoodState = (image: string) =>
  useNeighbourhoods((s) => s.byImage[image]) ?? defaultNeighbourhood();
export const useAnalysis = (nb: string | undefined) =>
  useNeighbourhoods((s) => (nb ? s.byNb[nb] : undefined)) ?? defaultAnalysis();
const stateOf = (image: string) => useNeighbourhoods.getState().byImage[image] ?? defaultNeighbourhood();
const analysisOf = (nb: string) => useNeighbourhoods.getState().byNb[nb] ?? defaultAnalysis();

type Files = Parameters<typeof allDatasets>[0];

/** The image a dataset of the column analysis belongs to (its Columns' parent). */
export function imageOfAtomic(files: Files, d: DatasetSummary | undefined): string | null {
  const all = allDatasets(files);
  let up = d;
  while (up && ["sublattice", "neighbourhood", "bonds"].includes(up.atomic?.output ?? "")) {
    const parent = up.parent_id;
    up = all.find((x) => x.id === parent);
  }
  return up?.atomic?.output === "columns" ? (up.parent_id ?? null) : null;
}

/** The neighbourhoods of an image's columns (under the Columns or its sublattices). */
export function neighbourhoodsOf(files: Files, image: string): DatasetSummary[] {
  return allDatasets(files).filter((d) => d.atomic?.output === "neighbourhood" && imageOfAtomic(files, d) === image);
}

/** The neighbourhood worked on for ``image``: the selected one, else the last made. */
export function shownNeighbourhood(files: Files, image: string, selected: string | null) {
  const mine = neighbourhoodsOf(files, image);
  return mine.find((d) => d.id === selected) ?? mine[mine.length - 1];
}

/** A neighbourhood's settings, as a draft ("" for the Columns). */
export function draftOf(files: Files, nb: DatasetSummary): NeighbourhoodDraft {
  const all = allDatasets(files);
  const isColumns = (id: string | null | undefined) => all.find((d) => d.id === id)?.atomic?.output === "columns";
  return {
    origins: isColumns(originsOf(nb)) ? "" : (originsOf(nb) ?? ""),
    neighbours: isColumns(nb.atomic?.neighbours) ? "" : (nb.atomic?.neighbours ?? ""),
    k: nb.atomic?.k ?? 20,
  };
}

/** A neighbourhood's origins (Columns or a sublattice; saved before they
 * were recorded: its parent). */
export const originsOf = (nb: DatasetSummary) => nb.atomic?.origins ?? nb.parent_id;

/** Work on (and plot) neighbourhood ``nb``; the form takes its settings. */
export function selectNeighbourhood(image: string, nb: DatasetSummary) {
  useNeighbourhoods.getState().patch(image, { selected: nb.id, draft: draftOf(useDatasets.getState().files, nb), error: null });
}

const fetching = new Set<string>();
/** A neighbourhood's vectors (fetched once per version). */
export function ensureNeighbourhoodData(nb: DatasetSummary) {
  const key = `${nb.revision}:${nb.atomic?.k}:${nb.atomic?.count}`;
  if (useNeighbourhoods.getState().data[nb.id]?.key === key || fetching.has(nb.id)) return;
  fetching.add(nb.id);
  api
    .neighbourhood(nb.id)
    .then(({ dy, dx }) =>
      useNeighbourhoods.setState((s) => ({
        data: { ...s.data, [nb.id]: { key, dy: Float32Array.from(dy), dx: Float32Array.from(dx) } },
      })),
    )
    .catch(() => {})
    .finally(() => fetching.delete(nb.id));
}

async function run(image: string, label: string, work: () => Promise<FileSummary>) {
  const { patch } = useNeighbourhoods.getState();
  patch(image, { busy: label, error: null });
  try {
    useDatasets.getState().replaceFile(await work());
    patch(image, { busy: null });
  } catch (e) {
    patch(image, { busy: null, error: (e as Error).message });
  }
}

/** A new neighbourhood with the form's settings (or ``replace`` made again
 * with them), selected and shown below the image. */
export function computeNeighbourhood(image: string, columns: DatasetSummary, replace?: DatasetSummary) {
  const { draft } = stateOf(image);
  return run(image, replace ? "Updating" : "Adding", async () => {
    const r = await api.makeNeighbourhood(draft.origins || columns.id, draft.neighbours || columns.id, draft.k, replace?.id);
    useNeighbourhoods.getState().patch(image, { selected: r.id });
    // Made again: its vectors may no longer hold; its picks stay.
    useUi.getState().setBottomTab("neighbourhood");
    return r.file;
  });
}

export function clusterNeighbourhood(image: string, nb: DatasetSummary) {
  return run(image, "Clustering", async () => (await api.clusterNeighbourhood(nb.id, analysisOf(nb.id).n)).file);
}

/** Set the picked vector of ``nb`` to a cluster centre and move on to the next unset one. */
export function pickVector(nb: string, at: [number, number]) {
  const a = analysisOf(nb);
  if (a.pick === null) return;
  const vectors = a.vectors.map((v, i) => (i === a.pick ? { ...v, dy: at[0], dx: at[1] } : v));
  const next = vectors.findIndex((v, i) => i > a.pick! && v.dy === null);
  useNeighbourhoods.getState().patchNb(nb, { vectors, pick: next >= 0 ? next : null });
}

/** Bonds along every set vector, each a dataset under the neighbourhood. */
export function measureBonds(image: string, nb: DatasetSummary) {
  const a = analysisOf(nb.id);
  return run(image, "Measuring", async () => {
    let file: FileSummary | null = null;
    for (const v of a.vectors) {
      if (v.dy === null || v.dx === null) continue;
      file = (await api.makeBonds(nb.id, v.name, [v.dy, v.dx], a.threshold)).file;
    }
    if (!file) throw new Error("Pick a vector first");
    return file;
  });
}
