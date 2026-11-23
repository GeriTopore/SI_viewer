import { create } from "zustand";
import { api, type CellSiteFit, type DatasetSummary } from "../api/client";
import { useDatasets } from "./datasets";
import { useUi } from "./ui";

/** Sublattices by position in cell (M22d): every column placed in the cell
 * of a reference sublattice's lattice (two lattice vectors, picked in a
 * neighbourhood), then given the sublattice of the nearest site picked in
 * that cell (the cell repeating). Per image. */

export interface CellSite {
  /** Place in the cell: fractions of the two lattice vectors, in [−½, ½). */
  u: number;
  v: number;
  /** Its sublattice (sites named alike make one). */
  name: string;
}

export type Vectors = [[number, number], [number, number]];

export interface CellState {
  /** The reference sublattice (its columns' lattice makes the cell), and
   * its title (after a split, the new sublattice of that name). */
  reference: string;
  referenceTitle: string;
  /** The neighbourhood whose first two lattice vectors span the cell. */
  nb: string;
  sites: CellSite[];
  /** The sublattice a click on the plot adds a site of. */
  name: string;
  /** How far (px) a column may be from its site (circles); with clusters,
   * how near a group must come to a site to be its. */
  radius: number;
  /** Circles of ``radius`` round each site, or each site the dense groups
   * of columns that come within ``radius`` of it, however far they reach
   * (e.g. a tilt streak): columns ``link`` px apart at most, groups of
   * ``least`` or more. */
  mode: "radius" | "clusters";
  link: number;
  least: number;
  /** The clusters for the sites as they are: each column's site (index, or
   * null), per site its count, groups no site took, and how many columns
   * none takes. */
  fits: { labels: (number | null)[]; sites: CellSiteFit[]; unpicked: { u: number; v: number; count: number }[]; leftOut: number } | null;
  /** The columns' places in the cell, for the plot (and what they were found with). */
  data: { key: string; u: Float32Array; v: Float32Array; vectors: Vectors } | null;
  busy: string | null;
  error: string | null;
}

const defaultCell = (): CellState => ({
  reference: "", referenceTitle: "", nb: "", sites: [], name: "A", radius: 8, mode: "radius", link: 4, least: 5, fits: null,
  data: null, busy: null, error: null,
});

interface Store {
  byImage: Record<string, CellState>;
  patch: (image: string, p: Partial<CellState>) => void;
}

export const useCell = create<Store>((set) => ({
  byImage: {},
  patch: (image, p) => set((s) => ({ byImage: { ...s.byImage, [image]: { ...(s.byImage[image] ?? defaultCell()), ...p } } })),
}));

export const useCellState = (image: string) => useCell((s) => s.byImage[image]) ?? defaultCell();
const stateOf = (image: string) => useCell.getState().byImage[image] ?? defaultCell();

export const cellKey = (columns: DatasetSummary, reference: string, vectors: Vectors) =>
  `${columns.id}:${columns.revision}:${reference}:${vectors.flat().join(",")}`;

/** Place every column in the cell and show the plot below the image. */
export async function showCell(image: string, columns: DatasetSummary, reference: DatasetSummary, vectors: Vectors, nb?: string) {
  const { patch } = useCell.getState();
  // What the cell is made with, kept (with the sites) for a reload.
  patch(image, { busy: "Placing", error: null, reference: reference.id, referenceTitle: reference.title, ...(nb ? { nb } : {}) });
  try {
    const r = await api.cellPositions(columns.id, reference.id, vectors);
    const f = (a: (number | null)[]) => Float32Array.from(a, (x) => (x == null ? NaN : x));
    patch(image, { busy: null, data: { key: cellKey(columns, reference.id, vectors), u: f(r.u), v: f(r.v), vectors } });
    useUi.getState().setBottomTab("cell");
  } catch (e) {
    patch(image, { busy: null, error: (e as Error).message });
  }
}

/** A site at (u, v) (wrapped into the cell), of the sublattice being added. */
export function addSite(image: string, u: number, v: number) {
  const st = stateOf(image);
  const wrap = (x: number) => x - Math.round(x);
  useCell.getState().patch(image, { sites: [...st.sites, { u: wrap(u), v: wrap(v), name: st.name }], fits: null });
}

export function removeSite(image: string, i: number) {
  const st = stateOf(image);
  useCell.getState().patch(image, { sites: st.sites.filter((_, j) => j !== i), fits: null });
}

const request = (st: CellState, reference: string) => ({
  reference, vectors: st.data!.vectors, sites: st.sites.map((s): [number, number] => [s.u, s.v]),
  names: st.sites.map((s) => s.name), radius: st.radius, mode: st.mode, link: st.link, least: st.least,
});

/** Find each site's clusters in the cell (shown on the Cell plot), without splitting. */
export async function fitCell(image: string, columns: DatasetSummary, reference: string) {
  const st = stateOf(image);
  if (!st.data || !st.sites.length) return;
  const { patch } = useCell.getState();
  patch(image, { busy: "Fitting", error: null });
  try {
    const r = await api.cellFit(columns.id, request(st, reference));
    patch(image, { busy: null, fits: { labels: r.labels, sites: r.sites, unpicked: r.unpicked, leftOut: r.left_out } });
  } catch (e) {
    patch(image, { busy: null, error: (e as Error).message });
  }
}

/** Split the columns by their sites (nearest within a circle, or fitted
 * clusters): the sites' sublattices replace those there were. */
export async function splitByCell(image: string, columns: DatasetSummary, reference: string) {
  const st = stateOf(image);
  if (!st.data) return;
  const { patch } = useCell.getState();
  patch(image, { busy: "Splitting", error: null });
  try {
    const r = await api.sublatticesByCell(columns.id, request(st, reference));
    useDatasets.getState().replaceFile(r.file);
    // The reference was one of the replaced sublattices: the section finds the new one by its name.
    patch(image, { busy: null });
  } catch (e) {
    patch(image, { busy: null, error: (e as Error).message });
  }
}
