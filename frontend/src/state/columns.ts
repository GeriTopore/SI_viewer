import { create } from "zustand";
import { api, type ColumnSettings, type ColumnTest, type DatasetSummary, type FileSummary, type JobSummary } from "../api/client";
import type { Frame } from "../api/arrays";
import { allDatasets, useDatasets } from "./datasets";

/** Atomic columns of one image (kemstem, M22d): how they are found and
 * fitted, the columns found (not kept: found again quickly), one test fit
 * and the job fitting them all. */
export interface ColumnsState {
  settings: ColumnSettings;
  /** Found columns, [y, x] pixels (centres at integers). */
  found: [number, number][] | null;
  finding: boolean;
  test: (ColumnTest & { at: [number, number] }) | null;
  testing: boolean;
  job: JobSummary | null;
  error: string | null;
  /** Splitting the fitted columns into sublattices by a value: thresholds
   * between them, largest first (sublattice A above the first). */
  split: { by: string; k: number; thresholds: number[] };
  splitting: boolean;
}

export const defaultColumns = (): ColumnsState => ({
  // Notebook 2's settings.
  settings: { blur: 0.5, distance: 4, threshold: 0.1, window: 7 },
  found: null,
  finding: false,
  test: null,
  testing: false,
  job: null,
  error: null,
  split: { by: "intensity", k: 2, thresholds: [] },
  splitting: false,
});

/** A fitted table for drawing: y, x and failed per column, every field of
 * its row (``values``, by name; NaN for a failed fit) and an index for
 * finding the column under the pointer. */
export interface ColumnPoints {
  y: Float32Array;
  x: Float32Array;
  failed: Uint8Array;
  values: Record<string, Float32Array>;
  /** Each column's sublattice (its place among the Columns' sublattices), -1: none. */
  sublattice: Int8Array;
  index: SiteIndex;
}

/** Sites in square cells of ``cell`` px (a spatial index). */
export interface SiteIndex {
  cell: number;
  cells: Map<number, number[]>;
}

const cellKey = (cx: number, cy: number) => cy * 100003 + cx;

function buildIndex(y: Float32Array, x: Float32Array, cell = 8): SiteIndex {
  const cells = new Map<number, number[]>();
  for (let i = 0; i < y.length; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) continue;
    const k = cellKey(Math.floor(x[i] / cell), Math.floor(y[i] / cell));
    const list = cells.get(k);
    if (list) list.push(i);
    else cells.set(k, [i]);
  }
  return { cell, cells };
}

/** The column nearest (x, y) (data px, pixel i spanning [i, i + 1]) within
 * ``r`` px, or -1. */
export function nearestColumn(p: ColumnPoints, x: number, y: number, r: number, accept?: (i: number) => boolean): number {
  const { cell, cells } = p.index;
  // Column centres are at integers; the pointer's pixel centre is at +0.5.
  const px = x - 0.5;
  const py = y - 0.5;
  let best = -1;
  let bestD = r * r;
  const n = Math.ceil(r / cell);
  const cx = Math.floor(px / cell);
  const cy = Math.floor(py / cell);
  for (let j = cy - n; j <= cy + n; j++)
    for (let i = cx - n; i <= cx + n; i++)
      for (const k of cells.get(cellKey(i, j)) ?? []) {
        const d = (p.x[k] - px) ** 2 + (p.y[k] - py) ** 2;
        if (d <= bestD && (!accept || accept(k))) [best, bestD] = [k, d];
      }
  return best;
}

/** Values the fitted columns can be coloured by (fields of the table, or
 * made from them). */
export const COLUMN_VALUES: { id: string; label: string; unit?: string }[] = [
  { id: "intensity", label: "Intensity" },
  { id: "amplitude", label: "Amplitude" },
  { id: "offset", label: "Offset" },
  { id: "sigma_1", label: "σ₁", unit: "px" },
  { id: "sigma_2", label: "σ₂", unit: "px" },
  { id: "ellipticity", label: "σ max/min" },
  { id: "angle", label: "Angle", unit: "°" },
];

const valueCache = new WeakMap<ColumnPoints, Map<string, Float32Array>>();
/** Value ``id`` (of COLUMN_VALUES) per column (NaN for a failed fit). */
export function columnValues(p: ColumnPoints, id: string): Float32Array {
  let byId = valueCache.get(p);
  if (!byId) valueCache.set(p, (byId = new Map()));
  let v = byId.get(id);
  if (v) return v;
  const n = p.y.length;
  v = new Float32Array(n);
  const s1 = p.values.sigma_1;
  const s2 = p.values.sigma_2;
  for (let i = 0; i < n; i++) {
    if (id === "ellipticity") {
      const a = Math.abs(s1[i]);
      const b = Math.abs(s2[i]);
      v[i] = Math.max(a, b) / Math.min(a, b);
    } else if (id === "angle") {
      // kemstem's angle (rad) of σ₁'s axis, as degrees in [0, 180).
      const deg = ((p.values.angle[i] * 180) / Math.PI) % 180;
      v[i] = deg < 0 ? deg + 180 : deg;
    } else v[i] = p.values[id]?.[i] ?? NaN;
    if (p.failed[i]) v[i] = NaN;
  }
  byId.set(id, v);
  return v;
}

/** A group of fitted columns: all ("") or a sublattice ("0", "1" …, "u":
 * none). */
export function inGroup(p: ColumnPoints, group: string, i: number): boolean {
  if (group === "") return true;
  if (group === "u") return p.sublattice[i] < 0;
  return p.sublattice[i] === Number(group);
}

const frameCache = new WeakMap<ColumnPoints, Map<string, Frame>>();
/** The finite values ``by`` of a group's columns as a 1 × n frame, for the
 * contrast histogram. */
export function valuesFrame(p: ColumnPoints, by: string, group = ""): Frame {
  let byKey = frameCache.get(p);
  if (!byKey) frameCache.set(p, (byKey = new Map()));
  const key = `${by}|${group}`;
  let f = byKey.get(key);
  if (!f) {
    const v = columnValues(p, by);
    const kept: number[] = [];
    for (let i = 0; i < v.length; i++) if (Number.isFinite(v[i]) && inGroup(p, group, i)) kept.push(v[i]);
    const data = Float32Array.from(kept);
    f = { data, width: data.length, height: 1, bin: 1 };
    byKey.set(key, f);
  }
  return f;
}

interface ColumnsStore {
  byId: Record<string, ColumnsState>;
  /** The image whose Atomic columns section is open (its columns are drawn). */
  shown: string | null;
  /** Fitted tables by Columns dataset id (and the data version they are of). */
  tables: Record<string, { key: string; points: ColumnPoints }>;
  setShown: (id: string | null) => void;
  patch: (id: string, patch: Partial<ColumnsState>) => void;
}

const SETTINGS_KEY = "si-viewer.columnSettings";

function loadSettings(): Record<string, ColumnsState> {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    const kept = raw ? (JSON.parse(raw) as Record<string, ColumnSettings>) : {};
    return Object.fromEntries(Object.entries(kept).map(([id, settings]) => [id, { ...defaultColumns(), settings }]));
  } catch {
    return {};
  }
}

export const useColumns = create<ColumnsStore>((set) => ({
  byId: loadSettings(),
  shown: null,
  tables: {},
  setShown: (shown) => set({ shown }),
  patch: (id, patch) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? defaultColumns()), ...patch } } })),
}));

// Each image's settings are kept in this browser.
useColumns.subscribe((s, prev) => {
  if (s.byId === prev.byId) return;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(Object.fromEntries(Object.entries(s.byId).map(([id, st]) => [id, st.settings]))));
  } catch {
    /* storage unavailable */
  }
});

export const useColumnsState = (id: string) => useColumns((s) => s.byId[id]) ?? defaultColumns();
const stateOf = (id: string) => useColumns.getState().byId[id] ?? defaultColumns();

/** The Columns dataset made from ``image`` (its fitted columns), if any. */
export function columnsOf(image: string): DatasetSummary | undefined {
  return allDatasets(useDatasets.getState().files).find(
    (d) => d.parent_id === image && d.atomic?.output === "columns",
  );
}
export const isColumns = (d: DatasetSummary | undefined) => d?.atomic?.output === "columns";

export async function findColumns(d: DatasetSummary) {
  const { patch } = useColumns.getState();
  patch(d.id, { finding: true, error: null });
  try {
    const r = await api.findColumns(d.id, stateOf(d.id).settings);
    patch(d.id, { found: r.columns, finding: false, test: null });
  } catch (e) {
    patch(d.id, { finding: false, error: (e as Error).message });
  }
}

/** Fit the found column nearest ``near`` ([y, x]) to show the fit. */
export async function testColumn(d: DatasetSummary, near: [number, number]) {
  const st = stateOf(d.id);
  if (!st.found?.length) return;
  let best = st.found[0];
  let bestD = Infinity;
  for (const c of st.found) {
    const dist = (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2;
    if (dist < bestD) [best, bestD] = [c, dist];
  }
  const { patch } = useColumns.getState();
  patch(d.id, { testing: true, error: null });
  try {
    const r = await api.testColumn(d.id, st.settings, best);
    patch(d.id, { test: { ...r, at: best }, testing: false });
  } catch (e) {
    patch(d.id, { testing: false, error: (e as Error).message });
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Fit every found column (a job); the table goes into the image's Columns dataset. */
export async function refineColumns(d: DatasetSummary) {
  const st = stateOf(d.id);
  if (!st.found?.length) return;
  const { patch } = useColumns.getState();
  patch(d.id, { error: null });
  try {
    let job = await api.refineColumns(d.id, st.settings, st.found);
    patch(d.id, { job });
    while (job.status === "queued" || job.status === "running") {
      await sleep(400);
      job = await api.getJob(job.id);
      patch(d.id, { job });
    }
    if (job.status === "done" && job.result) {
      const result = job.result as { file: FileSummary };
      useDatasets.getState().replaceFile(result.file);
      patch(d.id, { job: null });
    } else patch(d.id, { job: null, error: job.status === "failed" ? (job.error ?? "Failed") : null });
  } catch (e) {
    patch(d.id, { job: null, error: (e as Error).message });
  }
}

const tableFetches = new Set<string>();
/** The fitted columns of Columns dataset ``c`` (fetched once per version). */
export function ensureTable(c: DatasetSummary) {
  // Its sublattices (datasets under it) give each column's label.
  const subs = allDatasets(useDatasets.getState().files)
    .filter((d) => d.parent_id === c.id && d.atomic?.output === "sublattice")
    .map((d) => [d.id, d.revision]);
  const key = JSON.stringify([c.atomic?.count, c.atomic?.failed, c.atomic?.window, c.revision, subs]);
  const have = useColumns.getState().tables[c.id];
  if (have?.key === key || tableFetches.has(c.id)) return;
  tableFetches.add(c.id);
  api
    .columnsTable(c.id)
    .then(({ fields, rows }) => {
      const n = rows.length;
      const values: Record<string, Float32Array> = {};
      fields.forEach((f, j) => {
        const col = new Float32Array(n);
        for (let i = 0; i < n; i++) col[i] = rows[i][j] ?? NaN;
        values[f] = col;
      });
      const y = values.y ?? new Float32Array(n);
      const x = values.x ?? new Float32Array(n);
      const failed = new Uint8Array(n);
      const fc = values.failed;
      if (fc) for (let i = 0; i < n; i++) failed[i] = fc[i] ? 1 : 0;
      const sublattice = new Int8Array(n).fill(-1);
      const sc = values.sublattice;
      if (sc) for (let i = 0; i < n; i++) if (Number.isFinite(sc[i])) sublattice[i] = sc[i];
      const points: ColumnPoints = { y, x, failed, values, sublattice, index: buildIndex(y, x) };
      useColumns.setState((s) => ({ tables: { ...s.tables, [c.id]: { key, points } } }));
    })
    .catch(() => {})
    .finally(() => tableFetches.delete(c.id));
}

/** Default sublattice names and colours, the largest value first. */
export const SUBLATTICE_NAMES = ["A", "B", "C", "D", "E", "F"];
export const SUBLATTICE_COLOURS = ["#ff4d6d", "#4da3ff", "#5dd39e", "#c77dff", "#ff9f1c", "#2ec4b6"];

/** The sublattice of a value: thresholds largest first, so above the first
 * is 0 (A), between the first and second 1 (B) … */
export function sublatticeOf(v: number, thresholds: number[]): number | null {
  if (!Number.isFinite(v)) return null;
  let k = 0;
  while (k < thresholds.length && v <= thresholds[k]) k++;
  return k;
}

/** Thresholds half-way between k-means centres of the split's value. */
export async function kmeansSplit(d: DatasetSummary, points: ColumnPoints) {
  const st = stateOf(d.id);
  const { patch } = useColumns.getState();
  const v = Array.from(columnValues(points, st.split.by), (x) => (Number.isFinite(x) ? x : null));
  patch(d.id, { splitting: true, error: null });
  try {
    const { centres } = await api.kmeans(v, st.split.k);
    const thresholds = centres.slice(1).map((c, i) => (centres[i] + c) / 2);
    patch(d.id, { splitting: false, split: { ...stateOf(d.id).split, thresholds } });
  } catch (e) {
    patch(d.id, { splitting: false, error: (e as Error).message });
  }
}

/** Label the fitted columns (Columns dataset ``c``) by the split's thresholds. */
export async function applySplit(d: DatasetSummary, c: DatasetSummary, points: ColumnPoints) {
  const { split } = stateOf(d.id);
  const v = columnValues(points, split.by);
  const labels = Array.from(v, (x) => sublatticeOf(x, split.thresholds));
  const { patch } = useColumns.getState();
  patch(d.id, { splitting: true, error: null });
  try {
    const r = await api.setSublattices(c.id, labels, SUBLATTICE_NAMES.slice(0, split.thresholds.length + 1));
    useDatasets.getState().replaceFile(r.file);
    patch(d.id, { splitting: false });
  } catch (e) {
    patch(d.id, { splitting: false, error: (e as Error).message });
  }
}

/** Remove the sublattices of Columns dataset ``c``. */
export async function clearSplit(d: DatasetSummary, c: DatasetSummary) {
  const { patch } = useColumns.getState();
  patch(d.id, { splitting: true, error: null });
  try {
    const r = await api.setSublattices(c.id, new Array(c.atomic?.count ?? 0).fill(null), []);
    useDatasets.getState().replaceFile(r.file);
    patch(d.id, { splitting: false });
  } catch (e) {
    patch(d.id, { splitting: false, error: (e as Error).message });
  }
}

/** The sublattices of Columns dataset ``c`` (datasets under it), in order. */
export function sublatticesOf(files: Parameters<typeof allDatasets>[0], c: DatasetSummary): DatasetSummary[] {
  return allDatasets(files)
    .filter((d) => d.parent_id === c.id && d.atomic?.output === "sublattice")
    .sort((a, b) => (a.atomic?.index ?? 0) - (b.atomic?.index ?? 0));
}

/** A bonds table for drawing: each origin's row in its Columns table, the
 * vector to its neighbour (px) and its length (px) and angle (°); NaN where
 * it has none. */
export interface BondPoints {
  origin: Int32Array;
  dy: Float32Array;
  dx: Float32Array;
  length: Float32Array;
  angle: Float32Array;
}

export const BOND_VALUES: { id: "length" | "angle"; label: string; unit: string }[] = [
  { id: "length", label: "Length", unit: "px" },
  { id: "angle", label: "Angle", unit: "°" },
];

export const useBondTables = create<{ tables: Record<string, { key: string; points: BondPoints }> }>(() => ({ tables: {} }));

const bondFetches = new Set<string>();
/** The bonds of dataset ``b`` (fetched once per version). */
export function ensureBonds(b: DatasetSummary) {
  const key = `${b.revision}:${b.atomic?.count}:${b.atomic?.missing}`;
  if (useBondTables.getState().tables[b.id]?.key === key || bondFetches.has(b.id)) return;
  bondFetches.add(b.id);
  api
    .bondsTable(b.id)
    .then(({ rows }) => {
      const n = rows.length;
      const p: BondPoints = {
        origin: new Int32Array(n), dy: new Float32Array(n), dx: new Float32Array(n),
        length: new Float32Array(n), angle: new Float32Array(n),
      };
      rows.forEach((r, i) => {
        p.origin[i] = r[0] ?? -1;
        p.dy[i] = r[1] ?? NaN;
        p.dx[i] = r[2] ?? NaN;
        p.length[i] = r[3] ?? NaN;
        p.angle[i] = r[4] ?? NaN;
      });
      useBondTables.setState((s) => ({ tables: { ...s.tables, [b.id]: { key, points: p } } }));
    })
    .catch(() => {})
    .finally(() => bondFetches.delete(b.id));
}

const bondFrames = new WeakMap<BondPoints, Map<string, Frame>>();
/** A bonds value's finite values as a 1 × n frame, for the contrast histogram. */
export function bondFrame(p: BondPoints, by: "length" | "angle"): Frame {
  let m = bondFrames.get(p);
  if (!m) bondFrames.set(p, (m = new Map()));
  let f = m.get(by);
  if (!f) {
    const data = p[by].filter((v) => Number.isFinite(v));
    f = { data, width: data.length, height: 1, bin: 1 };
    m.set(by, f);
  }
  return f;
}
