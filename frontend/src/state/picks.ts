import { api, type DatasetSummary } from "../api/client";
import { useCell, type CellState, type Vectors } from "./cell";
import { sublatticesOf } from "./columns";
import { allDatasets, useDatasets } from "./datasets";
import { defaultAnalysis, useNeighbourhoods, type NeighbourhoodAnalysis } from "./neighbourhoods";

/** Picks made in the page for the column analysis, kept with their datasets
 * on the server (``atomic.picks``, saved with an analysis) so a reload or a
 * reopened analysis has them: a neighbourhood's clusters setting, lattice
 * vectors and cage; the Columns' cell (reference, vectors, sites). Saved a
 * moment after each change; read back for datasets the page has none for. */

const NB_FIELDS = ["n", "threshold", "vectors", "cage", "cageNearest", "cageThreshold"] as const;
type NbPicks = Pick<NeighbourhoodAnalysis, (typeof NB_FIELDS)[number]>;
const CELL_FIELDS = ["reference", "referenceTitle", "nb", "sites", "name", "radius", "mode", "link", "least"] as const;
type CellPicks = Pick<CellState, (typeof CELL_FIELDS)[number]> & { vectors?: Vectors };

const nbPicks = (a: NeighbourhoodAnalysis): NbPicks =>
  Object.fromEntries(NB_FIELDS.map((k) => [k, a[k]])) as unknown as NbPicks;
const cellPicks = (c: CellState): CellPicks => ({
  ...(Object.fromEntries(CELL_FIELDS.map((k) => [k, c[k]])) as unknown as CellPicks),
  vectors: c.data?.vectors,
});

// What each dataset last has on the server (JSON), so only changes are sent.
const sent = new Map<string, string>();
const timers = new Map<string, ReturnType<typeof setTimeout>>();
function send(id: string, picks: Record<string, unknown>) {
  const json = JSON.stringify(picks);
  if (sent.get(id) === json) return;
  clearTimeout(timers.get(id));
  timers.set(
    id,
    setTimeout(() => {
      sent.set(id, json);
      api.setPicks(id, picks).catch(() => sent.delete(id));
    }, 600),
  );
}

type Files = Parameters<typeof allDatasets>[0];
const columnsOfImage = (files: Files, image: string) =>
  allDatasets(files).find((d) => d.parent_id === image && d.atomic?.output === "columns");

/** Fill the page's state from the picks kept with the datasets (those it has none for). */
function restore(files: Files) {
  const all = allDatasets(files);
  const nbs = useNeighbourhoods.getState();
  for (const d of all) {
    const picks = d.atomic?.picks;
    if (!picks) continue;
    if (d.atomic?.output === "neighbourhood" && !nbs.byNb[d.id]) {
      sent.set(d.id, JSON.stringify(picks));
      nbs.patchNb(d.id, { ...defaultAnalysis(), ...(picks as Partial<NbPicks>), pick: null });
    }
    if (d.atomic?.output === "columns" && d.parent_id && picks.cell && !useCell.getState().byImage[d.parent_id]) {
      sent.set(d.id, JSON.stringify(picks));
      restoreCell(files, d, d.parent_id, picks.cell as CellPicks);
    }
  }
}

/** The cell as it was: its settings and sites, and the columns' places in it
 * (found again, without bringing the plot forward). */
function restoreCell(files: Files, columns: DatasetSummary, image: string, cell: CellPicks) {
  const { vectors, ...rest } = cell;
  const { patch } = useCell.getState();
  patch(image, { ...rest });
  const subs = sublatticesOf(files, columns);
  const ref = subs.find((x) => x.id === cell.reference) ?? subs.find((x) => x.title === cell.referenceTitle);
  if (!ref || !vectors) return;
  api
    .cellPositions(columns.id, ref.id, vectors)
    .then((r) => {
      const f = (a: (number | null)[]) => Float32Array.from(a, (x) => (x == null ? NaN : x));
      useCell.getState().patch(image, { data: { key: `${columns.id}:${columns.revision}:${ref.id}`, u: f(r.u), v: f(r.v), vectors } });
    })
    .catch(() => {});
}

let started = false;
/** Start keeping picks (once, at start-up). */
export function keepPicks() {
  if (started) return;
  started = true;
  restore(useDatasets.getState().files);
  useDatasets.subscribe((s, prev) => {
    if (s.files !== prev.files) restore(s.files);
  });
  useNeighbourhoods.subscribe((s, prev) => {
    if (s.byNb === prev.byNb) return;
    const files = useDatasets.getState().files;
    const exists = new Set(allDatasets(files).map((d) => d.id));
    for (const [id, a] of Object.entries(s.byNb)) {
      if (a !== prev.byNb[id] && exists.has(id)) send(id, nbPicks(a) as unknown as Record<string, unknown>);
    }
  });
  useCell.subscribe((s, prev) => {
    if (s.byImage === prev.byImage) return;
    const files = useDatasets.getState().files;
    for (const [image, c] of Object.entries(s.byImage)) {
      if (c === prev.byImage[image]) continue;
      const columns = columnsOfImage(files, image);
      if (columns) send(columns.id, { cell: cellPicks(c) });
    }
  });
}
