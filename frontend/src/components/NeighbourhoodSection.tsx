import { useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { sublatticesOf } from "../state/columns";
import { allDatasets, useDatasets } from "../state/datasets";
import {
  clusterNeighbourhood, computeNeighbourhood, measureBonds, measureCageDisplacement, neighbourhoodsOf, originsOf, seedCageCentres,
  selectNeighbourhood, shownNeighbourhood, useAnalysis, useNeighbourhoods, useNeighbourhoodState, type NeighbourhoodAnalysis,
  type NeighbourhoodDraft,
} from "../state/neighbourhoods";
import { pixelSize } from "./AtomicSection";
import { displacementText } from "./PldSection";
import { useUi } from "../state/ui";
import { EditableTitle } from "./RenameInput";
import { ProcessLink } from "./ProcessLink";

/** Process tab, images: neighbourhoods of the fitted atomic columns
 * (kemstem, M22e). An image can have several (any origins, neighbours and
 * k), each a dataset: listed here to select, rename or delete. The selected
 * one is plotted in the Neighbourhood tab below the image; its k-means
 * clusters, lattice vectors picked from them, and bonds measured along
 * those vectors are worked on below the list. */
export function NeighbourhoodSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const columns = useMemo(
    () => allDatasets(files).find((x) => x.parent_id === d.id && x.atomic?.output === "columns"),
    [files, d.id],
  );
  if (!columns)
    return (
      <p className="muted small">
        <ProcessLink to="columns" from="neighbourhoods" image={d.id}>Fit the atomic columns</ProcessLink> first.
      </p>
    );
  return <Neighbourhoods d={d} columns={columns} />;
}

function Neighbourhoods({ d, columns }: { d: DatasetSummary; columns: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const st = useNeighbourhoodState(d.id);
  const patch = (p: Partial<typeof st>) => useNeighbourhoods.getState().patch(d.id, p);
  const setDraft = (p: Partial<NeighbourhoodDraft>) => patch({ draft: { ...st.draft, ...p } });
  const subs = useMemo(() => sublatticesOf(files, columns), [files, columns]);
  const all = useMemo(() => allDatasets(files), [files]);
  const list = useMemo(() => neighbourhoodsOf(files, d.id), [files, d.id]);
  const nb = shownNeighbourhood(files, d.id, st.selected);
  // Settings naming a sublattice that has gone: back to all columns.
  const valid = (id: string) => (id && subs.some((s) => s.id === id) ? id : "");
  const draft = { ...st.draft, origins: valid(st.draft.origins), neighbours: valid(st.draft.neighbours) };
  const busy = !!st.busy;
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const sets = [{ id: "", label: "All columns" }, ...subs.map((s) => ({ id: s.id, label: s.title }))];
  const short = (id: string | null | undefined) => {
    const x = all.find((y) => y.id === id);
    return !x || x.atomic?.output === "columns" ? "all" : x.title.replace(/^Sublattice /, "");
  };
  const ready = () => patch({ draft });

  return (
    <section className="q-form">
      <h4 className="process-heading">Neighbourhoods</h4>
      {list.length > 0 ? (
        <ul className="nb-list">
          {list.map((x) => (
            <li
              key={x.id}
              className={`nb-row${x.id === nb?.id ? " active" : ""}`}
              onClick={() => selectNeighbourhood(d.id, x)}
              title="Work on this one (and plot it)"
            >
              <span className="nb-row-name" onClick={(e) => x.id === nb?.id && e.stopPropagation()}>
                {x.id === nb?.id ? <EditableTitle d={x} /> : x.title}
              </span>
              <span className="muted small">
                {short(originsOf(x))} → {short(x.atomic?.neighbours)} · {x.atomic?.k}
                {x.atomic?.clusters ? ` · ${x.atomic.clusters.length} cl.` : ""}
              </span>
              <button
                className="icon-button"
                title="Delete it (and its bonds)"
                onClick={(e) => {
                  e.stopPropagation();
                  useDatasets.getState().deleteDataset(x);
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted small">None yet.</p>
      )}
      <label title="The columns whose neighbourhoods are taken">
        <span>From</span>
        <select value={draft.origins} onChange={(e) => setDraft({ origins: e.target.value })}>
          {sets.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
      <label title="The columns counted as neighbours">
        <span>To</span>
        <select value={draft.neighbours} onChange={(e) => setDraft({ neighbours: e.target.value })}>
          {sets.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
      <label title="How many nearest neighbours of each column">
        <span>Neighbours</span>
        <input type="number" min={1} max={200} value={draft.k} onChange={(e) => setDraft({ k: Math.max(1, Math.min(200, Math.round(num(e.target.value, draft.k)))) })} />
      </label>
      <div className="process-actions">
        <button
          className="primary"
          disabled={busy}
          onClick={() => {
            ready();
            computeNeighbourhood(d.id, columns);
          }}
          title="A new neighbourhood with these settings"
        >
          {st.busy === "Adding" ? "Adding…" : "Add"}
        </button>
        {nb && (
          <button
            disabled={busy}
            onClick={() => {
              ready();
              computeNeighbourhood(d.id, columns, nb);
            }}
            title={`Make ${nb.title} again; its clusters and bonds go`}
          >
            {st.busy === "Updating" ? "Updating…" : "Update"}
          </button>
        )}
        {nb && (
          <button onClick={() => useUi.getState().setBottomTab("neighbourhood")} title="Show it in the pane below the image">
            Plot
          </button>
        )}
      </div>
      {nb && <Analysis d={d} nb={nb} busy={st.busy} />}
      {st.error && <p className="field-error">{st.error}</p>}
    </section>
  );
}

/** The selected neighbourhood's clusters, lattice vectors and bonds. */
function Analysis({ d, nb, busy }: { d: DatasetSummary; nb: DatasetSummary; busy: string | null }) {
  const a = useAnalysis(nb.id);
  const patch = (p: Partial<NeighbourhoodAnalysis>) => useNeighbourhoods.getState().patchNb(nb.id, p);
  const clusters = nb.atomic?.clusters;
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const setVector = (i: number, p: Partial<NeighbourhoodAnalysis["vectors"][number]>) =>
    patch({ vectors: a.vectors.map((v, j) => (j === i ? { ...v, ...p } : v)) });
  const showPlot = () => useUi.getState().setBottomTab("neighbourhood");

  return (
    <>
      <h4 className="process-heading">Clusters</h4>
      <label title="k-means clusters of the neighbourhood's vectors">
        <span>Clusters</span>
        <input type="number" min={2} max={200} value={a.n} onChange={(e) => patch({ n: Math.max(2, Math.min(200, Math.round(num(e.target.value, a.n)))) })} />
      </label>
      <div className="process-actions">
        <button disabled={!!busy} onClick={() => clusterNeighbourhood(d.id, nb)}>
          {busy === "Clustering" ? "Clustering…" : clusters ? "Cluster again" : "Cluster"}
        </button>
        {clusters && <span className="muted small">{clusters.length} clusters</span>}
      </div>

      <h4 className="process-heading">Lattice vectors</h4>
      <p className="muted small">{a.pick !== null ? `Click a cluster in the plot to set ${a.vectors[a.pick]?.name}` : " "}</p>
      {a.vectors.map((v, i) => (
        <div key={i} className={`lattice-vector${a.pick === i ? " picking" : ""}`}>
          <input className="lattice-vector-name" value={v.name} title="Name (of its bonds)" onChange={(e) => setVector(i, { name: e.target.value })} />
          <input
            type="number"
            step={0.1}
            value={v.dy ?? ""}
            placeholder="dy"
            title="dy (px, down)"
            onChange={(e) => setVector(i, { dy: e.target.value === "" ? null : num(e.target.value, 0) })}
          />
          <input
            type="number"
            step={0.1}
            value={v.dx ?? ""}
            placeholder="dx"
            title="dx (px, right)"
            onChange={(e) => setVector(i, { dx: e.target.value === "" ? null : num(e.target.value, 0) })}
          />
          <button
            className={`icon-toggle${a.pick === i ? " active" : ""}`}
            aria-pressed={a.pick === i}
            title="Pick it from a cluster in the plot"
            onClick={() => {
              patch({ pick: a.pick === i ? null : i });
              if (a.pick !== i) showPlot();
            }}
          >
            ⌖
          </button>
          <button
            className="icon-button"
            title="Remove"
            disabled={a.vectors.length < 2}
            onClick={() => patch({ vectors: a.vectors.filter((_, j) => j !== i), pick: null })}
          >
            ×
          </button>
        </div>
      ))}
      <div className="process-actions">
        <button
          onClick={() => patch({ vectors: [...a.vectors, { name: `v${subscript(a.vectors.length + 1)}`, dy: null, dx: null }], pick: a.vectors.length })}
        >
          Add vector
        </button>
      </div>
      <label title="How far (px) a neighbour may be from column + vector">
        <span>Within (px)</span>
        <input type="number" min={0.5} step={0.5} value={a.threshold} onChange={(e) => patch({ threshold: Math.max(0.1, num(e.target.value, a.threshold)) })} />
      </label>
      <div className="process-actions">
        <button
          className="primary"
          disabled={!!busy || !a.vectors.some((v) => v.dy !== null && v.dx !== null)}
          onClick={() => measureBonds(d.id, nb)}
        >
          {busy === "Measuring" ? "Measuring…" : "Measure bonds"}
        </button>
      </div>
      <BondsSummary nb={nb} />
      <Cage d={d} nb={nb} busy={busy} />
    </>
  );
}

/** A cage of neighbours round each origin (clusters picked in the plot, e.g.
 * the four A sites round a B site): its centre added as new columns, or
 * each origin's displacement from it measured (vectors, under the
 * neighbourhood). */
function Cage({ d, nb, busy }: { d: DatasetSummary; nb: DatasetSummary; busy: string | null }) {
  const a = useAnalysis(nb.id);
  const patch = (p: Partial<NeighbourhoodAnalysis>) => useNeighbourhoods.getState().patchNb(nb.id, p);
  const files = useDatasets((s) => s.files);
  const made = useMemo(
    () => allDatasets(files).find((x) => x.parent_id === nb.id && x.atomic?.output === "displacements"),
    [files, nb.id],
  );
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const centre = a.cage.length
    ? [a.cage.reduce((t, c) => t + c[0], 0) / a.cage.length, a.cage.reduce((t, c) => t + c[1], 0) / a.cage.length]
    : null;
  const px = pixelSize(d);
  const pm = (v: number | null | undefined) => (v == null ? "–" : displacementText(v, px));
  return (
    <>
      <h4 className="process-heading">Cage</h4>
      <label className="q-check-line small" title="Use each column's k nearest neighbours as the cage">
        <input type="checkbox" checked={a.cageNearest} onChange={(e) => patch({ cageNearest: e.target.checked, cagePick: false })} />
        The {nb.atomic?.k ?? "k"} nearest
      </label>
      {!a.cageNearest && <div className="process-actions">
        <button
          className={`icon-toggle${a.cagePick ? " active" : ""}`}
          aria-pressed={a.cagePick}
          title="Click clusters in the plot to add or remove them"
          onClick={() => {
            patch({ cagePick: !a.cagePick, pick: null });
            if (!a.cagePick) useUi.getState().setBottomTab("neighbourhood");
          }}
        >
          ⌖
        </button>
        <span className="muted small">
          {a.cage.length ? `${a.cage.length} clusters · centre (${centre!.map((v) => formatValue(v)).join(", ")}) px` : "No clusters picked"}
        </span>
        {a.cage.length > 0 && (
          <button className="icon-button" title="Clear the cage" onClick={() => patch({ cage: [], cageNote: null })}>
            ×
          </button>
        )}
      </div>}
      <label title={a.cageNearest ? "How far (px) a neighbour may be from the usual distance" : "How far (px) a neighbour may be from column + cluster"}>
        <span>Within (px)</span>
        <input type="number" min={0.5} step={0.5} value={a.cageThreshold} onChange={(e) => patch({ cageThreshold: Math.max(0.1, num(e.target.value, a.cageThreshold)) })} />
      </label>
      <div className="process-actions">
        <button
          disabled={!!busy || (!a.cageNearest && !a.cage.length)}
          onClick={() => seedCageCentres(d.id, nb)}
          title="Add each cage centre as a new column guess; fit in Atomic columns"
        >
          {busy === "Seeding" ? "Adding…" : "Add centres as columns"}
        </button>
        <button
          className="primary"
          disabled={!!busy || (!a.cageNearest && !a.cage.length)}
          onClick={() => measureCageDisplacement(d.id, nb)}
          title="Each origin's displacement from its cage centre, as vectors"
        >
          {busy === "Displacing" ? "Measuring…" : made ? "Measure again" : "Measure displacement"}
        </button>
      </div>
      {a.cageNote && (
        <p className="muted small">
          {a.cageNote}{" "}
          <ProcessLink to="columns" from="neighbourhoods" image={d.id}>Fit them</ProcessLink>
        </p>
      )}
      {made?.atomic && (
        <p className="muted small q-elements">
          median {pm(made.atomic.median)} · largest {pm(made.atomic.max)}
          {made.atomic.missing ? ` · ${made.atomic.missing} without` : ""}
        </p>
      )}
    </>
  );
}

/** The bonds measured from a neighbourhood: mean length and how many origins have none. */
function BondsSummary({ nb }: { nb: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const bonds = useMemo(() => allDatasets(files).filter((x) => x.parent_id === nb.id && x.atomic?.output === "bonds"), [files, nb.id]);
  if (!bonds.length) return null;
  return (
    <p className="muted small q-elements">
      {bonds
        .map((b) => `${b.atomic?.name} ${b.atomic?.mean_length != null ? formatValue(b.atomic.mean_length) : "–"} px${b.atomic?.missing ? ` (${b.atomic.missing} none)` : ""}`)
        .join(" · ")}
    </p>
  );
}

const subscript = (n: number) => String(n).replace(/\d/g, (c) => "₀₁₂₃₄₅₆₇₈₉"[Number(c)]);
