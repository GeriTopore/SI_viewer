import { useEffect, useMemo, useState } from "react";
import type { DatasetSummary } from "../api/client";
import {
  COLUMN_VALUES, SUBLATTICE_COLOURS, SUBLATTICE_NAMES, applySplit, clearSplit, ensureTable, kmeansSplit, maskSublattice, sublatticesOf, useColumns,
  useColumnsState, valuesFrame, type ColumnPoints,
} from "../state/columns";
import { allDatasets, useDatasets } from "../state/datasets";
import { masksFor } from "../state/masks";
import { fitCell, removeSite, showCell, splitByCell, useCell, useCellState, type Vectors } from "../state/cell";
import { neighbourhoodsOf, useNeighbourhoods } from "../state/neighbourhoods";
import { formatValue } from "../format";
import { useUi } from "../state/ui";
import { ThresholdHistogram } from "./ThresholdHistogram";
import { ProcessLink } from "./ProcessLink";

/** Process tab, images: sublattices of the fitted atomic columns (M22d),
 * each a dataset under the Columns. Made in one of several ways: split by
 * a value of the columns, one at a time from a mask (the columns in it), or
 * by position in the cell of a reference sublattice's lattice. */
export function SublatticesSection({ d }: { d: DatasetSummary }) {
  const made = useDatasets((st) => allDatasets(st.files).find((x) => x.parent_id === d.id && x.atomic?.output === "columns"));
  if (!made)
    return (
      <p className="muted small">
        <ProcessLink to="columns" from="sublattices" image={d.id}>Fit the atomic columns</ProcessLink> first.
      </p>
    );
  return (
    <section className="q-form">
      <ByValue d={d} made={made} />
      <ByMask d={d} made={made} />
      <ByCell d={d} made={made} />
      <SplitError d={d} />
    </section>
  );
}

/** Splitting by a value: thresholds on its histogram (dragged, or from
 * k-means). */
function ByValue({ d, made }: { d: DatasetSummary; made: DatasetSummary }) {
  const st = useColumnsState(d.id);
  const { split } = st;
  useEffect(() => ensureTable(made), [made]);
  const table = useColumns((s) => s.tables[made.id]?.points);
  const values = useMemo(() => (table ? valuesFrame(table, split.by).data : null), [table, split.by]);
  const setSplit = (p: Partial<typeof split>) => useColumns.getState().patch(d.id, { split: { ...split, ...p } });
  // Thresholds at even quantiles when there are none for this value and count.
  useEffect(() => {
    if (values && values.length && split.thresholds.length !== split.k - 1) {
      setSplit({ thresholds: quantileThresholds(values, split.k) });
    }
  }); // eslint-disable-line react-hooks/exhaustive-deps
  const files = useDatasets((s) => s.files);
  const subs = useMemo(() => sublatticesOf(files, made), [files, made]);
  const names = subs.length ? subs : null;

  return (
    <>
      <h4 className="process-heading">By value</h4>
      <label title="The value the columns are split by">
        <span>By</span>
        <select value={split.by} onChange={(e) => setSplit({ by: e.target.value, thresholds: [] })}>
          {COLUMN_VALUES.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>
      </label>
      <label title="How many sublattices">
        <span>Sublattices</span>
        <input
          type="number"
          min={2}
          max={SUBLATTICE_COLOURS.length}
          value={split.k}
          onChange={(e) => {
            const k = Math.round(Number(e.target.value));
            if (k >= 2 && k <= SUBLATTICE_COLOURS.length) setSplit({ k, thresholds: [] });
          }}
        />
      </label>
      {values && split.thresholds.length === split.k - 1 ? (
        <ThresholdHistogram
          values={values}
          thresholds={split.thresholds}
          colours={SUBLATTICE_COLOURS}
          onChange={(thresholds) => setSplit({ thresholds })}
        />
      ) : (
        <p className="muted small">Loading the columns…</p>
      )}
      <div className="process-actions">
        <button disabled={!table || st.splitting} onClick={() => table && kmeansSplit(d, table as ColumnPoints)} title="Thresholds half-way between k-means clusters of the value">
          k-means
        </button>
        <button className="primary" disabled={!table || st.splitting || split.thresholds.length !== split.k - 1} onClick={() => table && applySplit(d, made, table)}>
          {names ? "Split again" : "Split"}
        </button>
        {names && (
          <button disabled={st.splitting} onClick={() => clearSplit(d, made)}>
            Clear
          </button>
        )}
      </div>
      {names && (
        <p className="muted small q-elements">
          {names.map((n) => `${n.title} ${n.atomic?.count ?? 0}`).join(" · ")}
        </p>
      )}
    </>
  );
}

/** A sublattice of the columns in a mask the size of the image (any of the
 * file's), added next to the others. */
function ByMask({ d, made }: { d: DatasetSummary; made: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const masks = useMemo(() => masksFor(files, d), [files, d]);
  const [chosen, setChosen] = useState<string | null>(null);
  const mask = masks.find((m) => m.id === chosen) ?? masks[0];
  const splitting = useColumnsState(d.id).splitting;
  const all = useMemo(() => allDatasets(files), [files]);
  const from = (m: DatasetSummary) => all.find((x) => x.id === m.parent_id)?.title;
  return (
    <>
      <h4 className="process-heading">By mask</h4>
      {mask ? (
        <>
          <label title="Add a sublattice of the columns in the mask">
            <span>Mask</span>
            <select value={mask.id} onChange={(e) => setChosen(e.target.value)}>
              {masks.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.parent_id === d.id ? m.title : `${m.title} (${from(m)})`}
                </option>
              ))}
            </select>
          </label>
          <div className="process-actions">
            <button className="primary" disabled={splitting} onClick={() => maskSublattice(d, made, mask.id)}>
              Add sublattice
            </button>
          </div>
        </>
      ) : (
        <p className="muted small">
          <ProcessLink to="mask" from="sublattices" image={d.id}>Make a mask</ProcessLink> first.
        </p>
      )}
    </>
  );
}

/** Splitting by position in cell: every column placed in the cell of a
 * reference sublattice's lattice (the first two lattice vectors picked in
 * a neighbourhood), shown in the Cell tab below the image, where sites are
 * clicked; each column goes to the nearest site's sublattice. */
function ByCell({ d, made }: { d: DatasetSummary; made: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const st = useCellState(d.id);
  const patch = (p: Parameters<ReturnType<typeof useCell.getState>["patch"]>[1]) => useCell.getState().patch(d.id, p);
  const subs = useMemo(() => sublatticesOf(files, made), [files, made]);
  // The reference: as chosen (after a split, the new sublattice of its name).
  const ref = subs.find((x) => x.id === st.reference) ?? subs.find((x) => x.title === (st.referenceTitle || "Sublattice A")) ?? subs[0];
  const nbs = useMemo(() => neighbourhoodsOf(files, d.id), [files, d.id]);
  const byNb = useNeighbourhoods((s) => s.byNb);
  const withVectors = nbs.filter((n) => (byNb[n.id]?.vectors ?? []).filter((v) => v.dy !== null && v.dx !== null).length >= 2);
  const nb = withVectors.find((n) => n.id === st.nb) ?? withVectors[0];
  const set = nb ? (byNb[nb.id]?.vectors ?? []).filter((v) => v.dy !== null && v.dx !== null) : [];
  const vectors: Vectors | null = set.length >= 2 ? [[set[0].dy!, set[0].dx!], [set[1].dy!, set[1].dx!]] : null;
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const busy = !!st.busy;
  const names = SUBLATTICE_NAMES;
  return (
    <>
      <h4 className="process-heading">By position in cell</h4>
      {!subs.length ? (
        <p className="muted small">Make a reference sublattice first (e.g. the brightest columns, by value).</p>
      ) : !withVectors.length ? (
        <p className="muted small">
          Pick two lattice vectors of the reference in a{" "}
          <ProcessLink to="neighbourhoods" from="sublattices" image={d.id}>neighbourhood</ProcessLink>.
        </p>
      ) : (
        <>
          <label title="The columns whose lattice makes the cell">
            <span>Reference</span>
            <select
              value={ref?.id ?? ""}
              onChange={(e) => {
                patch({ reference: e.target.value, referenceTitle: subs.find((x) => x.id === e.target.value)?.title ?? "" });
              }}
            >
              {subs.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.title}
                </option>
              ))}
            </select>
          </label>
          <label title="The neighbourhood whose first two vectors span the cell">
            <span>Vectors</span>
            <select value={nb?.id ?? ""} onChange={(e) => patch({ nb: e.target.value })}>
              {withVectors.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.title}
                </option>
              ))}
            </select>
          </label>
          {vectors && (
            <p className="muted small">
              {set[0].name} ({vectors[0].map((v) => formatValue(v)).join(", ")}) · {set[1].name} ({vectors[1].map((v) => formatValue(v)).join(", ")}) px
            </p>
          )}
          <div className="process-actions">
            <button disabled={busy || !ref || !vectors} onClick={() => ref && vectors && showCell(d.id, made, ref, vectors, nb?.id)} title="Place every column in the cell and plot them below the image">
              {st.busy === "Placing" ? "Placing…" : st.data ? "Place again" : "Show cell"}
            </button>
          </div>
          {st.data && (
            <>
              <label title="Click the Cell plot to add a site; click a site to remove it">
                <span>Add sites as</span>
                <select value={st.name} onChange={(e) => patch({ name: e.target.value })}>
                  {names.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              {st.sites.length ? (
                <ul className="nb-list">
                  {st.sites.map((x, i) => (
                    <li key={i} className="nb-row" onClick={() => useUi.getState().setBottomTab("cell")}>
                      <span className="nb-row-name">
                        <span className="pld-swatch" style={{ background: SUBLATTICE_COLOURS[names.indexOf(x.name)] }} /> {x.name}
                      </span>
                      <span className="muted small">
                        ({formatValue(x.u)}, {formatValue(x.v)})
                      </span>
                      <button
                        className="icon-button"
                        title="Remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeSite(d.id, i);
                        }}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted small">Click the sites on the Cell plot below the image.</p>
              )}
              <label title="Assign columns to sites by radius or by cluster">
                <span>Sites as</span>
                <select value={st.mode} onChange={(e) => patch({ mode: e.target.value as "radius" | "clusters", fits: null })}>
                  <option value="radius">Circles</option>
                  <option value="clusters">Clusters</option>
                </select>
              </label>
              <label title={st.mode === "radius" ? "How far (px) a column may be from its site" : "How near (px) a group must come to a site to be its"}>
                <span>{st.mode === "radius" ? "Within (px)" : "Reach (px)"}</span>
                <input type="number" min={0.5} step={1} value={st.radius} onChange={(e) => patch({ radius: Math.max(0.1, num(e.target.value, st.radius)), fits: null })} />
              </label>
              {st.mode === "clusters" && (
                <>
                  <label title="Columns this close (px) are of one group">
                    <span>Link (px)</span>
                    <input type="number" min={0.5} step={0.5} value={st.link} onChange={(e) => patch({ link: Math.max(0.5, num(e.target.value, st.link)), fits: null })} />
                  </label>
                  <label title="Fewest columns in a group's core; fewer are left out">
                    <span>Smallest group</span>
                    <input type="number" min={1} step={1} value={st.least} onChange={(e) => patch({ least: Math.max(1, Math.round(num(e.target.value, st.least))), fits: null })} />
                  </label>
                </>
              )}
              <div className="process-actions">
                {st.mode === "clusters" && (
                  <button disabled={busy || !st.sites.length} onClick={() => ref && fitCell(d.id, made, ref.id)} title="Find each site's clusters, without splitting">
                    {st.busy === "Fitting" ? "Finding…" : "Preview"}
                  </button>
                )}
                <button
                  className="primary"
                  disabled={busy || !st.sites.length}
                  onClick={() => ref && splitByCell(d.id, made, ref.id)}
                  title={st.mode === "clusters" ? "Each site's clusters form its sublattice; replaces existing sublattices" : "Each column to its nearest site; replaces existing sublattices"}
                >
                  {st.busy === "Splitting" ? "Splitting…" : "Split"}
                </button>
              </div>
              {st.mode === "clusters" && st.fits && (
                <p className="muted small q-elements">
                  {st.sites.map((x, i) => `${x.name} ${st.fits!.sites[i]?.count ?? 0}${st.fits!.sites[i]?.empty ? " (none near)" : ""}`).join(" · ")} ·{" "}
                  {st.fits.leftOut} left out
                  {st.fits.unpicked.length > 0 &&
                    ` · no site: ${st.fits.unpicked.map((g) => `${g.count} at (${g.u.toFixed(2)}, ${g.v.toFixed(2)})`).join(", ")}`}
                </p>
              )}
            </>
          )}
          {st.error && <p className="field-error">{st.error}</p>}
        </>
      )}
    </>
  );
}

function SplitError({ d }: { d: DatasetSummary }) {
  const error = useColumnsState(d.id).error;
  return error ? <p className="field-error">{error}</p> : null;
}

/** k − 1 thresholds at even quantiles of ``values``, largest first. */
function quantileThresholds(values: Float32Array, k: number): number[] {
  const sorted = Float32Array.from(values).sort();
  const out: number[] = [];
  for (let i = k - 1; i >= 1; i--) out.push(sorted[Math.floor((i / k) * (sorted.length - 1))]);
  return out;
}

