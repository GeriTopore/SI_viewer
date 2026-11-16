import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import {
  COLUMN_VALUES, SUBLATTICE_COLOURS, applySplit, clearSplit, ensureTable, kmeansSplit, sublatticesOf, useColumns,
  useColumnsState, valuesFrame, type ColumnPoints,
} from "../state/columns";
import { allDatasets, useDatasets } from "../state/datasets";
import { ThresholdHistogram } from "./ThresholdHistogram";

/** Process tab, images: sublattices of the fitted atomic columns (M22d),
 * each a dataset under the Columns. Made in one of several ways; so far by
 * a value of the columns. */
export function SublatticesSection({ d }: { d: DatasetSummary }) {
  const made = useDatasets((st) => allDatasets(st.files).find((x) => x.parent_id === d.id && x.atomic?.output === "columns"));
  if (!made) return <p className="muted small">Fit the atomic columns first.</p>;
  return (
    <section className="q-form">
      <ByValue d={d} made={made} />
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

/** k − 1 thresholds at even quantiles of ``values``, largest first. */
function quantileThresholds(values: Float32Array, k: number): number[] {
  const sorted = Float32Array.from(values).sort();
  const out: number[] = [];
  for (let i = k - 1; i >= 1; i--) out.push(sorted[Math.floor((i / k) * (sorted.length - 1))]);
  return out;
}

