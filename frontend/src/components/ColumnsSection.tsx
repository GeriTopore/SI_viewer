import { useEffect } from "react";
import type { ColumnSettings, DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { findColumns, refineColumns, testColumn, useColumns, useColumnsState } from "../state/columns";
import { allDatasets, useDatasets } from "../state/datasets";
import { useViewerImage } from "../state/viewerImage";
import { pixelSize } from "./AtomicSection";
import { MiniImage } from "./MiniImage";

/** Process tab, images: atomic columns with kemstem (M22d). Find them
 * (``find_columns``, drawn on the image as they are found), test the fit on
 * one, then fit them all with 2D Gaussians (``refine_columns``, a job) into
 * the image's Columns dataset. */
export function ColumnsSection({ d }: { d: DatasetSummary }) {
  const st = useColumnsState(d.id);
  const patch = (p: Parameters<ReturnType<typeof useColumns.getState>["patch"]>[1]) => useColumns.getState().patch(d.id, p);
  const set = (s: Partial<ColumnSettings>) => patch({ settings: { ...st.settings, ...s } });
  // The fitted columns (their dataset), if any.
  const made = useDatasets((st) => allDatasets(st.files).find((x) => x.parent_id === d.id && x.atomic?.output === "columns"));
  // The viewer's part of this image in view: the test fits the column nearest its centre.
  const view = useViewerImage((s) => (s.image?.datasetId === d.id ? s.image : null));
  const px = pixelSize(d);

  useEffect(() => {
    const { setShown } = useColumns.getState();
    setShown(d.id);
    return () => {
      if (useColumns.getState().shown === d.id) setShown(null);
    };
  }, [d.id]);

  const { settings: s, found, job, test } = st;
  const windowOk = s.window >= 3 && s.window % 2 === 1;
  const running = !!job;
  const num = (v: string, fallback: number) => (Number.isFinite(Number(v)) && v.trim() !== "" ? Number(v) : fallback);
  const fields = test?.fields ?? [];
  const value = (name: string) => test?.row[fields.indexOf(name)] ?? null;
  const residual = test ? test.data.map((row, i) => row.map((v, j) => v - test.fit[i][j])) : null;

  return (
    <section className="q-form">
      <h4 className="process-heading">Find</h4>
      <label title="Gaussian blur (σ, px) of the normalised image, before finding and fitting">
        <span>Blur σ (px)</span>
        <input type="number" min={0} step={0.25} value={s.blur} onChange={(e) => set({ blur: Math.max(0, num(e.target.value, s.blur)) })} />
      </label>
      <label title="Least distance between two columns (px)">
        <span>Distance (px)</span>
        <input type="number" min={1} step={1} value={s.distance} onChange={(e) => set({ distance: Math.max(1, Math.round(num(e.target.value, s.distance))) })} />
      </label>
      <label title="Columns dimmer than this (0–1 of the normalised image) are left out">
        <span>Threshold</span>
        <input type="number" min={0} max={0.99} step={0.05} value={s.threshold} onChange={(e) => set({ threshold: Math.min(0.99, Math.max(0, num(e.target.value, s.threshold))) })} />
      </label>
      <div className="process-actions">
        <button className="primary" disabled={st.finding || running} onClick={() => findColumns(d)}>
          {st.finding ? "Finding…" : "Find"}
        </button>
        {found && <span className="muted small">{found.length} columns</span>}
      </div>

      <h4 className="process-heading">Fit</h4>
      <label title="Width of the window fitted round each column (odd, px)">
        <span>Window (px)</span>
        <input type="number" min={3} step={2} value={s.window} onChange={(e) => set({ window: Math.round(num(e.target.value, s.window)) })} />
      </label>
      {!windowOk && <p className="field-error">The window must be odd, 3 or more</p>}
      <div className="process-actions">
        <button
          disabled={!found?.length || !windowOk || st.testing || running}
          onClick={() =>
            testColumn(d, view ? [(view.y0 + view.y1) / 2, (view.x0 + view.x1) / 2] : [(d.axes[0]?.size ?? 0) / 2, (d.axes[1]?.size ?? 0) / 2])
          }
          title="Fit the column nearest the centre of the view, to check the window"
        >
          {st.testing ? "Testing…" : "Test one"}
        </button>
      </div>
      {test && residual && (
        <div className="columns-test">
          <figure>
            <MiniImage data={test.data} colormap="gray" size={64} title="Data" />
            <figcaption>Data</figcaption>
          </figure>
          <figure>
            <MiniImage data={test.fit} colormap="gray" size={64} title="Fit" />
            <figcaption>Fit</figcaption>
          </figure>
          <figure>
            <MiniImage data={residual} colormap="rdbu" size={64} range={symmetric(residual)} title="Data − fit" />
            <figcaption>Residual</figcaption>
          </figure>
          <p className="muted small">
            ({formatValue(test.at[1])}, {formatValue(test.at[0])}) ·{" "}
            {value("failed") ? "fit failed" : `σ ${fmt(value("sigma_1"), px)}, ${fmt(value("sigma_2"), px)} · I ${formatValue(value("intensity") ?? NaN)}`}
          </p>
        </div>
      )}
      {st.error && <p className="field-error">{st.error}</p>}
      {running && job ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>{job.message || "Waiting"}</span>
          </div>
        </div>
      ) : (
        <button className="primary" disabled={!found?.length || !windowOk} onClick={() => refineColumns(d)}>
          Fit {found?.length ? `${found.length} columns` : "the columns"}
        </button>
      )}
      {made?.atomic && (
        <p className="muted small q-elements">
          {made.atomic.count} fitted{made.atomic.failed ? `, ${made.atomic.failed} failed` : ""} (window {made.atomic.window} px)
        </p>
      )}
    </section>
  );
}

/** A width in px, with Å when calibrated. */
function fmt(v: number | null, px: { x: number; units: string }) {
  if (v === null) return "–";
  return px.units === "px" ? `${formatValue(v)} px` : `${formatValue(v * px.x)} ${px.units}`;
}

function symmetric(rows: number[][]): [number, number] {
  let m = 0;
  for (const r of rows) for (const v of r) m = Math.max(m, Math.abs(v));
  return [-(m || 1), m || 1];
}
