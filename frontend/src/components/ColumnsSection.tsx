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
  const method = s.method ?? "maxima";
  const windowOk = s.window >= 3 && s.window % 2 === 1;
  const running = !!job;
  const num = (v: string, fallback: number) => (Number.isFinite(Number(v)) && v.trim() !== "" ? Number(v) : fallback);
  const fields = test?.fields ?? [];
  const value = (name: string) => test?.row[fields.indexOf(name)] ?? null;
  const residual = test ? test.data.map((row, i) => row.map((v, j) => v - test.fit[i][j])) : null;

  return (
    <section className="q-form">
      <h4 className="process-heading">Find</h4>
      <label title="Local maxima of the image (kemstem), or of its correlation with a Gaussian the size of a column (finds dim columns as well as bright ones)">
        <span>Method</span>
        <select
          value={method}
          onChange={(e) => {
            const m = e.target.value as "maxima" | "template";
            // Each its own kind of threshold: of the image (0–1), or a correlation.
            set({ method: m, threshold: m === "template" ? 0.6 : 0.1 });
          }}
        >
          <option value="maxima">Local maxima</option>
          <option value="template">Template correlation</option>
        </select>
      </label>
      {method === "template" && (
        <label title="σ (px) of the Gaussian template: about a column's">
          <span>Template σ (px)</span>
          <input type="number" min={0.3} step={0.5} value={s.template ?? 1.5} onChange={(e) => set({ template: Math.max(0.3, num(e.target.value, s.template ?? 1.5)) })} />
        </label>
      )}
      <label title="Gaussian blur (σ, px) of the normalised image, before finding and fitting">
        <span>Blur σ (px)</span>
        <input type="number" min={0} step={0.25} value={s.blur} onChange={(e) => set({ blur: Math.max(0, num(e.target.value, s.blur)) })} />
      </label>
      <label title="Finding only: a broad Gaussian (σ, px) subtracted first, leaving the brightest columns (e.g. the A sites); 0: none">
        <span>Background σ (px)</span>
        <input type="number" min={0} step={5} value={s.background ?? 0} onChange={(e) => set({ background: Math.max(0, num(e.target.value, s.background ?? 0)) })} />
      </label>
      <label title="Least distance between two columns (px)">
        <span>Distance (px)</span>
        <input type="number" min={1} step={1} value={s.distance} onChange={(e) => set({ distance: Math.max(1, Math.round(num(e.target.value, s.distance))) })} />
      </label>
      {method === "template" ? (
        <label title="Places correlating less than this with the template (−1 to 1) are left out">
          <span>Correlation</span>
          <input type="number" min={-0.99} max={0.99} step={0.05} value={s.threshold} onChange={(e) => set({ threshold: Math.min(0.99, Math.max(-0.99, num(e.target.value, s.threshold))) })} />
        </label>
      ) : (
        <label title="Columns dimmer than this (0–1 of the normalised image) are left out">
          <span>Threshold</span>
          <input type="number" min={0} max={0.99} step={0.05} value={s.threshold} onChange={(e) => set({ threshold: Math.min(0.99, Math.max(0, num(e.target.value, s.threshold))) })} />
        </label>
      )}
      <div className="process-actions">
        <button className="primary" disabled={st.finding || running} onClick={() => findColumns(d)}>
          {st.finding ? "Finding…" : "Find"}
        </button>
        {found && (
          <span className="muted small">
            {found.length} {st.append ? `new columns (${st.append.name})` : "columns"}
          </span>
        )}
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
          {st.append ? `Fit and add ${found?.length ?? 0} columns as ${st.append.name}` : `Fit ${found?.length ? `${found.length} columns` : "the columns"}`}
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
