import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { useDatasets } from "../state/datasets";
import {
  deletePldPeak, makeReference, measureDisplacements, pldPeakAt, pldResults, refinePldPeaks, togglePeak, usePld, usePldState, type PldSettings,
} from "../state/pld";
import { pixelSize } from "./AtomicSection";
import { ProcessLink } from "./ProcessLink";

/** Process tab, images: periodic lattice displacements (kemstem, M22f;
 * notebook 3). 1. Pick the superlattice peaks on the FFT and refine them
 * (automatic finding is hidden for now: ``detectPeaks`` in state/pld.ts and
 * ``/atomic/pld/detect`` stay); ticked peaks are damped. 2. Damp them into a reference
 * image ("PLD reference", and "PLD difference"). 3. Fit the columns again
 * in the reference and measure each one's displacement ("Displacements"
 * under the Columns, drawn as vectors). */
export function PldSection({ d }: { d: DatasetSummary }) {
  const st = usePldState(d.id);
  const patch = (p: Partial<typeof st>) => usePld.getState().patch(d.id, p);
  const set = (p: Partial<PldSettings>) => patch({ settings: { ...st.settings, ...p } });
  const files = useDatasets((s) => s.files);
  const { columns, reference, displacements } = useMemo(() => pldResults(files, d.id), [files, d.id]);
  const px = pixelSize(d);
  useEffect(() => {
    const { setShown } = usePld.getState();
    setShown(d.id);
    return () => {
      if (usePld.getState().shown === d.id) setShown(null);
    };
  }, [d.id]);

  const s = st.settings;
  const num = (v: string, fallback: number) => (v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const damped = st.peaks.filter((p) => p.damp).length;
  const windowOk = s.window >= 3 && s.window % 2 === 1;
  const busy = !!st.busy || !!st.job;
  const pm = (v: number | null | undefined) => (v == null ? "–" : displacementText(v, px));

  return (
    <section className="q-form">
      <h4 className="process-heading">Superlattice peaks</h4>
      <div className="process-actions">
        <label className="q-check" title="Click the FFT to add a peak, or to switch one between damped and not; right-click a peak to delete it">
          <input type="checkbox" checked={st.picking} onChange={(e) => patch({ picking: e.target.checked })} />
          Pick on the FFT
        </label>
        <label className="q-check" title="Each click also adds the peak at −k">
          <input type="checkbox" checked={s.conjugates} onChange={(e) => set({ conjugates: e.target.checked })} />
          Conjugates
        </label>
      </div>
      {st.peaks.length === 0 ? (
        <p className="muted small q-elements">No peaks</p>
      ) : (
        <table className="q-table">
          <thead>
            <tr>
              <th title="Damped in the reference">Damp</th>
              <th>#</th>
              <th className="num">d ({px.units})</th>
              <th className="num">Angle</th>
              <th />
              <th />
            </tr>
          </thead>
          <tbody>
            {st.peaks.map((p, i) => {
              const sp = spacing(pldPeakAt(p), px);
              return (
                <tr key={i}>
                  <td>
                    <input type="checkbox" checked={p.damp} onChange={() => togglePeak(d.id, i)} aria-label={`Damp peak ${i + 1}`} />
                  </td>
                  <td>{i + 1}</td>
                  <td className="num">{formatValue(sp.d)}</td>
                  <td className="num">{formatValue(Math.round(sp.angle * 10) / 10)}°</td>
                  <td className="muted small" title={p.refined ? (p.refined.failed ? "Not fitted (as picked)" : "Refined") : "As picked"}>
                    {p.refined ? (p.refined.failed ? "✕" : "✓") : ""}
                  </td>
                  <td>
                    <button className="link-button" onClick={() => deletePldPeak(d.id, i)} title="Delete the peak (and its conjugate)">
                      ×
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <label title="Width of the window fitted round each peak (odd, FFT px)">
        <span>Window (px)</span>
        <input type="number" min={3} step={2} value={s.window} onChange={(e) => set({ window: Math.round(num(e.target.value, s.window)) })} />
      </label>
      <label title="Gaussian blur of |F| before fitting (σ, FFT px)">
        <span>Blur σ (px)</span>
        <input type="number" min={0} step={0.5} value={s.blur} onChange={(e) => set({ blur: Math.max(0, num(e.target.value, s.blur)) })} />
      </label>
      {!windowOk && <p className="field-error">The window must be odd, 3 or more</p>}
      <div className="process-actions">
        <button className="primary" disabled={busy || !st.peaks.length || !windowOk} onClick={() => refinePldPeaks(d)} title="Fit a 2D Gaussian to each peak, from where it is">
          {st.busy === "Refining" ? "Refining…" : "Refine"}
        </button>
        <button disabled={!st.peaks.length} onClick={() => patch({ peaks: [], families: [], error: null })}>
          Clear
        </button>
        {st.peaks.length > 0 && <span className="muted small">{damped} of {st.peaks.length} damped</span>}
      </div>

      <h4 className="process-heading">Reference</h4>
      <label title="Each damped peak is set to its background within this radius (FFT px)">
        <span>Radius (px)</span>
        <input type="number" min={1} step={1} value={s.radius} onChange={(e) => set({ radius: Math.max(0.5, num(e.target.value, s.radius)) })} />
      </label>
      <div className="process-actions">
        <button className="primary" disabled={busy || !damped || !windowOk} onClick={() => makeReference(d)} title="The image without the damped peaks: PLD reference (and difference) under it">
          {st.busy === "Damping" ? "Damping…" : reference ? "Make again" : "Make reference"}
        </button>
      </div>

      <h4 className="process-heading">Displacements</h4>
      {!columns ? (
        <p className="muted small">
          <ProcessLink to="columns" from="pld" image={d.id}>Fit the atomic columns</ProcessLink> first.
        </p>
      ) : (
        <>
          <label title="How far (px) a column may have moved between the reference and the image">
            <span>Within (px)</span>
            <input type="number" min={0.05} step={0.1} value={s.threshold} onChange={(e) => set({ threshold: Math.max(0.01, num(e.target.value, s.threshold)) })} />
          </label>
          {st.job ? (
            <div className="q-job">
              <div className="q-progress" role="progressbar" aria-valuenow={Math.round(st.job.progress * 100)}>
                <div style={{ width: `${Math.round(st.job.progress * 100)}%` }} />
              </div>
              <div className="q-job-row small">
                <span>{st.job.message || "Waiting"}</span>
              </div>
            </div>
          ) : (
            <div className="process-actions">
              <button className="primary" disabled={busy || !reference} onClick={() => measureDisplacements(d, columns)} title={reference ? "Fit the columns again in the reference and measure how far each moved" : "Make the reference first"}>
                {displacements ? "Measure again" : "Measure"}
              </button>
            </div>
          )}
          {displacements?.atomic && (
            <p className="muted small q-elements">
              median {pm(displacements.atomic.median)} · largest {pm(displacements.atomic.max)}
              {displacements.atomic.missing ? ` · ${displacements.atomic.missing} without` : ""}
            </p>
          )}
        </>
      )}
      {st.error && <p className="field-error">{st.error}</p>}
    </section>
  );
}

/** A displacement (px) in pm when calibrated (in Å), else px. */
export function displacementText(v: number, px: { x: number; units: string }) {
  if (px.units === "Å") return `${formatValue(v * px.x * 100)} pm`;
  return px.units === "px" ? `${formatValue(v)} px` : `${formatValue(v * px.x)} ${px.units}`;
}

/** A peak's lattice spacing and angle (y up), in the image's units. */
function spacing({ fy, fx }: { fy: number; fx: number }, px: { y: number; x: number }) {
  const ky = fy / px.y;
  const kx = fx / px.x;
  return { d: 1 / Math.hypot(kx, ky), angle: (Math.atan2(-ky, kx) * 180) / Math.PI };
}
