import { useEffect, useMemo } from "react";
import type { DatasetSummary } from "../api/client";
import { formatValue } from "../format";
import { useDatasets } from "../state/datasets";
import {
  detectPeaks, makeReference, measureDisplacements, pldResults, setFamilyDamp, usePld, usePldState, type PldSettings,
} from "../state/pld";
import { pixelSize } from "./AtomicSection";
import { ProcessLink } from "./ProcessLink";

/** Process tab, images: periodic lattice displacements (kemstem, M22f;
 * notebook 3). 1. Label the FFT's peaks: found and grouped automatically
 * (Bragg; off-lattice families q₁, q₂ …), or picked on the FFT; ticked
 * families and clicked peaks are damped. 2. Damp them into a reference
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
  const bragg = st.peaks.filter((p) => p.family === 0).length;
  const busy = !!st.busy || !!st.job;
  const famDamped = (id: number) => {
    const m = st.peaks.filter((p) => p.family === id);
    return m.length > 0 && m.every((p) => p.damp);
  };
  const pm = (v: number | null | undefined) => (v == null ? "–" : displacementText(v, px));

  return (
    <section className="q-form">
      <h4 className="process-heading">Superlattice peaks</h4>
      <label title="How many peaks to find at most">
        <span>Peaks</span>
        <input type="number" min={10} max={500} step={10} value={s.count} onChange={(e) => set({ count: Math.max(10, Math.min(500, Math.round(num(e.target.value, s.count)))) })} />
      </label>
      <label title="How far from the FFT's centre (cycles/px; 0.5 is Nyquist)">
        <span>Within (1/px)</span>
        <input type="number" min={0.02} max={0.5} step={0.05} value={s.within} onChange={(e) => set({ within: Math.max(0.02, Math.min(0.5, num(e.target.value, s.within))) })} />
      </label>
      <div className="process-actions">
        <button className="primary" disabled={busy} onClick={() => detectPeaks(d)} title="Find the FFT's peaks, index them in the lattice and group the others by their offset from it">
          {st.busy === "Detecting" ? "Finding…" : "Find peaks"}
        </button>
        {st.peaks.length > 0 && (
          <button onClick={() => patch({ peaks: [], families: [] })} title="Remove all the peaks">
            Clear
          </button>
        )}
      </div>
      {st.families.length > 0 && (
        <div className="pld-families">
          <span className="muted small">Damp</span>
          {st.families.map((f) => (
            <label key={f.id} className="q-check-line small" title={`Peaks at G ± q, q = (${f.q.join(", ")}) in the lattice; ${f.count} peaks`}>
              <input type="checkbox" checked={famDamped(f.id)} onChange={(e) => setFamilyDamp(d.id, f.id, e.target.checked)} />
              <span className={`pld-swatch family-${(f.id - 1) % 6}`} />
              q{subscript(f.id)} ({f.q.map((v) => formatValue(v)).join(", ")}) · {f.count}
            </label>
          ))}
        </div>
      )}
      <label className="q-check-line small" title="Clicks on the FFT add a peak (with its conjugate), or switch a peak between damped and not; right-click removes one">
        <input type="checkbox" checked={st.picking} onChange={(e) => patch({ picking: e.target.checked })} />
        Pick on the FFT
        <input type="checkbox" checked={s.conjugates} onChange={(e) => set({ conjugates: e.target.checked })} style={{ marginLeft: 10 }} />
        Conjugates
      </label>
      {st.peaks.length > 0 && (
        <p className="muted small">
          {damped} to damp · {bragg} Bragg · {st.peaks.length} in all
        </p>
      )}

      <h4 className="process-heading">Reference</h4>
      <label title="Each damped peak is set to its background within this radius (FFT px)">
        <span>Radius (px)</span>
        <input type="number" min={1} step={1} value={s.radius} onChange={(e) => set({ radius: Math.max(0.5, num(e.target.value, s.radius)) })} />
      </label>
      <div className="process-actions">
        <button className="primary" disabled={busy || !damped} onClick={() => makeReference(d)} title="The image without the damped peaks: PLD reference (and difference) under it">
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

const subscript = (n: number) => String(n).replace(/\d/g, (c) => "₀₁₂₃₄₅₆₇₈₉"[Number(c)]);
