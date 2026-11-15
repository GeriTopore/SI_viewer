import { useEffect } from "react";
import type { DatasetSummary, FftPeakOptions } from "../api/client";
import { formatValue, inAngstrom } from "../format";
import { deletePeak, peakAt, refinePeaks, useAtomic, useAtomicState, type FftPeak } from "../state/atomic";
import { useUi } from "../state/ui";

/** Images only (not maps of a spectrum image). */
export const canAnalyseAtomic = (d: DatasetSummary) => d.kind === "image" && !d.is_element_map;

/** Pixel size along y and x (Å when a length, else its units or px). */
function pixelSize(d: DatasetSummary) {
  const [ay, ax] = d.axes;
  if (!ax?.units) return { y: 1, x: 1, units: "px" };
  const x = inAngstrom(ax.scale, ax.units);
  const y = inAngstrom(ay.scale, ay.units || ax.units);
  return { y: y.size, x: x.size, units: x.units };
}

/** A peak's spacing d (1/|k|) and direction (°, anticlockwise from x). */
export function peakSpacing(p: FftPeak, px: { y: number; x: number }) {
  const { fy, fx } = peakAt(p);
  const ky = fy / px.y;
  const kx = fx / px.x;
  return { d: 1 / Math.hypot(kx, ky), angle: (Math.atan2(-ky, kx) * 180) / Math.PI };
}

/** Process tab, images: atomic-resolution analysis with kemstem (M22).
 * FFT peaks: picked on the FFT panel (which shows the whole image's FFT
 * while this is open), refined with 2D Gaussians or centre of mass. */
export function AtomicSection({ d }: { d: DatasetSummary }) {
  const st = useAtomicState(d.id);
  const patch = (p: Parameters<ReturnType<typeof useAtomic.getState>["patch"]>[1]) => useAtomic.getState().patch(d.id, p);
  const setOptions = (o: Partial<FftPeakOptions>) => patch({ options: { ...st.options, ...o } });
  const px = pixelSize(d);

  useEffect(() => {
    const { setShown } = useAtomic.getState();
    setShown(d.id);
    useUi.getState().setFftOpen(true);
    return () => {
      if (useAtomic.getState().shown === d.id) setShown(null);
    };
  }, [d.id]);

  const refined = st.peaks.filter((p) => p.refined && !p.refined.failed).length;
  const windowOk = st.options.window >= 3 && st.options.window % 2 === 1;

  return (
    <section className="q-form">
      <h4 className="process-heading">FFT peaks</h4>
      <div className="process-actions">
        <label className="q-check" title="Click the FFT to add a peak; right-click a peak to delete it">
          <input type="checkbox" checked={st.picking} onChange={(e) => patch({ picking: e.target.checked })} />
          Pick on the FFT
        </label>
        <label className="q-check" title="Each click also adds the peak at −k">
          <input type="checkbox" checked={st.conjugates} onChange={(e) => patch({ conjugates: e.target.checked })} />
          Conjugates
        </label>
      </div>
      {st.peaks.length === 0 ? (
        <p className="muted small q-elements">No peaks</p>
      ) : (
        <table className="q-table">
          <thead>
            <tr>
              <th>#</th>
              <th className="num">d ({px.units})</th>
              <th className="num">Angle</th>
              <th />
              <th />
            </tr>
          </thead>
          <tbody>
            {st.peaks.map((p, i) => {
              const s = peakSpacing(p, px);
              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td className="num">{formatValue(s.d)}</td>
                  <td className="num">{formatValue(Math.round(s.angle * 10) / 10)}°</td>
                  <td className="muted small" title={p.refined ? (p.refined.failed ? "Not refined (as clicked)" : "Refined") : "As clicked"}>
                    {p.refined ? (p.refined.failed ? "✕" : "✓") : ""}
                  </td>
                  <td>
                    <button className="link-button" onClick={() => deletePeak(d.id, i)} title="Delete the peak">
                      ×
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <div className="segmented process-choice" role="group" aria-label="Refinement">
        {([["gauss", "Gaussian", "2D Gaussian fits (kemstem refine_peaks_gf)"], ["com", "Centre of mass", "Iterated centre of mass (kemstem refine_peaks_com)"]] as const).map(([id, label, tip]) => (
          <button key={id} className={st.options.method === id ? "active" : undefined} aria-pressed={st.options.method === id} onClick={() => setOptions({ method: id })} title={tip}>
            {label}
          </button>
        ))}
      </div>
      <label title="Width of the window around each peak (odd, FFT pixels)">
        <span>Window (px)</span>
        <input type="number" min={3} step={2} value={st.options.window} onChange={(e) => setOptions({ window: Math.round(Number(e.target.value)) })} />
      </label>
      <label title="Gaussian blur of the log FFT before refining (σ, FFT pixels)">
        <span>Blur σ (px)</span>
        <input type="number" min={0} step={0.5} value={st.options.blur} onChange={(e) => setOptions({ blur: Math.max(0, Number(e.target.value)) })} />
      </label>
      <label title="Reduce the FFT's edge streaks first">
        <span>Edges</span>
        <select value={st.options.edge} onChange={(e) => setOptions({ edge: e.target.value as FftPeakOptions["edge"] })}>
          <option value="none">As is</option>
          <option value="hann">Hann window</option>
          <option value="pplus">Periodic + smooth</option>
        </select>
      </label>
      {!windowOk && <p className="field-error">The window must be odd, 3 or more</p>}
      {st.error && <p className="field-error">{st.error}</p>}
      <div className="process-actions">
        <button className="primary" disabled={st.busy || !st.peaks.length || !windowOk} onClick={() => refinePeaks(d)}>
          {st.busy ? "Refining…" : "Refine"}
        </button>
        <button disabled={!st.peaks.length} onClick={() => patch({ peaks: [], error: null })}>
          Clear
        </button>
        {refined > 0 && <span className="muted small">{refined} of {st.peaks.length} refined</span>}
      </div>
    </section>
  );
}
