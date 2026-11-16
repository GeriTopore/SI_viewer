import { useEffect } from "react";
import type { AtomicOutput, DatasetSummary, FftPeakOptions, FourierSettings } from "../api/client";
import { formatValue, inAngstrom } from "../format";
import {
  deletePeak, makeFourierMaps, peakAt, refinePeaks, useAtomic, useAtomicState, usedPeaks, type FftPeak,
} from "../state/atomic";
import { cropDataset } from "../state/datasets";
import { useUi } from "../state/ui";

/** Images only (not maps of a spectrum image). */
export const canAnalyseAtomic = (d: DatasetSummary) => d.kind === "image" && !d.is_element_map && d.origin !== "atomic";

/** Pixel size along y and x (Å when a length, else its units or px). */
export function pixelSize(d: DatasetSummary) {
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
              <th title="Used for Fourier filtering">Use</th>
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
                  <td>
                    <input
                      type="checkbox"
                      checked={p.use !== false}
                      onChange={(e) => patch({ peaks: st.peaks.map((q, j) => (j === i ? { ...q, use: e.target.checked } : q)) })}
                      aria-label={`Use peak ${i + 1}`}
                    />
                  </td>
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
      <FourierFiltering d={d} />
    </section>
  );
}

const OUTPUTS: [AtomicOutput, string, string][] = [
  ["amplitude", "Amplitude", "|filtered image|: where the lattice planes are (domains)"],
  ["grating", "Grating", "The filtered image's real part: the lattice planes as fringes"],
  ["phase", "Phase", "The planes' local phase (lock-in or GPA), 0–2π"],
  ["strain", "Strain ε∥, ε⊥", "From the phase's gradient, along and across the peak's direction"],
];

/** Fourier filtering round the peaks used (kemstem), and the phase and
 * strain from it: maps under the image (a job). */
function FourierFiltering({ d }: { d: DatasetSummary }) {
  const st = useAtomicState(d.id);
  const patch = (p: Parameters<ReturnType<typeof useAtomic.getState>["patch"]>[1]) => useAtomic.getState().patch(d.id, p);
  const set = (f: Partial<FourierSettings>) => patch({ fourier: { ...st.fourier, ...f } });
  const used = usedPeaks(st);
  const px = pixelSize(d);
  const [rows, cols] = [d.axes[0]?.size ?? 0, d.axes[1]?.size ?? 0];
  // kemstem's coarsening length (it uses the first axis).
  const r = (rows / (2 * Math.PI * st.fourier.sigma)) * Math.sqrt(Math.log(4));
  const tensorOk = used.length === 2;
  const tensor = st.tensor && tensorOk;
  const running = !!st.job;
  const phaseNeeded = st.outputs.includes("phase") || st.outputs.includes("strain") || tensor;
  const nothing = !(used.length && st.outputs.length) && !tensor;
  const toggle = (o: AtomicOutput, on: boolean) =>
    patch({ outputs: on ? [...st.outputs, o] : st.outputs.filter((x) => x !== o) });
  const num = (v: string) => (v.trim() === "" ? null : Number(v));
  const square = Math.min(rows, cols);

  return (
    <>
      <h4 className="process-heading">Fourier filtering</h4>
      <label title="Gaussian round each peak (σ, FFT pixels): larger keeps finer detail, and more noise">
        <span>σ (px)</span>
        <input type="number" min={0.5} step={1} value={st.fourier.sigma} onChange={(e) => set({ sigma: Math.max(0.5, Number(e.target.value)) })} />
      </label>
      <p className="muted small q-elements" title="Real-space coarsening radius (kemstem's coarsening_length)">
        Coarsening {formatValue(r * px.x)} {px.units}
        {px.units !== "px" && ` (${formatValue(r)} px)`}
      </p>
      <div className="process-actions">
        {OUTPUTS.map(([o, label, tip]) => (
          <label key={o} className="q-check" title={tip}>
            <input type="checkbox" checked={st.outputs.includes(o)} onChange={(e) => toggle(o, e.target.checked)} />
            {label}
          </label>
        ))}
        <label
          className="q-check"
          title={tensorOk ? "εxx, εyy, εxy and the rotation ωxy from the two peaks used" : "Use exactly two peaks in different directions"}
        >
          <input type="checkbox" checked={tensor} disabled={!tensorOk} onChange={(e) => patch({ tensor: e.target.checked })} />
          Strain tensor
        </label>
      </div>
      {phaseNeeded && (
        <>
          <div className="segmented process-choice" role="group" aria-label="Phase">
            {([["lockin", "Lock-in", "Phase lock-in (kemstem phaselock)"], ["gpa", "GPA", "Geometric phase (kemstem gpa)"]] as const).map(([id, label, tip]) => (
              <button key={id} className={st.fourier.phase === id ? "active" : undefined} aria-pressed={st.fourier.phase === id} onClick={() => set({ phase: id })} title={tip}>
                {label}
              </button>
            ))}
          </div>
          {st.fourier.phase === "lockin" && (
            <label title="The lock-in's low-pass (σ, FFT pixels); empty: twice the filter's">
              <span>Lock-in σ (px)</span>
              <input
                type="number"
                min={0.5}
                step={1}
                placeholder={formatValue(2 * st.fourier.sigma)}
                value={st.fourier.lock_sigma ?? ""}
                onChange={(e) => set({ lock_sigma: num(e.target.value) })}
              />
            </label>
          )}
          {(st.outputs.includes("strain") || tensor) && (
            <label title="Strains beyond ±this are replaced by their median-filtered surroundings; empty: kept">
              <span>Outlier limit |ε|</span>
              <input
                type="number"
                min={0}
                step={0.01}
                placeholder="None"
                value={st.fourier.mask_threshold ?? ""}
                onChange={(e) => set({ mask_threshold: num(e.target.value) })}
              />
            </label>
          )}
        </>
      )}
      {rows !== cols && (st.outputs.includes("strain") || tensor) && (
        <div className="process-actions">
          <span className="muted small">kemstem assumes a square image</span>
          <button
            onClick={() => {
              const x0 = Math.floor((cols - square) / 2);
              const y0 = Math.floor((rows - square) / 2);
              void cropDataset(d, { region: { kind: "rect", x0, y0, x1: x0 + square, y1: y0 + square } });
            }}
            title={`A ${square}×${square} copy of its centre, under it`}
          >
            Crop to a square
          </button>
        </div>
      )}
      {st.fourierError && <p className="field-error">{st.fourierError}</p>}
      {running && st.job ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(st.job.progress * 100)}>
            <div style={{ width: `${Math.round(st.job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>{st.job.message || "Waiting"}</span>
          </div>
        </div>
      ) : (
        <button
          className="primary"
          disabled={nothing}
          onClick={() => makeFourierMaps(d)}
          title={used.length ? `From peak${used.length > 1 ? "s" : ""} ${used.map((u) => u.n).join(", ")}` : "Tick the peaks to use"}
        >
          Make maps
        </button>
      )}
    </>
  );
}
