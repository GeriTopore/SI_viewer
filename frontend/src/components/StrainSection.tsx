import { useEffect } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import {
  bvmVersion, fitOf, latticeGrid, peakOptionsOf, peaksOnLattice, strainJobRunning, toleranceOf, useStrain, type BvmPeakOptions,
} from "../state/strain";
import { formatValue } from "../format";

const PEAK_FIELDS: { key: keyof BvmPeakOptions; label: string; title: string; min: number; max: number; step: number }[] = [
  { key: "sigma", label: "Blur (px)", title: "Gaussian blur of the map before its peaks are found (0: none)", min: 0, max: 50, step: 0.25 },
  { key: "threshold", label: "Threshold", title: "Lowest peak kept, as a fraction of the brightest", min: 0, max: 1, step: 0.01 },
  { key: "min_spacing", label: "Spacing (px)", title: "Closest two peaks may be", min: 0.5, max: 1000, step: 0.5 },
];

/** Strain from the Bragg vectors. Step 1, the lattice on the Bragg vector
 * map (shown in the pattern pane): its peaks found live, O, g1 and g2
 * proposed from them and dragged on the map (snapping to peaks), the
 * tolerance within which disks are indexed to the lattice points. Step 2,
 * the fit: tried on the patterns at the cursors, then every pattern's disks
 * indexed and the lattice fitted at every position (a job). */
export function StrainSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const cube = allDatasets(files).find((x) => x.id === d.id) ?? d;
  const hasVectors = allDatasets(files).some((x) => x.origin === "disks" && x.parent_id === d.id);
  const opts = useStrain((s) => peakOptionsOf(s, d.id));
  const peaks = useStrain((s) => s.peaks[d.id]);
  const lattice = useStrain((s) => s.lattice[d.id]);
  const tol = useStrain((s) => toleranceOf(s, d.id));
  const busy = useStrain((s) => !!s.busy[d.id]);
  const error = useStrain((s) => s.errors[d.id] ?? null);
  const fit = useStrain((s) => fitOf(s, d.id));
  const preview = useStrain((s) => s.preview[d.id] ?? true);
  const job = useStrain((s) => s.jobs[d.id]);
  const { setOptions, setLattice, setTolerance, findPeaks, setFit, setPreview, runFit, cancelFit } = useStrain.getState();
  const running = strainJobRunning(job);
  const fitted = allDatasets(files).find((x) => x.origin === "strain-fit" && x.strain?.source_id === d.id && x.strain.output === "lattice");
  // A lattice fitted earlier (an analysis reopened): its lattice and settings back.
  useEffect(() => {
    const st = useStrain.getState();
    if (!fitted?.strain || st.lattice[d.id]) return;
    st.setLattice(d.id, fitted.strain.lattice);
    st.setTolerance(d.id, fitted.strain.options.tolerance);
    st.setFit(d.id, { min_disks: fitted.strain.options.min_disks, weighted: fitted.strain.options.weighted });
  }, [fitted?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Found again as the settings or the map (disks, centring) change.
  const key = `${bvmVersion(files, cube)}|${JSON.stringify(opts)}`;
  useEffect(() => {
    if (!hasVectors || peaks?.key === key) return;
    const id = setTimeout(() => void findPeaks(d.id, key), 200);
    return () => clearTimeout(id);
  }, [key, hasVectors]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!hasVectors) return <p className="muted small">Find the disks first (Disk detection).</p>;

  const [w, h] = cube.signal_shape;
  const points = lattice ? latticeGrid(lattice, w, h) : [];
  const on = peaks && lattice ? peaksOnLattice(peaks.list, points, tol).size : 0;
  const cal = cube.calibration;
  // The pattern's reciprocal pixel size (the ellipse not corrected here):
  // the calibration's, else the pattern axes' if they are in 1/nm or 1/Å.
  const kx = cube.axes.find((a) => !a.navigate);
  const axisCal = kx && (kx.units === "1/nm" || kx.units === "1/Å") ? { scale: kx.scale, units: kx.units } : null;
  const scale = cal?.scale ?? axisCal?.scale ?? null;
  const units = cal?.units ?? axisCal?.units ?? "";
  const length = (gx: number, gy: number) => {
    const px = Math.hypot(gx, gy);
    if (!scale) return `${formatValue(px)} px`;
    const g = px * scale;
    const dA = units === "1/nm" ? 10 / g : 1 / g;
    return `${formatValue(px)} px · ${formatValue(g)} ${units} · d ${formatValue(dA)} Å`;
  };
  // Angles on screen (y down), counter-clockwise from x as seen.
  const angle = (gx: number, gy: number) => (Math.atan2(-gy, gx) * 180) / Math.PI;
  const between = lattice
    ? Math.abs(((angle(lattice.g2x, lattice.g2y) - angle(lattice.g1x, lattice.g1y) + 540) % 360) - 180)
    : 0;

  return (
    <section className="q-form virtual-form">
      <p className="process-heading small">Bragg vector map peaks</p>
      {PEAK_FIELDS.map(({ key: k, label, title, min, max, step }) => (
        <label key={k} title={title}>
          <span>{label}</span>
          <input
            type="number"
            min={min}
            max={max}
            step={step}
            value={opts[k]}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && e.target.value !== "") setOptions(d.id, { [k]: Math.min(max, Math.max(min, v)) });
            }}
          />
        </label>
      ))}
      {peaks && (
        <p className="muted small">
          {peaks.list.length} peaks{lattice ? ` · ${on} on the lattice` : ""}{busy ? " · finding…" : ""}
        </p>
      )}
      <p className="process-heading small">Lattice</p>
      {lattice ? (
        <dl className="props">
          <dt>O</dt>
          <dd>({formatValue(lattice.ox)}, {formatValue(lattice.oy)}) px</dd>
          <dt>g1</dt>
          <dd>{length(lattice.g1x, lattice.g1y)} · {formatValue(angle(lattice.g1x, lattice.g1y))}°</dd>
          <dt>g2</dt>
          <dd>{length(lattice.g2x, lattice.g2y)} · {formatValue(angle(lattice.g2x, lattice.g2y))}°</dd>
          <dt>g1 ∠ g2</dt>
          <dd>{formatValue(between)}°</dd>
        </dl>
      ) : (
        <p className="muted small">{busy ? "Finding the peaks…" : "No lattice found: lower the threshold or the spacing"}</p>
      )}
      <label title="Largest distance from a lattice point at which a disk is indexed to it">
        <span>Tolerance (px)</span>
        <input
          type="number"
          min={0.25}
          max={100}
          step={0.25}
          value={tol}
          onChange={(e) => e.target.value !== "" && setTolerance(d.id, Math.min(100, Math.max(0.25, Number(e.target.value))))}
        />
      </label>
      {peaks?.proposed && (
        <div className="process-actions">
          <button
            className="link-button"
            onClick={() => setLattice(d.id, peaks.proposed!)}
            title="O at the direct beam, g1 and g2 the shortest pair of peaks from it"
          >
            Propose again
          </button>
        </div>
      )}
      <p className="process-heading small">Fit</p>
      <label title="Fewest indexed disks a position needs to be fitted (not all on one line)">
        <span>Min disks</span>
        <input
          type="number"
          min={3}
          max={10000}
          step={1}
          value={fit.min_disks}
          onChange={(e) => e.target.value !== "" && setFit(d.id, { min_disks: Math.min(10000, Math.max(3, Math.round(Number(e.target.value)))) })}
        />
      </label>
      <label className="q-check-line small" title="Disks weighted by their intensity (the correlation), as py4DSTEM does">
        <input type="checkbox" checked={fit.weighted} onChange={(e) => setFit(d.id, { weighted: e.target.checked })} />
        Weighted
      </label>
      <label className="q-check-line small" title="The indexed disks and the lattice fitted at each cursor, on its pattern (pick a cursor in the pattern pane)">
        <input type="checkbox" checked={preview} onChange={(e) => setPreview(d.id, e.target.checked)} />
        Show on the patterns
      </label>
      {fitted && (
        <p className="muted small">
          Fitted with tol {formatValue(fitted.strain!.options.tolerance)} px, min {fitted.strain!.options.min_disks}
          {fitted.strain!.frame.ellipse ? ", ellipse" : ""}{fitted.strain!.frame.rotation ? `, QR ${formatValue(fitted.strain!.frame.rotation.theta)}°` : ", no QR rotation"}
        </p>
      )}
      {error && <p className="field-error">{error}</p>}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job!.progress * 100)}>
            <div style={{ width: `${Math.round(job!.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>{job!.message || "Starting…"} {Math.round(job!.progress * 100)}%</span>
            <button className="link-button" onClick={() => void cancelFit(d.id)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="process-actions">
          <button
            className="primary"
            disabled={!lattice}
            onClick={() => void runFit(d.id)}
            title="Index every pattern's disks to the lattice and fit it at every position: Lattice fit, Indexed disks and Fit residual under the Bragg vectors"
          >
            {fitted ? "Fit again" : "Fit every pattern"}
          </button>
        </div>
      )}
      {job?.status === "done" && !running && job.result && (
        <p className="muted small">
          {(job.result as unknown as { fitted: number }).fitted} of {(job.result as unknown as { positions: number }).positions} positions fitted
          · {(job.result as unknown as { indexed: number }).indexed} of {(job.result as unknown as { disks: number }).disks} disks indexed
        </p>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
      {job?.status === "cancelled" && !running && <p className="muted small">Cancelled.</p>}
    </section>
  );
}
