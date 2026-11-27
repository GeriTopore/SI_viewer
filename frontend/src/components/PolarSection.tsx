import type { DatasetSummary } from "../api/client";
import { jobRunning, perAngstrom, polarOptionsOf, useOrientation } from "../state/orientation";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";

const FIELDS: { key: "npt" | "npt_azim" | "r_min" | "r_max" | "gamma"; label: string; title: string; min: number; max: number; step: number; whole?: boolean }[] = [
  { key: "npt", label: "Radial points", title: "Points along the radius", min: 10, max: 1000, step: 10, whole: true },
  { key: "npt_azim", label: "Azimuthal points", title: "Points around the circle", min: 36, max: 3600, step: 36, whole: true },
  { key: "r_min", label: "From (1/Å)", title: "Start of the radial range", min: 0, max: 10, step: 0.05 },
  { key: "r_max", label: "To (1/Å)", title: "End of the radial range; empty: the whole pattern", min: 0.05, max: 10, step: 0.05 },
  { key: "gamma", label: "Gamma", title: "The transform raised to this power; 1: none", min: 0.05, max: 2, step: 0.05 },
];

/** The polar transform of every pattern, as pyxem makes it
 * (``get_azimuthal_integral2d``, its orientation mapping tutorial's
 * defaults): 4D data of radius × azimuth, as a copy or in place. */
export function PolarSection({ d }: { d: DatasetSummary }) {
  const polar = polarOptionsOf(useOrientation((s) => s.polar[d.id]));
  const job = useOrientation((s) => s.jobs[d.id]);
  const error = useOrientation((s) => s.errors[d.id] ?? null);
  const { setPolar, transform, cancel } = useOrientation.getState();
  const running = jobRunning(job);
  const [inPlace, setInPlace] = useInPlace("polar");
  const calibrated = perAngstrom(d) !== null;

  return (
    <section className="q-form virtual-form">
      {FIELDS.map(({ key, label, title, min, max, step, whole }) => (
        <label key={key} title={title}>
          <span>{label}</span>
          <input type="number" min={min} max={max} step={step} value={polar[key] ?? ""}
            placeholder={key === "r_max" ? "whole" : undefined}
            onChange={(e) => {
              if (e.target.value === "") return key === "r_max" && setPolar(d.id, { r_max: null });
              const v = Number(e.target.value);
              if (Number.isFinite(v)) setPolar(d.id, { [key]: Math.min(max, Math.max(min, whole ? Math.round(v) : v)) });
            }} />
        </label>
      ))}
      {d.calibration?.ring && (
        <label className="q-check-line small" title="Make the ring circular using the calibrated ellipse">
          <input type="checkbox" checked={polar.ellipse} onChange={(e) => setPolar(d.id, { ellipse: e.target.checked })} />
          Correct the ellipticity
        </label>
      )}
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={running} what={d.title} />
      <div className="rebin-warning" role="note" hidden={!inPlace}>
        Overwrites the loaded data; the file is unchanged.
      </div>
      {!calibrated && <p className="field-error">Calibrate the pattern's pixel size first (1/Å or 1/nm).</p>}
      {error && <p className="field-error">{error}</p>}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job!.progress * 100)}>
            <div style={{ width: `${Math.round(job!.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>{job!.message || "Starting…"} {Math.round(job!.progress * 100)}%</span>
            <button className="link-button" onClick={() => void cancel(d.id)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="process-actions">
          <button className="primary" disabled={!calibrated} onClick={() => void transform(d.id, polar, inPlace)}
            title="Transform every pattern to radius × azimuth">
            Transform
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
