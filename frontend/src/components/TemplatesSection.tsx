import type { DatasetSummary, TemplateOptions, ZoneAxis } from "../api/client";
import { jobRunning, templateOptionsOf, useOrientation } from "../state/orientation";

type NumberKey = Exclude<keyof TemplateOptions, "zone_axes">;

const FIELDS: { key: NumberKey; label: string; title: string; min: number; max: number; step: number }[] = [
  { key: "kv", label: "Voltage (kV)", title: "Accelerating voltage (diffsims SimulationGenerator)", min: 1, max: 1000, step: 10 },
  { key: "precession_angle", label: "Precession (°)", title: "Precession angle; 0: none", min: 0, max: 5, step: 0.5 },
  { key: "minimum_intensity", label: "Minimum intensity", title: "Reflections weaker than this are left out", min: 0, max: 1, step: 1e-5 },
  { key: "resolution", label: "Resolution (°)", title: "Angular step of the orientations over the reduced fundamental zone (orix get_sample_reduced_fundamental)", min: 0.1, max: 10, step: 0.5 },
  { key: "reciprocal_radius", label: "Reciprocal radius (1/Å)", title: "Largest reflection simulated", min: 0.05, max: 10, step: 0.1 },
  { key: "max_excitation_error", label: "Excitation error (1/Å)", title: "Largest excitation error s of a reflection (relrod length)", min: 0.0001, max: 1, step: 0.001 },
];

/** A Structure's templates (diffsims) over orix's orientations, optionally
 * only near some zone axes: each run a new Templates object under it. */
export function TemplatesSection({ d }: { d: DatasetSummary }) {
  const changed = useOrientation((s) => s.options[d.id]);
  const o = templateOptionsOf(changed);
  const job = useOrientation((s) => s.jobs[d.id]);
  const error = useOrientation((s) => s.errors[d.id] ?? null);
  const { setOptions, makeTemplates, cancel } = useOrientation.getState();
  const running = jobRunning(job);
  const setZones = (zone_axes: ZoneAxis[]) => setOptions(d.id, { zone_axes });

  return (
    <section className="q-form virtual-form">
      {FIELDS.map(({ key, label, title, min, max, step }) => (
        <label key={key} title={title}>
          <span>{label}</span>
          <input type="number" min={min} max={max} step={step} value={o[key]}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && e.target.value !== "") setOptions(d.id, { [key]: Math.min(max, Math.max(min, v)) });
            }} />
        </label>
      ))}
      <h4 className="process-heading">Near zone axes</h4>
      {o.zone_axes.map((z, k) => (
        <div key={k} className="zone-axis-row">
          <span className="uvw-input">
            {z.uvw.map((v, i) => (
              <input key={i} type="number" step={1} value={v} aria-label={"uvw"[i]}
                onChange={(e) => {
                  if (e.target.value === "") return;
                  const uvw = [...z.uvw];
                  uvw[i] = Math.round(Number(e.target.value));
                  setZones(o.zone_axes.map((x, j) => (j === k ? { ...x, uvw } : x)));
                }} />
            ))}
          </span>
          <input type="number" className="zone-axis-angle" min={0.1} max={90} step={1} value={z.angle} title="Within this angle (°)"
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v > 0) setZones(o.zone_axes.map((x, j) => (j === k ? { ...x, angle: Math.min(90, v) } : x)));
            }} />
          <span className="muted small">°</span>
          <button className="link-button" title="Remove" onClick={() => setZones(o.zone_axes.filter((_, j) => j !== k))}>✕</button>
        </div>
      ))}
      <button className="link-button" onClick={() => setZones([...o.zone_axes, { uvw: [1, 0, 1], angle: 5 }])}
        title="Only the orientations within an angle of a zone axis (none: all of them)">
        + Zone axis
      </button>
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
          <button className="primary" disabled={o.zone_axes.some((z) => z.uvw.every((x) => x === 0))}
            onClick={() => void makeTemplates(d.id, o)}
            title="Simulate the templates (diffsims): a new Templates object under the Structure">
            Make templates
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
