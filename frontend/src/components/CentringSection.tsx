import { useEffect, useState } from "react";
import { dataVersion, type DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { beamOf, jobRunning, planeOf, useCentring } from "../state/centring";
import { formatValue } from "../format";
import { segmented } from "../views/FourDView";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";

const range = (r: [number, number]) => `${formatValue(r[0])} … ${formatValue(r[1])}`;

/** Centring 4D-STEM data, step by step: (1) find the direct beam in every
 * pattern (its shifts kept as "Beam shift x / y" images, the beam marked on
 * the pattern), (2) choose the descan (each pattern's own, or a plane fitted
 * to the whole scan or its corners: marked too, with how far it is from what
 * was found), (3) shift every pattern so the beam is at its centre. */
export function CentringSection({ d }: { d: DatasetSummary }) {
  const beam = useCentring((s) => beamOf(s, d.id));
  const plane = useCentring((s) => planeOf(s, d.id));
  const job = useCentring((s) => s.jobs[d.id]);
  const error = useCentring((s) => s.errors[d.id] ?? null);
  const { setBeam, setPlane, find, centre, cancel } = useCentring.getState();
  const running = jobRunning(job);
  const [inPlace, setInPlace] = useInPlace("centre");
  const files = useDatasets((s) => s.files);
  const shifts = allDatasets(files).filter((x) => x.origin === "beam-shift" && x.parent_id === d.id);
  const found = shifts.find((x) => x.beam)?.beam ?? null;
  // Found on this data as it is now: not before a rebin (its shift images are
  // the scan's size) or before it was centred or changed (its revision).
  const current = !!found && shifts.every((x) => x.signal_shape.join() === d.navigation_shape.join())
    && (found.revision === undefined || found.revision === d.revision);

  // How far the descan is from the beam found: the RMS over the scan.
  const [residual, setResidual] = useState<number | null>(null);
  const key = current ? `${shifts.map((x) => `${x.id}@${dataVersion(x)}`).join()}|${plane.mode}|${plane.corners}` : "";
  useEffect(() => {
    if (!key || plane.mode === "none") return setResidual(null);
    let stale = false;
    fetch(`/api/datasets/${d.id}/4d/beam?x=0&y=0&mode=${plane.mode}&corners=${plane.corners}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => !stale && setResidual(b ? b.residual : null))
      .catch(() => undefined);
    return () => {
      stale = true;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="q-form virtual-form">
      <h4 className="process-heading">Step 1 · Find the beam</h4>
      <div className="virtual-row">
        {segmented("Method", [
          { id: "center_of_mass", label: "Centre of mass", title: "The centre of mass of the bright-field disk (sub-pixel)" },
          { id: "blur", label: "Blur", title: "The brightest pixel after a Gaussian blur (whole pixels)" },
        ], beam.method, (v) => setBeam(d.id, { method: v }))}
      </div>
      {beam.method === "center_of_mass" ? (
        <label title="Ignore intensities below this × the mean">
          <span>Threshold (× mean)</span>
          <input type="number" min={0} step={0.5} value={beam.threshold}
            onChange={(e) => e.target.value !== "" && setBeam(d.id, { threshold: Math.max(0, Number(e.target.value)) })} />
        </label>
      ) : (
        <label title="The Gaussian blur's sigma">
          <span>Sigma (px)</span>
          <input type="number" min={0.5} step={0.5} value={beam.sigma}
            onChange={(e) => e.target.value !== "" && setBeam(d.id, { sigma: Math.max(0.5, Number(e.target.value)) })} />
        </label>
      )}
      <label title="Half the side of the search square; empty: 3 × the disk radius">
        <span>Search (± px)</span>
        <input type="number" min={2} step={1} placeholder="3 r" value={beam.half_width ?? ""}
          onChange={(e) => setBeam(d.id, { half_width: e.target.value === "" ? null : Math.max(2, Math.round(Number(e.target.value))) })} />
      </label>
      <div className="process-actions">
        <button className={found ? "secondary" : "primary"} disabled={running} onClick={() => void find(d.id)}
          title="Find the beam in every pattern">
          {found ? "Find again" : "Find the beam"}
        </button>
      </div>
      {found && (
        <p className="muted small">
          {current ? `Beam moves x ${range(found.range.x)}, y ${range(found.range.y)} px` : "Found before the data changed: find it again"}
        </p>
      )}

      <h4 className="process-heading">Step 2 · Descan</h4>
      <div className="virtual-row">
        {segmented("Descan", [
          { id: "none", label: "Each pattern", title: "Every pattern centred on its own beam" },
          { id: "plane", label: "Plane", title: "A linear plane fitted to the beam over the whole scan" },
          { id: "corners", label: "Corners", title: "A plane fitted to the scan's corners only" },
        ], plane.mode, (v) => setPlane(d.id, { mode: v }))}
      </div>
      {plane.mode === "corners" && (
        <label title="The part of each side of the scan taken as its corners">
          <span>Corners (fraction)</span>
          <input type="number" min={0.02} max={0.5} step={0.05} value={plane.corners}
            onChange={(e) => e.target.value !== "" && setPlane(d.id, { corners: Math.min(0.5, Math.max(0.02, Number(e.target.value))) })} />
        </label>
      )}
      {residual !== null && <p className="muted small">Plane − beam found: {formatValue(residual)} px RMS</p>}

      <h4 className="process-heading">Step 3 · Centre</h4>
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={running} what={d.title} />
      <div className="rebin-warning" role="note" hidden={!inPlace}>
        Overwrites the loaded data; the file is unchanged.
      </div>
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
          <button className="primary" disabled={!current} onClick={() => void centre(d.id, inPlace)}
            title="Shift every pattern so the beam is at its centre">
            Centre the patterns
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
