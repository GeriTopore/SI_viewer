import { useState } from "react";
import type { DatasetSummary } from "../api/client";
import { setQrRotation } from "../state/calibration";
import { allDatasets, useDatasets } from "../state/datasets";
import { comJobRunning, comOptionsOf, useCom } from "../state/com";
import { formatValue } from "../format";
import { segmented } from "../views/FourDView";

/** DPC and iCOM from 4D-STEM data: every pattern's centre of mass (around
 * the bright-field disk, above a threshold), less the descan plane, rotated
 * into the scan's frame (the rotation leaving the least curl, or as given):
 * "DPC (COM)" (its direction in a circular colour map, magnitude as
 * brightness; COM x and y under it) and "iCOM" (the integral), under the data. */
export function ComSection({ d }: { d: DatasetSummary }) {
  const opts = useCom((s) => comOptionsOf(s, d.id));
  const result = useCom((s) => s.results[d.id]);
  const job = useCom((s) => s.jobs[d.id]);
  const error = useCom((s) => s.errors[d.id] ?? null);
  const { setOptions, run, cancel } = useCom.getState();
  const running = comJobRunning(job);
  const files = useDatasets((s) => s.files);
  const made = allDatasets(files).some((x) => x.origin === "dpc" && x.dpc?.cube === d.id);
  const [keeping, setKeeping] = useState(false);
  const [keepError, setKeepError] = useState<string | null>(null);
  const stored = (allDatasets(files).find((x) => x.id === d.id) ?? d).calibration?.rotation;
  const kept = !!result && !!stored && Math.abs(stored.theta - (((result.theta % 360) + 360) % 360)) < 1e-6 && stored.mirror === result.mirror;

  const keep = async () => {
    if (!result) return;
    const found = Math.abs(result.theta - result.found.theta) < 1e-9 && result.mirror === result.found.mirror;
    setKeeping(true);
    setKeepError(null);
    try {
      await setQrRotation(d.id, {
        theta: result.theta,
        mirror: result.mirror,
        how: found ? "com" : "entered",
        ...(found ? { curl_ratio: result.curl_ratio } : {}),
      });
    } catch (e) {
      setKeepError((e as Error).message);
    } finally {
      setKeeping(false);
    }
  };

  return (
    <section className="q-form virtual-form">
      <p className="process-heading small">Centre of mass</p>
      <label title="Pixels below this × the pattern's mean intensity are left out (0: all)">
        <span>Threshold (× mean)</span>
        <input
          type="number"
          min={0}
          max={1000}
          step={0.5}
          value={opts.beam.threshold}
          onChange={(e) => e.target.value !== "" && setOptions(d.id, { beam: { threshold: Math.max(0, Number(e.target.value)) } })}
        />
      </label>
      <label title="Half the side of the square around the bright-field disk of the mean pattern it is found in; empty: 3 × its radius">
        <span>Search (px)</span>
        <input
          type="number"
          min={2}
          step={1}
          placeholder="3 × BF"
          value={opts.beam.half_width ?? ""}
          onChange={(e) => setOptions(d.id, { beam: { half_width: e.target.value === "" ? null : Math.max(2, Math.round(Number(e.target.value))) } })}
        />
      </label>
      <p className="process-heading small">Descan</p>
      <div className="virtual-row">
        {segmented("Descan", [
          { id: "none", label: "None", title: "Only the mean deflection removed" },
          { id: "plane", label: "Plane", title: "A plane fitted over the whole scan removed (it also takes the specimen's own linear part)" },
          { id: "corners", label: "Corners", title: "A plane fitted to the scan's corners removed (where there is no specimen, e.g. vacuum)" },
        ], opts.plane.mode, (v) => setOptions(d.id, { plane: { mode: v } }))}
      </div>
      {opts.plane.mode === "corners" && (
        <label title="The corners' size: this fraction of each side">
          <span>Corners</span>
          <input
            type="number"
            min={0.02}
            max={0.5}
            step={0.05}
            value={opts.plane.corners}
            onChange={(e) => e.target.value !== "" && setOptions(d.id, { plane: { corners: Math.min(0.5, Math.max(0.02, Number(e.target.value))) } })}
          />
        </label>
      )}
      <p className="process-heading small">Rotation</p>
      <label className="q-check-line small" title="The detector's rotation (and mirror) from the scan: the one leaving the least curl">
        <input type="checkbox" checked={opts.rotation.auto} onChange={(e) => setOptions(d.id, { rotation: { auto: e.target.checked } })} />
        Find (least curl)
      </label>
      {!opts.rotation.auto && (
        <>
          <label title="Rotation of the pattern's axes into the scan's (degrees, counter-clockwise)">
            <span>θ (°)</span>
            <input
              type="number"
              min={-360}
              max={360}
              step={0.5}
              value={opts.rotation.theta}
              onChange={(e) => e.target.value !== "" && setOptions(d.id, { rotation: { theta: Number(e.target.value) } })}
            />
          </label>
          <label className="q-check-line small" title="The pattern's y axis flipped first">
            <input type="checkbox" checked={opts.rotation.mirror} onChange={(e) => setOptions(d.id, { rotation: { mirror: e.target.checked } })} />
            Mirror
          </label>
        </>
      )}
      {result && (
        <p className="small">
          θ {formatValue(result.theta)}°{result.mirror ? ", mirrored" : ""}
          <span className="muted">
            {" "}· least curl at {formatValue(result.found.theta)}°{result.found.mirror ? " mirrored" : ""} (curl left {formatValue(result.curl_ratio * 100)} %)
          </span>
        </p>
      )}
      {result && (
        <div className="process-actions">
          {kept ? (
            <span className="muted small">QR rotation in the calibration</span>
          ) : (
            <button
              className="link-button"
              disabled={keeping}
              onClick={() => void keep()}
              title="Keep θ and the mirror as the data's QR rotation (Info tab): the Bragg vectors turned into the scan's frame"
            >
              Set as QR rotation
            </button>
          )}
        </div>
      )}
      {keepError && <p className="field-error">{keepError}</p>}
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
          <button className="primary" onClick={() => void run(d.id)} title="Find every pattern's centre of mass; make DPC (COM) and iCOM under the data">
            {made ? "Make again" : "Make DPC"}
          </button>
        </div>
      )}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
      {job?.status === "cancelled" && !running && <p className="muted small">Cancelled.</p>}
    </section>
  );
}
