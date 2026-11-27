import { useEffect } from "react";
import { api, type DatasetSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { datasetKey, useQuant } from "../state/quant";

const ACTIVE = new Set(["queued", "running"]);

/** Quantify tab, spectrum images: fit every pixel as a background job,
 * making a fitted map per element (and at% maps). */
export function FitMapsSection({ ds }: { ds: DatasetSummary }) {
  const key = datasetKey(ds);
  const q = useQuant();
  const job = q.jobs[key];
  const running = !!job && ACTIVE.has(job.status);

  // Poll the job while it runs; on completion bring in the new maps.
  useEffect(() => {
    if (!job || !ACTIVE.has(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const next = await api.getJob(job.id);
        useQuant.getState().setJob(key, next);
        if (next.status === "done" && next.result) {
          useDatasets.getState().replaceFile(next.result.file);
          useLayers.getState().notify("Per-pixel maps added to the dataset pane.");
        }
      } catch {
        /* backend busy or restarted: try again */
      }
    }, 300);
    return () => clearInterval(timer);
  }, [job?.id, job?.status, key]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    const lines: Record<string, string> = { ...(q.lineChoice[key] ?? {}) };
    try {
      const started = await api.startFitMaps(ds.id, q.options, {
        composition: q.mapComposition,
        kfactors: q.kfactors,
        lines,
        exclude: q.excluded[key] ?? [],
        min_counts: q.minCounts,
      });
      q.setJob(key, started);
    } catch (e) {
      useLayers.getState().notify(`Could not start: ${(e as Error).message}`);
    }
  };

  return (
    <section className="q-maps">
      <h3>Maps of every pixel</h3>
      <p className="muted small">
        Fits the summed spectrum, then solves every pixel for the line intensities with those line shapes.
        Adds a fitted map per element to the dataset pane.
      </p>
      <label className="q-check-line">
        <input
          type="checkbox"
          checked={q.mapComposition}
          onChange={(e) => q.setMapComposition(e.target.checked)}
          disabled={running}
        />
        Also composition maps (at%) with the k-factors and ticked elements above
      </label>
      {q.mapComposition && (
        <label
          className="q-mask"
          title="Pixels with fewer fitted counts than this are left blank"
        >
          Mask pixels under
          <input
            type="number"
            min={0}
            step={1}
            value={q.minCounts}
            onChange={(e) => q.setMinCounts(Math.max(0, Number(e.target.value) || 0))}
            disabled={running}
          />
          counts
        </label>
      )}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job.message || "Starting…"} {Math.round(job.progress * 100)}%
            </span>
            <button className="link-button" onClick={() => api.cancelJob(job.id).then((j) => q.setJob(key, j))}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button className="secondary" onClick={start} disabled={!ds.eds?.elements.length}>
          {job?.status === "done" ? "Fit every pixel again" : "Fit every pixel"}
        </button>
      )}
      {job?.status === "done" && (
        <p className="muted small">
          Done: fitted maps
          {q.mapComposition ? " and at% maps" : ""} are in the dataset pane.
          {job.result?.skipped?.length ? ` Not fitted: ${job.result.skipped.join(", ")}.` : ""}
        </p>
      )}
      {job?.status === "cancelled" && <p className="muted small">Cancelled.</p>}
      {job?.status === "failed" && <p className="field-error">{job.error}</p>}
    </section>
  );
}
