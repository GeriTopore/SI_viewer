import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { isInsituView } from "../format";
import { lowLossOf, useZlp, useZlpOptions } from "../state/zlp";

const ACTIVE = new Set(["queued", "running"]);

/** Process tab, EELS spectrum images and line scans: aligns the zero-loss peak of
 * the low loss (of the shown one, or of its group) to 0 eV. Its linked core
 * losses get the same shifts. The loaded data is overwritten, like a rebin. */
export function ZlpSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const ll = lowLossOf(d, files);
  const options = useZlpOptions(ll);
  const job = useZlp((s) => (ll ? s.jobs[ll.id] : undefined));
  const startError = useZlp((s) => (ll ? s.startError[ll.id] : null));
  const { patch, run, cancel } = useZlp.getState();

  const running = !!job && ACTIVE.has(job.status);
  const locked = !ll || running;
  // A frame of an in-situ series would lose the alignment with the next frame.
  const byId = new Map(allDatasets(files).map((x) => [x.id, x]));
  const inSitu = !!ll && [ll, ...(ll.eels?.linked ?? []).map((id) => byId.get(id))].some((x) => x && isInsituView(x));

  return (
    <section className="q-form">
      {!ll && <p className="muted small q-elements">No low loss in its group</p>}
      <label className="q-check" title="Refine the alignment by cross-correlation around the peak">
        <input
          type="checkbox"
          checked={options.subpixel}
          disabled={locked}
          onChange={(e) => ll && patch(ll, { subpixel: e.target.checked })}
        />
        Sub-pixel
      </label>
      <label className="q-check" title="Remove the channels the shifts leave empty">
        <input
          type="checkbox"
          checked={options.crop}
          disabled={locked}
          onChange={(e) => ll && patch(ll, { crop: e.target.checked })}
        />
        Crop
      </label>
      <div className="rebin-warning" role="note">
        Overwrites the loaded data; the file is unchanged.
      </div>
      {startError && <p className="field-error">{startError}</p>}
      {running ? (
        <div className="q-job">
          <div className="q-progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <div className="q-job-row small">
            <span>
              {job.message || "Starting…"} {Math.round(job.progress * 100)}%
            </span>
            <button className="link-button" onClick={() => ll && cancel(ll.id)}>
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          className="primary"
          disabled={!ll || inSitu}
          onClick={() => ll && run(ll)}
          title={inSitu ? "Not for in-situ frames: the next frame would undo it" : undefined}
        >
          Align
        </button>
      )}
      {job?.status === "cancelled" && !running && <p className="muted small q-elements">Cancelled.</p>}
      {job?.status === "failed" && !running && <p className="field-error">{job.error}</p>}
    </section>
  );
}
