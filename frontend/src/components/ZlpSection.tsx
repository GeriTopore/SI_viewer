import { useEffect, useState } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { isInsituView } from "../format";
import { lowLossOf, useZlp, useZlpOptions } from "../state/zlp";
import { InPlaceChoice, useInPlace } from "./InPlaceChoice";
import { spectralAxis } from "../views/geometry";

const ACTIVE = new Set(["queued", "running"]);

/** Process tab, EELS spectrum images and line scans: aligns the zero-loss peak of
 * the low loss (of the shown one, or of its group) to 0 eV. Its linked core
 * losses get the same shifts. In place, the loaded data is overwritten (like
 * a rebin); as a new dataset, the aligned ones are added under them. Without
 * a low loss, the energy axis can be shifted by hand. */
export function ZlpSection({ d }: { d: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const ll = lowLossOf(d, files);
  if (!ll) return <ManualShift d={d} />;
  return <ZlpAlign ll={ll} />;
}

/** No low loss to align to: shift the energy axis by hand, drawn live in the
 * spectrum viewer while it is typed. */
function ManualShift({ d }: { d: DatasetSummary }) {
  const [text, setText] = useState("");
  const [inPlace, setInPlace] = useInPlace("zlp");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setPreview, shiftEnergy } = useZlp.getState();
  const units = spectralAxis(d)?.units ?? "";
  const shift = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(shift) && shift !== 0;
  const inSitu = isInsituView(d);

  useEffect(() => {
    setPreview(d.id, valid ? shift : null);
  }, [d.id, valid, shift, setPreview]);
  useEffect(() => () => setPreview(d.id, null), [d.id, setPreview]);

  const apply = async () => {
    setBusy(true);
    setError(null);
    try {
      await shiftEnergy(d, shift, inPlace);
      setText("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="q-form">
      <p className="field-warning">No ZLP available</p>
      <label title="Added to the energy axis: positive moves the spectra up">
        <span>Shift{units && ` (${units})`}</span>
        <input
          type="number"
          step="any"
          value={text}
          disabled={busy}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && valid && !inSitu && apply()}
          aria-label="Energy shift"
        />
      </label>
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={busy} what={d.title} />
      {inPlace && (
        <div className="rebin-warning" role="note">
          Overwrites the loaded data; the file is unchanged.
        </div>
      )}
      {error && <p className="field-error">{error}</p>}
      <button
        className="primary"
        disabled={!valid || busy || inSitu}
        onClick={apply}
        title={inSitu ? "Not for in-situ frames: the next frame would undo it" : undefined}
      >
        {busy ? "Shifting…" : "Apply shift"}
      </button>
    </section>
  );
}

/** The low loss ``ll`` (the one shown, or the one of its group) aligned to 0 eV. */
function ZlpAlign({ ll }: { ll: DatasetSummary }) {
  const files = useDatasets((s) => s.files);
  const options = useZlpOptions(ll);
  const job = useZlp((s) => (ll ? s.jobs[ll.id] : undefined));
  const startError = useZlp((s) => (ll ? s.startError[ll.id] : null));
  const [inPlace, setInPlace] = useInPlace("zlp");
  const { patch, run, cancel } = useZlp.getState();

  const running = !!job && ACTIVE.has(job.status);
  const locked = !ll || running;
  // A frame of an in-situ series would lose the alignment with the next frame.
  const byId = new Map(allDatasets(files).map((x) => [x.id, x]));
  const inSitu = !!ll && [ll, ...(ll.eels?.linked ?? []).map((id) => byId.get(id))].some((x) => x && isInsituView(x));

  return (
    <section className="q-form">
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
      <InPlaceChoice inPlace={inPlace} onChange={setInPlace} disabled={locked} what="the low loss and its core losses" />
      {inPlace && (
        <div className="rebin-warning" role="note">
          Overwrites the loaded data; the file is unchanged.
        </div>
      )}
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
          onClick={() => ll && run(ll, inPlace)}
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
