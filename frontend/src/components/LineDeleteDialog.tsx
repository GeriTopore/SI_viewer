import { useEffect } from "react";
import type { DatasetSummary } from "../api/client";
import { allDatasets, useDatasets } from "../state/datasets";
import { getViewProfiles, useProfiles } from "../state/profiles";

/** The line scans generated along line ``id`` of ``view`` that still exist. */
function lineScansOf(view: string, id: string): DatasetSummary[] {
  const ids = getViewProfiles(view).lines.find((l) => l.id === id)?.lineScans ?? [];
  const byId = new Map(allDatasets(useDatasets.getState().files).map((d) => [d.id, d]));
  return ids.flatMap((i) => byId.get(i) ?? []);
}

/** The user deletes a profile line: at once, or after confirming when line
 * scans were generated along it (they are deleted with it). */
export function askDeleteLine(view: string, id: string) {
  const { deleteLine, setConfirming } = useProfiles.getState();
  if (getViewProfiles(view).lines.find((l) => l.id === id)?.fixed) return;
  if (lineScansOf(view, id).length) setConfirming({ view, id });
  else deleteLine(view, id);
}

/** Confirms deleting a profile line together with its line scans. */
export function LineDeleteDialog() {
  const confirming = useProfiles((s) => s.confirming);
  useDatasets((s) => s.files); // the list follows the datasets
  const { setConfirming, deleteLine } = useProfiles.getState();
  const close = () => setConfirming(null);
  useEffect(() => {
    if (!confirming) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setConfirming(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirming, setConfirming]);
  if (!confirming) return null;
  const { view, id } = confirming;
  const line = getViewProfiles(view).lines.find((l) => l.id === id);
  if (!line) return null;
  const scans = lineScansOf(view, id);
  if (line.fixed) return null;
  const confirm = async () => {
    close();
    const { deleteDataset } = useDatasets.getState();
    for (const d of scans) await deleteDataset(d);
    deleteLine(view, id);
  };
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal confirm-dialog" role="alertdialog" aria-label={`Delete ${line.name}`}>
        <h2>Delete {line.name}?</h2>
        <div className="rebin-warning" role="note">
          {scans.length > 1 ? "These LineScans are deleted too:" : "This LineScan is deleted too:"}
          <ul>
            {scans.map((d) => (
              <li key={d.id}>{d.title}</li>
            ))}
          </ul>
        </div>
        <div className="rebin-actions">
          <button className="secondary" onClick={close} autoFocus>
            Cancel
          </button>
          <button className="primary danger" onClick={confirm}>
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}
