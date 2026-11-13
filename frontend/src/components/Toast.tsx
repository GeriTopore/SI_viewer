import { useEffect } from "react";
import { useLayers } from "../state/layers";

/** Short-lived message, e.g. why a map can't be added to the view; above it
 * a background task's progress bar while it runs. */
export function Toast() {
  const notice = useLayers((s) => s.notice);
  const notify = useLayers((s) => s.notify);
  const progress = useLayers((s) => s.progress);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => notify(null), 4000);
    return () => clearTimeout(t);
  }, [notice, notify]);

  if (!notice && !progress) return null;
  return (
    <div className="toasts">
      {progress && (
        <div className="toast toast-progress" role="progressbar" aria-valuenow={Math.round(progress.fraction * 100)}>
          <span>{progress.label}</span>
          <div className="toast-bar">
            <div style={{ width: `${Math.round(progress.fraction * 100)}%` }} />
          </div>
        </div>
      )}
      {notice && (
        <div className="toast" role="status" onClick={() => notify(null)}>
          {notice}
        </div>
      )}
    </div>
  );
}
