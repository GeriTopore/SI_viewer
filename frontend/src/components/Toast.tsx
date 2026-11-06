import { useEffect } from "react";
import { useLayers } from "../state/layers";

/** Short-lived message, e.g. why a map can't be added to the view. */
export function Toast() {
  const notice = useLayers((s) => s.notice);
  const notify = useLayers((s) => s.notify);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => notify(null), 4000);
    return () => clearTimeout(t);
  }, [notice, notify]);

  if (!notice) return null;
  return (
    <div className="toast" role="status" onClick={() => notify(null)}>
      {notice}
    </div>
  );
}
