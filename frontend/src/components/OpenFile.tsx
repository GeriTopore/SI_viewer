import { useEffect } from "react";
import { useDatasets } from "../state/datasets";
import { useUi } from "../state/ui";

/** "Open…" button (⌘/Ctrl+O) plus progress while a batch of files loads. */
export function OpenFile() {
  const progress = useDatasets((s) => s.progress);
  const setBrowserOpen = useUi((s) => s.setBrowserOpen);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "o") {
        e.preventDefault();
        setBrowserOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setBrowserOpen]);

  return (
    <div className="open-file">
      <button className="primary" onClick={() => setBrowserOpen(true)} disabled={!!progress}>
        Open…
      </button>
      {progress && (
        <span className="open-progress">
          <span className="spinner" aria-hidden />
          Opening {progress.total > 1 && `${progress.done + 1} of ${progress.total}: `}
          {progress.current}
        </span>
      )}
    </div>
  );
}
