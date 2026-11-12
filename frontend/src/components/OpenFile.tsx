import { useEffect } from "react";
import { useDatasets } from "../state/datasets";
import { useUi } from "../state/ui";
import { workspaceToSave } from "../state/project";
import { ExamplesMenu } from "./ExamplesMenu";

/** "Open…" button (⌘/Ctrl+O), opening and saving an analysis (⇧⌘O, ⇧⌘S),
 * plus progress while a batch of files loads. */
export function OpenFile() {
  const progress = useDatasets((s) => s.progress);
  const setBrowserOpen = useUi((s) => s.setBrowserOpen);
  // Save analysis saves the active workspace (or the only one).
  const canSave = useDatasets((s) => s.files.length > 0 && (s.active !== null || workspaceToSave() !== null));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || useUi.getState().browserOpen) return;
      const key = e.key.toLowerCase();
      if (key === "o") {
        e.preventDefault();
        setBrowserOpen(true, e.shiftKey ? "open-analysis" : "data");
      } else if (key === "s" && e.shiftKey) {
        e.preventDefault();
        if (workspaceToSave()) setBrowserOpen(true, "save-analysis");
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
      <ExamplesMenu />
      <button className="secondary" onClick={() => setBrowserOpen(true, "open-analysis")} disabled={!!progress} title="⇧⌘O">
        Open analysis…
      </button>
      <button
        className="secondary"
        onClick={() => setBrowserOpen(true, "save-analysis")}
        disabled={!canSave || !!progress}
        title="The workspace shown, with what it links to · ⇧⌘S"
      >
        Save analysis…
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
