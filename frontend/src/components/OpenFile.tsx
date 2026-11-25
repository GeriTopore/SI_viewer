import { useCallback, useEffect } from "react";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { useUi } from "../state/ui";
import { projectToSave, saveAnalysis, workspaceToSave } from "../state/project";
import { runCommand, usePlugins } from "../plugins/host";
import { ExamplesMenu } from "./ExamplesMenu";

/** "Open…" button (⌘/Ctrl+O), opening and saving an analysis (⇧⌘O, ⇧⌘S),
 * the plugins' commands, plus progress while a batch of files loads. */
export function OpenFile() {
  const progress = useDatasets((s) => s.progress);
  const setBrowserOpen = useUi((s) => s.setBrowserOpen);
  const commands = usePlugins((s) => s.commands);
  // Save analysis saves the active workspace (or the only one).
  const canSave = useDatasets((s) => s.files.length > 0 && (s.active !== null || workspaceToSave() !== null));

  // A plugin's project is saved in its folder; anything else asks where.
  const save = useCallback(() => {
    const project = projectToSave();
    if (!project) return setBrowserOpen(true, "save-analysis");
    saveAnalysis(project.path, project.fileIds, true).catch((e) =>
      useLayers.getState().notify(`Could not save: ${(e as Error).message}`),
    );
  }, [setBrowserOpen]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || useUi.getState().browserOpen) return;
      const key = e.key.toLowerCase();
      if (key === "o") {
        e.preventDefault();
        setBrowserOpen(true, e.shiftKey ? "open-analysis" : "data");
      } else if (key === "s" && e.shiftKey) {
        e.preventDefault();
        if (workspaceToSave()) save();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setBrowserOpen, save]);

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
        onClick={save}
        disabled={!canSave || !!progress}
        title="The workspace shown, with what it links to · ⇧⌘S"
      >
        Save analysis…
      </button>
      {commands.map((c) => (
        <button key={c.id} className="secondary" title={c.title} disabled={!!progress} onClick={() => void runCommand(c)}>
          {c.label}
        </button>
      ))}
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
