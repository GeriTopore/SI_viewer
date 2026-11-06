import { useEffect, useState } from "react";
import { api, type Health } from "./api/client";
import { DatasetList, DatasetListHeader } from "./components/DatasetList";
import { FileBrowser } from "./components/FileBrowser";
import { Inspector } from "./components/Inspector";
import { OpenFile } from "./components/OpenFile";
import { useDatasets, useSelectedDataset } from "./state/datasets";
import { useUi } from "./state/ui";
import { kindLabel, shapeLabel } from "./format";

export function App() {
  const { refresh, selectRelative, errors, clearErrors } = useDatasets();
  const browserOpen = useUi((s) => s.browserOpen);
  const selected = useSelectedDataset();
  const [health, setHealth] = useState<Health | null>(null);
  const [backendDown, setBackendDown] = useState(false);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setHealth(h);
        refresh();
      })
      .catch(() => setBackendDown(true));
  }, [refresh]);

  // Up/down arrows switch datasets, unless typing in a field or a dialog is open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (useUi.getState().browserOpen) return;
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (e.key === "ArrowDown") selectRelative(1);
      else if (e.key === "ArrowUp") selectRelative(-1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectRelative]);

  return (
    <div className="app">
      <header className="topbar">
        <span className="app-name">SI Viewer</span>
        <OpenFile />
      </header>

      {errors.length > 0 && (
        <div className="error-banner" role="alert">
          <ul>
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
          <button className="icon-button" onClick={clearErrors} title="Dismiss">
            ×
          </button>
        </div>
      )}

      <aside className="sidebar panel">
        <DatasetListHeader />
        <DatasetList />
      </aside>

      <main className="main panel">
        {selected ? (
          <div className="placeholder">
            <span className={`badge badge-${selected.is_element_map ? "map" : selected.kind}`}>
              {kindLabel(selected)}
            </span>
            <h1>{selected.title}</h1>
            <p className="muted">{shapeLabel(selected)}</p>
            <p className="muted small">The viewer for this dataset arrives in milestone M1.</p>
          </div>
        ) : (
          <div className="placeholder">
            <h1>No dataset selected</h1>
            <p className="muted">Open .emd files or TIFF stacks to start (⌘/Ctrl+O).</p>
          </div>
        )}
      </main>

      <aside className="inspector-panel panel">
        <h2>Inspector</h2>
        <Inspector />
      </aside>

      <footer className="statusbar">
        {backendDown ? (
          <span className="status-dot down">Backend not reachable — is si-viewer running?</span>
        ) : health ? (
          <span className="status-dot up">
            {Object.entries(health.versions)
              .map(([k, v]) => `${k} ${v}`)
              .join(" · ")}
          </span>
        ) : (
          <span className="muted">Connecting…</span>
        )}
      </footer>

      {browserOpen && <FileBrowser />}
    </div>
  );
}
