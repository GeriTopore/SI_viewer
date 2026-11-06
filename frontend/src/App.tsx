import { useEffect, useState } from "react";
import { api, type Health } from "./api/client";
import { DatasetList, DatasetListHeader } from "./components/DatasetList";
import { FileBrowser } from "./components/FileBrowser";
import { Inspector } from "./components/Inspector";
import { OpenFile } from "./components/OpenFile";
import { useDatasets } from "./state/datasets";
import { useUi } from "./state/ui";
import { MainView } from "./views/MainView";

export function App() {
  const { refresh, selectRelative, errors, clearErrors } = useDatasets();
  const browserOpen = useUi((s) => s.browserOpen);
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
        <MainView />
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
