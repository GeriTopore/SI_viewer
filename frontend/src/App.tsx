import { useEffect, useState } from "react";
import { api, type Health } from "./api/client";
import { DatasetList, DatasetListHeader } from "./components/DatasetList";
import { FileBrowser } from "./components/FileBrowser";
import { CollapsedStrip } from "./components/PaneToggle";
import { RightPanel } from "./components/RightPanel";
import { OpenFile } from "./components/OpenFile";
import { ThemeToggle } from "./components/ThemeToggle";
import { Toast } from "./components/Toast";
import { useDatasets } from "./state/datasets";
import { useUi } from "./state/ui";
import { MainView } from "./views/MainView";
import { isTyping } from "./keyboard";

export function App() {
  const { refresh, selectRelative, errors, clearErrors } = useDatasets();
  const browserOpen = useUi((s) => s.browserOpen);
  const leftCollapsed = useUi((s) => s.leftCollapsed);
  const rightCollapsed = useUi((s) => s.rightCollapsed);
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
      if (e.defaultPrevented || useUi.getState().browserOpen) return;
      if (isTyping(e)) return;
      if (e.key === "ArrowDown") selectRelative(1);
      else if (e.key === "ArrowUp") selectRelative(-1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectRelative]);

  return (
    <div className={`app${leftCollapsed ? " left-collapsed" : ""}${rightCollapsed ? " right-collapsed" : ""}`}>
      <header className="topbar">
        <span className="app-name">SI Viewer</span>
        <OpenFile />
        <span className="topbar-spacer" />
        <ThemeToggle />
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

      <aside className={`sidebar panel${leftCollapsed ? " collapsed" : ""}`}>
        {leftCollapsed ? (
          <CollapsedStrip side="left" label="Datasets" />
        ) : (
          <>
            <DatasetListHeader />
            <DatasetList />
          </>
        )}
      </aside>

      <main className="main panel">
        <MainView />
      </main>

      <aside className={`inspector-panel panel${rightCollapsed ? " collapsed" : ""}`}>
        {rightCollapsed ? <CollapsedStrip side="right" label="Info · Quantify" /> : <RightPanel />}
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
      <Toast />
    </div>
  );
}
