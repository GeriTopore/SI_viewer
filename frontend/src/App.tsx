import { useEffect, useRef, useState } from "react";
import { FocusLayer } from "./components/FocusLayer";
import { api, type Health } from "./api/client";
import { useUndo } from "./state/undo";
import { DatasetList, DatasetListHeader } from "./components/DatasetList";
import { DropZone } from "./components/DropZone";
import { FileBrowser } from "./components/FileBrowser";
import { CollapsedStrip } from "./components/PaneToggle";
import { RightPanel } from "./components/RightPanel";
import { OpenFile } from "./components/OpenFile";
import { ThemeToggle } from "./components/ThemeToggle";
import { Toast } from "./components/Toast";
import { useDatasets, useFocusedDataset } from "./state/datasets";
import { PANE_SIZES, useUi } from "./state/ui";
import { Splitter } from "./components/Splitter";
import { ShortcutsHelp } from "./components/ShortcutsHelp";
import { MainView } from "./views/MainView";
import { isTyping, useShortcuts } from "./keyboard";
import { LineDeleteDialog } from "./components/LineDeleteDialog";

// Width of a folded side pane.
const COLLAPSED = 34;

export function App() {
  const { refresh, selectRelative, errors, clearErrors } = useDatasets();
  const browserOpen = useUi((s) => s.browserOpen);
  const leftCollapsed = useUi((s) => s.leftCollapsed);
  const rightCollapsed = useUi((s) => s.rightCollapsed);
  const leftWidth = useUi((s) => s.leftWidth);
  const rightWidth = useUi((s) => s.rightWidth);
  const setPaneWidth = useUi((s) => s.setPaneWidth);
  const sideRef = useRef<HTMLElement>(null);
  const setCollapsed = useUi((s) => s.setCollapsed);
  useShortcuts({
    "[": () => setCollapsed("left", !leftCollapsed),
    "]": () => setCollapsed("right", !rightCollapsed),
  });
  const inspRef = useRef<HTMLElement>(null);
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

  // The window title names the dataset in focus.
  const focused = useFocusedDataset();
  useEffect(() => {
    document.title = focused ? `${focused.title} — SI Viewer` : "SI Viewer";
  }, [focused?.title]); // eslint-disable-line react-hooks/exhaustive-deps

  // ⌘/Ctrl Z undoes the last deletion (not while typing, nor in a dialog).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || !(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey || e.key.toLowerCase() !== "z") return;
      if (isTyping(e) || document.querySelector(".modal-backdrop")) return;
      e.preventDefault();
      useUndo.getState().undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
    <div
      className="app"
      style={{
        gridTemplateColumns: `${leftCollapsed ? COLLAPSED : leftWidth}px minmax(0, 1fr) ${rightCollapsed ? COLLAPSED : rightWidth}px`,
      }}
    >
      <header className="topbar">
        <span className="app-name">SI Viewer</span>
        <OpenFile />
        <span className="topbar-spacer" />
        <ShortcutsHelp />
        <ThemeToggle />
      </header>
      <LineDeleteDialog />

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

      <aside ref={sideRef} className={`sidebar panel${leftCollapsed ? " collapsed" : ""}`}>
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

      {!leftCollapsed && (
        <Splitter
          direction="col"
          className="pane-resizer left"
          label="Resize the datasets pane"
          onDrag={(x) => setPaneWidth("left", x - sideRef.current!.getBoundingClientRect().left)}
          onReset={() => setPaneWidth("left", PANE_SIZES.left.initial)}
        />
      )}
      {!rightCollapsed && (
        <Splitter
          direction="col"
          className="pane-resizer right"
          label="Resize the right-hand pane"
          onDrag={(x) => setPaneWidth("right", inspRef.current!.getBoundingClientRect().right - x)}
          onReset={() => setPaneWidth("right", PANE_SIZES.right.initial)}
        />
      )}

      <aside ref={inspRef} className={`inspector-panel panel${rightCollapsed ? " collapsed" : ""}`}>
        {rightCollapsed ? <CollapsedStrip side="right" label="FFT · Info · Quantify · Process" /> : <RightPanel />}
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
      <DropZone />
      <FocusLayer />
      <Toast />
    </div>
  );
}
