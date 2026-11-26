import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { api, ApiError, type FsEntry, type FsListing, type MissingRaw, type SaveLevel } from "../api/client";
import { openAnalysis, saveAnalysis, workspaceToSave } from "../state/project";
import { useDatasets } from "../state/datasets";
import { useUi } from "../state/ui";
import { bytesLabel } from "../format";

/** Modal for picking one or more files (or a whole folder) on the backend's
 * disk; or an analysis to open, or where to save one (``browserMode``). */
export function FileBrowser() {
  const { lastDir, setLastDir, setBrowserOpen, browserMode: mode } = useUi();
  // Save analysis: the active workspace (with what it links to).
  const [toSave] = useState(() => (mode === "save-analysis" ? workspaceToSave() : null));
  const [name, setName] = useState(() => `${toSave?.name ?? "analysis"} analysis.hspy`);
  const [replace, setReplace] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [level, setLevel] = useState<SaveLevel>("link");
  // Opening an analysis whose raw files moved: locating them one by one.
  const [locate, setLocate] = useState<{
    project: string;
    missing: MissingRaw[];
    locations: Record<string, string>;
  } | null>(null);
  const [sizes, setSizes] = useState<(Record<SaveLevel, number> & { file_ids: string[] }) | null>(null);
  useEffect(() => {
    if (toSave) api.projectEstimate(toSave.fileIds).then(setSizes).catch((e) => setError((e as Error).message));
  }, [toSave]);
  const { files: openFiles, openFiles: open, openFolder } = useDatasets();
  const [listing, setListing] = useState<FsListing | null>(null);
  const [pathInput, setPathInput] = useState(lastDir);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const close = useCallback(() => setBrowserOpen(false), [setBrowserOpen]);

  const navigate = useCallback(
    async (path?: string): Promise<boolean> => {
      try {
        const l = await api.listDir(path);
        setListing(l);
        setPathInput(l.path);
        setSelected(new Set());
        setAnchor(null);
        setError(null);
        setLastDir(l.path);
        listRef.current?.focus();
        return true;
      } catch (e) {
        setError((e as Error).message);
        return false;
      }
    },
    [setLastDir],
  );

  // On open: the last folder used, or the home folder if that has gone.
  useEffect(() => {
    navigate(lastDir || undefined).then((ok) => ok || navigate());
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const entries = listing?.entries ?? [];
  const dataFiles = useMemo(() => entries.filter((e) => !e.is_dir), [entries]);
  const openPaths = useMemo(() => new Set(openFiles.map((f) => f.path)), [openFiles]);

  const openPathsAndClose = (paths: string[]) => {
    if (!paths.length) return;
    if (mode === "open-analysis") {
      if (locate) return void startOpen(locate.project, { ...locate.locations, [locate.missing[0].id]: paths[0] });
      return void startOpen(paths[0]);
    }
    if (mode === "save-analysis") return void save(paths[0]);
    close();
    open(paths);
  };

  /** Run an analysis open or save, closing on success. */
  const runAnalysis = async (job: () => Promise<unknown>, label: string) => {
    setBusy(label);
    setError(null);
    try {
      await job();
      close();
    } catch (e) {
      if (mode === "open-analysis") useDatasets.getState().refresh();
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  /** Open ``project``, asking for the raw files that can't be found first. */
  const startOpen = async (project: string, locations: Record<string, string> = {}) => {
    setBusy("Opening…");
    setError(null);
    let missing: MissingRaw[];
    try {
      missing = (await api.checkProject(project, locations)).missing;
    } catch (e) {
      setError((e as Error).message);
      setBusy(null);
      return;
    }
    setBusy(null);
    if (missing.length) {
      setLocate({ project, missing, locations });
      return;
    }
    setLocate(null);
    await runAnalysis(() => openAnalysis(project, locations), "Opening…");
  };

  const join = (dir: string, file: string) => (file.startsWith("/") ? file : `${dir.replace(/\/$/, "")}/${file}`);

  const save = async (path: string, overwrite = false) => {
    setReplace(null);
    setBusy("Saving…");
    setError(null);
    try {
      if (!sizes) return;
      await saveAnalysis(path, sizes.file_ids, overwrite, level);
      close();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && !overwrite && e.message.endsWith("already exists")) setReplace(path);
      else setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const onClick = (e: MouseEvent, index: number, entry: FsEntry) => {
    if (entry.is_dir) {
      setSelected(new Set());
      setAnchor(null);
      return;
    }
    const next = new Set(e.metaKey || e.ctrlKey ? selected : []);
    if (e.shiftKey && anchor !== null) {
      const [a, b] = [Math.min(anchor, index), Math.max(anchor, index)];
      entries.slice(a, b + 1).forEach((x) => !x.is_dir && next.add(x.path));
    } else if ((e.metaKey || e.ctrlKey) && next.has(entry.path)) {
      next.delete(entry.path);
    } else {
      next.add(entry.path);
    }
    if (!e.shiftKey) setAnchor(index);
    if (mode !== "data") {
      setSelected(new Set([entry.path]));
      if (mode === "save-analysis") setName(entry.name);
      return;
    }
    setSelected(next);
  };

  const onDoubleClick = (entry: FsEntry) => {
    if (entry.is_dir) navigate(entry.path);
    else openPathsAndClose([entry.path]);
  };

  // Typing a file path opens it; a directory path navigates there.
  const submitPath = async () => {
    const p = pathInput.trim();
    if (!p) return;
    try {
      await api.listDir(p);
      navigate(p);
    } catch {
      openPathsAndClose([p]);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  const selectedPaths = entries.filter((e) => selected.has(e.path)).map((e) => e.path);

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div
        className="modal file-browser"
        role="dialog"
        aria-label={mode === "save-analysis" ? "Save analysis" : mode === "open-analysis" ? "Open analysis" : "Open files"}
      >
        <header className="fb-header">
          <button
            className="icon-button"
            title="Up one folder"
            disabled={!listing?.parent}
            onClick={() => listing?.parent && navigate(listing.parent)}
          >
            ↑
          </button>
          <input
            className="fb-path"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submitPath()}
            spellCheck={false}
            aria-label="Folder or file path"
          />
          <button className="icon-button" title="Close" onClick={close}>
            ×
          </button>
        </header>

        {error && <div className="fb-error">{error}</div>}

        <ul
          className="fb-list"
          ref={listRef}
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && openPathsAndClose(selectedPaths)}
        >
          {entries.length === 0 && listing && (
            <li className="muted fb-empty">No folders or supported files here.</li>
          )}
          {entries.map((entry, i) => (
            <li
              key={entry.path}
              className={`fb-entry${selected.has(entry.path) ? " selected" : ""}`}
              onClick={(e) => onClick(e, i, entry)}
              onDoubleClick={() => onDoubleClick(entry)}
            >
              <span className={`fb-icon ${entry.is_dir ? "dir" : "file"}`} aria-hidden />
              <span className="fb-name">{entry.name}</span>
              {openPaths.has(entry.path) && <span className="fb-open-tag">open</span>}
              <span className="fb-size">{entry.size !== null ? bytesLabel(entry.size) : ""}</span>
              <span className="fb-date">{new Date(entry.modified * 1000).toLocaleDateString()}</span>
            </li>
          ))}
        </ul>

        {mode !== "data" ? (
          <footer className="fb-footer">
            {mode === "save-analysis" ? (
              <input
                className="fb-name-input"
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setReplace(null);
                }}
                onKeyDown={(e) => e.key === "Enter" && listing && name.trim() && save(join(listing.path, name.trim()))}
                spellCheck={false}
                aria-label="File name"
              />
            ) : locate ? (
              <span className="small fb-locate" title={locate.missing[0].path}>
                Locate <strong>{locate.missing[0].name}</strong>
                {locate.missing.length > 1 && <span className="muted"> · {locate.missing.length} not found</span>}
              </span>
            ) : (
              <span className="muted small">{busy ?? "A saved analysis (.hspy)"}</span>
            )}
            <div className="fb-actions">
              {mode === "save-analysis" && !replace && (
                <div className="segmented" role="group" aria-label="What to save">
                  {(
                    [
                      ["link", "Link raw data", "The raw files are referenced: open it where they are"],
                      ["results", "Include results", "Every dataset's data is stored: opens without the raw files"],
                      ["raw", "Include raw data", "The raw files are embedded too (in-situ series: their results)"],
                    ] as const
                  ).map(([value, label, title]) => (
                    <button
                      key={value}
                      className={level === value ? "active" : undefined}
                      aria-pressed={level === value}
                      disabled={!!busy}
                      onClick={() => setLevel(value)}
                      title={sizes ? `${title} · up to ${bytesLabel(sizes[value])}` : title}
                    >
                      {label}
                      {sizes && <span className="muted"> {bytesLabel(sizes[value])}</span>}
                    </button>
                  ))}
                </div>
              )}
              {replace ? (
                <>
                  <span className="small">{replace.split("/").pop()} exists</span>
                  <button className="secondary" onClick={() => setReplace(null)}>
                    Cancel
                  </button>
                  <button className="primary" onClick={() => save(replace, true)}>
                    Replace
                  </button>
                </>
              ) : mode === "save-analysis" ? (
                <button
                  className="primary"
                  disabled={!listing || !name.trim() || !!busy || !sizes}
                  onClick={() => listing && save(join(listing.path, name.trim()))}
                >
                  {busy ?? "Save"}
                </button>
              ) : (
                <>
                  {locate && (
                    <button className="secondary" onClick={() => setLocate(null)} disabled={!!busy}>
                      Cancel
                    </button>
                  )}
                  <button
                    className="primary"
                    disabled={selectedPaths.length !== 1 || !!busy}
                    onClick={() => openPathsAndClose(selectedPaths)}
                  >
                    {busy ?? (locate ? "Use" : "Open")}
                  </button>
                </>
              )}
            </div>
          </footer>
        ) : (
        <footer className="fb-footer">
          <span className="muted small">
            .emd, .tif/.tiff, .hspy, .dm3/.dm4, .bcf, .h5 (py4DSTEM), .mib, .blo · ⌘/Ctrl- or Shift-click to select several
          </span>
          <div className="fb-actions">
            <button
              className="secondary"
              disabled={!dataFiles.length || !listing}
              title="Open every file here as one experiment"
              onClick={() => {
                if (!listing) return;
                close();
                openFolder(listing.path);
              }}
            >
              Open all in folder
            </button>
            <button
              className="primary"
              disabled={!selectedPaths.length}
              onClick={() => openPathsAndClose(selectedPaths)}
            >
              Open{selectedPaths.length > 1 ? ` ${selectedPaths.length} files` : ""}
            </button>
          </div>
        </footer>
        )}
      </div>
    </div>
  );
}
