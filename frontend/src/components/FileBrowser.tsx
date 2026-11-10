import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { api, type FsEntry, type FsListing } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useUi } from "../state/ui";
import { bytesLabel } from "../format";

/** Modal for picking one or more files (or a whole folder) on the backend's disk. */
export function FileBrowser() {
  const { lastDir, setLastDir, setBrowserOpen } = useUi();
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
    close();
    open(paths);
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
      <div className="modal file-browser" role="dialog" aria-label="Open files">
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

        <footer className="fb-footer">
          <span className="muted small">
            .emd, .tif/.tiff, .hspy, .dm3/.dm4, .bcf · ⌘/Ctrl- or Shift-click to select several
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
      </div>
    </div>
  );
}
