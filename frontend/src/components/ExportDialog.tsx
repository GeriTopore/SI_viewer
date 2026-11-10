import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api, type DatasetSummary, type ExportFormat, type FsEntry, type FsListing } from "../api/client";
import { safeFileName, useExportFormats } from "../download";
import { bytesLabel } from "../format";
import { useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { useUi } from "../state/ui";

/** The formats every one of ``datasets`` can be saved in. */
function commonFormats(datasets: DatasetSummary[], byKind: Record<string, ExportFormat[]>): ExportFormat[] {
  const [first, ...rest] = datasets.map((d) => byKind[d.kind] ?? []);
  return (first ?? []).filter((f) => rest.every((fs) => fs.some((g) => g.format === f.format)));
}

const parentFolder = (path: string) => path.replace(/[/\\][^/\\]*$/, "") || "/";
const join = (folder: string, name: string) => `${folder.replace(/[/\\]+$/, "")}/${name}`;

/**
 * Save one dataset (a file name and type in a folder you browse to) or
 * several (a type; each named after its title) on the backend's disk.
 */
export function ExportDialog({ datasets, onClose }: { datasets: DatasetSummary[]; onClose: () => void }) {
  const batch = datasets.length > 1;
  const byKind = useExportFormats();
  const formats = useMemo(() => commonFormats(datasets, byKind), [datasets, byKind]);
  const files = useDatasets((s) => s.files);
  const { lastExportDir, setLastExportDir } = useUi();
  const [format, setFormat] = useState<string>("");
  const fmt = formats.find((f) => f.format === format) ?? formats[0];
  const [name, setName] = useState(safeFileName(datasets[0]?.title ?? "data"));
  const [listing, setListing] = useState<FsListing | null>(null);
  const [pathInput, setPathInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  const navigate = useCallback(
    async (path?: string): Promise<boolean> => {
      try {
        const l = await api.listDir(path, fmt?.extension);
        setListing(l);
        setPathInput(l.path);
        setError(null);
        return true;
      } catch (e) {
        setError((e as Error).message);
        return false;
      }
    },
    [fmt?.extension],
  );

  // Start in the last export folder, else the folder of the dataset's file.
  const start = useMemo(() => {
    const file = files.find((f) => f.id === datasets[0]?.file_id);
    return lastExportDir || (file && !file.path.startsWith("example:") ? parentFolder(file.path) : "");
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const folder = listing?.path ?? start;
  // Relist when the type changes (the files shown are those of that type).
  useEffect(() => {
    if (!fmt) return;
    navigate(folder || undefined).then((ok) => ok || navigate());
  }, [fmt?.extension]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    nameRef.current?.select();
  }, []);

  const fileName = `${name.replace(new RegExp(`\\.${fmt?.extension}$`), "")}.${fmt?.extension ?? ""}`;

  const save = async (overwrite = false) => {
    if (!fmt || !listing) return;
    setBusy(true);
    setError(null);
    try {
      if (batch) {
        const r = await api.batchExport(datasets.map((d) => d.id), fmt.format, listing.path, overwrite);
        useLayers.getState().notify(`Exported ${r.paths.length} files to ${listing.path}`);
      } else {
        const r = await api.saveDataset(datasets[0].id, fmt.format, join(listing.path, fileName), overwrite);
        useLayers.getState().notify(`Saved ${r.path}`);
      }
      setLastExportDir(listing.path);
      onClose();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setConflict((e as Error).message);
      else setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onEntry = (entry: FsEntry, open: boolean) => {
    if (entry.is_dir) {
      if (open) navigate(entry.path);
    } else if (!batch) {
      setName(entry.name);
      if (open) save();
    }
  };

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
      // The dialog's keys stay in it (not the app's dataset switching).
      onKeyDown={(e) => {
        e.stopPropagation();
        if (e.key === "Escape") onClose();
      }}
    >
      <div className="modal file-browser export-dialog" role="dialog" aria-label="Export">
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
            onKeyDown={(e) => e.key === "Enter" && navigate(pathInput.trim())}
            spellCheck={false}
            aria-label="Folder"
          />
          <button className="icon-button" title="Close" onClick={onClose}>
            ×
          </button>
        </header>

        <ul className="fb-list" tabIndex={0}>
          {listing && listing.entries.length === 0 && <li className="muted fb-empty">Empty folder</li>}
          {listing?.entries.map((entry) => (
            <li
              key={entry.path}
              className={`fb-entry${!entry.is_dir && entry.name === fileName ? " selected" : ""}`}
              onClick={() => onEntry(entry, false)}
              onDoubleClick={() => onEntry(entry, true)}
            >
              <span className={`fb-icon ${entry.is_dir ? "dir" : "file"}`} aria-hidden />
              <span className="fb-name">{entry.name}</span>
              <span className="fb-size">{entry.size !== null ? bytesLabel(entry.size) : ""}</span>
              <span className="fb-date">{new Date(entry.modified * 1000).toLocaleDateString()}</span>
            </li>
          ))}
        </ul>

        {(error || conflict) && (
          <div className="fb-error">
            {conflict ? (
              <>
                {conflict}.{" "}
                <button className="link-button" onClick={() => (setConflict(null), save(true))} disabled={busy}>
                  Replace
                </button>
              </>
            ) : (
              error
            )}
          </div>
        )}

        <footer className="fb-footer export-footer">
          <div className="export-fields">
            {batch ? (
              <span className="muted">{datasets.length} files, named after their titles</span>
            ) : (
              <input
                ref={nameRef}
                className="fb-path export-name"
                value={name}
                onChange={(e) => (setName(e.target.value), setConflict(null))}
                onKeyDown={(e) => e.key === "Enter" && save()}
                spellCheck={false}
                aria-label="File name"
              />
            )}
            <select
              value={fmt?.format ?? ""}
              onChange={(e) => (setFormat(e.target.value), setConflict(null))}
              aria-label="File type"
              disabled={!formats.length}
            >
              {formats.map((f) => (
                <option key={f.format} value={f.format}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
          <div className="fb-actions">
            <button className="secondary" onClick={onClose}>
              Cancel
            </button>
            <button
              className="primary"
              disabled={busy || !fmt || !listing || (!batch && !name.trim())}
              onClick={() => save()}
            >
              {busy ? "Exporting…" : "Export"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
