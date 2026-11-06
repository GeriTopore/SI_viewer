import { useState } from "react";
import { thumbnailUrl, type DatasetSummary } from "../api/client";
import { useDatasets } from "../state/datasets";
import { useUi } from "../state/ui";
import { kindLabel, shapeLabel } from "../format";

const badgeClass = (d: DatasetSummary) => `badge badge-${d.is_element_map ? "map" : d.kind}`;

export function DatasetList() {
  const { files, selectedId, select, closeFile } = useDatasets();
  const listMode = useUi((s) => s.listMode);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  if (!files.length) {
    return <p className="muted empty-list">No files open. Use Open… (⌘/Ctrl+O).</p>;
  }

  return (
    <ul className="file-list">
      {files.map((file) => {
        const isCollapsed = collapsed[file.id];
        return (
          <li key={file.id} className="file">
            <div className="file-header">
              <button
                className="disclosure"
                aria-expanded={!isCollapsed}
                onClick={() => setCollapsed((c) => ({ ...c, [file.id]: !isCollapsed }))}
              >
                {isCollapsed ? "▸" : "▾"}
              </button>
              <span className="file-name" title={file.path}>
                {file.name}
              </span>
              <button className="icon-button" title="Close file" onClick={() => closeFile(file.id)}>
                ×
              </button>
            </div>
            {!isCollapsed &&
              (listMode === "grid" ? (
                <ul className="dataset-grid">
                  {file.datasets.map((d) => (
                    <li key={d.id}>
                      <button
                        className={`dataset-card${d.id === selectedId ? " selected" : ""}`}
                        onClick={() => select(d.id)}
                        title={`${d.title} — ${shapeLabel(d)}`}
                      >
                        <span className={`thumb thumb-${d.kind}`}>
                          <img src={thumbnailUrl(d.id)} alt="" loading="lazy" draggable={false} />
                          <span className={badgeClass(d)}>{kindLabel(d)}</span>
                        </span>
                        <span className="dataset-title">{d.title}</span>
                        <span className="dataset-shape">{shapeLabel(d)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <ul className="dataset-list">
                  {file.datasets.map((d) => (
                    <li key={d.id}>
                      <button
                        className={`dataset${d.id === selectedId ? " selected" : ""}`}
                        onClick={() => select(d.id)}
                      >
                        <span className={badgeClass(d)}>{kindLabel(d)}</span>
                        <span className="dataset-title">{d.title}</span>
                        <span className="dataset-shape">{shapeLabel(d)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              ))}
          </li>
        );
      })}
    </ul>
  );
}

/** Header of the dataset sidebar: count, list/icon toggle, close all. */
export function DatasetListHeader() {
  const { files, closeAll } = useDatasets();
  const { listMode, setListMode } = useUi();
  const count = files.reduce((n, f) => n + f.datasets.length, 0);

  return (
    <div className="sidebar-header">
      <h2>
        Datasets {count > 0 && <span className="count">{count}</span>}
      </h2>
      <div className="segmented" role="group" aria-label="Dataset view">
        <button
          className={listMode === "list" ? "active" : undefined}
          aria-pressed={listMode === "list"}
          title="List view"
          onClick={() => setListMode("list")}
        >
          <ListIcon />
        </button>
        <button
          className={listMode === "grid" ? "active" : undefined}
          aria-pressed={listMode === "grid"}
          title="Large icon view"
          onClick={() => setListMode("grid")}
        >
          <GridIcon />
        </button>
      </div>
      {files.length > 1 && (
        <button className="link-button" onClick={closeAll} title="Close all files">
          Close all
        </button>
      )}
    </div>
  );
}

const ListIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
    <path d="M1 2.5h12M1 7h12M1 11.5h12" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

const GridIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
    <rect x="1" y="1" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="8" y="1" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="1" y="8" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="8" y="8" width="5" height="5" rx="1" fill="currentColor" />
  </svg>
);
