import { useState, type DragEvent, type KeyboardEvent, type MouseEvent } from "react";
import { thumbnailUrl, type DatasetSummary } from "../api/client";
import { setMapColour, useDatasets } from "../state/datasets";
import { useDisplay } from "../state/display";
import { useUi } from "../state/ui";
import { isColour, swatchBackground } from "../viewer/colormaps";
import { kindLabel, shapeLabel } from "../format";
import { DATASET_MIME } from "./LayeredViewer";

const badgeClass = (d: DatasetSummary) => `badge badge-${d.is_element_map ? "map" : d.kind}`;

export function DatasetList() {
  const { files, selection, clickDataset, toggleSelect, closeFile } = useDatasets();
  const listMode = useUi((s) => s.listMode);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const highlighted = new Set(selection);

  // Rows are div[role=button] (not <button>) so the colour picker can sit inside.
  // Click: maps/images are added to or removed from the view; a spectrum
  // image, stack or spectrum becomes the view. ⌘/Ctrl-click toggles anything
  // (e.g. removes the spectrum image to leave just its maps). Images can also
  // be dragged onto the viewer.
  const activate = (d: DatasetSummary, toggle: boolean) => (toggle ? toggleSelect(d) : clickDataset(d));
  const itemProps = (d: DatasetSummary, className: string) => ({
    role: "button",
    tabIndex: 0,
    "aria-pressed": highlighted.has(d.id),
    className: `${className}${highlighted.has(d.id) ? " selected" : ""}`,
    onClick: (e: MouseEvent) => activate(d, e.metaKey || e.ctrlKey),
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      activate(d, e.metaKey || e.ctrlKey);
    },
    draggable: d.kind === "image",
    onDragStart: (e: DragEvent) => {
      e.dataTransfer.setData(DATASET_MIME, d.id);
      e.dataTransfer.effectAllowed = "copy";
    },
  });

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
                      <div {...itemProps(d, "dataset-card")} title={`${d.title} — ${shapeLabel(d)}`}>
                        <span className={`thumb thumb-${d.kind}`}>
                          <img src={thumbnailUrl(d.id)} alt="" loading="lazy" draggable={false} />
                          {d.is_element_map && <MapTint d={d} />}
                          <span className={badgeClass(d)}>{kindLabel(d)}</span>
                        </span>
                        <span className="dataset-title">
                          {d.is_element_map && <ColourDot d={d} />}
                          {d.title}
                        </span>
                        <span className="dataset-shape">{shapeLabel(d)}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <ul className="dataset-list">
                  {file.datasets.map((d) => (
                    <li key={d.id}>
                      <div {...itemProps(d, "dataset")}>
                        <span className={badgeClass(d)}>{kindLabel(d)}</span>
                        <span className="dataset-title">
                          {d.is_element_map && <ColourDot d={d} />}
                          {d.title}
                        </span>
                        <span className="dataset-shape">{shapeLabel(d)}</span>
                      </div>
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

/** The colormap an element map is shown with (its element colour by default). */
const useMapColormap = (d: DatasetSummary) =>
  useDisplay((s) => s.display[d.id]?.colormap ?? s.elementColours[d.title] ?? "#ffffff");

/** Tints a map's greyscale thumbnail with its colour: multiplying turns the
 * grey ramp into the same black → colour scale the viewer uses. */
function MapTint({ d }: { d: DatasetSummary }) {
  const colormap = useMapColormap(d);
  return isColour(colormap) ? <span className="thumb-tint" style={{ background: colormap }} /> : null;
}

/** An element map's colour; click to pick another (applies to that element). */
function ColourDot({ d }: { d: DatasetSummary }) {
  const colormap = useMapColormap(d);
  return (
    <label
      className="colour-dot"
      style={{ background: swatchBackground(colormap) }}
      title={`${d.title} colour — click to change`}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      <input
        type="color"
        value={isColour(colormap) ? colormap : "#ffffff"}
        onChange={(e) => setMapColour(d, e.target.value)}
        aria-label={`${d.title} colour`}
      />
    </label>
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
