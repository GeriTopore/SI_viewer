import { useEffect, useState, type CSSProperties, type DragEvent, type KeyboardEvent, type MouseEvent } from "react";
import { api, exportUrl, mapsZipUrl, thumbnailUrl, type DatasetSummary, type EelsPartner, type FileSummary } from "../api/client";
import { allDatasets, canDelete, elementOf, isColourMap, layoutFile, layoutPairs, setMapColour, useDatasets } from "../state/datasets";
import { useLayers } from "../state/layers";
import { useDisplay } from "../state/display";
import { useUi } from "../state/ui";
import { isColour, swatchBackground } from "../viewer/colormaps";
import { datasetTooltip, isWindowMap, kindLabel, shapeLabel } from "../format";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { CollapseButton } from "./PaneToggle";
import { DATASET_MIME } from "./LayeredViewer";
import { RebinDialog, canRebin } from "./RebinDialog";
import { openAlignment } from "./AlignSection";
import { downloadUrl, useExportFormats } from "../download";

const pairStyle = (colour: number) => ({ "--pair": `var(--pair-${colour})` }) as CSSProperties;

const badgeClass = (d: DatasetSummary) => `badge badge-${isWindowMap(d) ? "window" : d.is_element_map ? "map" : d.kind}`;

const deleteTitle = (d: DatasetSummary) =>
  d.origin === "fitted" || d.origin === "composition" || isWindowMap(d)
    ? "Delete this map"
    : `Delete this map (also unticks ${elementOf(d)} in the periodic table)` +
      (d.origin === "file" ? ". The file on disk is not changed." : "");

export function DatasetList() {
  const { files, selection, clickDataset, toggleSelect, closeFile, deleteDataset } = useDatasets();
  const listMode = useUi((s) => s.listMode);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  // Right-click menu (Rebin…, Delete) and the rebin dialog it opens.
  const [menu, setMenu] = useState<{ x: number; y: number; d: DatasetSummary } | null>(null);
  const [rebinTarget, setRebinTarget] = useState<DatasetSummary | null>(null);
  const exportFormats = useExportFormats();
  // Low/core loss pairs: candidates for the right-clicked EELS spectrum image.
  const [partners, setPartners] = useState<EelsPartner[] | null>(null);
  const menuId = menu?.d.eels ? menu.d.id : null;
  useEffect(() => {
    setPartners(null);
    if (!menuId) return;
    let stale = false;
    api
      .eelsPartners(menuId)
      .then((p) => !stale && setPartners(p))
      .catch((e) => {
        if (stale) return;
        setPartners([]);
        useLayers.getState().notify(`Could not list partners: ${(e as Error).message}`);
      });
    return () => {
      stale = true;
    };
  }, [menuId]);
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
    onContextMenu: (e: MouseEvent) => {
      e.preventDefault();
      setMenu({ x: e.clientX, y: e.clientY, d });
    },
    onKeyDown: (e: KeyboardEvent) => {
      if (canDelete(d) && (e.key === "Delete" || e.key === "Backspace")) {
        e.preventDefault();
        deleteDataset(d);
        return;
      }
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

  const { colourOf, items } = layoutPairs(files);
  const titles = new Map(allDatasets(files).map((x) => [x.id, x.title]));
  const parentTitle = (d: DatasetSummary) => (d.parent_id ? titles.get(d.parent_id) : undefined);
  const renderDataset = (d: DatasetSummary) =>
    listMode === "grid" ? (
      <li key={d.id}>
        <div {...itemProps(d, "dataset-card")} title={datasetTooltip(d, parentTitle(d))}>
          <span className={`thumb thumb-${d.kind}`}>
            <img src={thumbnailUrl(d)} alt="" loading="lazy" draggable={false} />
            {isColourMap(d) && <MapTint d={d} />}
            <span className={badgeClass(d)}>{kindLabel(d)}</span>
            {canDelete(d) && <DeleteButton d={d} className="thumb-delete" onDelete={deleteDataset} />}
          </span>
          <span className="dataset-title">
            {isColourMap(d) && <ColourDot d={d} />}
            <LinkGlyph d={d} files={files} colour={colourOf.get(d.id)} />
            {d.title}
          </span>
          <span className="dataset-shape">{shapeLabel(d)}</span>
        </div>
      </li>
    ) : (
      <li key={d.id}>
        <div {...itemProps(d, "dataset")} title={datasetTooltip(d, parentTitle(d))}>
          <span className={badgeClass(d)}>{kindLabel(d)}</span>
          <span className="dataset-title">
            {isColourMap(d) && <ColourDot d={d} />}
            <LinkGlyph d={d} files={files} colour={colourOf.get(d.id)} />
            {d.title}
          </span>
          <span className="dataset-shape">{shapeLabel(d)}</span>
          {canDelete(d) ? <DeleteButton d={d} className="row-delete" onDelete={deleteDataset} /> : <span />}
        </div>
      </li>
    );
  const listClass = listMode === "grid" ? "dataset-grid" : "dataset-list";
  const renderFile = (file: FileSummary) => {
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
        {!isCollapsed && (
          <ul className={listClass}>
            {layoutFile(file, colourOf).map((it) =>
              it.type === "dataset" ? (
                renderDataset(it.d)
              ) : (
                // Two linked datasets of this file, side by side in a box.
                <li key={`${it.ds[0].id}+${it.ds[1].id}`} className="pair-box pair-rows" style={pairStyle(it.colour)}>
                  <span className="pair-label">EELS pair</span>
                  <ul className={listClass}>{it.ds.map(renderDataset)}</ul>
                </li>
              ),
            )}
          </ul>
        )}
      </li>
    );
  };

  return (
    <>
      <ul className="file-list">
        {items.map((it) =>
          it.type === "file" ? (
            renderFile(it.file)
          ) : (
            <li key={`${it.files[0].id}+${it.files[1].id}`} className="pair-box" style={pairStyle(it.colour)}>
              <span className="pair-label">EELS pair</span>
              <ul className="file-list">{it.files.map(renderFile)}</ul>
            </li>
          ),
        )}
      </ul>
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            ...eelsItems(menu.d, partners, files),
            ...(menu.d.kind === "stack" ? [{ label: "Align frames…", onSelect: () => openAlignment(menu.d) }] : []),
            ...(canRebin(menu.d) ? [{ label: "Rebin…", onSelect: () => setRebinTarget(menu.d) }] : []),
            ...(exportFormats[menu.d.kind] ?? []).map((f) => ({
              label: `Save as ${f.label}`,
              onSelect: () => downloadUrl(exportUrl(menu.d.id, f.format)),
            })),
            ...(menu.d.is_element_map
              ? [{ label: "Save all maps of this file (TIFF, .zip)", onSelect: () => downloadUrl(mapsZipUrl(menu.d.file_id)) }]
              : []),
            ...(canDelete(menu.d) ? [{ label: "Delete", danger: true, onSelect: () => deleteDataset(menu.d) }] : []),
          ]}
        />
      )}
      {rebinTarget && <RebinDialog d={rebinTarget} onClose={() => setRebinTarget(null)} />}
    </>
  );
}

const linkedTo = (d: DatasetSummary, files: FileSummary[]) => {
  const id = d.eels?.partner_id;
  const partner = id ? allDatasets(files).find((x) => x.id === id) : undefined;
  const file = partner && files.find((f) => f.id === partner.file_id);
  return partner && file ? { partner, file } : null;
};

const ROLE_NAME = { low_loss: "low loss", core_loss: "core loss" } as const;

/** Right-click entries for an EELS spectrum image: link it with a dataset of
 * the other kind (the candidates come from the backend), or unlink it. */
function eelsItems(d: DatasetSummary, partners: EelsPartner[] | null, files: FileSummary[]): MenuItem[] {
  if (!d.eels) return [];
  const want = ROLE_NAME[d.eels.role === "low_loss" ? "core_loss" : "low_loss"];
  const notify = useLayers.getState().notify;
  const fileName = (id: string) => files.find((f) => f.id === id)?.name ?? "";
  const nameOf = (id: string) => allDatasets(files).find((x) => x.id === id)?.title ?? "another dataset";
  const items: MenuItem[] = [{ label: `Link ${want}…`, disabled: true }];
  if (!partners) items.push({ label: "Looking…", disabled: true, indent: true });
  else if (!partners.length) items.push({ label: `No compatible ${want} datasets open`, disabled: true, indent: true });
  else {
    for (const p of partners) {
      const current = p.id === d.eels.partner_id;
      const other = p.linked_to && !current ? ` (linked to ${nameOf(p.linked_to)})` : "";
      items.push({
        label: `${p.title} (${fileName(p.file_id)})${current ? " ✓" : other}`,
        indent: true,
        disabled: current,
        title: other ? "Linking replaces its current link" : undefined,
        onSelect: () =>
          useDatasets
            .getState()
            .linkEels(d, p.id)
            .catch((e) => notify(`Could not link ${d.title}: ${(e as Error).message}`)),
      });
    }
  }
  if (d.eels.partner_id) {
    const role = ROLE_NAME[d.eels.role === "low_loss" ? "core_loss" : "low_loss"];
    items.push({
      label: `Unlink ${role}`,
      onSelect: () =>
        useDatasets
          .getState()
          .linkEels(d, null)
          .catch((e) => notify(`Could not unlink ${d.title}: ${(e as Error).message}`)),
    });
  }
  return items;
}

/** Chain mark on a low/core loss that is linked with a partner. */
function LinkGlyph({ d, files, colour }: { d: DatasetSummary; files: FileSummary[]; colour?: number }) {
  if (!d.eels?.partner_id) return null;
  const link = linkedTo(d, files);
  const text = link ? `Linked to ${link.partner.title} (${link.file.name})` : "Linked to a low/core loss";
  return (
    <span className="link-glyph" title={text} aria-label={text} style={colour === undefined ? undefined : { color: `var(--pair-${colour})` }}>
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
        <path d="M5 7a2 2 0 0 0 2.8 0l2-2a2 2 0 0 0-2.8-2.8l-.7.7" />
        <path d="M7 5a2 2 0 0 0-2.8 0l-2 2A2 2 0 0 0 5 9.8l.7-.7" />
      </svg>
    </span>
  );
}

/** × that deletes an element map (without selecting the row). */
function DeleteButton({
  d, className, onDelete,
}: { d: DatasetSummary; className: string; onDelete: (d: DatasetSummary) => void }) {
  return (
    <button
      className={`icon-button ${className}`}
      title={deleteTitle(d)}
      aria-label={`Delete ${d.title}`}
      onClick={(e) => {
        e.stopPropagation();
        onDelete(d);
      }}
      onKeyDown={(e) => e.stopPropagation()}
      draggable={false}
    >
      ×
    </button>
  );
}

/** The colormap an element map is shown with (its element colour by default). */
const useMapColormap = (d: DatasetSummary) =>
  useDisplay(
    (s) =>
      s.display[d.id]?.colormap ??
      (isWindowMap(d) ? s.windowColours[d.id] : s.elementColours[elementOf(d)]) ??
      "#ffffff",
  );

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
      title={`${isWindowMap(d) ? d.title : elementOf(d)} colour — click to change`}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => e.stopPropagation()}
    >
      <input
        type="color"
        value={isColour(colormap) ? colormap : "#ffffff"}
        onChange={(e) => setMapColour(d, e.target.value)}
        aria-label={`${isWindowMap(d) ? d.title : elementOf(d)} colour`}
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
      <CollapseButton side="left" label="Datasets" />
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
