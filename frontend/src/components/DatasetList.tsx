import {
  Fragment, useEffect, useState, type CSSProperties, type DragEvent, type KeyboardEvent, type MouseEvent, type ReactNode,
} from "react";
import { api, thumbnailUrl, type DatasetSummary, type EelsPartner, type FileSummary } from "../api/client";
import {
  allDatasets, canDelete, elementOf, isColourMap, layoutPane, layoutTree, paneFiles, paneKey, setMapColour,
  useDatasets, workspaceOf, type FileTree, type PaneItem, type RowItem,
} from "../state/datasets";
import { useLayers } from "../state/layers";
import { useDisplay } from "../state/display";
import { useUi } from "../state/ui";
import { isColour, swatchBackground } from "../viewer/colormaps";
import { datasetTooltip, isBackgroundRemoved, isDeconvolved, isLineScan, isThickness, isWindowMap, isZlpShift, kindLabel, shapeLabel } from "../format";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { CollapseButton } from "./PaneToggle";
import { DATASET_MIME } from "./LayeredViewer";
import { RebinDialog, canRebin } from "./RebinDialog";
import { openAlignment } from "./AlignSection";
import { RenameInput } from "./RenameInput";
import { FitQuantitySwitch } from "./FitQuantitySwitch";
import { ExportDialog } from "./ExportDialog";
import { useExportFormats } from "../download";

/** How many datasets are under ``id`` in ``tree``, at any depth. */
function countUnder(id: string, tree: FileTree): number {
  const rows = tree.children.get(id) ?? [];
  return rows
    .flatMap((it) => (it.type === "dataset" ? [it.d] : it.ds))
    .reduce((n, d) => n + 1 + countUnder(d.id, tree), 0);
}

const pairStyle = (colour: number) => ({ "--pair": `var(--pair-${colour})` }) as CSSProperties;

const badgeClass = (d: DatasetSummary) =>
  `badge badge-${isWindowMap(d) ? "window" : isBackgroundRemoved(d) ? "background" : isDeconvolved(d) ? "deconvolved" : isThickness(d) ? "thickness" : isZlpShift(d) ? "shift" : d.survey ? "survey" : d.is_element_map ? "map" : d.kind}`;

const deleteTitle = (d: DatasetSummary) =>
  isBackgroundRemoved(d) || isDeconvolved(d) || d.origin === "line-scan"
    ? `Delete this ${isLineScan(d) ? "LineScan" : "spectrum image"} (and what was made from it)`
    : d.origin === "fitted" || isWindowMap(d) || isThickness(d) || isZlpShift(d)
      ? "Delete this map"
      : `Delete this map (also unticks ${elementOf(d)} in the periodic table)` +
        (d.origin === "file" ? ". The file on disk is not changed." : "");

export function DatasetList() {
  const { files, selection, focusId, clickDataset, toggleSelect, closeFile, deleteDataset, enterWorkspace } = useDatasets();
  // The workspace being viewed (a top-level entry); the others are entered first.
  const active = useDatasets((s) => s.active);
  const activeKey = active ? (workspaceOf(files, active)?.key ?? null) : null;
  const workspaceKey = new Map(
    layoutPane(files).items.flatMap((it) => paneFiles(it).map((f): [string, string] => [f.id, paneKey(it)])),
  );
  /** ``fileId`` is in a workspace other than the one being viewed. */
  const elsewhere = (fileId: string) => activeKey !== null && workspaceKey.get(fileId) !== activeKey;
  const wsClass = (it: PaneItem) =>
    activeKey === null ? "" : paneKey(it) === activeKey ? " workspace-active" : " workspace-inactive";
  // Another workspace takes no action of its own (folding, ×, renaming, menus,
  // colours, dragging, keys): any click on it only enters it.
  const wsProps = (it: PaneItem) => {
    if (activeKey === null || paneKey(it) === activeKey) return {};
    const enter = (e: React.SyntheticEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const first = paneFiles(it)[0];
      if (first) enterWorkspace(first.id);
    };
    const block = (e: React.SyntheticEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };
    return {
      onClickCapture: enter,
      onContextMenuCapture: enter,
      onDoubleClickCapture: block,
      onDragStartCapture: block,
      onKeyDownCapture: (e: KeyboardEvent) => {
        if (e.key === "Tab") return;
        if (e.key === "Enter" || e.key === " ") enter(e);
        else block(e);
      },
    };
  };
  const listMode = useUi((s) => s.listMode);
  const collapsedTypes = useUi((s) => s.collapsedTypes);
  const toggleTypeGroup = useUi((s) => s.toggleTypeGroup);
  const foldedNodes = useUi((s) => s.foldedNodes);
  const toggleNode = useUi((s) => s.toggleNode);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  // Right-click menu (Rebin…, Delete) and the rebin dialog it opens.
  const [menu, setMenu] = useState<{ x: number; y: number; d: DatasetSummary } | null>(null);
  const [rebinTarget, setRebinTarget] = useState<DatasetSummary | null>(null);
  // The dataset whose name is being edited (Rename…, F2 or double-clicking its name).
  const [renaming, setRenaming] = useState<string | null>(null);
  // Right-click on a type group's row (Batch export…), and the Export dialog.
  const [groupMenu, setGroupMenu] = useState<{ x: number; y: number; ds: DatasetSummary[] } | null>(null);
  const [exporting, setExporting] = useState<DatasetSummary[] | null>(null);
  const exportFormats = useExportFormats();
  // Link groups: candidates for the right-clicked EELS spectrum image.
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
        useLayers.getState().notify(`Could not list link candidates:${(e as Error).message}`);
      });
    return () => {
      stale = true;
    };
  }, [menuId]);
  // A linked EELS group in the viewer is highlighted as a whole; of several
  // highlighted, the one Info and Process show is marked "focused".
  const byIdAll = new Map(allDatasets(files).map((d) => [d.id, d]));
  const highlighted = new Set(selection.flatMap((id) => [id, ...(byIdAll.get(id)?.eels?.linked ?? [])]));

  // Rows are div[role=button] (not <button>) so the colour picker can sit inside.
  // Click: maps/images are added to or removed from the view; a spectrum
  // image, stack or spectrum becomes the view. ⌘/Ctrl-click toggles anything
  // (e.g. removes the spectrum image to leave just its maps). Images can also
  // be dragged onto the viewer.
  // A dataset of another workspace enters that workspace first (bringing back
  // what it showed); the next click acts on the dataset.
  const activate = (d: DatasetSummary, toggle: boolean) =>
    elsewhere(d.file_id) ? enterWorkspace(d.file_id) : toggle ? toggleSelect(d) : clickDataset(d);
  const itemProps = (d: DatasetSummary, className: string) => ({
    role: "button",
    tabIndex: 0,
    "aria-pressed": highlighted.has(d.id),
    className: `${className}${highlighted.has(d.id) ? " selected" : ""}${
      highlighted.size > 1 && d.id === focusId && highlighted.has(d.id) ? " focused" : ""
    }`,
    onClick: (e: MouseEvent) => activate(d, e.metaKey || e.ctrlKey),
    onContextMenu: (e: MouseEvent) => {
      e.preventDefault();
      setMenu({ x: e.clientX, y: e.clientY, d });
    },
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key === "F2") {
        e.preventDefault();
        setRenaming(d.id);
        return;
      }
      if (canDelete(d) && (e.key === "Delete" || e.key === "Backspace")) {
        e.preventDefault();
        deleteDataset(d);
        return;
      }
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      activate(d, e.metaKey || e.ctrlKey);
    },
    draggable: d.kind === "image" && renaming !== d.id && !elsewhere(d.file_id),
    onDragStart: (e: DragEvent) => {
      e.dataTransfer.setData(DATASET_MIME, d.id);
      e.dataTransfer.effectAllowed = "copy";
    },
  });

  if (!files.length) {
    return <p className="muted empty-list">No files open. Use Open… (⌘/Ctrl+O).</p>;
  }

  const { colourOf, items } = layoutPane(files);
  const titles = new Map(allDatasets(files).map((x) => [x.id, x.title]));
  const titleOf = (d: DatasetSummary) =>
    renaming === d.id ? (
      <RenameInput d={d} onDone={() => setRenaming(null)} />
    ) : (
      <span className="dataset-name" onDoubleClick={() => setRenaming(d.id)}>
        {d.title}
      </span>
    );
  const parentTitle = (d: DatasetSummary) => (d.parent_id ? titles.get(d.parent_id) : undefined);
  // A dataset with results (anything made from it) folds them away.
  const nodeToggle = (d: DatasetSummary, tree: FileTree) => {
    const n = countUnder(d.id, tree);
    if (!n) return null;
    const folded = !!foldedNodes[d.id];
    return (
      <button
        className={`node-toggle${folded ? " folded" : ""}`}
        aria-expanded={!folded}
        aria-label={folded ? `Show ${n} results` : `Hide ${n} results`}
        title={folded ? `${n} result${n > 1 ? "s" : ""}` : "Hide its results"}
        onClick={(e) => {
          e.stopPropagation();
          toggleNode(d.id);
        }}
        onKeyDown={(e) => e.stopPropagation()}
      >
        <span className="process-chevron" aria-hidden />
      </button>
    );
  };
  // `inGroup`: a member of a type group, whose header already names the type
  // (list view only; the cards keep their badge). The dataset's results
  // follow it, one level in.
  const renderDataset = (d: DatasetSummary, tree: FileTree, inGroup = false) => {
    const kids = tree.children.get(d.id);
    return (
      <Fragment key={d.id}>
        {renderRow(d, tree, inGroup)}
        {kids && !foldedNodes[d.id] && (
          <li className="tree-children-row">
            <ul className={listClass}>{renderRows(kids, tree)}</ul>
          </li>
        )}
      </Fragment>
    );
  };
  const renderRow = (d: DatasetSummary, tree: FileTree, inGroup: boolean) =>
    listMode === "grid" ? (
      <li className={tree.children.has(d.id) ? "tree-node" : undefined}>
        <div {...itemProps(d, "dataset-card")} title={datasetTooltip(d, parentTitle(d))}>
          <span className={`thumb thumb-${d.kind}`}>
            <img src={thumbnailUrl(d)} alt="" loading="lazy" draggable={false} />
            {isColourMap(d) && <MapTint d={d} />}
            <span className={badgeClass(d)}>{kindLabel(d)}</span>
            {canDelete(d) && <DeleteButton d={d} className="thumb-delete" onDelete={deleteDataset} />}
          </span>
          <span className="dataset-title">
            {nodeToggle(d, tree)}
            {isColourMap(d) && <ColourDot d={d} />}
            <LinkGlyph d={d} files={files} colour={colourOf.get(d.id)} />
            {titleOf(d)}
          </span>
          <span className="dataset-shape">{shapeLabel(d)}</span>
        </div>
      </li>
    ) : (
      <li className={tree.children.has(d.id) ? "tree-node" : undefined}>
        <div
          {...itemProps(d, `${inGroup ? "dataset no-badge" : "dataset"}${tree.children.has(d.id) ? " has-results" : ""}`)}
          title={datasetTooltip(d, parentTitle(d))}
        >
          {/* Before the badge, as a type group's chevron is. */}
          {nodeToggle(d, tree)}
          {!inGroup && <span className={badgeClass(d)}>{kindLabel(d)}</span>}
          <span className="dataset-title">
            {isColourMap(d) && <ColourDot d={d} />}
            <LinkGlyph d={d} files={files} colour={colourOf.get(d.id)} />
            {titleOf(d)}
          </span>
          <span className="dataset-shape">{shapeLabel(d)}</span>
          {canDelete(d) ? <DeleteButton d={d} className="row-delete" onDelete={deleteDataset} /> : <span />}
        </div>
      </li>
    );
  const listClass = listMode === "grid" ? "dataset-grid" : "dataset-list";
  // Two or more datasets of one type: a foldable group.
  const renderTypes = (g: Extract<RowItem, { type: "types" }>, tree: FileTree) => {
    const isCollapsed = !!collapsedTypes[g.key];
    const active = isCollapsed && g.ds.some((d) => highlighted.has(d.id) || d.id === focusId);
    return (
      <li key={g.key} className={`type-group${isCollapsed ? " collapsed" : ""}${active ? " active" : ""}`}>
        <div className="type-group-header">
          <button
            className="type-group-toggle"
            aria-expanded={!isCollapsed}
            title={`${g.ds.length} × ${g.label}`}
            onClick={() => toggleTypeGroup(g.key)}
            onContextMenu={(e) => {
              if (!exportFormats[g.ds[0].kind]?.length) return;
              e.preventDefault();
              setGroupMenu({ x: e.clientX, y: e.clientY, ds: g.ds });
            }}
          >
            <span className="process-chevron" aria-hidden />
            <span className={badgeClass(g.ds[0])}>{g.label}</span>
            <span className="dataset-shape">{g.ds.length}</span>
          </button>
          <FitQuantitySwitch ds={g.ds} />
        </div>
        {!isCollapsed && <ul className={listClass}>{g.ds.map((d) => renderDataset(d, tree, true))}</ul>}
      </li>
    );
  };
  // One level of a file's tree: datasets, type groups and link boxes.
  const renderRows = (rows: RowItem[], tree: FileTree): ReactNode[] =>
    rows.map((it) =>
      it.type === "dataset" ? (
        renderDataset(it.d, tree)
      ) : it.type === "types" ? (
        renderTypes(it, tree)
      ) : (
        // Linked datasets of this file, side by side in a box.
        <li key={it.ds.map((d) => d.id).join("+")} className="pair-box pair-rows" style={pairStyle(it.colour)}>
          <span className="pair-label">Linked EELS</span>
          <ul className={listClass}>{it.ds.map((d) => renderDataset(d, tree))}</ul>
        </li>
      ),
    );
  const renderFile = (file: FileSummary, extra = "", props = {}) => {
    const isCollapsed = collapsed[file.id];
    return (
      <li key={file.id} className={`file${extra}`} {...props}>
        <div className="file-header">
          <button
            className="disclosure"
            aria-expanded={!isCollapsed}
            onClick={() => setCollapsed((c) => ({ ...c, [file.id]: !isCollapsed }))}
          >
            {isCollapsed ? "▸" : "▾"}
          </button>
          {/* In an experiment: what the file is (Survey, ADF, EELS LL, …). */}
          <span className="file-name" title={file.path} onClick={() => enterWorkspace(file.id)}>
            {file.role ?? file.name}
          </span>
          <button className="icon-button" title="Close file" onClick={() => closeFile(file.id)}>
            ×
          </button>
        </div>
        {!isCollapsed && (() => {
          const tree = layoutTree(file, colourOf);
          return <ul className={listClass}>{renderRows(tree.rows, tree)}</ul>;
        })()}
      </li>
    );
  };

  // The files of one folder opened together, by role, on a tree.
  const renderExperiment = (it: Extract<PaneItem, { type: "experiment" }>) => {
    const { experiment: x } = it;
    const isCollapsed = collapsed[x.id];
    return (
      <li key={x.id} className={`experiment${wsClass(it)}`} {...wsProps(it)}>
        <div className="file-header">
          <button
            className="disclosure"
            aria-expanded={!isCollapsed}
            onClick={() => setCollapsed((c) => ({ ...c, [x.id]: !isCollapsed }))}
          >
            {isCollapsed ? "▸" : "▾"}
          </button>
          <span className="fb-icon dir" aria-hidden />
          <span className="file-name" title={x.path} onClick={() => it.files[0] && enterWorkspace(it.files[0].id)}>
            {x.name}
          </span>
          <button
            className="icon-button"
            title="Close the experiment's files"
            onClick={async () => {
              for (const f of it.files) await closeFile(f.id);
            }}
          >
            ×
          </button>
        </div>
        {!isCollapsed && <ul className="file-list experiment-files">{it.files.map((f) => renderFile(f))}</ul>}
      </li>
    );
  };

  return (
    <>
      <ul className="file-list">
        {items.map((it) =>
          it.type === "file" ? (
            renderFile(it.file, wsClass(it), wsProps(it))
          ) : it.type === "experiment" ? (
            renderExperiment(it)
          ) : (
            <li
              key={it.files.map((f) => f.id).join("+")}
              className={`pair-box${wsClass(it)}`}
              style={pairStyle(it.colour)}
              {...wsProps(it)}
            >
              <span className="pair-label">Linked EELS</span>
              <ul className="file-list">{it.files.map((f) => renderFile(f))}</ul>
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
            { label: "Rename…", onSelect: () => setRenaming(menu.d.id) },
            ...eelsItems(menu.d, partners, files),
            ...(menu.d.kind === "stack" ? [{ label: "Align frames…", onSelect: () => openAlignment(menu.d) }] : []),
            ...(canRebin(menu.d) ? [{ label: "Rebin…", onSelect: () => setRebinTarget(menu.d) }] : []),
            ...(exportFormats[menu.d.kind]?.length ? [{ label: "Export…", onSelect: () => setExporting([menu.d]) }] : []),
            ...(canDelete(menu.d) ? [{ label: "Delete", danger: true, onSelect: () => deleteDataset(menu.d) }] : []),
          ]}
        />
      )}
      {groupMenu && (
        <ContextMenu
          x={groupMenu.x}
          y={groupMenu.y}
          onClose={() => setGroupMenu(null)}
          items={[{ label: "Batch export…", onSelect: () => setExporting(groupMenu.ds) }]}
        />
      )}
      {rebinTarget && <RebinDialog d={rebinTarget} onClose={() => setRebinTarget(null)} />}
      {exporting && <ExportDialog datasets={exporting} onClose={() => setExporting(null)} />}
    </>
  );
}

const ROLE_NAME = { low_loss: "low loss", core_loss: "core loss" } as const;

/** Right-click entries for an EELS spectrum image: link it with another one
 * of the same scan (the candidates come from the backend) or unlink it (its
 * role is set in the Info tab). */
function eelsItems(d: DatasetSummary, partners: EelsPartner[] | null, files: FileSummary[]): MenuItem[] {
  if (!d.eels) return [];
  const notify = useLayers.getState().notify;
  const fileName = (id: string) => files.find((f) => f.id === id)?.name ?? "";
  const items: MenuItem[] = [{ label: "Link with…", disabled: true }];
  if (!partners) items.push({ label: "Looking…", disabled: true, indent: true });
  else if (!partners.length) items.push({ label: "No compatible EELS datasets open", disabled: true, indent: true });
  else {
    for (const p of partners) {
      const current = d.eels.linked.includes(p.id);
      const elsewhere = p.group && !current ? ", linked elsewhere" : "";
      items.push({
        label: `${p.title} (${fileName(p.file_id)}, ${ROLE_NAME[p.role]}${elsewhere})${current ? " ✓" : ""}`,
        indent: true,
        disabled: current,
        title: elsewhere ? "Linking moves this dataset to its group" : undefined,
        onSelect: () =>
          useDatasets
            .getState()
            .linkEels(d, p.id)
            .catch((e) => notify(`Could not link ${d.title}: ${(e as Error).message}`)),
      });
    }
  }
  if (d.eels.linked.length) {
    items.push({
      label: "Unlink",
      onSelect: () =>
        useDatasets
          .getState()
          .linkEels(d, null)
          .catch((e) => notify(`Could not unlink ${d.title}: ${(e as Error).message}`)),
    });
  }
  return items;
}

/** Chain mark on a spectrum image that is linked with others. */
function LinkGlyph({ d, files, colour }: { d: DatasetSummary; files: FileSummary[]; colour?: number }) {
  if (!d.eels?.linked.length) return null;
  const all = allDatasets(files);
  const named = d.eels.linked.map((id) => {
    const o = all.find((x) => x.id === id);
    const file = o && files.find((f) => f.id === o.file_id);
    return o ? `${o.title}${file ? ` (${file.name})` : ""}` : "another dataset";
  });
  const text = `Linked to ${named.join(", ")}`;
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

export const ListIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
    <path d="M1 2.5h12M1 7h12M1 11.5h12" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

export const GridIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
    <rect x="1" y="1" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="8" y="1" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="1" y="8" width="5" height="5" rx="1" fill="currentColor" />
    <rect x="8" y="8" width="5" height="5" rx="1" fill="currentColor" />
  </svg>
);
