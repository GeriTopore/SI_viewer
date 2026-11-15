import { useMemo, useRef, useState, type MouseEvent, type PointerEvent } from "react";
import type { DatasetSummary } from "../api/client";
import { datasetTooltip, kindLabel, shapeLabel } from "../format";
import { canDelete, layoutPane, layoutTree, useDatasets, workspaceOf, type FileTree, type RowItem } from "../state/datasets";
import { useExportFormats } from "../download";
import { useUi } from "../state/ui";
import { ContextMenu } from "./ContextMenu";
import { badgeClass } from "./DatasetList";
import { ExportDialog } from "./ExportDialog";
import { RenameInput } from "./RenameInput";

/** Focus mode of the datasets pane (Shift F over it): everything in the
 * highlighted dataset's workspace (its file, or the folder or link group it
 * belongs to) as family trees, one per raw dataset, each branching down to
 * everything made from it. Nodes fold; results of one type are one node that
 * opens into a list. Drag to pan. Right-click a dataset to show it (or
 * double-click it: focus mode closes and the main view shows it), rename,
 * export or delete it, as in the list. */
export function ProcessTree() {
  const files = useDatasets((s) => s.files);
  const focusId = useDatasets((s) => s.focusId);
  const selection = useDatasets((s) => s.selection);
  const current = focusId ?? selection[0] ?? null;
  const [folded, setFolded] = useState<Set<string>>(new Set());
  // Type groups start folded; opened ones are listed here.
  const [opened, setOpened] = useState<Set<string>>(new Set());
  const [menu, setMenu] = useState<{ x: number; y: number; d: DatasetSummary } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [exporting, setExporting] = useState<DatasetSummary[] | null>(null);
  const exportFormats = useExportFormats();
  const pan = usePan();

  const data = useMemo(() => {
    const file = files.find((f) => f.datasets.some((d) => d.id === current));
    if (!file) return null;
    const ws = workspaceOf(files, file.id);
    const { colourOf } = layoutPane(files);
    const forest = (ws?.files ?? [file]).map((f) => ({ file: f, tree: layoutTree(f, colourOf) }));
    const titles = new Map(forest.flatMap(({ file: f }) => f.datasets.map((d) => [d.id, d.title] as const)));
    const name = forest.length > 1 ? (file.experiment?.name ?? `${forest.length} files`) : file.name;
    return { forest, titles, name };
  }, [files, current]);

  if (!data) {
    return (
      <div className="process-tree-panel">
        <p className="muted process-tree-empty">Highlight a dataset to see its processing.</p>
      </div>
    );
  }
  const { forest, titles } = data;

  // Everything with children (dataset ids) and every type group (keys).
  const all = { nodes: [] as string[], groups: [] as string[] };
  const collect = (rows: RowItem[], tree: FileTree) => {
    for (const it of rows) {
      if (it.type === "types") all.groups.push(it.key);
      for (const d of members(it)) {
        const under = tree.children.get(d.id) ?? [];
        if (under.length) all.nodes.push(d.id);
        collect(under, tree);
      }
    }
  };
  for (const { tree } of forest) collect(tree.rows, tree);
  const collapseAll = () => {
    setFolded(new Set(all.nodes));
    setOpened(new Set());
  };
  const expandAll = () => {
    setFolded(new Set());
    setOpened(new Set(all.groups));
  };
  const toggle = (set: Set<string>, key: string, apply: (s: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    apply(next);
  };

  const show = (d: DatasetSummary) => {
    useUi.getState().setFocus(null);
    useDatasets.getState().select(d.id);
  };
  const onMenu = (e: MouseEvent, d: DatasetSummary) => {
    e.preventDefault();
    setMenu({ x: e.clientX, y: e.clientY, d });
  };

  const card = (d: DatasetSummary, extra = "") => (
    <div
      className={`process-tree-card${d.id === current ? " current" : ""}${extra}`}
      title={datasetTooltip(d, d.parent_id ? titles.get(d.parent_id) : undefined)}
      onContextMenu={(e) => onMenu(e, d)}
      onDoubleClick={() => renaming !== d.id && show(d)}
    >
      <span className={badgeClass(d)}>{kindLabel(d)}</span>
      {renaming === d.id ? (
        <RenameInput d={d} onDone={() => setRenaming(null)} />
      ) : (
        <span className="process-tree-title">{d.title}</span>
      )}
      <span className="process-tree-shape">{shapeLabel(d)}</span>
    </div>
  );

  const node = (d: DatasetSummary, tree: FileTree) => {
    const rows = tree.children.get(d.id) ?? [];
    const isFolded = folded.has(d.id);
    const count = rows.reduce((n, it) => n + members(it).length, 0);
    return (
      <li key={d.id}>
        {card(d)}
        {rows.length > 0 && (
          <button
            className="process-tree-fold"
            onClick={() => toggle(folded, d.id, setFolded)}
            aria-expanded={!isFolded}
            title={isFolded ? `Show what was made from it (${count})` : "Hide what was made from it"}
          >
            {isFolded ? `+${count}` : "−"}
          </button>
        )}
        {rows.length > 0 && !isFolded && <ul className="process-tree-level">{rows.map((it) => row(it, tree))}</ul>}
      </li>
    );
  };

  const row = (it: RowItem, tree: FileTree) => {
    if (it.type !== "types") return members(it).map((d) => node(d, tree));
    // Results of one type with nothing made from them: one node, a list.
    if (it.ds.some((d) => tree.children.get(d.id)?.length)) return it.ds.map((d) => node(d, tree));
    const isOpen = opened.has(it.key);
    return (
      <li key={it.key}>
        <div className="process-tree-group">
          <button
            className="process-tree-group-head"
            onClick={() => toggle(opened, it.key, setOpened)}
            aria-expanded={isOpen}
            title={isOpen ? "Fold" : `Show the ${it.ds.length}`}
          >
            <span className={badgeClass(it.ds[0])}>{it.label}</span>
            <span className="process-tree-shape">{it.ds.length}</span>
            <span className="muted" aria-hidden>{isOpen ? "▾" : "▸"}</span>
          </button>
          {isOpen && (
            <div className="process-tree-group-list">
              {it.ds.map((d) => (
                <div key={d.id}>{card(d, " in-group")}</div>
              ))}
            </div>
          )}
        </div>
      </li>
    );
  };

  return (
    <div className="process-tree-panel">
      <div className="process-tree-header">
        <h2>Processing of {data.name}</h2>
        <span className="process-tree-actions">
          <button className="secondary" onClick={collapseAll} disabled={!all.nodes.length && !all.groups.length}>
            Collapse all
          </button>
          <button className="secondary" onClick={expandAll} disabled={!all.nodes.length && !all.groups.length}>
            Expand all
          </button>
        </span>
      </div>
      <div ref={pan.ref} className={`process-tree-body${pan.dragging ? " panning" : ""}`} {...pan.handlers}>
        <div className="process-tree-forest">
          {forest.map(({ file, tree }) => (
            <section key={file.id} className="process-tree-file">
              {forest.length > 1 && <h3 title={file.path}>{file.name}</h3>}
              {/* Each raw dataset (or group of them) of the file: its own tree. */}
              <div className="process-tree-roots">
                {tree.rows.map((it, i) => (
                  <ul key={i} className="process-tree-level root">
                    {row(it, tree)}
                  </ul>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            { label: "Show", title: "Leave focus mode and show it", onSelect: () => show(menu.d) },
            { label: "Rename…", onSelect: () => setRenaming(menu.d.id) },
            ...(exportFormats[menu.d.kind]?.length ? [{ label: "Export…", onSelect: () => setExporting([menu.d]) }] : []),
            ...(canDelete(menu.d)
              ? [{ label: "Delete", danger: true, onSelect: () => useDatasets.getState().deleteDataset(menu.d) }]
              : []),
          ]}
        />
      )}
      {exporting && <ExportDialog datasets={exporting} onClose={() => setExporting(null)} />}
    </div>
  );
}

/** The datasets of a row (a link box holds several). */
function members(it: RowItem): DatasetSummary[] {
  return it.type === "dataset" ? [it.d] : it.ds;
}

/** Dragging with the left button pans the element (its scroll position);
 * buttons and the group lists keep their clicks. */
function usePan() {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const handlers = {
    onPointerDown: (e: PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0 || (e.target as Element).closest("button, input")) return;
      const el = ref.current!;
      start.current = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop };
    },
    onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
      const s = start.current;
      if (!s) return;
      const dx = e.clientX - s.x;
      const dy = e.clientY - s.y;
      if (!dragging) {
        if (Math.hypot(dx, dy) < 3) return; // a click, not a drag (clicks stay with the cards)
        ref.current!.setPointerCapture(e.pointerId);
        setDragging(true);
      }
      ref.current!.scrollLeft = s.left - dx;
      ref.current!.scrollTop = s.top - dy;
    },
    onPointerUp: (e: PointerEvent<HTMLDivElement>) => {
      start.current = null;
      setDragging(false);
      if (ref.current?.hasPointerCapture(e.pointerId)) ref.current.releasePointerCapture(e.pointerId);
    },
  };
  return { ref, dragging, handlers };
}
