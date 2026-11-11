import { useLayoutEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import { safeFileName } from "../download";
import { useDisplay } from "../state/display";
import { useUi } from "../state/ui";
import { isColour } from "../viewer/colormaps";
import type { SnapshotPanel } from "../viewer/snapshot";
import { PngExportDialog } from "./PngExportDialog";
import { ImageView, type AxisCal, type Interaction, type Transform, type ViewControls, type ViewLayer } from "./ImageView";

/** One image of the grid: its layer, size and calibration, and the key under
 * which its pan/zoom is stored (shared by panels of the same pixel size). */
export interface GridPanel {
  layer: ViewLayer;
  width: number;
  height: number;
  calX: AxisCal;
  calY: AxisCal;
  viewKey: string;
  loading?: string | null;
  error?: string | null;
}

interface Props {
  /** The panels to show, in order (one per layer, hidden or not). */
  panels: GridPanel[];
  /** Key of the layer the toolbar edits; its panel is outlined and publishes
   * its image for the FFT. */
  active: string | undefined;
  onActivate: (key: string) => void;
  interaction?: Interaction;
  renderOverlay?: (t: Transform, p: GridPanel) => ReactNode;
  /** Removes a panel's image from the view (× on its label); null for
   * panels that can't be removed. */
  onRemove?: (key: string) => (() => void) | null;
  empty?: string | null;
  /** Panels over a navigator k times finer than the scan show the
   * spectrum-image pixel under the pointer too. */
  scanFactor?: number;
  /** Key of the FFT boxes drawn on the panels (published with the image). */
  fftView?: string;
  /** Key under which a hand-made arrangement of the panels is kept; without
   * it the panels can't be moved. */
  layoutKey?: string;
}

const GAP = 6;

/** Drag-and-drop type of a panel being moved (its key is kept in the grid). */
const PANEL_MIME = "application/x-si-viewer-panel";

/** Where a moved panel goes: beside the target in its row, or in a new row
 * above or below the target's. */
type DropSide = "left" | "right" | "above" | "below";

/** The rows to show: the saved arrangement (panels no longer shown dropped,
 * new ones added to the last row), else ``keys`` in rows of ``cols``. */
function arrange(keys: string[], saved: string[][] | undefined, cols: number): string[][] {
  if (saved) {
    const shown = new Set(keys);
    const rows = saved.map((r) => r.filter((k) => shown.has(k))).filter((r) => r.length);
    if (rows.length) {
      const placed = new Set(rows.flat());
      const rest = keys.filter((k) => !placed.has(k));
      if (rest.length) rows[rows.length - 1] = [...rows[rows.length - 1], ...rest];
      return rows;
    }
  }
  return Array.from({ length: Math.ceil(keys.length / cols) }, (_, i) => keys.slice(i * cols, (i + 1) * cols));
}

/** ``rows`` with the panel ``key`` moved to ``side`` of ``target``. */
function movePanel(rows: string[][], key: string, target: string, side: DropSide): string[][] {
  const out = rows.map((r) => r.filter((k) => k !== key));
  const ri = out.findIndex((r) => r.includes(target));
  if (ri < 0) return rows;
  if (side === "left" || side === "right") {
    const row = [...out[ri]];
    row.splice(row.indexOf(target) + (side === "right" ? 1 : 0), 0, key);
    out[ri] = row;
  } else {
    out.splice(side === "above" ? ri : ri + 1, 0, [key]);
  }
  return out.filter((r) => r.length);
}

/** The side of the panel under the pointer: its top or bottom quarter makes a
 * new row, else the nearer half. */
function sideOf(e: DragEvent): DropSide {
  const r = e.currentTarget.getBoundingClientRect();
  const fy = (e.clientY - r.top) / r.height;
  if (fy < 0.25) return "above";
  if (fy > 0.75) return "below";
  return e.clientX - r.left < r.width / 2 ? "left" : "right";
}

/** Number of columns giving the largest panels (the smallest panel area,
 * images fitted into equal cells) in a ``w`` × ``h`` area. */
export function bestColumns(aspects: number[], w: number, h: number): number {
  const n = aspects.length;
  if (n < 2 || w <= 0 || h <= 0) return 1;
  let best = 1;
  let bestArea = -1;
  for (let c = 1; c <= n; c++) {
    const rows = Math.ceil(n / c);
    const cw = (w - GAP * (c - 1)) / c;
    const ch = (h - GAP * (rows - 1)) / rows;
    if (cw <= 0 || ch <= 0) continue;
    const area = Math.min(
      ...aspects.map((a) => {
        const iw = Math.min(cw, ch * a);
        return (iw * iw) / a;
      }),
    );
    if (area > bestArea * 1.001) {
      best = c;
      bestArea = area;
    }
  }
  return best;
}

/** One panel per image, in a grid; panels of the same pixel size share
 * pan and zoom. Clicking a panel makes it the one being edited. */
export function ViewerGrid({
  panels, active, onActivate, interaction, renderOverlay, onRemove, empty, scanFactor, fftView, layoutKey,
}: Props) {
  const areaRef = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = areaRef.current!;
    const ro = new ResizeObserver(() => setArea({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const cols = bestColumns(
    panels.map((p) => p.width / Math.max(1, p.height)),
    area.w,
    area.h,
  );
  // Rows of panels: as arranged by hand, else automatic.
  const saved = useUi((s) => (layoutKey ? s.gridLayouts[layoutKey] : undefined));
  const setGridLayout = useUi((s) => s.setGridLayout);
  const keys = panels.map((p) => p.layer.key);
  const rows = arrange(keys, saved, cols);
  const byKey = new Map(panels.map((p) => [p.layer.key, p]));
  const ordered = rows.flat().map((k) => byKey.get(k)!);

  // Moving a panel: dragged by the strip along its top edge, dropped beside
  // another panel or in a new row above or below it.
  const [dragged, setDragged] = useState<string | null>(null);
  const [drop, setDrop] = useState<{ key: string; side: DropSide } | null>(null);
  const movable = !!layoutKey && panels.length > 1;
  const dropProps = (key: string) =>
    !movable
      ? {}
      : {
          onDragOver: (e: DragEvent) => {
            if (!dragged || !e.dataTransfer.types.includes(PANEL_MIME)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            const side = sideOf(e);
            const next = key === dragged ? null : { key, side };
            if (next?.key !== drop?.key || next?.side !== drop?.side) setDrop(next);
          },
          onDragLeave: (e: DragEvent) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrop((d) => (d?.key === key ? null : d));
          },
          onDrop: (e: DragEvent) => {
            e.preventDefault();
            if (dragged && dragged !== key) setGridLayout(layoutKey!, movePanel(rows, dragged, key, sideOf(e)));
            setDragged(null);
            setDrop(null);
          },
        };
  const gripProps = (key: string) =>
    !movable
      ? null
      : {
          draggable: true,
          onDragStart: (e: DragEvent) => {
            e.dataTransfer.setData(PANEL_MIME, key);
            e.dataTransfer.effectAllowed = "move";
            // The whole panel as the drag image.
            const panel = e.currentTarget.parentElement;
            if (panel) e.dataTransfer.setDragImage(panel, e.nativeEvent.offsetX, e.nativeEvent.offsetY);
            setDragged(key);
          },
          onDragEnd: () => {
            setDragged(null);
            setDrop(null);
          },
        };

  // The edited panel publishes for the FFT (the first one if it is hidden).
  const publishKey = panels.some((p) => p.layer.key === active) ? active : panels[0]?.layer.key;

  // Fit and 1:1 for every distinct view (panels sharing a key follow each other).
  const controls = useRef(new Map<string, { current: ViewControls | null }>());
  const controlRef = (viewKey: string) => {
    let r = controls.current.get(viewKey);
    if (!r) controls.current.set(viewKey, (r = { current: null }));
    return r;
  };
  const eachView = (fn: (c: ViewControls) => void) => {
    for (const k of new Set(panels.map((p) => p.viewKey))) {
      const c = controls.current.get(k)?.current;
      if (c) fn(c);
    }
  };

  const first = ordered[0];
  const zoom = useDisplay((s) => (first ? s.views[first.viewKey]?.zoom : undefined));

  // The export dialog, opened on the panels shown at that moment.
  const [exporting, setExporting] = useState<{ panels: SnapshotPanel[]; cols: number[]; name: string } | null>(null);
  const saveImage = () => {
    const ready = (p: GridPanel) => !!(p.layer.frame && p.layer.display);
    const shown = ordered.filter(ready);
    if (!shown.length) return;
    // The figure keeps the rows as shown.
    const lengths = rows.map((r) => r.filter((k) => ready(byKey.get(k)!)).length).filter((n) => n > 0);
    setExporting({
      name: `${safeFileName(shown.map((p) => p.layer.label).join("+"))}.png`,
      cols: lengths,
      panels: shown.map((p) => ({
        label: p.layer.label,
        colour: isColour(p.layer.display!.colormap) ? p.layer.display!.colormap : undefined,
        layers: [{ key: p.layer.key, frame: p.layer.frame!, display: p.layer.display!, opacity: 1 }],
        width: p.width,
        height: p.height,
        calX: p.calX,
      })),
    });
  };
  const save = useRef(saveImage);
  save.current = saveImage;
  const onSave = useRef(() => save.current()).current;

  return (
    <div className="image-view">
      <div className="viewer-grid-wrap">
        <div ref={areaRef} className={`viewer-grid${dragged ? " moving" : ""}`} style={{ gap: GAP }}>
          {rows.map((row) => (
            <div key={row.join("|")} className="viewer-grid-row" style={{ gap: GAP }}>
              {row.map((k) => {
                const p = byKey.get(k)!;
                return (
                  <GridCell
                    key={k}
                    p={p}
                    active={k === publishKey}
                    first={p === first}
                    onActivate={() => onActivate(k)}
                    interaction={interaction}
                    renderOverlay={renderOverlay}
                    controlRef={controlRef(p.viewKey)}
                    onSave={onSave}
                    onRemove={onRemove?.(k) ?? null}
                    scanFactor={scanFactor}
                    fftView={fftView}
                    grip={gripProps(k)}
                    dropProps={dropProps(k)}
                    dropSide={drop?.key === k ? drop.side : null}
                    dragging={dragged === k}
                  />
                );
              })}
              {/* The automatic grid keeps its columns: a short last row is padded. */}
              {!saved && Array.from({ length: cols - row.length }, (_, i) => <div key={`pad${i}`} className="grid-pad" />)}
            </div>
          ))}
          {!panels.length && empty && <div className="image-message">{empty}</div>}
        </div>
      </div>
      <div className="readout">
        <span className="readout-pos muted">
          {panels.length} panels{first && ` · ${first.width} × ${first.height} px`}
        </span>
        <span className="readout-zoom">
          {zoom !== undefined && `${Math.round(zoom * 100)}%`}
          <button className="link-button" onClick={() => eachView((c) => c.fit())} title="Fit all panels to their windows (F, or double-click a panel)">
            Fit
          </button>
          <button className="link-button" onClick={() => eachView((c) => c.oneToOne())} title="One screen pixel per data pixel">
            1:1
          </button>
          {saved && (
            <button className="link-button" onClick={() => setGridLayout(layoutKey!, null)} title="Back to the automatic grid">
              Reset layout
            </button>
          )}
          <button
            className="link-button"
            onClick={saveImage}
            disabled={!panels.some((p) => p.layer.frame)}
            title="Save the grid as shown (colours, contrast, a label per panel) at full resolution as PNG"
          >
            Save PNG
          </button>
        </span>
      </div>
      {exporting && (
        <PngExportDialog
          panels={exporting.panels}
          cols={exporting.cols}
          grid
          name={exporting.name}
          onClose={() => setExporting(null)}
        />
      )}
    </div>
  );
}

function GridCell({
  p, active, first, onActivate, interaction, renderOverlay, controlRef, onSave, onRemove, scanFactor, fftView,
  grip, dropProps, dropSide, dragging,
}: {
  p: GridPanel;
  active: boolean;
  first: boolean;
  onActivate: () => void;
  interaction?: Interaction;
  renderOverlay?: (t: Transform, p: GridPanel) => ReactNode;
  controlRef: { current: ViewControls | null };
  onSave: () => void;
  onRemove: (() => void) | null;
  scanFactor?: number;
  fftView?: string;
  /** The strip along the top edge that moves the panel (null: it can't move). */
  grip: object | null;
  dropProps: object;
  /** A moved panel would land on this side. */
  dropSide: DropSide | null;
  dragging: boolean;
}) {
  const { layer } = p;
  const layers = useMemo(
    () => [{ ...layer, opacity: 1, visible: true }],
    [layer.key, layer.label, layer.frame, layer.display], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const colormap = layer.display?.colormap;
  return (
    <div
      className={`grid-panel${active ? " active" : ""}${dragging ? " dragging" : ""}`}
      onPointerDownCapture={onActivate}
      {...dropProps}
    >
      {grip && <div className="panel-grip" title="Drag to move the panel" {...grip} />}
      {dropSide && <div className={`panel-drop ${dropSide}`} aria-hidden />}
      <ImageView
        viewKey={p.viewKey}
        follow
        layers={layers}
        fullWidth={p.width}
        fullHeight={p.height}
        calX={p.calX}
        calY={p.calY}
        loading={p.loading ?? (layer.frame ? null : "Loading…")}
        error={p.error}
        interaction={interaction}
        renderOverlay={renderOverlay && ((t) => renderOverlay(t, p))}
        analysisKey={layer.key}
        publish={active}
        fftView={fftView}
        scaleBar={first}
        readout={false}
        hoverChip
        scanFactor={scanFactor}
        onSave={onSave}
        controlRef={controlRef}
      >
        <span className="panel-label">
          {colormap && isColour(colormap) && <span className="editing-swatch" style={{ background: colormap }} />}
          {layer.label}
          {onRemove && (
            <button
              className="panel-remove"
              title={`Remove ${layer.label} from the view`}
              aria-label={`Remove ${layer.label} from the view`}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={onRemove}
            >
              ×
            </button>
          )}
        </span>
      </ImageView>
    </div>
  );
}
