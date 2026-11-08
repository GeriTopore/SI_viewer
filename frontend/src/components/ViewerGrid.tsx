import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { downloadBlob, safeFileName } from "../download";
import { useDisplay } from "../state/display";
import { useLayers } from "../state/layers";
import { isColour } from "../viewer/colormaps";
import { gridPng } from "../viewer/snapshot";
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
}

const GAP = 6;

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
export function ViewerGrid({ panels, active, onActivate, interaction, renderOverlay, onRemove, empty }: Props) {
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

  const first = panels[0];
  const zoom = useDisplay((s) => (first ? s.views[first.viewKey]?.zoom : undefined));

  const saveImage = () => {
    const shown = panels.filter((p) => p.layer.frame && p.layer.display);
    if (!shown.length) return;
    const name = `${safeFileName(shown.map((p) => p.layer.label).join("+"))}.png`;
    gridPng(
      shown.map((p) => ({
        label: p.layer.label,
        colour: isColour(p.layer.display!.colormap) ? p.layer.display!.colormap : undefined,
        layer: { key: p.layer.key, frame: p.layer.frame!, display: p.layer.display!, opacity: 1 },
        width: p.width,
        height: p.height,
        calX: p.calX,
      })),
      Math.min(cols, shown.length),
    )
      .then((blob) => downloadBlob(blob, name))
      .catch((e) => useLayers.getState().notify(`Could not save the image: ${(e as Error).message}`));
  };
  const save = useRef(saveImage);
  save.current = saveImage;
  const onSave = useRef(() => save.current()).current;

  return (
    <div className="image-view">
      <div className="viewer-grid-wrap">
        <div
          ref={areaRef}
          className="viewer-grid"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: GAP }}
        >
          {panels.map((p, i) => (
            <GridCell
              key={p.layer.key}
              p={p}
              active={p.layer.key === publishKey}
              first={i === 0}
              onActivate={() => onActivate(p.layer.key)}
              interaction={interaction}
              renderOverlay={renderOverlay}
              controlRef={controlRef(p.viewKey)}
              onSave={onSave}
              onRemove={onRemove?.(p.layer.key) ?? null}
            />
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
    </div>
  );
}

function GridCell({
  p, active, first, onActivate, interaction, renderOverlay, controlRef, onSave, onRemove,
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
}) {
  const { layer } = p;
  const layers = useMemo(
    () => [{ ...layer, opacity: 1, visible: true }],
    [layer.key, layer.label, layer.frame, layer.display], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const colormap = layer.display?.colormap;
  return (
    <div className={`grid-panel${active ? " active" : ""}`} onPointerDownCapture={onActivate}>
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
        scaleBar={first}
        readout={false}
        hoverChip
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
