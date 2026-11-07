import { useEffect, useState, type DragEvent, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import type { DatasetSummary } from "../api/client";
import { allDatasets, initialColormap, setMapColour, useDatasets } from "../state/datasets";
import { DEFAULT_COLORMAP, useDisplay } from "../state/display";
import { DEFAULT_LAYER, baseLayerKey, useLayers } from "../state/layers";
import { autoRange, sortedSample } from "../viewer/contrast";
import { isColour, swatchBackground } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";
import type { ImageGeometry } from "../views/geometry";
import { useImageDisplay } from "../views/useImageDisplay";
import { useOverlayFrames } from "../views/useOverlayFrames";
import { ContrastControls } from "./ContrastControls";
import { ImageView, type Interaction, type Transform, type ViewLayer } from "./ImageView";
import { LayersPanel } from "./LayersPanel";

/** Drag-and-drop type for datasets dragged from the sidebar. */
export const DATASET_MIME = "application/x-si-viewer-dataset";

/** The image a view is built on (an SI navigator, a stack frame). */
export interface BaseImage {
  /** Display-settings key (the dataset id when ``dataset`` is set). */
  key: string;
  label: string;
  /** The image ``key`` describes, or null while it loads. */
  frame: Frame | null;
  /** The image dataset shown as the base, if it is one (e.g. a map as SI
   * navigator): it then keeps its own contrast and element colour. */
  dataset?: DatasetSummary;
}

interface Props {
  /** Pan/zoom key (and base layer key). */
  viewKey: string;
  base: BaseImage | null;
  /** Highlighted image datasets, drawn on top of the base in colour. */
  images: DatasetSummary[];
  geometry: ImageGeometry;
  loading?: string | null;
  error?: string | null;
  /** Top-left overlay, e.g. a frame label. */
  children?: ReactNode;
  /** If given, the base layer can be removed (× in the layer list). */
  onRemoveBase?: () => void;
  /** Shown when there is nothing to draw. */
  empty?: string;
  /** Pointer tool and SVG overlay passed to the image view. */
  interaction?: Interaction;
  renderOverlay?: (t: Transform) => ReactNode;
}

const BASE = "base";

/** Contrast toolbar + image view composing a base image and highlighted maps. */
export function LayeredViewer({
  viewKey, base, images, geometry, loading, error, children, onRemoveBase, empty, interaction, renderOverlay,
}: Props) {
  const focusId = useDatasets((s) => s.focusId);
  const { setFocus, deselect, addToSelection } = useDatasets.getState();
  const files = useDatasets((s) => s.files);
  const displays = useDisplay((s) => s.display);
  const setDisplay = useDisplay((s) => s.setDisplay);
  const layerProps = useLayers((s) => s.props);
  const setProps = useLayers((s) => s.setProps);

  const frames = useOverlayFrames(images);
  const baseDataset = base?.dataset;
  const baseDisplay = useImageDisplay(
    base?.key ?? `${viewKey}:none`,
    base?.frame ?? null,
    baseDataset ? () => initialColormap(baseDataset) : undefined,
  );

  // Each image gets auto contrast (and its element colour) once its data arrives.
  useEffect(() => {
    for (const d of images) {
      const f = frames[d.id];
      if (f && !useDisplay.getState().display[d.id]) {
        const [lo, hi] = autoRange(sortedSample(f.data));
        useDisplay.getState().initDisplay(d.id, { lo, hi, gamma: 1, colormap: initialColormap(d) });
      }
    }
  }, [frames, images]);

  const baseProps = layerProps[baseLayerKey(viewKey)] ?? DEFAULT_LAYER;
  const layers: ViewLayer[] = [
    ...(base
      ? [{ key: BASE, label: base.label, frame: base.frame, display: baseDisplay.display, ...baseProps }]
      : []),
    ...images.map((d) => ({
      key: d.id,
      label: d.title,
      frame: frames[d.id] ?? null,
      display: displays[d.id],
      ...(layerProps[d.id] ?? DEFAULT_LAYER),
    })),
  ];

  // The toolbar edits the focused image if it's in the view, else the base.
  const [active, setActive] = useState<string>(BASE);
  useEffect(() => {
    if (focusId && images.some((d) => d.id === focusId)) setActive(focusId);
  }, [focusId, images]);
  const activeLayer = layers.find((l) => l.key === active) ?? layers[0];
  const isBase = activeLayer?.key === BASE;
  const activeDataset = isBase ? baseDataset : images.find((d) => d.id === activeLayer?.key);

  const onDisplayChange = (patch: Partial<DisplayParams>) => {
    if (!activeLayer) return;
    const key = isBase ? base!.key : activeLayer.key;
    if (activeDataset && patch.colormap && isColour(patch.colormap)) {
      // A solid colour on an element map becomes that element's colour.
      setMapColour(activeDataset, patch.colormap);
      const rest = { ...patch };
      delete rest.colormap;
      if (Object.keys(rest).length) setDisplay(key, rest);
    } else setDisplay(key, patch);
  };

  const [dragOver, setDragOver] = useState(false);
  const accepts = (e: DragEvent) => e.dataTransfer.types.includes(DATASET_MIME);
  const onDrop = (e: DragEvent) => {
    setDragOver(false);
    const id = e.dataTransfer.getData(DATASET_MIME);
    const d = allDatasets(files).find((x) => x.id === id);
    if (d) {
      e.preventDefault();
      addToSelection(d);
    }
  };

  // The layer list appears as soon as any map is shown.
  const showPanel = images.length > 0;
  const panel = showPanel ? (
    <LayersPanel
      layers={layers.map((l) => ({
        id: l.key,
        label: l.label,
        visible: l.visible,
        opacity: l.opacity,
        display: l.display,
        removable: l.key !== BASE || !!onRemoveBase,
      }))}
      active={activeLayer?.key ?? BASE}
      onActivate={(id) => {
        setActive(id);
        if (id !== BASE) setFocus(id);
      }}
      onChange={(id, patch) => setProps(id === BASE ? baseLayerKey(viewKey) : id, patch)}
      onColour={(id, colour) => {
        const d = id === BASE ? baseDataset : images.find((x) => x.id === id);
        if (d) setMapColour(d, colour);
        else baseDisplay.setDisplay({ colormap: colour });
      }}
      onRemove={(id) => (id === BASE ? onRemoveBase?.() : deselect(id))}
    />
  ) : null;

  return (
    <div
      className={`layered-viewer${dragOver ? " drop-target" : ""}`}
      onDragOver={(e) => {
        if (!accepts(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        setDragOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragOver(false);
      }}
      onDrop={onDrop}
    >
      <div className="layer-toolbar">
        {showPanel && activeLayer && (
          <span className="editing-chip" title="Layer being edited (pick another in the Layers panel)">
            <span
              className="editing-swatch"
              style={{ background: swatchBackground(activeLayer.display?.colormap ?? DEFAULT_COLORMAP) }}
            />
            {activeLayer.label}
          </span>
        )}
        <ContrastControls
          frame={activeLayer?.frame ?? null}
          display={activeLayer?.display}
          onChange={onDisplayChange}
        />
      </div>
      <ImageView
        viewKey={viewKey}
        layers={layers}
        fullWidth={geometry.width}
        fullHeight={geometry.height}
        calX={geometry.calX}
        calY={geometry.calY}
        loading={loading}
        error={error}
        empty={layers.length ? null : (empty ?? null)}
        panel={panel}
        interaction={interaction}
        renderOverlay={renderOverlay}
        hint={showPanel ? undefined : "click or drag maps to overlay them"}
      >
        {children}
      </ImageView>
      {dragOver && <div className="drop-hint">Drop to overlay</div>}
    </div>
  );
}
