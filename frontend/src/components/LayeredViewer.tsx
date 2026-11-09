import { useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { dataVersion, type DatasetSummary } from "../api/client";
import { allDatasets, initialColormap, planeSize, setMapColour, useDatasets } from "../state/datasets";
import { DEFAULT_COLORMAP, useDisplay } from "../state/display";
import { DEFAULT_LAYER, baseLayerKey, useLayers } from "../state/layers";
import { isColour, swatchBackground } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";
import { imageGeometry, type ImageGeometry } from "../views/geometry";
import { autoContrast, useImageDisplay } from "../views/useImageDisplay";
import { useOverlayFrames } from "../views/useOverlayFrames";
import { ContrastControls } from "./ContrastControls";
import { ImageView, type Interaction, type Transform, type ViewLayer } from "./ImageView";
import { LayersPanel } from "./LayersPanel";
import { LineOverlay } from "./LineOverlay";
import { ProfilePanel } from "./ProfilePanel";
import { useProfiles, useViewProfiles } from "../state/profiles";
import { useLineTool } from "../views/useLineTool";
import { useUi, type BottomTab, type GridContext, type GridMode } from "../state/ui";
import { BottomPane } from "./BottomPane";
import { ViewerGrid, type GridPanel } from "./ViewerGrid";
import { useShortcuts } from "../keyboard";

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
  /** Version of the data in ``frame``: contrast is redone when it changes. */
  version?: string;
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
  /** Shown when there is nothing to draw. */
  empty?: string;
  /** Pointer tool and SVG overlay passed to the image view. */
  interaction?: Interaction;
  renderOverlay?: (t: Transform) => ReactNode;
  /** The line (profile) tool is active. Without this prop the viewer offers
   * its own Pan / Line tool buttons. */
  lineTool?: boolean;
  /** Points picked on the profile plot (spectrum images: their spectrum). */
  profilePick?: boolean;
  /** Shown right under the image (e.g. a stack's frame slider). */
  footer?: ReactNode;
  /** Tabs of the pane under the image besides "Line profile" (e.g. the
   * spectrum). Without them the pane only appears for line profiles. */
  bottomTabs?: { id: BottomTab; label: string; content: ReactNode }[];
  /** Maps can be overlaid here (drop target and hint); default true. */
  overlays?: boolean;
  /** Several images can be shown merged or as a grid; the choice is
   * remembered per kind of view. Without it they are always merged. */
  gridContext?: GridContext;
  /** Makes this layer the one being edited (``BASE_LAYER`` or an image's id)
   * when the object changes: lets the view link another control to the
   * layers (e.g. a window on the spectrum). */
  requestActive?: { key: string } | null;
  /** The layer being edited changed, or a layer was clicked (key as above). */
  onActiveChange?: (key: string) => void;
}

const BASE = "base";
/** Key of the base layer in ``requestActive`` / ``onActiveChange``. */
export const BASE_LAYER = BASE;

/** Contrast toolbar + image view composing a base image and highlighted maps. */
export function LayeredViewer({
  viewKey, base, images, geometry, loading, error, children, empty, interaction, renderOverlay,
  lineTool, profilePick, footer, bottomTabs = [], overlays = true, gridContext, requestActive, onActiveChange,
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
    base?.version ?? (baseDataset && dataVersion(baseDataset)),
  );

  // Each image gets auto contrast (and its element colour) once its data
  // arrives, and again when it's filtered.
  useEffect(() => {
    for (const d of images) {
      const f = frames[d.id];
      if (f) autoContrast(d.id, f, dataVersion(d), () => initialColormap(d));
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

  // Merged or grid: the user's choice for this kind of view, else the
  // default (maps over a spectrum image merge; plain images get a grid).
  // Merged needs images of one pixel size.
  const gridPref = useUi((s) => (gridContext ? s.gridMode[gridContext] : null));
  const setGridMode = useUi((s) => s.setGridMode);
  const sizes = layers.map((l) => {
    const d = l.key === BASE ? undefined : images.find((x) => x.id === l.key);
    if (!d) return { geometry, w: geometry.width, h: geometry.height };
    const [w, h] = planeSize(d);
    const own = w === geometry.width && h === geometry.height ? geometry : imageGeometry(d, "signal");
    return { geometry: own, w: own.width, h: own.height };
  });
  const sameSize = sizes.every((z) => z.w === sizes[0].w && z.h === sizes[0].h);
  const canSwitch = !!gridContext && layers.length > 1;
  const mode: GridMode = canSwitch
    ? !sameSize
      ? "grid"
      : (gridPref ?? (gridContext === "images" ? "grid" : "merged"))
    : "merged";
  const grid = mode === "grid";
  // Every layer gets a panel: switching to the grid shows any hidden ones again.
  const switchMode = (m: GridMode) => {
    if (m === "grid") {
      for (const l of layers) if (!l.visible) setProps(l.key === BASE ? baseLayerKey(viewKey) : l.key, { visible: true });
    }
    setGridMode(gridContext!, m);
  };
  useShortcuts({ g: () => (sameSize ? switchMode(grid ? "merged" : "grid") : false) }, canSwitch);

  // The toolbar edits the focused image if it's in the view, else the base.
  const [active, setActive] = useState<string>(BASE);
  const imageIds = images.map((d) => d.id).join(",");
  useEffect(() => {
    if (focusId && images.some((d) => d.id === focusId)) setActive(focusId);
  }, [focusId, imageIds]); // eslint-disable-line react-hooks/exhaustive-deps
  const activeLayer = layers.find((l) => l.key === active) ?? layers[0];
  // Tell the view which layer is edited (not on mount), and honour its requests.
  const onActive = useRef(onActiveChange);
  onActive.current = onActiveChange;
  const reported = useRef<string | undefined>(activeLayer?.key);
  useEffect(() => {
    if (reported.current === activeLayer?.key) return;
    reported.current = activeLayer?.key;
    if (activeLayer) onActive.current?.(activeLayer.key);
  }, [activeLayer?.key]); // eslint-disable-line react-hooks/exhaustive-deps
  const layerKeys = layers.map((l) => l.key).join(",");
  useEffect(() => {
    if (!requestActive || !layerKeys.split(",").includes(requestActive.key)) return;
    setActive(requestActive.key);
    if (requestActive.key !== BASE) setFocus(requestActive.key);
  }, [requestActive]); // eslint-disable-line react-hooks/exhaustive-deps
  const activate = (key: string) => {
    setActive(key);
    if (key !== BASE) setFocus(key);
    onActive.current?.(key);
  };
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

  // Profile lines: drawn with the line tool, profiled below the image.
  const imageTool = useProfiles((s) => s.imageTool);
  const setImageTool = useProfiles((s) => s.setImageTool);
  const drawingLines = lineTool ?? imageTool === "line";
  const lineInteraction = useLineTool(viewKey);
  const hasLines = useViewProfiles(viewKey).lines.length > 0;

  // Pane under the image: the given tabs, plus line profiles (with the line
  // tool, or once there are lines). Its height is a fraction of the viewer.
  const setBottomTab = useUi((s) => s.setBottomTab);
  const showProfiles = hasLines || drawingLines || bottomTabs.length > 0;
  const spectrumTab = bottomTabs.find((t) => t.id !== "alignment");
  const tabs = [
    ...bottomTabs,
    ...(showProfiles
      ? [
          {
            id: "profile" as const,
            label: "Line profile",
            content: (
              <ProfilePanel
                view={viewKey}
                layers={layers}
                calX={geometry.calX}
                calY={geometry.calY}
                title={base?.label ?? images[0]?.title ?? "image"}
                pickable={profilePick}
                onShowSpectrum={spectrumTab ? () => setBottomTab(spectrumTab.id) : undefined}
              />
            ),
          },
        ]
      : []),
  ];
  const viewerRef = useRef<HTMLDivElement>(null);
  // Image and stack views: their own tool keys.
  useShortcuts({ l: () => setImageTool("line"), h: () => setImageTool("pan") }, lineTool === undefined);

  const [dragOver, setDragOver] = useState(false);
  const accepts = (e: DragEvent) => overlays && e.dataTransfer.types.includes(DATASET_MIME);
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
        removable: l.key !== BASE,
      }))}
      active={activeLayer?.key ?? BASE}
      onActivate={activate}
      onChange={(id, patch) => setProps(id === BASE ? baseLayerKey(viewKey) : id, patch)}
      onColour={(id, colour) => {
        const d = id === BASE ? baseDataset : images.find((x) => x.id === id);
        if (d) setMapColour(d, colour);
        else baseDisplay.setDisplay({ colormap: colour });
      }}
      onRemove={(id) => id !== BASE && deselect(id)}
      opacity={!grid}
    />
  ) : null;

  return (
    <div
      ref={viewerRef}
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
      <div className="viewer-top">
        <div className="layer-toolbar">
          {lineTool === undefined && (
            <div className="segmented tool-buttons" role="group" aria-label="Image tool">
              {(["pan", "line"] as const).map((t) => (
                <button
                  key={t}
                  className={imageTool === t ? "active" : undefined}
                  aria-pressed={imageTool === t}
                  onClick={() => setImageTool(t)}
                  title={
                    t === "pan"
                      ? "Pan (or hold Space / middle-drag with any tool)"
                      : "Line profile: drag to draw (Shift snaps the angle), drag ends or the line to change it, Delete removes it"
                  }
                >
                  {t === "pan" ? "✋" : "╱"}
                </button>
              ))}
            </div>
          )}
          {canSwitch && (
            <div className="segmented" role="group" aria-label="Several images">
              {(["merged", "grid"] as const).map((m) => (
                <button
                  key={m}
                  className={mode === m ? "active" : undefined}
                  aria-pressed={mode === m}
                  disabled={m === "merged" && !sameSize}
                  onClick={() => switchMode(m)}
                  title={
                    m === "grid"
                      ? "Grid: one panel per image (G)"
                      : sameSize
                        ? "Merged: all images in one composite (G)"
                        : "Merged needs images of the same pixel size"
                  }
                >
                  {m === "merged" ? "Merged" : "Grid"}
                </button>
              ))}
            </div>
          )}
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
        {grid ? (
          <ViewerGrid
            panels={layers
              .map((layer, i): GridPanel => {
                const z = sizes[i];
                const first = sizes[0];
                return {
                  layer,
                  width: z.w,
                  height: z.h,
                  calX: z.geometry.calX,
                  calY: z.geometry.calY,
                  // Panels of the first image's size share the view's key.
                  viewKey: z.w === first.w && z.h === first.h ? viewKey : `${viewKey}:${z.w}x${z.h}`,
                  ...(layer.key === BASE && { loading, error }),
                };
              })}
            active={activeLayer?.key}
            onActivate={activate}
            interaction={drawingLines ? lineInteraction : interaction}
            renderOverlay={(t, p) => (
              <>
                {renderOverlay?.(t)}
                <LineOverlay t={t} view={viewKey} calX={p.calX} calY={p.calY} />
              </>
            )}
            onRemove={(key) => (key === BASE ? null : () => deselect(key))}
            empty="No images to show"
          />
        ) : (
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
            interaction={drawingLines ? lineInteraction : interaction}
            renderOverlay={(t) => (
              <>
                {renderOverlay?.(t)}
                <LineOverlay t={t} view={viewKey} calX={geometry.calX} calY={geometry.calY} />
              </>
            )}
            analysisKey={activeLayer?.key}
            hint={showPanel || !overlays ? undefined : "click or drag maps to overlay them"}
          >
            {children}
          </ImageView>

        )}
        {footer}
        {dragOver && <div className="drop-hint">Drop to overlay</div>}
      </div>
      {tabs.length > 0 && (
        <BottomPane tabs={tabs} viewerRef={viewerRef} resizeLabel="Resize the image and the pane below it" />
      )}
    </div>
  );
}
