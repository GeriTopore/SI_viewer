import { useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { contrastVersion, type DatasetSummary } from "../api/client";
import { allDatasets, cropDataset, initialColormap, planeSize, setMapColour, useDatasets } from "../state/datasets";
import { containsPoint, getSiState, shapeToRegion, useSi, useSiState, type Roi } from "../state/si";
import { useNavigatorTools } from "../views/si/useNavigatorTools";
import { RoiOverlay } from "./RoiOverlay";
import { useDisplay } from "../state/display";
import { DEFAULT_LAYER, baseLayerKey, useLayers } from "../state/layers";
import { useViewerImage } from "../state/viewerImage";
import { isColour } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";
import { imageGeometry, type ImageGeometry } from "../views/geometry";
import { autoContrast, useImageDisplay } from "../views/useImageDisplay";
import { useOverlayFrames } from "../views/useOverlayFrames";
import { ContrastControls } from "./ContrastControls";
import { ImageView, type Interaction, type Pt, type Transform, type ViewLayer } from "./ImageView";
import { LayersPanel } from "./LayersPanel";
import { LineOverlay } from "./LineOverlay";
import { FftBoxOverlay } from "./FftBoxOverlay";
import { FftCompare } from "./FftCompare";
import { SurveyRegions } from "./SurveyRegions";
import { HistogramPanel } from "./HistogramPanel";
import { ProfilePanel } from "./ProfilePanel";
import { getViewProfiles, useProfiles, useViewProfiles, type ProfileLine } from "../state/profiles";
import { lineAt, useLineTool } from "../views/useLineTool";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { AnnotationsToggle } from "./AnnotationsToggle";
import { lineMenuItems } from "./ProfilePanel";
import { useFftBoxTool } from "../views/useFftBoxTool";
import { useUi, type BottomTab, type GridContext, type GridMode } from "../state/ui";
import { BottomPane } from "./BottomPane";
import { ViewerGrid, type GridPanel } from "./ViewerGrid";
import { FitQuantitySwitch } from "./FitQuantitySwitch";
import { stepKeys, useShortcuts } from "../keyboard";
import { stretch } from "../views/si/scan";

/** Tooltip of the FFT box tool (the image views' and the navigator's). */
export const FFT_TOOL_TITLE =
  "FFT region: drag to draw a square (its FFT shows in the FFT panel), drag a corner or the inside to change it, Delete removes it";

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
  /** The colormap it starts with (e.g. a DPC stack's circular one). */
  colormap?: () => string;
  /** The dataset it shows (e.g. a stack), for its region menu (crop). */
  source?: DatasetSummary;
}

interface Props {
  /** Pan/zoom key (and base layer key). */
  viewKey: string;
  base: BaseImage | null;
  /** Highlighted image datasets, drawn on top of the base in colour. */
  images: DatasetSummary[];
  /** Other stacks shown with the base, at the same frame (their frames come
   * from the view): a layer each, before the images, with the contrast kept
   * under ``displayKey``. */
  stacks?: StackLayer[];
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
   * its own Pan / Line / FFT tool buttons. */
  lineTool?: boolean;
  /** The FFT box tool is active (with ``lineTool``: the view's own tool bar). */
  fftTool?: boolean;
  /** Points picked on the profile plot (spectrum images: their spectrum). */
  profilePick?: boolean;
  /** Cuts a line scan along a profile line (spectrum images). */
  onLineScan?: (line: ProfileLine) => void;
  /** Entries of the menu of a right-click on the image (``p`` in image
   * pixels) away from a profile line; none: no menu. */
  imageMenu?: (p: Pt) => MenuItem[];
  /** The view's regions for the Histogram tab when the view keeps its own
   * (a spectrum image's; ``scale``: image pixels per region unit). Views with
   * the viewer's own tools have regions of their own. */
  histogramRegions?: { rois: Roi[]; scale: number };
  /** Shown right under the image (e.g. a stack's frame slider). */
  footer?: ReactNode;
  /** Tabs of the pane under the image besides "Line profile" (e.g. the
   * spectrum). Without them the pane only appears for line profiles. */
  bottomTabs?: { id: BottomTab; label: string; content: ReactNode }[];
  /** In focus mode's navigator flap: a menu in the image area's top-left
   * corner (e.g. which navigator is shown), fixed there whatever the zoom. */
  focusCorner?: ReactNode;
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
  /** The base image is k times finer than the scan the ``images`` (maps) are
   * on (a sub-scan navigator): they are stretched over it, k × k image pixels
   * per map pixel (``views/si/scan.ts``). Default 1. */
  scanFactor?: number;
}

/** A stack shown next to the base stack (see ``stacks``), or one frame of a
 * stack shown as a panel of its own (several from one dataset, each with its
 * own ``key``, e.g. a PCA's components). */
export interface StackLayer {
  dataset: DatasetSummary;
  /** Layer key (default: the dataset's id). */
  key?: string;
  /** Panel label (default: the dataset's title). */
  label?: string;
  displayKey: string;
  frame: Frame | null;
  /** Contrast is redone when this changes (not for the next frame). */
  version?: string;
  /** Its panel can be closed (deselecting the dataset); default true. */
  removable?: boolean;
}

/** One layer of the view besides the base: a stack's frame or a map. */
interface Entry {
  key: string;
  dataset: DatasetSummary;
  label: string;
  frame: Frame | null;
  displayKey: string;
  stack: boolean;
  removable: boolean;
}

const BASE = "base";
/** Key of the base layer in ``requestActive`` / ``onActiveChange``. */
export const BASE_LAYER = BASE;

const TOOL_LABELS = { line: "╱", fft: "FFT", rect: "▭", ellipse: "◯", polygon: "⬠" } as const;
const TOOL_TITLES = {
  line: "Line profile: drag to draw (Shift snaps the angle), drag ends or the line to change it, Delete removes it",
  fft: FFT_TOOL_TITLE,
  rect: "Rectangle region: drag to draw, drag inside to move; right-click it to crop",
  ellipse: "Ellipse region: drag to draw, drag inside to move",
  polygon: "Polygon region: click vertices; click the first, double-click or Enter to close",
} as const;

/** Contrast toolbar + image view composing a base image and highlighted maps. */
/** Narrowest the navigator's flap gets in focus mode (room for its toolbar). */
export const FLAP_MIN_W = 220;

export function LayeredViewer({
  viewKey, base, images: maps, stacks = [], geometry, loading, error, children, empty, interaction, renderOverlay,
  lineTool, fftTool, profilePick, onLineScan, imageMenu, footer, bottomTabs = [], focusCorner, overlays = true, gridContext, requestActive, onActiveChange,
  histogramRegions,
  scanFactor = 1,
}: Props) {
  const focusId = useDatasets((s) => s.focusId);
  const { setFocus, deselect, addToSelection } = useDatasets.getState();
  const files = useDatasets((s) => s.files);
  const displays = useDisplay((s) => s.display);
  const setDisplay = useDisplay((s) => s.setDisplay);
  const layerProps = useLayers((s) => s.props);
  const setProps = useLayers((s) => s.setProps);

  const overlayFrames = useOverlayFrames(maps);
  // The stacks come first (their frames are given), then the maps.
  const entries: Entry[] = [
    ...stacks.map((s) => ({
      key: s.key ?? s.dataset.id, dataset: s.dataset, label: s.label ?? s.dataset.title, frame: s.frame,
      displayKey: s.displayKey, stack: true, removable: s.removable ?? true,
    })),
    ...maps.map((d) => ({
      key: d.id, dataset: d, label: d.title, frame: overlayFrames[d.id] ?? null, displayKey: d.id, stack: false,
      removable: true,
    })),
  ];
  const entryOf = (key: string) => entries.find((e) => e.key === key);
  // The datasets shown (once each).
  const images = [...new Map(entries.map((e) => [e.dataset.id, e.dataset])).values()];
  const displayKeyOf = (key: string) => entryOf(key)?.displayKey ?? key;
  const isStack = (key: string) => !!entryOf(key)?.stack;
  const baseDataset = base?.dataset;
  const baseDisplay = useImageDisplay(
    base?.key ?? `${viewKey}:none`,
    base?.frame ?? null,
    base?.colormap ?? (baseDataset ? () => initialColormap(baseDataset) : undefined),
    base?.version ?? (baseDataset && contrastVersion(baseDataset)),
  );

  // Each image gets auto contrast (and its element colour) once its data
  // arrives, and again when it's filtered.
  useEffect(() => {
    for (const d of maps) {
      const f = overlayFrames[d.id];
      if (f) autoContrast(d.id, f, contrastVersion(d), () => initialColormap(d));
    }
    for (const s of stacks) {
      if (s.frame) autoContrast(s.displayKey, s.frame, s.version ?? contrastVersion(s.dataset), () => initialColormap(s.dataset));
    }
  }); // every render: cheap, it only acts when a version changes

  const baseProps = layerProps[baseLayerKey(viewKey)] ?? DEFAULT_LAYER;
  // Survey images show where their spectrum image was taken.
  const surveys = [...(baseDataset ? [baseDataset] : []), ...images].filter(
    (d) => d.survey?.regions.length && (layerProps[d.id]?.visible ?? true),
  );
  const layers: ViewLayer[] = [
    ...(base
      ? [{ key: BASE, label: base.label, frame: base.frame, display: baseDisplay.display, ...baseProps }]
      : []),
    ...entries.map((e) => ({
      key: e.key,
      label: e.label,
      frame: e.frame ? stretch(e.frame, scanFactor) : null,
      display: displays[e.displayKey],
      ...(layerProps[e.key] ?? DEFAULT_LAYER),
    })),
  ];

  // Merged or grid: the user's choice for this kind of view, else the
  // default (maps over a spectrum image merge; plain images get a grid).
  // Merged needs images of one pixel size.
  const gridPref = useUi((s) => (gridContext ? s.gridMode[gridContext] : null));
  // In focus mode's navigator flap (this viewer has the pane below it):
  // merged only, no grid (too cluttered there).
  const focusNav = useUi((s) => s.focus === "bottom" && s.focusNav);
  const flapOpen = focusNav && bottomTabs.length > 0;
  const setGridMode = useUi((s) => s.setGridMode);
  const sizes = layers.map((l) => {
    const d = l.key === BASE ? undefined : entryOf(l.key)?.dataset;
    if (!d) return { geometry, w: geometry.width, h: geometry.height };
    const [w0, h0] = planeSize(d);
    const [w, h] = [w0 * scanFactor, h0 * scanFactor]; // stretched over the base
    const own = w === geometry.width && h === geometry.height ? geometry : imageGeometry(d, "signal");
    return { geometry: own, w: own.width, h: own.height };
  });
  const sameSize = sizes.every((z) => z.w === sizes[0].w && z.h === sizes[0].h);
  const canSwitch = !!gridContext && layers.length > 1 && !(flapOpen && sameSize);
  const mode: GridMode = canSwitch
    ? !sameSize
      ? "grid"
      : (gridPref ?? (gridContext === "si" ? "merged" : "grid"))
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
  const imageIds = entries.map((e) => e.key).join(",");
  useEffect(() => {
    if (focusId && entries.some((e) => e.key === focusId)) setActive(focusId);
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
    if (requestActive.key !== BASE) setFocus(entryOf(requestActive.key)?.dataset.id ?? requestActive.key);
  }, [requestActive]); // eslint-disable-line react-hooks/exhaustive-deps
  const activate = (key: string) => {
    setActive(key);
    if (key !== BASE) setFocus(entryOf(key)?.dataset.id ?? key);
    onActive.current?.(key);
  };
  const isBase = activeLayer?.key === BASE;
  const activeDataset = isBase ? baseDataset : activeLayer && entryOf(activeLayer.key)?.dataset;

  const onDisplayChange = (patch: Partial<DisplayParams>) => {
    if (!activeLayer) return;
    const key = isBase ? base!.key : displayKeyOf(activeLayer.key);
    if (activeDataset && !isStack(activeLayer.key) && patch.colormap && isColour(patch.colormap)) {
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
  // FFT boxes: squares whose FFT the FFT panel can show instead of the view.
  const drawingBoxes = lineTool === undefined ? imageTool === "fft" : !!fftTool;
  const boxInteraction = useFftBoxTool(viewKey, geometry.width, geometry.height);
  // Regions (rectangle, ellipse, polygon) drawn on images and stacks, kept per view.
  const ownRegions = lineTool === undefined;
  const regionTool = imageTool === "rect" || imageTool === "ellipse" || imageTool === "polygon" ? imageTool : null;
  const drawingRegions = ownRegions && !!regionTool;
  const regionTools = useNavigatorTools(viewKey, geometry.width, geometry.height, false, regionTool ?? "rect");
  const regionState = useSiState(viewKey);
  const annotationsHidden = useUi((s) => !!s.annotationsHidden[viewKey]);
  const toolInteraction0 = drawingLines
    ? lineInteraction
    : drawingBoxes
      ? boxInteraction
      : drawingRegions
        ? regionTools.interaction
        : interaction;
  // With another tool (or none), a press still selects the region under it
  // (its info in the Info tab) or, off the regions, deselects it.
  const activeInteraction: Interaction | undefined = ownRegions && !drawingRegions
    ? {
        ...toolInteraction0,
        onDown: (p, e, t) => {
          if (!annotationsHidden) regionTools.regionPress(p, t);
          return toolInteraction0?.onDown?.(p, e, t) ?? false;
        },
      }
    : toolInteraction0;
  // Shift +/-: the selected line one pixel wider or narrower (on a spectrum
  // image only with the line tool: otherwise its selected region is resized).
  useShortcuts(
    stepKeys((step) => {
      if (lineTool === false) return false;
      const { selected, lines } = getViewProfiles(viewKey);
      const line = lines.find((l) => l.id === selected);
      if (!line) return false;
      const width = Math.min(200, Math.max(1, Math.round(line.width) + step));
      if (width !== line.width) useProfiles.getState().updateLine(viewKey, line.id, { width });
      return true;
    }),
  );
  // Annotations hidden with the eye: drawing lines or boxes shows them again.
  useEffect(() => {
    if (drawingLines || drawingBoxes || drawingRegions) useUi.getState().setAnnotationsHidden(viewKey, false);
  }, [drawingLines, drawingBoxes, drawingRegions, viewKey]);
  // Right-clicking a profile line (with any tool) opens its menu; elsewhere
  // on the image, the view's own menu (if it has entries there).
  const [menu, setMenu] = useState<{ x: number; y: number; items: MenuItem[] } | null>(null);
  // A region right-clicked: crop the image shown to it (a rectangle), or delete it.
  const regionItems = (p: Pt): MenuItem[] => {
    if (!ownRegions || annotationsHidden) return [];
    const hits = getSiState(viewKey).rois.filter((r) => r.visible && containsPoint(r.shape, p.x, p.y));
    const target = isBase ? base?.source : activeLayer && entryOf(activeLayer.key)?.dataset;
    return hits.flatMap((r) => [
      ...(r.shape.kind === "rect" && target
        ? [{
            label: `Crop ${target.title} to ${r.name}`,
            title: "A new dataset of the pixels in this rectangle (a stack: every frame)",
            onSelect: () => void cropDataset(target, { region: shapeToRegion(r.shape) }),
          }]
        : []),
      { label: `Delete ${r.name}`, danger: true, onSelect: () => useSi.getState().deleteRois(viewKey, [r.id]) },
    ]);
  };
  const toolInteraction: Interaction | undefined = onLineScan || imageMenu || ownRegions
    ? {
        ...activeInteraction,
        onContextMenu: (p, e, t) => {
          const line = annotationsHidden || !onLineScan ? undefined : lineAt(viewKey, p, t.zoom);
          const items = line ? lineMenuItems(line, onLineScan!) : [...regionItems(p), ...(imageMenu?.(p) ?? [])];
          if (!items.length) return activeInteraction?.onContextMenu?.(p, e, t) ?? false;
          setMenu({ x: e.clientX, y: e.clientY, items });
          return true;
        },
      }
    : activeInteraction;
  // The FFTs of the view and its boxes, beside the image (when it is the one the FFT panel follows).
  const showCompare = useViewerImage((s) => s.compare && s.image?.fftView === viewKey);
  const hasLines = useViewProfiles(viewKey).lines.length > 0;

  // Pane under the image: the given tabs, plus line profiles (with the line
  // tool, or once there are lines), and the histogram of the edited layer
  // whenever there is a pane. Its height is a fraction of the viewer.
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
                onLineScan={onLineScan}
                onShowSpectrum={spectrumTab ? () => setBottomTab(spectrumTab.id) : undefined}
              />
            ),
          },
        ]
      : []),
  ];
  // With regions drawn, the histogram tab is offered (to see a region's).
  if (tabs.length || (ownRegions && regionState.rois.length > 0)) {
    // The layer's own pixels (not stretched over the scan).
    const own = isBase ? (base?.frame ?? null) : activeLayer ? (entryOf(activeLayer.key)?.frame ?? null) : null;
    tabs.push({
      id: "histogram" as const,
      label: "Histogram",
      content: (
        <HistogramPanel
          frame={own}
          label={activeLayer?.label ?? ""}
          display={activeLayer?.display}
          regions={ownRegions ? { rois: regionState.rois, scale: 1 } : histogramRegions}
        />
      ),
    });
  }
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
  const showPanel = entries.length > 0;
  const panel = showPanel ? (
    <LayersPanel
      layers={layers.map((l) => ({
        id: l.key,
        label: l.label,
        visible: l.visible,
        opacity: l.opacity,
        display: l.display,
        removable: l.key !== BASE && (entryOf(l.key)?.removable ?? true),
      }))}
      active={activeLayer?.key ?? BASE}
      onActivate={activate}
      onChange={(id, patch) => setProps(id === BASE ? baseLayerKey(viewKey) : id, patch)}
      onColour={(id, colour) => {
        const d = id === BASE ? baseDataset : entryOf(id)?.dataset;
        if (d && isStack(id)) setDisplay(displayKeyOf(id), { colormap: colour });
        else if (d) setMapColour(d, colour);
        else baseDisplay.setDisplay({ colormap: colour });
      }}
      onRemove={(id) => id !== BASE && (entryOf(id)?.removable ?? true) && deselect(entryOf(id)?.dataset.id ?? id)}
      opacity={!grid}
    />
  ) : null;

  // Focus mode with the navigator pulled out: the image goes into the flap,
  // sized to the image's shape (no dead space around it): as tall as the
  // window allows and as wide as that needs, or narrower and shorter for a
  // wide image (--focus-nav-w / --focus-nav-h, which the pane follows).
  const flapW = useUi((s) => s.focusNavW);
  const topRef = useRef<HTMLDivElement>(null);
  const lifted = focusNav && tabs.length > 0;
  useEffect(() => {
    if (!lifted || !geometry.width || !geometry.height) return;
    const root = document.documentElement.style;
    let frame = 0;
    const fit = (again: number) => {
      const top = topRef.current;
      const img = top?.querySelector<HTMLElement>(".image-view");
      if (!top || !img) return;
      // What the flap holds besides the image (toolbar, footer, padding).
      const chromeW = top.clientWidth - img.clientWidth;
      const chromeH = top.clientHeight - img.clientHeight;
      const maxH = window.innerHeight * 0.92;
      // The most it takes: as dragged, else up to 30 % of the window.
      const maxW = flapW ?? Math.min(560, Math.max(240, window.innerWidth * 0.3));
      const aspect = geometry.width / geometry.height;
      let ih = maxH - chromeH;
      let iw = ih * aspect;
      if (iw + chromeW > maxW) {
        iw = maxW - chromeW;
        ih = iw / aspect;
      }
      root.setProperty("--focus-nav-w", `${Math.round(Math.max(FLAP_MIN_W, iw + chromeW))}px`);
      root.setProperty("--focus-nav-h", `${Math.round(Math.min(maxH, ih + chromeH))}px`);
      // The toolbar may wrap at the new width: measure once more.
      if (again) frame = requestAnimationFrame(() => fit(again - 1));
    };
    const refit = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => fit(1));
    };
    refit();
    window.addEventListener("resize", refit);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", refit);
      root.removeProperty("--focus-nav-w");
      root.removeProperty("--focus-nav-h");
    };
  }, [lifted, geometry.width, geometry.height, flapW]);
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
      <div ref={topRef} className={`viewer-top${lifted ? " focus-nav-lifted" : ""}`}>
        <div className="layer-toolbar">
          {lineTool === undefined && (
            <div className="segmented tool-buttons" role="group" aria-label="Image tool">
              {/* A tool is switched on and off; with none on, dragging pans. */}
              {(["line", "fft", "rect", "ellipse", "polygon"] as const).map((t) => (
                <button
                  key={t}
                  className={imageTool === t ? "active" : undefined}
                  aria-pressed={imageTool === t}
                  onClick={() => setImageTool(imageTool === t ? "pan" : t)}
                  title={`${TOOL_TITLES[t]} · click again to pan`}
                >
                  {TOOL_LABELS[t]}
                </button>
              ))}
            </div>
          )}
          {lineTool === undefined && <AnnotationsToggle view={viewKey} />}
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
          <FitQuantitySwitch ds={baseDataset ? [baseDataset, ...images] : images} />
          <ContrastControls
            frame={activeLayer?.frame ?? null}
            display={activeLayer?.display}
            onChange={onDisplayChange}
          />
        </div>
        <div className="viewer-row">
          {lifted && focusCorner && <div className="focus-corner">{focusCorner}</div>}
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
              interaction={toolInteraction}
              fftView={viewKey}
              layoutKey={viewKey}
              renderOverlay={(t, p) => (
                <>
                  {renderOverlay?.(t)}
                  {!annotationsHidden && (
                    <>
                      <FftBoxOverlay t={t} view={viewKey} />
                      {ownRegions && (
                        <RoiOverlay t={t} state={regionState} cursorSize={1} draft={regionTools.draft} hover={regionTools.hover} />
                      )}
                      {surveys.filter((d) => d.id === p.layer.key).map((d) => (
                        <SurveyRegions key={d.id} t={t} regions={d.survey!.regions} />
                      ))}
                      <LineOverlay t={t} view={viewKey} calX={p.calX} calY={p.calY} />
                    </>
                  )}
                </>
              )}
              onRemove={(key) =>
                key === BASE || !(entryOf(key)?.removable ?? true) ? null : () => deselect(entryOf(key)?.dataset.id ?? key)
              }
              empty="No images to show"
              scanFactor={scanFactor}
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
              interaction={toolInteraction}
              fftView={viewKey}
              renderOverlay={(t) => (
                <>
                  {renderOverlay?.(t)}
                  {!annotationsHidden && (
                    <>
                      <FftBoxOverlay t={t} view={viewKey} />
                      {ownRegions && (
                        <RoiOverlay t={t} state={regionState} cursorSize={1} draft={regionTools.draft} hover={regionTools.hover} />
                      )}
                      {surveys.map((d) => (
                        <SurveyRegions key={d.id} t={t} regions={d.survey!.regions} />
                      ))}
                      <LineOverlay t={t} view={viewKey} calX={geometry.calX} calY={geometry.calY} />
                    </>
                  )}
                </>
              )}
              analysisKey={activeLayer?.key}
              scanFactor={scanFactor}
              hint={showPanel || !overlays ? undefined : "click or drag maps to overlay them"}
            >
              {children}
            </ImageView>

          )}
          {showCompare && <FftCompare view={viewKey} />}
        </div>
        {footer}
        {dragOver && <div className="drop-hint">Drop to overlay</div>}
      </div>
      {tabs.length > 0 && (
        <BottomPane tabs={tabs} viewerRef={viewerRef} resizeLabel="Resize the image and the pane below it" />
      )}
      {menu && <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)} items={menu.items} />}
    </div>
  );
}
