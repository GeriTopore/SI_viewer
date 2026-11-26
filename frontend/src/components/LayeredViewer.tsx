import { useEffect, useId, useRef, useState, type DragEvent, type ReactNode } from "react";
import type { Frame } from "../api/arrays";
import { contrastVersion, type DatasetSummary } from "../api/client";
import { allDatasets, cropDataset, initialColormap, planeSize, setMapColour, useDatasets } from "../state/datasets";
import { containsPoint, getSiState, shapeToRegion, useSi, useSiState, type Roi } from "../state/si";
import { useNavigatorTools } from "../views/si/useNavigatorTools";
import { RoiOverlay } from "./RoiOverlay";
import { useDisplay } from "../state/display";
import { DEFAULT_LAYER, baseLayerKey, useLayers } from "../state/layers";
import { useViewerImage } from "../state/viewerImage";
import { useLayersTab } from "../state/layersTab";
import { isColour } from "../viewer/colormaps";
import type { DisplayParams } from "../viewer/renderer";
import { imageGeometry, type ImageGeometry } from "../views/geometry";
import { autoContrast, useImageDisplay } from "../views/useImageDisplay";
import { useOverlayFrames } from "../views/useOverlayFrames";
import { ContrastControls } from "./ContrastControls";
import { ImageView, type Interaction, type Pt, type Transform, type ViewLayer } from "./ImageView";
import { LayersPanel, type LayersPanelProps, type PanelLayer } from "./LayersPanel";
import { LineOverlay } from "./LineOverlay";
import { FftBoxOverlay } from "./FftBoxOverlay";
import { FftCompare } from "./FftCompare";
import { SurveyRegions } from "./SurveyRegions";
import { HistogramPanel } from "./HistogramPanel";
import { ProfilePanel } from "./ProfilePanel";
import { getViewProfiles, useProfiles, useViewProfiles, type ProfileLine } from "../state/profiles";
import { lineAt, useLineTool } from "../views/useLineTool";
import { ContextMenu, type MenuItem } from "./ContextMenu";
import { LoupeToggle } from "./Loupe";
import { CoarseningMarker, PhaseContours } from "./PhaseContours";
import {
  ColumnsOverlay, foundLayerKey, foundPanelLayer, imageOfItem, isFoundKey, itemInfo, itemPanelLayer, itemProps,
  useItemContrast,
} from "./ColumnsOverlay";
import { useColumns } from "../state/columns";
import { neighbourhoodsOf } from "../state/neighbourhoods";
import { NeighbourhoodPlot } from "./NeighbourhoodPlot";
import { CellPlot } from "./CellPlot";
import { useCell } from "../state/cell";
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
  /** Column items (kemstem: Columns, sublattices, neighbourhoods, bonds),
   * drawn over the image they came from (one of ``images``): a panel each
   * in the grid with the image under it, layers over it when merged. */
  items?: DatasetSummary[];
  /** Images shown only under items: drawn when merged, no panel of their own. */
  implied?: string[];
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
  /** ``key``: the layer of the grid panel drawn on (undefined when merged). */
  renderOverlay?: (t: Transform, key?: string) => ReactNode;
  /** Always a grid (no merged choice), e.g. the 4D view's panels. */
  gridOnly?: boolean;
  /** Merged: the layers' colours added (default) or the brighter kept. */
  blend?: "add" | "max";
  /** The line (profile) tool is active. Without this prop the viewer offers
   * its own Pan / Line / FFT tool buttons. */
  lineTool?: boolean;
  /** The FFT box tool is active (with ``lineTool``: the view's own tool bar). */
  fftTool?: boolean;
  /** Show the viewer's tool buttons (default true). False when another viewer
   * beside it shows them: the tool chosen there (shared) works here too. */
  toolButtons?: boolean;
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
  /** Publish the layers to this section of the Layers tab (a view of several
   * panes, e.g. 4D-STEM's "Diffraction pattern"), not as its only list. */
  layersSection?: string;
  /** Rows the view adds to the Layers tab after its images (marks it draws
   * itself: a detector, disks, cursors), with their own actions. */
  extraLayers?: ExtraLayers;
}

/** Layers tab rows a view draws itself (see ``extraLayers``). */
export interface ExtraLayers {
  rows: PanelLayer[];
  onActivate?: (id: string) => void;
  onChange: (id: string, patch: { visible?: boolean; opacity?: number }) => void;
  onColour: (id: string, colour: string) => void;
  onRemove?: (id: string) => void;
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
  /** The panel's size and calibration (default: the dataset's image plane),
   * e.g. a 4D-STEM dataset's virtual image on its scan. */
  geometry?: ImageGeometry;
  /** Its starting colormap or colour (default: the dataset's). */
  colormap?: string;
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
  geometry?: ImageGeometry;
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

/** The viewer's image tools (line, FFT box, rectangle, ellipse, polygon):
 * one is switched on and off; with none on, dragging pans. The tool is
 * shared by every viewer (e.g. shown once over the 4D view's two panes). */
export function ImageToolButtons() {
  const imageTool = useProfiles((s) => s.imageTool);
  const setImageTool = useProfiles((s) => s.setImageTool);
  return (
    <div className="segmented tool-buttons" role="group" aria-label="Image tool">
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
  );
}

/** Contrast toolbar + image view composing a base image and highlighted maps. */
/** Narrowest the navigator's flap gets in focus mode (room for its toolbar). */
export const FLAP_MIN_W = 220;

export function LayeredViewer({
  viewKey, base, images: maps, items = [], implied = [], stacks = [], geometry, loading, error, children, empty, interaction, renderOverlay,
  gridOnly = false, blend, lineTool, fftTool, toolButtons = true, profilePick, onLineScan, imageMenu, footer, bottomTabs = [], focusCorner, overlays = true, gridContext, requestActive, onActiveChange,
  histogramRegions,
  scanFactor = 1,
  layersSection,
  extraLayers,
}: Props) {
  const focusId = useDatasets((s) => s.focusId);
  const { setFocus, deselect, addToSelection } = useDatasets.getState();
  const files = useDatasets((s) => s.files);
  const cellStates = useCell((s) => s.byImage);
  const displays = useDisplay((s) => s.display);
  const setDisplay = useDisplay((s) => s.setDisplay);
  const layerProps = useLayers((s) => s.props);
  const setProps = useLayers((s) => s.setProps);

  const overlayFrames = useOverlayFrames(maps);
  // The stacks come first (their frames are given), then the maps.
  const entries: Entry[] = [
    ...stacks.map((s) => ({
      key: s.key ?? s.dataset.id, dataset: s.dataset, label: s.label ?? s.dataset.title, frame: s.frame,
      displayKey: s.displayKey, stack: true, removable: s.removable ?? true, geometry: s.geometry,
    })),
    ...maps.map((d) => ({
      key: d.id, dataset: d, label: d.title, frame: overlayFrames[d.id] ?? null, displayKey: d.id, stack: false,
      removable: true,
    })),
  ];
  const entryOf = (key: string) => entries.find((e) => e.key === key);
  // The datasets shown (once each).
  const images = [...new Map(entries.map((e) => [e.dataset.id, e.dataset])).values()];
  const displayKeyOf = (key: string) => resolve(key)?.displayKey ?? key;
  const isStack = (key: string) => !!resolve(key)?.stack;
  // Phase maps (kemstem) get its contours drawn over them.
  const isPhase = (key: string) => entryOf(key)?.dataset.atomic?.output === "phase";
  // Fourier-filtered maps (kemstem): the coarsening radius, drawn as its marker.
  const coarseningOf = (key: string) => entryOf(key)?.dataset.atomic?.coarsening;
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
      if (s.frame) autoContrast(s.displayKey, s.frame, s.version ?? contrastVersion(s.dataset), () => s.colormap ?? initialColormap(s.dataset));
    }
  }); // every render: cheap, it only acts when a version changes

  const baseProps = layerProps[baseLayerKey(viewKey)] ?? DEFAULT_LAYER;
  // Survey images show where their spectrum image was taken.
  const surveys = [...(baseDataset ? [baseDataset] : []), ...images].filter(
    (d) => d.survey?.regions.length && (layerProps[d.id]?.visible ?? true),
  );
  const layers: ViewLayer[] = [
    ...(base
      ? [{ key: BASE, datasetId: baseDataset?.id, label: base.label, frame: base.frame, display: baseDisplay.display, ...baseProps }]
      : []),
    ...entries.map((e) => ({
      key: e.key,
      datasetId: e.dataset.id,
      label: e.label,
      frame: e.frame ? stretch(e.frame, scanFactor) : null,
      display: displays[e.displayKey],
      ...(layerProps[e.key] ?? DEFAULT_LAYER),
    })),
  ];

  // Column items: over their image (when it is in the view). Their own
  // props say whether they show; in the grid, ``bg:<item>`` props the
  // image under them.
  const itemImage = (key: string) => {
    const item = items.find((d) => d.id === key);
    return item ? imageOfItem(files, item) : null;
  };
  const imageLayers = layers;
  const itemLayers: ViewLayer[] = items.flatMap((item) => {
    const info = itemInfo(files, item);
    const under = imageLayers.find((l) => l.key === info?.columns.parent_id);
    if (!info || !under) return [];
    const q = itemProps(layerProps[item.id], info);
    return [{ ...under, key: item.id, label: info.label, visible: q.visible, opacity: q.opacity }];
  });
  const isItem = (key: string) => itemLayers.some((l) => l.key === key);
  const allLayers = [...imageLayers, ...itemLayers];
  // The grid's panels: images (not those only under items) and items.
  const panelLayers = allLayers.filter((l) => !implied.includes(l.key));
  // An item's entry is its image's.
  const resolve = (key: string) => entryOf(key) ?? entryOf(itemImage(key) ?? "");
  // The images that have column guesses (Atomic columns process): a layer each.
  const foundById = useColumns((s) => s.byId);
  const foundImages = imageLayers.filter((l) => l.datasetId && foundById[l.datasetId]?.found?.length).map((l) => l.datasetId!);

  // Merged or grid: the user's choice for this kind of view, else the
  // default (maps over a spectrum image merge; plain images get a grid).
  // Merged needs images of one pixel size.
  const gridPref = useUi((s) => (gridContext ? s.gridMode[gridContext] : null));
  // In focus mode's navigator flap (this viewer has the pane below it):
  // merged only, no grid (too cluttered there).
  const focusNav = useUi((s) => s.focus === "bottom" && s.focusNav);
  const flapOpen = focusNav && bottomTabs.length > 0;
  const setGridMode = useUi((s) => s.setGridMode);
  const sizes = panelLayers.map((l) => {
    const given = l.key === BASE ? undefined : entryOf(l.key)?.geometry;
    if (given) return { geometry: given, w: given.width, h: given.height };
    const d = l.key === BASE ? undefined : resolve(l.key)?.dataset;
    if (!d) return { geometry, w: geometry.width, h: geometry.height };
    const [w0, h0] = planeSize(d);
    const [w, h] = [w0 * scanFactor, h0 * scanFactor]; // stretched over the base
    const own = w === geometry.width && h === geometry.height ? geometry : imageGeometry(d, "signal");
    return { geometry: own, w: own.width, h: own.height };
  });
  const sameSize = sizes.every((z) => z.w === sizes[0].w && z.h === sizes[0].h);
  const canSwitch = !gridOnly && !!gridContext && panelLayers.length > 1 && !(flapOpen && sameSize);
  const mode: GridMode = gridOnly ? "grid" : canSwitch
    ? !sameSize
      ? "grid"
      : (gridPref ?? (gridContext === "si" ? "merged" : "grid"))
    : "merged";
  const grid = mode === "grid";
  // Every layer gets a panel: switching to the grid shows any hidden ones again.
  const switchMode = (m: GridMode) => {
    if (m === "grid") {
      for (const l of panelLayers) if (!l.visible) setProps(l.key === BASE ? baseLayerKey(viewKey) : l.key, { visible: true });
    }
    setGridMode(gridContext!, m);
  };
  useShortcuts({ g: () => (sameSize ? switchMode(grid ? "merged" : "grid") : false) }, canSwitch);

  // The toolbar edits the focused image if it's in the view, else the base.
  const [active, setActive] = useState<string>(BASE);
  // With an item, the toolbar edits its colours (when coloured by a value)
  // or, with ``editBg``, the image under it.
  const [editBg, setEditBg] = useState(false);
  const imageIds = allLayers.map((e) => e.key).join(",");
  useEffect(() => {
    if (focusId && allLayers.some((e) => e.key === focusId)) {
      setActive(focusId);
      setEditBg(false);
    }
  }, [focusId, imageIds]); // eslint-disable-line react-hooks/exhaustive-deps
  const activeLayer = allLayers.find((l) => l.key === active) ?? panelLayers[0] ?? allLayers[0];
  // Tell the view which layer is edited (not on mount), and honour its requests.
  const onActive = useRef(onActiveChange);
  onActive.current = onActiveChange;
  const reported = useRef<string | undefined>(activeLayer?.key);
  useEffect(() => {
    if (reported.current === activeLayer?.key) return;
    reported.current = activeLayer?.key;
    if (activeLayer) onActive.current?.(activeLayer.key);
  }, [activeLayer?.key]); // eslint-disable-line react-hooks/exhaustive-deps
  const layerKeys = allLayers.map((l) => l.key).join(",");
  useEffect(() => {
    if (!requestActive || !layerKeys.split(",").includes(requestActive.key)) return;
    setActive(requestActive.key);
    if (requestActive.key !== BASE) setFocus(entryOf(requestActive.key)?.dataset.id ?? requestActive.key);
  }, [requestActive]); // eslint-disable-line react-hooks/exhaustive-deps
  // The active item coloured by a value: the contrast toolbar and the
  // Histogram tab edit its limits.
  const activeItem = activeLayer && !editBg ? items.find((d) => d.id === activeLayer.key) ?? null : null;
  const itemContrast = useItemContrast(activeItem);
  const shapeKey = itemContrast && activeItem ? activeItem.id : null;
  const shapeContrast = itemContrast;
  const shapeDisplay = shapeKey ? displays[shapeKey] : undefined;
  const activate = (key: string) => {
    setEditBg(false);
    setActive(key);
    if (key !== BASE) setFocus(entryOf(key)?.dataset.id ?? key);
    onActive.current?.(key);
  };
  const isBase = activeLayer?.key === BASE;
  const activeDataset = isBase ? baseDataset : activeLayer && resolve(activeLayer.key)?.dataset;

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
  // Neighbourhoods of an image's columns (kemstem): their plot.
  const nbImage = [baseDataset?.id, ...images.map((d) => d.id)].find((id) => id && neighbourhoodsOf(files, id).length);
  if (nbImage) tabs.push({ id: "neighbourhood" as const, label: "Neighbourhood", content: <NeighbourhoodPlot image={nbImage} /> });
  // Sublattices by position in cell: the cell plot, once shown.
  const cellImage = [baseDataset?.id, ...images.map((d) => d.id)].find((id) => id && cellStates[id]?.data);
  if (cellImage) tabs.push({ id: "cell" as const, label: "Cell", content: <CellPlot image={cellImage} /> });
  // The histogram tab is always offered (with regions drawn, also theirs).
  {
    // The layer's own pixels (not stretched over the scan).
    const own = isBase ? (base?.frame ?? null) : activeLayer ? (resolve(activeLayer.key)?.frame ?? null) : null;
    tabs.push({
      id: "histogram" as const,
      label: "Histogram",
      content: (
        <HistogramPanel
          frame={shapeContrast ? shapeContrast.frame : own}
          label={shapeContrast ? shapeContrast.label : (activeLayer?.label ?? "")}
          display={shapeContrast ? shapeDisplay : activeLayer?.display}
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

  // The layer list (over the image as soon as any map or item is shown;
  // always in the Layers tab).
  const showPanel = entries.length > 0 || items.length > 0 || foundImages.length > 0;
  const imageRow = (l: ViewLayer) => ({
    id: l.key, label: l.label, visible: l.visible, opacity: l.opacity, display: l.display,
    removable: l.key !== BASE && !implied.includes(l.key) && (entryOf(l.key)?.removable ?? true),
  });
  const itemRow = (item: DatasetSummary) => {
    const info = itemInfo(files, item);
    return info ? [itemPanelLayer(item, info, layerProps[item.id], displays[item.id], true)] : [];
  };
  const foundRows = (ids: string[]) => ids.map((id) => foundPanelLayer(id, layerProps[foundLayerKey(id)]));
  const shownItems = items.filter((d) => isItem(d.id));
  const rowsAll = [
    ...(grid ? panelLayers.filter((l) => !isItem(l.key)) : imageLayers).map(imageRow),
    ...foundRows(foundImages),
    ...shownItems.flatMap(itemRow),
  ];
  const onPanelChange = (id: string, patch: { visible?: boolean; opacity?: number }) =>
    setProps(id === BASE ? baseLayerKey(viewKey) : id, patch);
  const panelProps: LayersPanelProps = {
      layers: rowsAll,
      active: shapeKey ?? activeLayer?.key ?? BASE,
      onActivate: (id) => {
        if (id.startsWith("bg:")) {
          activate(id.slice(3));
          setEditBg(true);
          return;
        }
        if (isFoundKey(id)) return;
        activate(id);
      },
      onChange: onPanelChange,
      onColour: (id, colour) => {
        if (isFoundKey(id) || isItem(id)) return setProps(id, { colour, colourBy: undefined });
        const key = id.startsWith("bg:") ? (itemImage(id.slice(3)) ?? id) : id;
        const d = key === BASE ? baseDataset : entryOf(key)?.dataset;
        if (d && isStack(key)) setDisplay(displayKeyOf(key), { colormap: colour });
        else if (d) setMapColour(d, colour);
        else baseDisplay.setDisplay({ colormap: colour });
      },
      onRemove: (id) => {
        if (isFoundKey(id)) return useColumns.getState().patch(id.slice("columns:found:".length), { found: null, test: null });
        if (isItem(id)) return deselect(id);
        if (id !== BASE && (entryOf(id)?.removable ?? true)) deselect(entryOf(id)?.dataset.id ?? id);
      },
      opacity: !grid,
  };
  // The Layers tab: what the highlighted part of the viewer draws. Merged:
  // everything; grid: the active panel (an item's: the image under it and
  // the item; an image's: it and its guesses).
  const tabRows = (() => {
    if (!grid || !activeLayer) return rowsAll;
    if (isItem(activeLayer.key)) {
      const imageId = itemImage(activeLayer.key) ?? "";
      const under = imageLayers.find((l) => l.key === imageId);
      const bg = { ...DEFAULT_LAYER, ...layerProps[`bg:${activeLayer.key}`] };
      const item = items.find((d) => d.id === activeLayer.key)!;
      return [
        ...(under ? [{ id: `bg:${activeLayer.key}`, label: under.label, visible: bg.visible, opacity: bg.opacity, display: under.display, removable: false }] : []),
        ...itemRow(item),
      ];
    }
    const l = imageLayers.find((x) => x.key === activeLayer.key);
    return l ? [imageRow(l), ...foundRows(l.datasetId && foundImages.includes(l.datasetId) ? [l.datasetId] : [])] : rowsAll;
  })();
  // The view's own rows (its marks) after the images, handled by the view.
  const extra = extraLayers;
  const isExtra = (id: string) => !!extra?.rows.some((r) => r.id === id);
  const tabProps: LayersPanelProps = {
    ...panelProps,
    layers: [...tabRows, ...(extra?.rows ?? [])],
    active: editBg && activeLayer ? `bg:${activeLayer.key}` : panelProps.active,
    opacity: true,
    onActivate: (id) => (isExtra(id) ? extra!.onActivate?.(id) : panelProps.onActivate(id)),
    onChange: (id, p) => (isExtra(id) ? extra!.onChange(id, p) : panelProps.onChange(id, p)),
    onColour: (id, c) => (isExtra(id) ? extra!.onColour(id, c) : panelProps.onColour(id, c)),
    onRemove: (id) => (isExtra(id) ? extra!.onRemove?.(id) : panelProps.onRemove(id)),
  };
  const panel = showPanel ? <LayersPanel {...panelProps} /> : null;
  // The Layers tab shows this list: the first viewer mounted owns it until it
  // goes; a viewer with a section publishes there instead.
  const tabOwner = useId();
  useEffect(() => {
    const { owner, publish, publishSection } = useLayersTab.getState();
    if (layersSection) publishSection(layersSection, tabOwner, tabProps);
    else if (owner === null || owner === tabOwner) publish(tabOwner, tabProps);
  });
  useEffect(
    () => () => (layersSection ? useLayersTab.getState().clearSection(layersSection, tabOwner) : useLayersTab.getState().clear(tabOwner)),
    [tabOwner, layersSection],
  );

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
          {lineTool === undefined && toolButtons && <ImageToolButtons />}
          {lineTool === undefined && <AnnotationsToggle view={viewKey} />}
          <LoupeToggle which="viewer" />
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
          {shapeContrast && shapeKey ? (
            <ContrastControls
              frame={shapeContrast.frame}
              display={shapeDisplay}
              onChange={(patch) => setDisplay(shapeKey, patch)}
            />
          ) : (
            <ContrastControls
              frame={activeLayer?.frame ?? null}
              display={activeLayer?.display}
              onChange={onDisplayChange}
            />
          )}
        </div>
        <div className="viewer-row">
          {lifted && focusCorner && <div className="focus-corner">{focusCorner}</div>}
          {grid ? (
            <ViewerGrid
              panels={panelLayers
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
                    // An item's panel: the image under it, as its props say.
                    ...(isItem(layer.key) && (() => {
                      const bg = { ...DEFAULT_LAYER, ...layerProps[`bg:${layer.key}`] };
                      return { background: bg.visible ? bg.opacity : 0 };
                    })()),
                  };
                })}
              active={activeLayer?.key}
              onActivate={activate}
              interaction={toolInteraction}
              fftView={viewKey}
              layoutKey={viewKey}
              renderOverlay={(t, p) => (
                <>
                  {renderOverlay?.(t, p.layer.key)}
                  {!annotationsHidden && (
                    <>
                      <FftBoxOverlay t={t} view={viewKey} />
                      {ownRegions && (
                        <RoiOverlay t={t} state={regionState} cursorSize={1} draft={regionTools.draft} hover={regionTools.hover} />
                      )}
                      {surveys.filter((d) => d.id === p.layer.key).map((d) => (
                        <SurveyRegions key={d.id} t={t} regions={d.survey!.regions} />
                      ))}
                      {isPhase(p.layer.key) && p.layer.frame && <PhaseContours t={t} frame={p.layer.frame} />}
                      {isItem(p.layer.key) ? (
                        <ColumnsOverlay
                          t={t}
                          image={p.layer.datasetId}
                          items={items.filter((d) => d.id === p.layer.key)}
                        />
                      ) : (
                        <ColumnsOverlay t={t} image={p.layer.datasetId} items={[]} found />
                      )}
                      {coarseningOf(p.layer.key) && (
                        <CoarseningMarker t={t} width={p.width} height={p.height} radius={coarseningOf(p.layer.key)!} />
                      )}
                      <LineOverlay t={t} view={viewKey} calX={p.calX} calY={p.calY} />
                    </>
                  )}
                </>
              )}
              onRemove={(key) =>
                isItem(key)
                  ? () => deselect(key)
                  : key === BASE || !(entryOf(key)?.removable ?? true) ? null : () => deselect(entryOf(key)?.dataset.id ?? key)
              }
              empty="No images to show"
              scanFactor={scanFactor}
            />
          ) : (
            <ImageView
              viewKey={viewKey}
              layers={imageLayers}
              fullWidth={geometry.width}
              fullHeight={geometry.height}
              calX={geometry.calX}
              calY={geometry.calY}
              loading={loading}
              error={error}
              empty={imageLayers.length ? null : (empty ?? null)}
              panel={panel}
              interaction={toolInteraction}
              blend={blend}
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
                      {layers.filter((l) => l.visible && l.frame && isPhase(l.key)).map((l) => (
                        <PhaseContours key={l.key} t={t} frame={l.frame!} />
                      ))}
                      {imageLayers.filter((l) => l.datasetId).map((l) => (
                        <ColumnsOverlay
                          key={`columns-${l.key}`}
                          t={t}
                          image={l.datasetId}
                          items={shownItems.filter((d) => itemImage(d.id) === l.datasetId)}
                          found={l.visible}
                        />
                      ))}
                      {(() => {
                        // One marker: the largest coarsening of the maps shown.
                        const r = Math.max(0, ...layers.filter((l) => l.visible).map((l) => coarseningOf(l.key) ?? 0));
                        return r > 0 && <CoarseningMarker t={t} width={geometry.width} height={geometry.height} radius={r} />;
                      })()}
                      <LineOverlay t={t} view={viewKey} calX={geometry.calX} calY={geometry.calY} />
                    </>
                  )}
                </>
              )}
              analysisKey={activeLayer && isItem(activeLayer.key) ? (itemImage(activeLayer.key) ?? undefined) : activeLayer?.key}
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
