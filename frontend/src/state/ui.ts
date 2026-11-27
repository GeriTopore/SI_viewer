import { create } from "zustand";
import { COLORMAPS, isColour } from "../viewer/colormaps";

/** How the FFT is shown: amplitude |F|, or log(1 + |F|²). */
export type FftScale = "linear" | "log";
export type ListMode = "list" | "grid";
export type Theme = "light" | "dark";
export type RightTab = "info" | "quantify" | "process" | "layers";
/** How several images are shown: one composite, or a panel each. */
export type GridMode = "merged" | "grid";
/** Kinds of view with their own mode: a spectrum image (with maps), plain images. */
export type GridContext = "si" | "images" | "stacks";
/** Tabs of the pane under the image; a linked EELS spectrum image has one per
 * member of its group, ``eels:<dataset id>`` (ids last for one session only:
 * a stored one that is not shown falls back, see BottomPane). */
export type BottomTab =
  | "spectrum" | "profile" | "histogram" | "exploded" | "alignment" | "pca-component" | "pca-scree" | "neighbourhood"
  | "cell" | "templates" | "symmetry" | "sites" | "structure" | "match" | "ipf-key" | `eels:${string}`;
const isBottomTab = (t: string | undefined): t is BottomTab =>
  t === "spectrum" || t === "profile" || t === "histogram" || t === "exploded" || t === "alignment" ||
  t === "pca-component" || t === "pca-scree" || t === "neighbourhood" || t === "cell" ||
  t === "templates" || t === "symmetry" || t === "sites" || t === "structure" || t === "match" || t === "ipf-key" ||
  (!!t && t.startsWith("eels:") && t.length > 5);
export const eelsTab = (datasetId: string): BottomTab => `eels:${datasetId}`;
/** Two tabs of the bottom pane shown side by side (left, right) or stacked (top, bottom). */
export type BottomSplit = [BottomTab, BottomTab];
/** A process split off in the Process tab (see ``processSplit``). */
export interface ProcessSplit {
  id: string;
  side: SplitSide;
  image: string | null;
}
export type SplitDir = "cols" | "rows";
/** Where a dragged tab lands in the pane. */
export type SplitSide = "left" | "right" | "top" | "bottom";

// Pane sizes: default and limits (side panes in px, the viewer's bottom pane
// as a fraction of the viewer's height).
export const PANE_SIZES = {
  left: { initial: 280, min: 180, max: 640 },
  right: { initial: 330, min: 260, max: 720 },
  bottom: { initial: 0.42, min: 0.15, max: 0.85 },
  // The split bottom pane's first half (left, or top for rows), as a fraction
  // of its width / height.
  split: { initial: 0.5, min: 0.2, max: 0.8 },
  splitRow: { initial: 0.5, min: 0.2, max: 0.8 },
  // The Process tab's split: the share of the split-off process.
  processSplit: { initial: 0.5, min: 0.15, max: 0.85 },
  // The alignment's row of compared cross-correlations, as a fraction of the step.
  compare: { initial: 0.38, min: 0.12, max: 0.8 },
};

const clamp = (v: number, { min, max }: { min: number; max: number }) => Math.min(max, Math.max(min, v));

// Per-browser conveniences, remembered across reloads when storage is available.
function load<T extends string>(key: string, fallback: T): T {
  try {
    return (localStorage.getItem(key) as T | null) ?? fallback;
  } catch {
    return fallback;
  }
}

function loadNumber(key: string, limits: { initial: number; min: number; max: number }): number {
  const v = Number(load(key, ""));
  return Number.isFinite(v) && v > 0 ? clamp(v, limits) : limits.initial;
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
}

function loadCollapsedTypes(): Record<string, true> {
  try {
    const v: unknown = JSON.parse(load("si-viewer.collapsedTypes", "{}"));
    if (!v || typeof v !== "object" || Array.isArray(v)) return {};
    return Object.fromEntries(Object.keys(v).map((k) => [k, true as const]));
  } catch {
    return {};
  }
}

/** Key of a type group in `collapsedTypes`: the file and the type badge. */
export const typeGroupKey = (filePath: string, label: string) => `${filePath}\n${label}`;

function loadFftGamma(): number {
  const v = Number(load("si-viewer.fftGamma", ""));
  return Number.isFinite(v) && v >= 0.01 && v <= 100 ? v : 1;
}

function loadFftColormap(): string {
  const c = load("si-viewer.fftColormap", "gray");
  return COLORMAPS.includes(c) || isColour(c) ? c : "gray";
}

function loadGridMode(context: GridContext): GridMode | null {
  const v = load<string>(`si-viewer.gridMode.${context}`, "auto");
  return v === "merged" || v === "grid" ? v : null;
}

function loadSplit(): BottomSplit | null {
  const [a, b] = load<string>("si-viewer.bottomSplit", "none").split("|");
  return isBottomTab(a) && isBottomTab(b) && a !== b ? [a, b] : null;
}

function loadProcessSplit(): ProcessSplit | null {
  try {
    const v = JSON.parse(load<string>("si-viewer.processSplit", "null"));
    return v && typeof v.id === "string" && ["left", "right", "top", "bottom"].includes(v.side)
      ? { id: v.id, side: v.side, image: typeof v.image === "string" ? v.image : null }
      : null;
  } catch {
    return null;
  }
}
const saveSplit = (split: BottomSplit | null) => save("si-viewer.bottomSplit", split ? split.join("|") : "none");

interface UiState {
  /** Set on <html data-theme> by index.html before first paint. */
  theme: Theme;
  setTheme: (theme: Theme) => void;
  listMode: ListMode;
  /** Dataset type groups folded in the list, by `typeGroupKey` (file path + type). */
  collapsedTypes: Record<string, true>;
  toggleTypeGroup: (key: string) => void;
  /** Datasets whose results are folded away in the list, by id (this session only). */
  foldedNodes: Record<string, true>;
  /** Grid views arranged by hand: rows of panel keys, per view (this session
   * only). Without one, a view's grid is automatic. */
  gridLayouts: Record<string, string[][]>;
  setGridLayout: (view: string, rows: string[][] | null) => void;
  toggleNode: (id: string) => void;
  /** Tab shown in the right-hand column. */
  rightTab: RightTab;
  setRightTab: (tab: RightTab) => void;
  /** Side panes folded to their edges, leaving the viewer. */
  leftCollapsed: boolean;
  rightCollapsed: boolean;
  setCollapsed: (side: "left" | "right", collapsed: boolean) => void;
  /** Focus mode (Shift F): "bottom", the bottom pane's active half lifted
   * over the blurred interface, with the right-hand pane beside it;
   * "datasets" (over the datasets pane), the highlighted dataset's
   * processing as a family tree. */
  focus: "bottom" | "datasets" | null;
  setFocus: (focus: "bottom" | "datasets" | null) => void;
  /** In focus mode, the navigator image pulled out on the left (remembered). */
  focusNav: boolean;
  setFocusNav: (open: boolean) => void;
  /** In focus mode, widths set by dragging (px; null: the default): the
   * right-hand pane, and the navigator's flap (the most it takes; it keeps
   * the image's shape). The pane takes the rest. Remembered. */
  focusRightW: number | null;
  focusNavW: number | null;
  setFocusWidth: (which: "right" | "nav", px: number | null) => void;
  /** Widths of the side panes (px). */
  leftWidth: number;
  rightWidth: number;
  setPaneWidth: (side: "left" | "right", width: number) => void;
  /** Height of the viewer's bottom pane (spectrum / line profile), as a
   * fraction of the viewer. */
  bottomFraction: number;
  setBottomFraction: (f: number) => void;
  compareFraction: number;
  setCompareFraction: (f: number) => void;
  /** The pane under the image: the focused tab (the only one when not
   * split, else the half last used; T and the line-scan image follow it). */
  bottomTab: BottomTab;
  /** Focus ``tab``. Split: a shown tab is just focused, any other replaces
   * the half holding the focused tab. */
  setBottomTab: (tab: BottomTab) => void;
  /** Focus ``tab`` without touching the split (a view showing it single). */
  focusBottomTab: (tab: BottomTab) => void;
  /** Two tabs (first, second) side by side (left, right) or stacked (top,
   * bottom), per ``splitDir``; null: one tab. Kept while a view lacks one. */
  bottomSplit: BottomSplit | null;
  /** Orientation of the split; remembered while unsplit. */
  splitDir: SplitDir;
  /** Split into ``first`` and ``second`` (in ``dir``, default the current
   * ``splitDir``); ``focus`` (default first) is focused. */
  splitBottom: (first: BottomTab, second: BottomTab, focus?: BottomTab, dir?: SplitDir) => void;
  /** Back to one tab, ``keep``. */
  unsplitBottom: (keep: BottomTab) => void;
  /** Put ``tab`` on a side of the split (orienting it, swapping when it is in
   * the other slot) and focus it. */
  placeBottomTab: (tab: BottomTab, side: SplitSide) => void;
  /** Size of the split's first half (width for columns, height for rows), as a fraction. */
  splitFraction: number;
  setSplitFraction: (f: number) => void;
  splitRowFraction: number;
  setSplitRowFraction: (f: number) => void;
  /** Ids of the Process tab's sections folded to their titles. */
  processCollapsed: string[];
  toggleProcess: (id: string) => void;
  /** The Process tab as a list of foldable sections or a grid of icons. */
  processMode: ListMode;
  setProcessMode: (mode: ListMode) => void;
  /** The process opened from the icon grid (null: the grid). */
  processOpen: string | null;
  setProcessOpen: (id: string | null) => void;
  /** A process split off beside the Process tab's own view (dragged there by
   * its icon): which, on which side, and the dataset it stays on (null:
   * whatever the tab is working on). */
  processSplit: ProcessSplit | null;
  setProcessSplit: (split: ProcessSplit | null) => void;
  /** Share of the Process tab the split-off process takes. */
  processSplitFraction: number;
  setProcessSplitFraction: (f: number) => void;
  /** The bottom pane's tabs now shown (so a process can bring its plot forward). */
  bottomTabsShown: BottomTab[];
  setBottomTabsShown: (tabs: BottomTab[]) => void;
  /** Image views whose annotations (regions, lines, FFT boxes) are hidden, by view key. */
  annotationsHidden: Record<string, true>;
  setAnnotationsHidden: (view: string, hidden: boolean) => void;
  /** FFT of the viewer's image, above the tabs. */
  fftOpen: boolean;
  setFftOpen: (open: boolean) => void;
  /** Clicking the FFT marks a spatial frequency (and shows the mark). */
  fftMeasure: boolean;
  setFftMeasure: (on: boolean) => void;
  /** The FFT is of the patch with a Hann window (no edge streaks). */
  fftHann: boolean;
  setFftHann: (on: boolean) => void;
  /** A magnifier follows the pointer over the FFT / over the image viewers. */
  fftLoupe: boolean;
  setFftLoupe: (on: boolean) => void;
  viewerLoupe: boolean;
  setViewerLoupe: (on: boolean) => void;
  /** FFT display: amplitude or log(1 + |F|²), gamma and colormap. */
  fftScale: FftScale;
  setFftScale: (scale: FftScale) => void;
  fftGamma: number;
  setFftGamma: (gamma: number) => void;
  fftColormap: string;
  setFftColormap: (colormap: string) => void;
  /** The FFT's contrast controls are shown (else folded to their title). */
  fftContrastOpen: boolean;
  setFftContrastOpen: (open: boolean) => void;
  /** Merged/Grid choice per kind of view; null follows the content
   * (spectrum image: merged, images: grid). */
  gridMode: Record<GridContext, GridMode | null>;
  setGridMode: (context: GridContext, mode: GridMode | null) => void;
  lastDir: string;
  /** The folder last exported to (Export… dialog). */
  lastExportDir: string;
  setLastExportDir: (dir: string) => void;
  browserOpen: boolean;
  /** What the file browser is for: data, opening an analysis or saving one. */
  browserMode: BrowserMode;
  setListMode: (mode: ListMode) => void;
  setLastDir: (dir: string) => void;
  setBrowserOpen: (open: boolean, mode?: BrowserMode) => void;
}

export type BrowserMode = "data" | "open-analysis" | "save-analysis";

export const useUi = create<UiState>((set, get) => ({
  theme: document.documentElement.dataset.theme === "dark" ? "dark" : "light",
  setTheme: (theme) => {
    document.documentElement.dataset.theme = theme;
    save("si-viewer.theme", theme);
    set({ theme });
  },
  listMode: load<ListMode>("si-viewer.listMode", "list"),
  collapsedTypes: loadCollapsedTypes(),
  toggleTypeGroup: (key) =>
    set((s) => {
      const collapsedTypes = { ...s.collapsedTypes };
      if (collapsedTypes[key]) delete collapsedTypes[key];
      else collapsedTypes[key] = true;
      save("si-viewer.collapsedTypes", JSON.stringify(collapsedTypes));
      return { collapsedTypes };
    }),
  gridLayouts: {},
  setGridLayout: (view, rows) =>
    set((s) => {
      const gridLayouts = { ...s.gridLayouts };
      if (rows) gridLayouts[view] = rows;
      else delete gridLayouts[view];
      return { gridLayouts };
    }),
  foldedNodes: {},
  toggleNode: (id) =>
    set((s) => {
      const foldedNodes = { ...s.foldedNodes };
      if (foldedNodes[id]) delete foldedNodes[id];
      else foldedNodes[id] = true;
      return { foldedNodes };
    }),
  rightTab: load<RightTab>("si-viewer.rightTab", "info"),
  leftCollapsed: load<string>("si-viewer.leftCollapsed", "false") === "true",
  rightCollapsed: load<string>("si-viewer.rightCollapsed", "false") === "true",
  setCollapsed: (side, collapsed) => {
    save(`si-viewer.${side}Collapsed`, String(collapsed));
    set(side === "left" ? { leftCollapsed: collapsed } : { rightCollapsed: collapsed });
  },
  focus: null,
  setFocus: (focus) => set({ focus }),
  focusNav: load<string>("si-viewer.focusNav", "false") === "true",
  setFocusNav: (open) => {
    save("si-viewer.focusNav", String(open));
    set({ focusNav: open });
  },
  focusRightW: Number(load("si-viewer.focusRightW", "")) || null,
  focusNavW: Number(load("si-viewer.focusNavW", "")) || null,
  setFocusWidth: (which, px) => {
    const v = px === null ? null : Math.round(px);
    save(`si-viewer.focus${which === "right" ? "RightW" : "NavW"}`, v === null ? "" : String(v));
    set(which === "right" ? { focusRightW: v } : { focusNavW: v });
  },
  leftWidth: loadNumber("si-viewer.leftWidth", PANE_SIZES.left),
  rightWidth: loadNumber("si-viewer.rightWidth", PANE_SIZES.right),
  setPaneWidth: (side, width) => {
    const w = Math.round(clamp(width, PANE_SIZES[side]));
    save(`si-viewer.${side}Width`, String(w));
    set(side === "left" ? { leftWidth: w } : { rightWidth: w });
  },
  bottomFraction: loadNumber("si-viewer.bottomFraction", PANE_SIZES.bottom),
  setBottomFraction: (f) => {
    const bottomFraction = clamp(f, PANE_SIZES.bottom);
    save("si-viewer.bottomFraction", bottomFraction.toFixed(3));
    set({ bottomFraction });
  },
  compareFraction: loadNumber("si-viewer.compareFraction", PANE_SIZES.compare),
  setCompareFraction: (f) => {
    const compareFraction = clamp(f, PANE_SIZES.compare);
    save("si-viewer.compareFraction", compareFraction.toFixed(3));
    set({ compareFraction });
  },
  bottomTab: ((t) => (isBottomTab(t) ? t : "spectrum"))(load<string>("si-viewer.bottomTab", "spectrum")),
  setBottomTab: (tab) => {
    const { bottomSplit: split, bottomTab } = get();
    const i = split && !split.includes(tab) ? split.indexOf(bottomTab) : -1;
    if (split && i >= 0) {
      const bottomSplit = split.map((t, j) => (j === i ? tab : t)) as BottomSplit;
      saveSplit(bottomSplit);
      set({ bottomSplit });
    }
    save("si-viewer.bottomTab", tab);
    set({ bottomTab: tab });
  },
  focusBottomTab: (bottomTab) => {
    save("si-viewer.bottomTab", bottomTab);
    set({ bottomTab });
  },
  bottomSplit: loadSplit(),
  splitDir: load<string>("si-viewer.splitDir", "cols") === "rows" ? "rows" : "cols",
  splitBottom: (first, second, focus = first, dir = get().splitDir) => {
    if (first === second) return;
    saveSplit([first, second]);
    save("si-viewer.splitDir", dir);
    save("si-viewer.bottomTab", focus);
    set({ bottomSplit: [first, second], splitDir: dir, bottomTab: focus });
  },
  unsplitBottom: (keep) => {
    saveSplit(null);
    save("si-viewer.bottomTab", keep);
    set({ bottomSplit: null, bottomTab: keep });
  },
  placeBottomTab: (tab, side) => {
    const split = get().bottomSplit;
    if (!split) return;
    const splitDir: SplitDir = side === "left" || side === "right" ? "cols" : "rows";
    const [i, j] = side === "left" || side === "top" ? [0, 1] : [1, 0];
    const bottomSplit: BottomSplit = [...split];
    bottomSplit[i] = tab;
    if (split[j] === tab) bottomSplit[j] = split[i];
    saveSplit(bottomSplit);
    save("si-viewer.splitDir", splitDir);
    save("si-viewer.bottomTab", tab);
    set({ bottomSplit, splitDir, bottomTab: tab });
  },
  splitFraction: loadNumber("si-viewer.splitFraction", PANE_SIZES.split),
  setSplitFraction: (f) => {
    const splitFraction = clamp(f, PANE_SIZES.split);
    save("si-viewer.splitFraction", splitFraction.toFixed(3));
    set({ splitFraction });
  },
  splitRowFraction: loadNumber("si-viewer.splitRowFraction", PANE_SIZES.splitRow),
  setSplitRowFraction: (f) => {
    const splitRowFraction = clamp(f, PANE_SIZES.splitRow);
    save("si-viewer.splitRowFraction", splitRowFraction.toFixed(3));
    set({ splitRowFraction });
  },
  processCollapsed: load<string>("si-viewer.processCollapsed", "").split(",").filter(Boolean),
  toggleProcess: (id) => {
    const cur = get().processCollapsed;
    const processCollapsed = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id];
    save("si-viewer.processCollapsed", processCollapsed.join(","));
    set({ processCollapsed });
  },
  processMode: load<ListMode>("si-viewer.processMode", "list"),
  setProcessMode: (processMode) => {
    save("si-viewer.processMode", processMode);
    set({ processMode });
  },
  processOpen: null,
  setProcessOpen: (processOpen) => set({ processOpen }),
  processSplit: loadProcessSplit(),
  setProcessSplit: (processSplit) => {
    save("si-viewer.processSplit", processSplit ? JSON.stringify(processSplit) : "none");
    set({ processSplit });
  },
  processSplitFraction: loadNumber("si-viewer.processSplitFraction", PANE_SIZES.processSplit),
  setProcessSplitFraction: (f) => {
    const processSplitFraction = clamp(f, PANE_SIZES.processSplit);
    save("si-viewer.processSplitFraction", processSplitFraction.toFixed(3));
    set({ processSplitFraction });
  },
  bottomTabsShown: [],
  setBottomTabsShown: (tabs) => {
    const cur = get().bottomTabsShown;
    if (cur.length !== tabs.length || cur.some((t, i) => t !== tabs[i])) set({ bottomTabsShown: tabs });
  },
  annotationsHidden: {},
  setAnnotationsHidden: (view, hidden) =>
    set((s) => {
      const annotationsHidden = { ...s.annotationsHidden };
      if (hidden) annotationsHidden[view] = true;
      else delete annotationsHidden[view];
      return { annotationsHidden };
    }),
  fftOpen: load<string>("si-viewer.fftOpen", "true") === "true",
  fftMeasure: load<string>("si-viewer.fftMeasure", "true") === "true",
  fftScale: load<string>("si-viewer.fftScale", "log") === "linear" ? "linear" : "log",
  setFftScale: (fftScale) => {
    save("si-viewer.fftScale", fftScale);
    set({ fftScale });
  },
  fftGamma: loadFftGamma(),
  setFftGamma: (fftGamma) => {
    save("si-viewer.fftGamma", String(fftGamma));
    set({ fftGamma });
  },
  fftColormap: loadFftColormap(),
  setFftColormap: (fftColormap) => {
    save("si-viewer.fftColormap", fftColormap);
    set({ fftColormap });
  },
  fftHann: load<string>("si-viewer.fftHann", "true") === "true",
  setFftHann: (fftHann) => {
    save("si-viewer.fftHann", String(fftHann));
    set({ fftHann });
  },
  // Off whenever the app starts (not remembered).
  fftLoupe: false,
  setFftLoupe: (fftLoupe) => set({ fftLoupe }),
  viewerLoupe: false,
  setViewerLoupe: (viewerLoupe) => set({ viewerLoupe }),
  fftContrastOpen: load<string>("si-viewer.fftContrastOpen", "true") === "true",
  setFftContrastOpen: (fftContrastOpen) => {
    save("si-viewer.fftContrastOpen", String(fftContrastOpen));
    set({ fftContrastOpen });
  },
  setFftMeasure: (fftMeasure) => {
    save("si-viewer.fftMeasure", String(fftMeasure));
    set({ fftMeasure });
  },
  setFftOpen: (fftOpen) => {
    save("si-viewer.fftOpen", String(fftOpen));
    set({ fftOpen });
  },
  setRightTab: (rightTab) => {
    save("si-viewer.rightTab", rightTab);
    set({ rightTab });
  },
  gridMode: { si: loadGridMode("si"), images: loadGridMode("images"), stacks: loadGridMode("stacks") },
  setGridMode: (context, mode) => {
    save(`si-viewer.gridMode.${context}`, mode ?? "auto");
    set((s) => ({ gridMode: { ...s.gridMode, [context]: mode } }));
  },
  lastDir: load("si-viewer.lastDir", ""),
  lastExportDir: load("si-viewer.lastExportDir", ""),
  setLastExportDir: (lastExportDir) => {
    save("si-viewer.lastExportDir", lastExportDir);
    set({ lastExportDir });
  },
  browserOpen: false,
  browserMode: "data",
  setListMode: (listMode) => {
    save("si-viewer.listMode", listMode);
    set({ listMode });
  },
  setLastDir: (lastDir) => {
    save("si-viewer.lastDir", lastDir);
    set({ lastDir });
  },
  setBrowserOpen: (browserOpen, browserMode = "data") => set({ browserOpen, browserMode }),
}));

/** Whether process ``id`` is on screen in the Process tab: opened from the
 * icon view, unfolded in the list view, or split off beside the others. */
export const useProcessShown = (id: string): boolean =>
  useUi((s) =>
    s.rightTab === "process"
    && (s.processSplit?.id === id || (s.processMode === "list" ? !s.processCollapsed.includes(id) : s.processOpen === id)),
  );
