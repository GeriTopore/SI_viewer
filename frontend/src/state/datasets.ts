import { create } from "zustand";
import { api, type DatasetSummary, type FileSummary } from "../api/client";
import { DEFAULT_COLORMAP, useDisplay } from "./display";
import { useLayers } from "./layers";
import { useSi } from "./si";

interface OpenProgress {
  done: number;
  total: number;
  current: string;
}

interface DatasetsState {
  files: FileSummary[];
  /** Highlighted datasets, in the order they were added. The viewer shows
   * exactly these: at most one spectrum image/stack/spectrum, plus images. */
  selection: string[];
  /** Most recently clicked dataset: shown in the Inspector, edited in the toolbar. */
  focusId: string | null;
  progress: OpenProgress | null;
  errors: string[];
  refresh: () => Promise<void>;
  openFiles: (paths: string[]) => Promise<void>;
  closeFile: (fileId: string) => Promise<void>;
  closeAll: () => Promise<void>;
  /** Replace a dataset's summary (after loading, calibration, …). */
  updateDataset: (summary: DatasetSummary) => void;
  /** Replace a file's datasets (maps added, removed or recomputed). */
  replaceFile: (file: FileSummary) => void;
  /** Delete an element map (the backend also unticks its element). */
  deleteDataset: (d: DatasetSummary) => Promise<void>;
  /** Rebin the loaded data in place (one factor per axis). */
  rebin: (d: DatasetSummary, factors: number[]) => Promise<void>;
  /** Show only this dataset. */
  select: (datasetId: string) => void;
  /** Plain click in the dataset pane: images toggle in and out of the view;
   * a spectrum image, stack or spectrum becomes the view. */
  clickDataset: (d: DatasetSummary) => void;
  /** ⌘/Ctrl-click: add to or remove from the highlighted set. */
  toggleSelect: (d: DatasetSummary) => void;
  /** Drop onto the viewer: add to the highlighted set. */
  addToSelection: (d: DatasetSummary) => void;
  deselect: (datasetId: string) => void;
  setFocus: (datasetId: string) => void;
  selectRelative: (step: 1 | -1) => void;
  clearErrors: () => void;
}

export const allDatasets = (files: FileSummary[]): DatasetSummary[] =>
  files.flatMap((f) => f.datasets);

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

/** Width × height of the 2D plane a dataset is shown on. */
export function planeSize(d: DatasetSummary): [number, number] {
  const shape = d.kind === "spectrum_image" ? d.navigation_shape : d.signal_shape;
  return [shape[0] ?? 0, shape[1] ?? 0];
}

/** Why ``d`` can't be shown together with ``current``, or null if it can. */
function combineError(current: DatasetSummary[], d: DatasetSummary): string | null {
  if (d.kind === "other") return `${d.title} can't be displayed.`;
  const all = [...current, d];
  const primaries = all.filter((x) => x.kind !== "image");
  if (primaries.length > 1) {
    return "Only one spectrum image, stack or spectrum can be shown at a time. Click it without ⌘ to switch.";
  }
  const images = all.filter((x) => x.kind === "image");
  const primary = primaries[0];
  if (primary?.kind === "spectrum" && images.length) return "Images can't be overlaid on a spectrum.";
  const ref = primary ?? images[0];
  const [w, h] = planeSize(ref);
  for (const im of images) {
    const [iw, ih] = planeSize(im);
    if (iw !== w || ih !== h) return `${im.title} is ${iw}×${ih} px; ${ref.title} is ${w}×${h} px.`;
  }
  return null;
}

const notify = (msg: string) => useLayers.getState().notify(msg);

// Each file lists its spectrum image, spectrum and stack first, then plain
// images, then element maps (otherwise in file order).
const KIND_ORDER: Record<string, number> = { spectrum_image: 0, spectrum: 1, stack: 2, image: 3, other: 5 };
const rank = (d: DatasetSummary) => (d.is_element_map ? 4 : (KIND_ORDER[d.kind] ?? 5));

function sortFile(f: FileSummary): FileSummary {
  return { ...f, datasets: [...f.datasets].sort((a, b) => rank(a) - rank(b)) };
}

/** Pre-assign a colour to every element map (one colour per element). */
function assignElementColours(files: FileSummary[]) {
  const { ensureElementColour } = useDisplay.getState();
  for (const d of allDatasets(files)) if (d.is_element_map) ensureElementColour(elementOf(d));
}

export const useDatasets = create<DatasetsState>((set, get) => {
  const byId = (id: string) => allDatasets(get().files).find((d) => d.id === id);
  const selected = () => get().selection.map(byId).filter((d): d is DatasetSummary => !!d);

  return {
    files: [],
    selection: [],
    focusId: null,
    progress: null,
    errors: [],

    refresh: async () => {
      try {
        const files = (await api.listFiles()).map(sortFile);
        assignElementColours(files);
        set({ files });
      } catch (e) {
        set({ errors: [(e as Error).message] });
      }
    },

    // Files open one at a time so each appears in the list as soon as it loads.
    openFiles: async (paths) => {
      if (!paths.length || get().progress) return;
      set({ errors: [] });
      const errors: string[] = [];
      for (const [i, path] of paths.entries()) {
        set({ progress: { done: i, total: paths.length, current: baseName(path) } });
        try {
          const entry = sortFile(await api.openFile(path));
          const files = get().files.filter((f) => f.id !== entry.id).concat(entry);
          assignElementColours([entry]);
          set({ files });
          // Follow the newest file, so the user sees what they just opened
          // (sorted, so a spectrum image comes first if there is one).
          const first = entry.datasets[0];
          if (first) get().select(first.id);
        } catch (e) {
          errors.push(`${baseName(path)}: ${(e as Error).message}`);
        }
      }
      set({ progress: null, errors });
    },

    closeFile: async (fileId) => {
      try {
        await api.closeFile(fileId);
      } catch (e) {
        set({ errors: [(e as Error).message] });
        return;
      }
      const files = get().files.filter((f) => f.id !== fileId);
      const alive = new Set(allDatasets(files).map((d) => d.id));
      const selection = get().selection.filter((id) => alive.has(id));
      const focusId = alive.has(get().focusId ?? "") ? get().focusId : (selection.at(-1) ?? null);
      set({ files, selection, focusId });
    },

    closeAll: async () => {
      for (const f of get().files) await get().closeFile(f.id);
    },

    replaceFile: (file) => {
      const sorted = sortFile(file);
      assignElementColours([sorted]);
      const before = new Map(allDatasets(get().files).map((d) => [d.id, d.revision]));
      // Recomputed or rebinned data gets fresh auto contrast and zoom; a
      // rebinned spectrum image's cursor and regions no longer fit.
      const changed = sorted.datasets.filter((d) => before.has(d.id) && before.get(d.id) !== d.revision);
      if (changed.length) {
        useDisplay.getState().resetDisplay(changed.map((d) => d.id));
        changed.forEach((d) => d.kind === "spectrum_image" && useSi.getState().reset(d.id));
      }
      const files = get().files.map((f) => (f.id === file.id ? sorted : f));
      const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
      // Drop highlighted datasets that are gone, or that no longer fit the
      // view's size (e.g. a HAADF image over a rebinned spectrum image).
      const kept = get().selection.map((id) => byId.get(id)).filter((d): d is DatasetSummary => !!d);
      const base = kept.find((d) => d.kind !== "image") ?? kept[0];
      const [bw, bh] = base ? planeSize(base) : [0, 0];
      const selection = kept
        .filter((d) => d === base || d.kind !== "image" || (planeSize(d)[0] === bw && planeSize(d)[1] === bh))
        .map((d) => d.id);
      const focusId = byId.has(get().focusId ?? "") ? get().focusId : (selection.at(-1) ?? null);
      set({ files, selection, focusId });
    },

    deleteDataset: async (d) => {
      try {
        get().replaceFile((await api.deleteDataset(d.id)).file);
      } catch (e) {
        notify(`Could not delete ${d.title}: ${(e as Error).message}`);
      }
    },

    rebin: async (d, factors) => {
      get().replaceFile((await api.rebin(d.id, factors)).file);
    },

    updateDataset: (summary) =>
      set({
        files: get().files.map((f) =>
          f.id !== summary.file_id
            ? f
            : { ...f, datasets: f.datasets.map((d) => (d.id === summary.id ? summary : d)) },
        ),
      }),

    select: (id) => set({ selection: [id], focusId: id }),

    clickDataset: (d) => {
      const { selection } = get();
      if (d.kind !== "image") {
        if (selection.includes(d.id)) set({ focusId: d.id });
        else get().select(d.id);
        return;
      }
      if (selection.includes(d.id)) get().deselect(d.id);
      else if (!combineError(selected(), d)) set({ selection: [...selection, d.id], focusId: d.id });
      // Can't be overlaid on what's shown (e.g. a different size): show it on its own.
      else get().select(d.id);
    },

    toggleSelect: (d) => {
      if (get().selection.includes(d.id)) get().deselect(d.id);
      else get().addToSelection(d);
    },

    addToSelection: (d) => {
      if (get().selection.includes(d.id)) {
        set({ focusId: d.id });
        return;
      }
      const why = combineError(selected(), d);
      if (why) {
        notify(why);
        return;
      }
      set({ selection: [...get().selection, d.id], focusId: d.id });
    },

    deselect: (id) => {
      const selection = get().selection.filter((x) => x !== id);
      const focusId = get().focusId === id ? (selection.at(-1) ?? null) : get().focusId;
      set({ selection, focusId });
    },

    setFocus: (focusId) => set({ focusId }),

    // Arrow keys: step the focus; spectrum images, stacks and spectra also
    // become the view, while images are only inspected (click adds them).
    selectRelative: (step) => {
      const list = allDatasets(get().files);
      if (!list.length) return;
      const i = list.findIndex((d) => d.id === get().focusId);
      const next = list[i < 0 ? 0 : Math.min(list.length - 1, Math.max(0, i + step))];
      if (next.kind === "image") set({ focusId: next.id });
      else if (get().selection.includes(next.id)) set({ focusId: next.id });
      else get().select(next.id);
    },

    clearErrors: () => set({ errors: [] }),
  };
});

/** Only element maps can be deleted. */
export const canDelete = (d: DatasetSummary) => d.is_element_map;

/** Element symbol of an element map (computed maps are titled e.g. "Fe Kα"). */
export const elementOf = (d: DatasetSummary) => d.element ?? d.title;

/** Colormap an image starts with: its element colour for element maps. */
export function initialColormap(d: DatasetSummary): string {
  return d.is_element_map ? useDisplay.getState().ensureElementColour(elementOf(d)) : DEFAULT_COLORMAP;
}

/** Recolour an element map, and every other open map of the same element. */
export function setMapColour(d: DatasetSummary, colour: string) {
  if (!d.is_element_map) {
    useDisplay.getState().setDisplay(d.id, { colormap: colour });
    return;
  }
  const ids = allDatasets(useDatasets.getState().files)
    .filter((x) => x.is_element_map && elementOf(x) === elementOf(d))
    .map((x) => x.id);
  useDisplay.getState().setElementColour(elementOf(d), colour, ids);
}

/** The focused dataset (last clicked), shown in the Inspector. */
export const useFocusedDataset = (): DatasetSummary | undefined =>
  useDatasets((s) => allDatasets(s.files).find((d) => d.id === s.focusId));
