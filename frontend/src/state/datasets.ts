import { create } from "zustand";
import { api, type DatasetSummary, type EelsRole, type ExperimentSummary, type FileSummary } from "../api/client";
import { DEFAULT_COLORMAP, useDisplay } from "./display";
import { useProfiles } from "./profiles";
import { useFftBoxes } from "./fftBoxes";
import { useLayers } from "./layers";
import { useSi } from "./si";
import { isLineScan, isWindowMap, kindLabel } from "../format";
import { typeGroupKey, useUi } from "./ui";

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
  /** Open files one by one (into an experiment: see ``openFolder``). */
  openFiles: (paths: string[], experimentId?: string) => Promise<void>;
  /** Open a folder's files as one experiment (listed together, their EELS
   * spectrum images linked). */
  openFolder: (folder: string) => Promise<void>;
  /** Open a bundled example dataset (see the Examples menu). */
  openExample: (id: string, title: string) => Promise<void>;
  closeFile: (fileId: string) => Promise<void>;
  closeAll: () => Promise<void>;
  /** Replace a dataset's summary (after loading, calibration, …). */
  updateDataset: (summary: DatasetSummary) => void;
  /** Replace a file's datasets (maps added, removed or recomputed). */
  replaceFile: (file: FileSummary) => void;
  /** Refetch every file and apply the ones that changed (linked EELS spectrum
   * images change together, possibly in other files). */
  syncFiles: () => Promise<void>;
  /** Put an EELS spectrum image in a partner's link group (null: leave its group). Throws on failure. */
  linkEels: (d: DatasetSummary, partnerId: string | null) => Promise<void>;
  /** Say by hand that an EELS spectrum image is a low or core loss. Throws on failure. */
  setEelsRole: (d: DatasetSummary, role: EelsRole) => Promise<void>;
  /** Delete an element or window map (the backend also unticks an element). */
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

// --- Display order of linked EELS groups (the store keeps the opening order) ---

/** Number of group colours (--pair-0 … in styles.css). */
export const PAIR_COLOURS = 6;

/** A rendered stretch of one file's datasets: a single dataset, or the
 * members of a link group that live in this file placed side by side (low
 * loss first) and boxed. */
export type DatasetItem =
  | { type: "dataset"; d: DatasetSummary }
  | { type: "group"; colour: number; ds: DatasetSummary[] };

/** A rendered stretch of the file list: one file, or the files holding a link
 * group, placed one after the other and boxed. */
export type FileItem =
  | { type: "file"; file: FileSummary }
  | { type: "group"; colour: number; files: FileSummary[] };

export interface GroupLayout {
  /** Group colour index for every dataset that is linked. */
  colourOf: Map<string, number>;
  items: FileItem[];
}

const isLow = (d: DatasetSummary) => d.eels?.role === "low_loss";

/** First energy channel of a spectrum image (the order of its core losses). */
const energyOffset = (d: DatasetSummary) => d.axes.find((a) => !a.navigate)?.offset ?? 0;

/** The link group ``d`` is in (itself included), among the datasets in
 * ``byId``: low loss first, then core losses by energy offset. Just ``d`` when
 * it is not linked. */
export function groupOf(d: DatasetSummary, byId: Map<string, DatasetSummary>): DatasetSummary[] {
  const members = [d, ...(d.eels?.linked ?? []).flatMap((id) => byId.get(id) ?? [])];
  return members.sort((a, b) => Number(!isLow(a)) - Number(!isLow(b)) || energyOffset(a) - energyOffset(b));
}

/** Order the files so that files with a link group are adjacent, and give every
 * group a colour (by the order in which the groups are found in the store, so
 * a colour stays while the group stays). */
export function layoutGroups(files: FileSummary[]): GroupLayout {
  const all = allDatasets(files);
  const byId = new Map(all.map((d) => [d.id, d]));
  const colourOf = new Map<string, number>();
  let n = 0;
  for (const d of all) {
    if (colourOf.has(d.id)) continue;
    const members = groupOf(d, byId);
    if (members.length < 2) continue;
    for (const m of members) colourOf.set(m.id, n % PAIR_COLOURS);
    n += 1;
  }
  const fileOf = new Map<string, FileSummary>();
  for (const f of files) for (const d of f.datasets) fileOf.set(d.id, f);

  const items: FileItem[] = [];
  const placed = new Set<string>();
  for (const f of files) {
    if (placed.has(f.id)) continue;
    placed.add(f.id);
    let item: FileItem = { type: "file", file: f };
    search: for (const d of f.datasets) {
      // The group's files in its order (low loss first), this one in its place.
      const members = groupOf(d, byId);
      const group = [...new Set(members.flatMap((m) => fileOf.get(m.id) ?? []))];
      if (!group.some((g) => g.id !== f.id && !placed.has(g.id))) continue;
      for (const g of group) placed.add(g.id);
      item = { type: "group", colour: colourOf.get(d.id)!, files: group };
      break search;
    }
    items.push(item);
  }
  return { colourOf, items };
}

/** A top-level stretch of the dataset pane: an experiment (the files of one
 * folder opened together, by role: survey, scan images, low loss, core
 * losses, …) or, for other files, a file or the files of a link group. */
export type PaneItem = FileItem | { type: "experiment"; experiment: ExperimentSummary; files: FileSummary[] };

const roleRank = (f: FileSummary) => {
  const r = f.role ?? "";
  if (r === "Survey") return 0;
  if (r === "EELS LL") return 2;
  if (r.startsWith("EELS")) return 3;
  if (r === "EDS") return 4;
  return r ? 1 : 5;
};

/** The dataset pane's layout: experiments, files and the files of link
 * groups in the order they were opened (link colours as ``layoutGroups``). */
export function layoutPane(files: FileSummary[]): { colourOf: Map<string, number>; items: PaneItem[] } {
  const { colourOf } = layoutGroups(files);
  const loose = layoutGroups(files.filter((f) => !f.experiment)).items.map((it): FileItem => {
    if (it.type !== "group") return it;
    const linked = it.files.flatMap((f) => f.datasets).find((d) => colourOf.has(d.id));
    return { ...it, colour: linked ? colourOf.get(linked.id)! : it.colour };
  });
  const items: PaneItem[] = [];
  const done = new Set<string>();
  for (const f of files) {
    if (f.experiment) {
      const id = f.experiment.id;
      if (done.has(id)) continue;
      done.add(id);
      const members = files.filter((g) => g.experiment?.id === id);
      const sorted = members.map((g, i) => [g, i] as const).sort(([a, i], [b, j]) => roleRank(a) - roleRank(b) || i - j);
      items.push({ type: "experiment", experiment: f.experiment, files: sorted.map(([g]) => g) });
    } else {
      const it = loose.find((x) => (x.type === "file" ? [x.file] : x.files).some((g) => g.id === f.id));
      const key = it && (it.type === "file" ? it.file.id : it.files.map((g) => g.id).join("+"));
      if (!it || done.has(key!)) continue;
      done.add(key!);
      items.push(it);
    }
  }
  return { colourOf, items };
}

/** The files of a pane item, in order. */
export const paneFiles = (it: PaneItem): FileSummary[] => (it.type === "file" ? [it.file] : it.files);

/** Datasets of one level of a file's tree, with the members of a link group
 * that are on this level side by side (low loss first, at the place of
 * whichever comes first). */
export function layoutFile(ds: DatasetSummary[], colourOf: Map<string, number>): DatasetItem[] {
  const byId = new Map(ds.map((d) => [d.id, d]));
  const out: DatasetItem[] = [];
  const done = new Set<string>();
  for (const d of ds) {
    if (done.has(d.id)) continue;
    const members = groupOf(d, byId);
    for (const m of members) done.add(m.id);
    out.push(
      members.length < 2
        ? { type: "dataset", d }
        : { type: "group", colour: colourOf.get(d.id) ?? 0, ds: members },
    );
  }
  return out;
}

/** Rows of one level: plain datasets, link groups, and type groups (two or
 * more unlinked datasets with the same badge, at the place of the first). */
export type RowItem = DatasetItem | { type: "types"; key: string; label: string; ds: DatasetSummary[] };

/** A file's datasets as a tree: everything made from a dataset (``parent_id``,
 * or ``derived_from``) under it, to any depth. ``rows`` is the top level;
 * ``children`` has the rows under each dataset that has any. */
export interface FileTree {
  rows: RowItem[];
  children: Map<string, RowItem[]>;
}

/** The dataset ``d`` was made from, if it is in ``ids``. */
const parentIn = (d: DatasetSummary, ids: Set<string>) => {
  const p = d.parent_id ?? d.derived_from;
  return p && p !== d.id && ids.has(p) ? p : null;
};

export function layoutTree(f: FileSummary, colourOf: Map<string, number>): FileTree {
  const ids = new Set(f.datasets.map((d) => d.id));
  const under = new Map<string | null, DatasetSummary[]>();
  for (const d of f.datasets) {
    const p = parentIn(d, ids);
    under.set(p, [...(under.get(p) ?? []), d]);
  }
  const level = (ds: DatasetSummary[], parent: string | null): RowItem[] => {
    const items = layoutFile(ds, colourOf);
    const byLabel = new Map<string, DatasetSummary[]>();
    for (const it of items) {
      if (it.type !== "dataset") continue;
      const label = kindLabel(it.d);
      byLabel.set(label, [...(byLabel.get(label) ?? []), it.d]);
    }
    const out: RowItem[] = [];
    const placed = new Set<string>();
    for (const it of items) {
      if (it.type !== "dataset") {
        out.push(it);
        continue;
      }
      const label = kindLabel(it.d);
      const same = byLabel.get(label)!;
      if (same.length < 2) out.push(it);
      else if (!placed.has(label)) {
        placed.add(label);
        // The top level keeps the key its folding was saved under.
        const key = typeGroupKey(f.path, parent === null ? label : `${parent}\n${label}`);
        out.push({ type: "types", key, label, ds: same });
      }
    }
    return out;
  };
  const children = new Map<string, RowItem[]>();
  for (const [p, ds] of under) if (p !== null) children.set(p, level(ds, p));
  return { rows: level(under.get(null) ?? [], null), children };
}

/** The datasets in the order the list shows them. With ``collapsed`` (folded
 * type groups) and ``folded`` (datasets whose children are hidden), the ones
 * hidden are left out. */
export function displayDatasets(
  files: FileSummary[], collapsed?: Record<string, true>, folded?: Record<string, true>,
): DatasetSummary[] {
  const { colourOf, items } = layoutPane(files);
  return items.flatMap(paneFiles).flatMap((f) => {
    const { rows, children } = layoutTree(f, colourOf);
    const node = (d: DatasetSummary): DatasetSummary[] => [
      d,
      ...(folded?.[d.id] ? [] : flat(children.get(d.id) ?? [])),
    ];
    const flat = (rs: RowItem[]): DatasetSummary[] =>
      rs.flatMap((it) =>
        it.type === "dataset" ? node(it.d) : it.type === "types" && collapsed?.[it.key] ? [] : it.ds.flatMap(node),
      );
    return flat(rows);
  });
}

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

/** Width × height of the 2D plane a dataset is shown on. */
export function planeSize(d: DatasetSummary): [number, number] {
  const shape = d.kind === "spectrum_image" ? d.navigation_shape : d.signal_shape;
  return [shape[0] ?? 0, shape[1] ?? 0];
}

/** Why ``d`` can't be shown together with ``current``, or null if it can.
 * Images and maps go over a spectrum image (same scan size) or side by side
 * with other images (any size). Anything else is shown alone. */
function combineError(current: DatasetSummary[], d: DatasetSummary): string | null {
  if (d.kind === "other") return `${d.title} can't be displayed.`;
  const all = [...current, d];
  const primaries = all.filter((x) => x.kind !== "image");
  if (primaries.length > 1) {
    return "Only one spectrum image, stack or spectrum can be shown at a time. Click it without ⌘ to switch.";
  }
  const images = all.filter((x) => x.kind === "image");
  const primary = primaries[0];
  if (!primary) return null;
  if (isLineScan(primary) && images.length) return "Images can't be shown with a line scan.";
  if (primary.kind !== "spectrum_image" && images.length) return `Images can't be shown with a ${primary.kind}.`;
  const [w, h] = planeSize(primary);
  for (const im of images) {
    const [iw, ih] = planeSize(im);
    if (iw !== w || ih !== h) return `${im.title} is ${iw}×${ih} px; ${primary.title} is ${w}×${h} px.`;
  }
  return null;
}

/** A newly opened file with an EELS dataset linked to one in another file
 * (the others' summaries changed too). */
const linkedElsewhere = (f: FileSummary) =>
  f.datasets.some((d) => d.eels?.linked.some((id) => !f.datasets.some((x) => x.id === id)));

const notify = (msg: string) => useLayers.getState().notify(msg);

// Each file lists its spectrum image, spectrum and stack first, then plain
// images, then element maps (otherwise in file order).
const KIND_ORDER: Record<string, number> = { spectrum_image: 0, spectrum: 1, stack: 2, image: 3, other: 5 };
const rank = (d: DatasetSummary) =>
  isWindowMap(d) ? 4.5 : d.is_element_map ? 4 : (KIND_ORDER[d.kind] ?? 5);

const madeByProcess = (d: DatasetSummary) =>
  d.origin === "background" || d.origin === "background-r" || d.origin === "thickness" || d.origin === "zlp-shift" ||
  d.origin === "deconvolved";

/** A file's datasets in list order; what background removal made (the new
 * spectrum image, then its r map), the thickness and the ZLP shift sit right under their source. */
function sortFile(f: FileSummary): FileSummary {
  const sorted = [...f.datasets].sort((a, b) => rank(a) - rank(b));
  if (!sorted.some(madeByProcess)) return { ...f, datasets: sorted };
  const ids = new Set(sorted.map((d) => d.id));
  const out: DatasetSummary[] = [];
  const add = (d: DatasetSummary) => {
    out.push(d);
    sorted.filter((x) => madeByProcess(x) && x.parent_id === d.id).forEach(add);
  };
  sorted.filter((d) => !madeByProcess(d) || !ids.has(d.parent_id ?? "")).forEach(add);
  return { ...f, datasets: out };
}

/** Pre-assign a colour to every element map (one colour per element) and window map. */
function assignElementColours(files: FileSummary[]) {
  const { ensureElementColour, ensureWindowColour } = useDisplay.getState();
  for (const d of allDatasets(files)) {
    if (d.is_element_map) ensureElementColour(elementOf(d));
    else if (isWindowMap(d)) ensureWindowColour(d.id);
  }
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

    syncFiles: async () => {
      let fresh: FileSummary[];
      try {
        fresh = await api.listFiles();
      } catch (e) {
        notify(`Could not refresh the files: ${(e as Error).message}`);
        return;
      }
      const have = new Map(get().files.map((f) => [f.id, f]));
      for (const f of fresh) {
        const old = have.get(f.id);
        if (old && JSON.stringify(old) !== JSON.stringify(sortFile(f))) get().replaceFile(f);
      }
    },

    linkEels: async (d, partnerId) => {
      const { files } = await api.setEelsPartner(d.id, partnerId);
      files.forEach((f) => get().replaceFile(f));
    },

    setEelsRole: async (d, role) => {
      const { files } = await api.setEelsRole(d.id, role);
      files.forEach((f) => get().replaceFile(f));
    },

    // Files open one at a time so each appears in the list as soon as it loads.
    openFiles: async (paths, experimentId) => {
      if (!paths.length || get().progress) return;
      set({ errors: [] });
      const errors: string[] = [];
      const opened: FileSummary[] = [];
      for (const [i, path] of paths.entries()) {
        set({ progress: { done: i, total: paths.length, current: baseName(path) } });
        try {
          const entry = sortFile(await api.openFile(path, experimentId));
          const files = get().files.filter((f) => f.id !== entry.id).concat(entry);
          assignElementColours([entry]);
          set({ files });
          opened.push(entry);
          // Follow the newest file, so the user sees what they just opened
          // (sorted, so a spectrum image comes first if there is one).
          const first = entry.datasets[0];
          if (first && !experimentId) get().select(first.id);
          // Linking changes the others (and their roles in the experiment).
          if (experimentId || linkedElsewhere(entry)) await get().syncFiles();
        } catch (e) {
          errors.push(`${baseName(path)}: ${(e as Error).message}`);
        }
      }
      // An experiment: show its (first) spectrum image.
      const all = opened.flatMap((f) => f.datasets);
      const shown = all.find((d) => d.kind === "spectrum_image") ?? all[0];
      if (experimentId && shown) get().select(shown.id);
      set({ progress: null, errors });
    },

    openFolder: async (folder) => {
      if (get().progress) return;
      let plan: Awaited<ReturnType<typeof api.openFolder>>;
      try {
        plan = await api.openFolder(folder);
      } catch (e) {
        set({ errors: [`${baseName(folder)}: ${(e as Error).message}`] });
        return;
      }
      if (!plan.paths.length) {
        set({ errors: [`${baseName(folder)}: no files to open`] });
        return;
      }
      await get().openFiles(plan.paths, plan.experiment.id);
    },

    openExample: async (id, title) => {
      if (get().progress) return;
      set({ errors: [], progress: { done: 0, total: 1, current: title } });
      try {
        const entry = sortFile(await api.openExample(id));
        assignElementColours([entry]);
        set({ files: get().files.filter((f) => f.id !== entry.id).concat(entry) });
        if (entry.datasets[0]) get().select(entry.datasets[0].id);
        if (linkedElsewhere(entry)) await get().syncFiles();
      } catch (e) {
        set({ errors: [`${title}: ${(e as Error).message}`] });
      } finally {
        set({ progress: null });
      }
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
        useProfiles.getState().reset(changed.map((d) => d.id));
        useFftBoxes.getState().reset(changed.map((d) => d.id));
        changed.forEach((d) => d.kind === "spectrum_image" && useSi.getState().reset(d.id));
      }
      const files = get().files.map((f) => (f.id === file.id ? sorted : f));
      const byId = new Map(allDatasets(files).map((d) => [d.id, d]));
      // Drop highlighted datasets that are gone, or that no longer fit the
      // view's size (e.g. a HAADF image over a rebinned spectrum image).
      const kept = get().selection.map((id) => byId.get(id)).filter((d): d is DatasetSummary => !!d);
      const primary = kept.find((d) => d.kind !== "image");
      const [bw, bh] = primary ? planeSize(primary) : [0, 0];
      const selection = kept
        .filter((d) => !primary || d === primary || (planeSize(d)[0] === bw && planeSize(d)[1] === bh))
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
      const r = await api.rebin(d.id, factors);
      get().replaceFile(r.file);
      // The datasets linked to it are rebinned with it.
      if (d.eels?.linked.length || r.dataset.eels?.linked.length) await get().syncFiles();
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
    // Datasets hidden in folded type groups or under folded datasets are skipped.
    selectRelative: (step) => {
      const files = get().files;
      const list = displayDatasets(files);
      const { collapsedTypes, foldedNodes } = useUi.getState();
      const visible = new Set(displayDatasets(files, collapsedTypes, foldedNodes).map((d) => d.id));
      if (!visible.size) return;
      const i = list.findIndex((d) => d.id === get().focusId);
      let j = i < 0 ? -1 : i;
      // Nearest visible row in the direction of travel (the focus itself may be hidden).
      for (let k = i < 0 ? 0 : i + step; k >= 0 && k < list.length; k += step) {
        if (visible.has(list[k].id)) {
          j = k;
          break;
        }
      }
      if (j < 0) j = list.findIndex((d) => visible.has(d.id));
      const next = list[j];
      if (next.kind === "image") set({ focusId: next.id });
      else if (get().selection.includes(next.id)) set({ focusId: next.id });
      else get().select(next.id);
    },

    clearErrors: () => set({ errors: [] }),
  };
});

/** Datasets made in the app that can be deleted: maps, thickness, ZLP shifts and background-removed SIs (with their r maps). */
export const canDelete = (d: DatasetSummary) =>
  d.is_element_map || isWindowMap(d) || d.origin === "background" || d.origin === "background-r" ||
  d.origin === "thickness" || d.origin === "zlp-shift" || d.origin === "deconvolved";

/** Maps shown in a solid colour: element maps and window maps. */
export const isColourMap = (d: DatasetSummary) => d.is_element_map || isWindowMap(d);

/** Element symbol of an element map (computed maps are titled e.g. "Fe Kα"). */
export const elementOf = (d: DatasetSummary) => d.element ?? d.title;

/** Colormap an image starts with: its element colour for element maps. */
export function initialColormap(d: DatasetSummary): string {
  if (isWindowMap(d)) return useDisplay.getState().ensureWindowColour(d.id);
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
