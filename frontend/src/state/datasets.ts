import { create } from "zustand";
import { api, type DatasetSummary, type FileSummary } from "../api/client";

interface OpenProgress {
  done: number;
  total: number;
  current: string;
}

interface DatasetsState {
  files: FileSummary[];
  selectedId: string | null;
  progress: OpenProgress | null;
  errors: string[];
  refresh: () => Promise<void>;
  openFiles: (paths: string[]) => Promise<void>;
  closeFile: (fileId: string) => Promise<void>;
  closeAll: () => Promise<void>;
  select: (datasetId: string | null) => void;
  selectRelative: (step: 1 | -1) => void;
  clearErrors: () => void;
}

export const allDatasets = (files: FileSummary[]): DatasetSummary[] =>
  files.flatMap((f) => f.datasets);

const baseName = (path: string) => path.split(/[\\/]/).pop() ?? path;

export const useDatasets = create<DatasetsState>((set, get) => ({
  files: [],
  selectedId: null,
  progress: null,
  errors: [],

  refresh: async () => {
    try {
      set({ files: await api.listFiles() });
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
        const entry = await api.openFile(path);
        const files = get().files.filter((f) => f.id !== entry.id).concat(entry);
        const first =
          entry.datasets.find((d) => d.kind === "spectrum_image") ?? entry.datasets[0];
        // Follow the newest file, so the user sees what they just opened.
        set({ files, selectedId: first?.id ?? get().selectedId });
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
    const stillThere = allDatasets(files).some((d) => d.id === get().selectedId);
    set({ files, selectedId: stillThere ? get().selectedId : null });
  },

  closeAll: async () => {
    for (const f of get().files) await get().closeFile(f.id);
  },

  select: (selectedId) => set({ selectedId }),

  selectRelative: (step) => {
    const list = allDatasets(get().files);
    if (!list.length) return;
    const i = list.findIndex((d) => d.id === get().selectedId);
    const next = i < 0 ? 0 : Math.min(list.length - 1, Math.max(0, i + step));
    set({ selectedId: list[next].id });
  },

  clearErrors: () => set({ errors: [] }),
}));

export const useSelectedDataset = (): DatasetSummary | undefined =>
  useDatasets((s) => allDatasets(s.files).find((d) => d.id === s.selectedId));
