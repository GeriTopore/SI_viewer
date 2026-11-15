import { create } from "zustand";
import type { EelsFitOptions, EelsFitResult, JobSummary } from "../api/client";
import type { SourceChoice } from "./quant";

/** The fit of one spectrum: the whole, the cursor or a region (its id). */
export interface StoredEelsFit {
  source: SourceChoice;
  label: string;
  result: EelsFitResult;
}

export const EELS_FIT_DEFAULTS: EelsFitOptions = {
  gos: "dft", fine_structure: false, fine_structure_width: 30, onsets: {}, lo: null, hi: null, convolve: true,
  bg_lo: null, bg_hi: null,
};

interface EelsFitState {
  /** Settings per dataset id (onsets and the range belong to its spectrum). */
  options: Record<string, EelsFitOptions>;
  /** Latest fits per dataset, one per spectrum fitted (key: "id@revision", as ``datasetKey``). */
  fits: Record<string, StoredEelsFit[]>;
  /** The spectra fitted together, per dataset key (default: the whole). */
  sources: Record<string, SourceChoice[]>;
  /** Per-pixel fit job per dataset key, and whether it also makes at% maps. */
  jobs: Record<string, JobSummary>;
  composition: boolean;
  /** The dataset whose EELS fit panel is open: its background window is drawn. */
  shown: string | null;
  setShown: (id: string | null) => void;
  setOptions: (id: string, patch: Partial<EelsFitOptions>) => void;
  setFits: (key: string, fits: StoredEelsFit[]) => void;
  setSources: (key: string, sources: SourceChoice[]) => void;
  setJob: (key: string, job: JobSummary) => void;
  setComposition: (on: boolean) => void;
}

export const useEelsFit = create<EelsFitState>((set) => ({
  options: {},
  fits: {},
  sources: {},
  jobs: {},
  composition: true,
  shown: null,
  setShown: (shown) => set({ shown }),
  setOptions: (id, patch) =>
    set((s) => ({ options: { ...s.options, [id]: { ...(s.options[id] ?? EELS_FIT_DEFAULTS), ...patch } } })),
  setFits: (key, fits) => set((s) => ({ fits: { ...s.fits, [key]: fits } })),
  setSources: (key, sources) => set((s) => ({ sources: { ...s.sources, [key]: sources } })),
  setJob: (key, job) => set((s) => ({ jobs: { ...s.jobs, [key]: job } })),
  setComposition: (composition) => set({ composition }),
}));
