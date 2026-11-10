import { create } from "zustand";
import type { EelsFitOptions, EelsFitResult, JobSummary } from "../api/client";

export interface StoredEelsFit {
  label: string;
  result: EelsFitResult;
}

export const EELS_FIT_DEFAULTS: EelsFitOptions = {
  gos: "dft", fine_structure: false, fine_structure_width: 30, onsets: {}, lo: null, hi: null, convolve: true,
};

interface EelsFitState {
  /** Settings per dataset id (onsets and the range belong to its spectrum). */
  options: Record<string, EelsFitOptions>;
  /** Latest fit per dataset (key: "id@revision", as ``datasetKey``). */
  fits: Record<string, StoredEelsFit>;
  /** Per-pixel fit job per dataset key, and whether it also makes at% maps. */
  jobs: Record<string, JobSummary>;
  composition: boolean;
  setOptions: (id: string, patch: Partial<EelsFitOptions>) => void;
  setFit: (key: string, fit: StoredEelsFit) => void;
  setJob: (key: string, job: JobSummary) => void;
  setComposition: (on: boolean) => void;
}

export const useEelsFit = create<EelsFitState>((set) => ({
  options: {},
  fits: {},
  jobs: {},
  composition: true,
  setOptions: (id, patch) =>
    set((s) => ({ options: { ...s.options, [id]: { ...(s.options[id] ?? EELS_FIT_DEFAULTS), ...patch } } })),
  setFit: (key, fit) => set((s) => ({ fits: { ...s.fits, [key]: fit } })),
  setJob: (key, job) => set((s) => ({ jobs: { ...s.jobs, [key]: job } })),
  setComposition: (composition) => set({ composition }),
}));
