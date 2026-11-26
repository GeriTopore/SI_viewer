import { create } from "zustand";
import { api, type JobSummary } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** Background subtraction of diffraction patterns: the backend's
 * ``processing/pattern_background.py`` ``PatternBackgroundOptions`` (pyxem's
 * ``subtract_diffraction_background`` methods). */
export interface PatternBackgroundOptions {
  method: "median kernel" | "difference of gaussians" | "radial median" | "h-dome";
  /** Median kernel: its size (px). */
  footprint: number;
  /** Difference of Gaussians: the narrow and wide blurs (px). */
  min_sigma: number;
  max_sigma: number;
  /** h-dome: the peaks' height kept (of the pattern's maximum). */
  h: number;
}

export const DEFAULT_PATTERN_BACKGROUND: PatternBackgroundOptions = {
  method: "median kernel", footprint: 19, min_sigma: 1, max_sigma: 55, h: 0.25,
};

const ACTIVE = new Set(["queued", "running"]);

interface State {
  options: Record<string, PatternBackgroundOptions>;
  /** The patterns shown with their background subtracted (while tuning). */
  preview: Record<string, boolean>;
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  setOptions: (id: string, patch: Partial<PatternBackgroundOptions>) => void;
  setPreview: (id: string, on: boolean) => void;
  /** Subtract it from every pattern (a job), in place or as a copy. */
  run: (id: string, inPlace: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const backgroundOptionsOf = (s: State, id: string) => s.options[id] ?? DEFAULT_PATTERN_BACKGROUND;
export const backgroundJobRunning = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);

export const usePatternBackground = create<State>((set, get) => ({
  options: {},
  preview: {},
  jobs: {},
  errors: {},
  setOptions: (id, patch) => set((s) => ({ options: { ...s.options, [id]: { ...backgroundOptionsOf(s, id), ...patch } } })),
  setPreview: (id, on) => set((s) => ({ preview: { ...s.preview, [id]: on } })),
  run: async (id, inPlace) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await api.subtractBackground(id, backgroundOptionsOf(get(), id), inPlace);
      setJob(job);
      // Polled here, not in the Process tab, so it finishes if the tab is closed.
      while (ACTIVE.has(job.status)) {
        await new Promise((r) => setTimeout(r, 300));
        try {
          job = await api.getJob(job.id);
          setJob(job);
        } catch {
          /* backend busy or restarted: try again */
        }
      }
      if (job.status === "done" && job.result) {
        useDatasets.getState().replaceFile(job.result.file);
        // Done in place: the patterns shown are the result (no preview on top of it).
        if (job.result.id === id) set((s) => ({ preview: { ...s.preview, [id]: false } }));
        useLayers.getState().notify(job.result.id === id ? "Background subtracted." : "Background subtracted: a copy added under the data.");
      }
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    }
  },
  cancel: async (id) => {
    const job = get().jobs[id];
    if (job) set((s) => ({ jobs: { ...s.jobs, [id]: undefined } })), await api.cancelJob(job.id);
  },
}));
