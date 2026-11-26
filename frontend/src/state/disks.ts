import { create } from "zustand";
import { api, type JobSummary } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** Bragg disk detection options: the backend's ``processing/disks.py``
 * ``DiskOptions``. */
export interface DiskOptions {
  /** "probe": the bright-field disk of the mean pattern; "disc": a flat disc of ``radius``. */
  template: "probe" | "disc";
  /** Disc radius (pattern px); null: the bright-field disk's. */
  radius: number | null;
  /** Lowest correlation a disk may have (−1…1). */
  threshold: number;
  min_spacing: number;
  edge: number;
  max_disks: number;
  subpixel: boolean;
}

export const DEFAULT_DISK_OPTIONS: DiskOptions = {
  template: "probe", radius: null, threshold: 0.4, min_spacing: 5, edge: 2, max_disks: 100, subpixel: true,
};

/** Disks found in a pattern: [ky, kx, score] (pattern px, pixel i spanning [i, i + 1)). */
export type Disk = [number, number, number];

const ACTIVE = new Set(["queued", "running"]);

interface DisksState {
  options: Record<string, DiskOptions>;
  /** The disks are found live on the patterns shown (while tuning). */
  preview: Record<string, boolean>;
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  setOptions: (id: string, patch: Partial<DiskOptions>) => void;
  setPreview: (id: string, on: boolean) => void;
  /** Find the disks of every pattern (a job); the table and its map join the dataset list. */
  run: (id: string) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const diskOptionsOf = (s: DisksState, id: string) => s.options[id] ?? DEFAULT_DISK_OPTIONS;

export const useDisks = create<DisksState>((set, get) => ({
  options: {},
  preview: {},
  jobs: {},
  errors: {},
  setOptions: (id, patch) => set((s) => ({ options: { ...s.options, [id]: { ...diskOptionsOf(s, id), ...patch } } })),
  setPreview: (id, on) => set((s) => ({ preview: { ...s.preview, [id]: on } })),
  run: async (id) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await api.findDisks(id, diskOptionsOf(get(), id));
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
        useLayers.getState().notify(`${job.result.count} disks found: Bragg vectors and their map added to the dataset pane.`);
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

export const diskJobRunning = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);
