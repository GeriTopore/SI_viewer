import { create } from "zustand";
import { api, type JobSummary } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";

const ACTIVE = new Set(["queued", "running"]);

interface InsituStore {
  /** The frame being read (0-based), per dataset; absent when none. */
  reading: Record<string, number>;
  /** Why the last frame could not be shown. */
  errors: Record<string, string | null>;
  /** The latest frame-range job per dataset. */
  jobs: Record<string, JobSummary>;
  /** Show frame ``k``: one request at a time, the latest wanted frame wins
   * (frames asked for in between are skipped). */
  showFrame: (id: string, k: number) => void;
  /** Mean or sum of frames start…end (0-based): a job whose result becomes the view. */
  reduce: (id: string, mode: "mean" | "sum", start: number, end: number) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

// Latest frame wanted per dataset, and the datasets with a request in flight.
const wanted = new Map<string, number>();
const busy = new Set<string>();

export const useInsitu = create<InsituStore>((set, get) => {
  const setJob = (id: string, job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
  const setError = (id: string, e: string | null) => set((s) => ({ errors: { ...s.errors, [id]: e } }));
  const viewOf = (id: string) => allDatasets(useDatasets.getState().files).find((d) => d.id === id)?.insitu?.view;

  async function pump(id: string) {
    if (busy.has(id)) return;
    busy.add(id);
    try {
      while (wanted.has(id)) {
        const k = wanted.get(id)!;
        wanted.delete(id);
        const v = viewOf(id);
        if (v && v.mode === "frame" && v.start === k) continue; // already there
        set((s) => ({ reading: { ...s.reading, [id]: k } }));
        try {
          useDatasets.getState().replaceFile((await api.setInsituFrame(id, k)).file);
          setError(id, null);
        } catch (e) {
          setError(id, (e as Error).message);
        }
      }
    } finally {
      busy.delete(id);
      set((s) => {
        const { [id]: _gone, ...reading } = s.reading;
        return { reading };
      });
    }
  }

  return {
    reading: {},
    errors: {},
    jobs: {},

    showFrame: (id, k) => {
      wanted.set(id, k);
      void pump(id);
    },

    reduce: async (id, mode, start, end) => {
      setError(id, null);
      try {
        let job = await api.reduceInsitu(id, mode, start, end);
        setJob(id, job);
        // Polled here, not in the bar, so it finishes if the dataset is left.
        while (ACTIVE.has(job.status)) {
          await new Promise((r) => setTimeout(r, 300));
          try {
            job = await api.getJob(job.id);
            if (ACTIVE.has(job.status)) setJob(id, job);
          } catch {
            /* backend busy or restarted: try again */
          }
        }
        // The new view first, then the job's end: the bar never shows the old view in between.
        if (job.status === "done" && job.result) useDatasets.getState().replaceFile(job.result.file);
        else if (job.status === "failed") setError(id, job.error);
        setJob(id, job);
      } catch (e) {
        setError(id, (e as Error).message);
      }
    },

    cancel: async (id) => {
      const job = get().jobs[id];
      if (job) setJob(id, await api.cancelJob(job.id));
    },
  };
});

/** Whether a frame-range job of ``id`` is running. */
export const reducing = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);
