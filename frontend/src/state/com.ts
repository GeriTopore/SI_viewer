import { create } from "zustand";
import { api, type JobSummary } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** DPC from 4D-STEM data's centre of mass: the backend's
 * ``processing/com.py`` ``ComOptions``. */
export interface ComOptions {
  beam: { method: "center_of_mass"; threshold: number; half_width: number | null };
  plane: { mode: "none" | "plane" | "corners"; corners: number };
  rotation: { auto: boolean; theta: number; mirror: boolean };
}

export const DEFAULT_COM: ComOptions = {
  beam: { method: "center_of_mass", threshold: 1, half_width: null },
  plane: { mode: "plane", corners: 0.1 },
  rotation: { auto: true, theta: 0, mirror: false },
};

/** What a run found: the rotation used, and the curl left with the one found. */
export interface ComResult {
  theta: number;
  mirror: boolean;
  curl_ratio: number;
  found: { theta: number; mirror: boolean };
}

const ACTIVE = new Set(["queued", "running"]);

interface State {
  options: Record<string, ComOptions>;
  results: Record<string, ComResult | undefined>;
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  setOptions: (id: string, patch: { beam?: Partial<ComOptions["beam"]>; plane?: Partial<ComOptions["plane"]>; rotation?: Partial<ComOptions["rotation"]> }) => void;
  run: (id: string) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const comOptionsOf = (s: State, id: string) => s.options[id] ?? DEFAULT_COM;
export const comJobRunning = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);

export const useCom = create<State>((set, get) => ({
  options: {},
  results: {},
  jobs: {},
  errors: {},
  setOptions: (id, patch) =>
    set((s) => {
      const o = comOptionsOf(s, id);
      return {
        options: {
          ...s.options,
          [id]: { beam: { ...o.beam, ...patch.beam }, plane: { ...o.plane, ...patch.plane }, rotation: { ...o.rotation, ...patch.rotation } },
        },
      };
    }),
  run: async (id) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await api.comDpc(id, comOptionsOf(get(), id));
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
        const r = job.result as ComResult & { file: Parameters<ReturnType<typeof useDatasets.getState>["replaceFile"]>[0] };
        useDatasets.getState().replaceFile(r.file);
        set((s) => ({ results: { ...s.results, [id]: { theta: r.theta, mirror: r.mirror, curl_ratio: r.curl_ratio, found: r.found } } }));
        // The rotation used stays in the boxes, to adjust from.
        get().setOptions(id, { rotation: { theta: Math.round(r.theta * 10) / 10, mirror: r.mirror } });
        useLayers.getState().notify("DPC (COM) and iCOM made under the data.");
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
