import { create } from "zustand";
import { api, type JobSummary } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** Finding the direct beam: the backend's ``processing/centring.py`` ``BeamOptions``. */
export interface BeamOptions {
  method: "center_of_mass" | "blur";
  /** Centre of mass: intensities below this × the mean left out. */
  threshold: number;
  /** Blur: the Gaussian's sigma (px). */
  sigma: number;
  /** Half the side of the square searched around the bright-field disk (px); null: 3 × its radius. */
  half_width: number | null;
}

/** The descan used for centring: ``PlaneOptions``. */
export interface PlaneOptions {
  mode: "none" | "plane" | "corners";
  corners: number;
}

export const DEFAULT_BEAM: BeamOptions = { method: "center_of_mass", threshold: 1, sigma: 2, half_width: null };
export const DEFAULT_PLANE: PlaneOptions = { mode: "plane", corners: 0.1 };

const ACTIVE = new Set(["queued", "running"]);
export const jobRunning = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);

interface CentringState {
  beam: Record<string, BeamOptions>;
  plane: Record<string, PlaneOptions>;
  /** The running or last job of each dataset (finding, or centring). */
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  setBeam: (id: string, patch: Partial<BeamOptions>) => void;
  setPlane: (id: string, patch: Partial<PlaneOptions>) => void;
  /** Find the beam in every pattern (a job): its shift images join the dataset list. */
  find: (id: string) => Promise<void>;
  /** Shift every pattern so the beam is at the centre (a job). */
  centre: (id: string, inPlace: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const beamOf = (s: CentringState, id: string) => s.beam[id] ?? DEFAULT_BEAM;
export const planeOf = (s: CentringState, id: string) => s.plane[id] ?? DEFAULT_PLANE;

export const useCentring = create<CentringState>((set, get) => {
  /** Run a job to its end, kept in ``jobs``; ``done`` with its result. */
  const follow = async (id: string, start: () => Promise<JobSummary>, done: (job: JobSummary) => void) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await start();
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
      if (job.status === "done" && job.result) done(job);
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    }
  };
  return {
    beam: {},
    plane: {},
    jobs: {},
    errors: {},
    setBeam: (id, patch) => set((s) => ({ beam: { ...s.beam, [id]: { ...beamOf(s, id), ...patch } } })),
    setPlane: (id, patch) => set((s) => ({ plane: { ...s.plane, [id]: { ...planeOf(s, id), ...patch } } })),
    find: (id) =>
      follow(id, () => api.findBeam(id, beamOf(get(), id)), (job) => {
        useDatasets.getState().replaceFile(job.result!.file);
        useLayers.getState().notify("Beam found: Beam shift x and y added to the dataset pane.");
      }),
    centre: (id, inPlace) =>
      follow(id, () => api.centreBeam(id, planeOf(get(), id), inPlace), (job) => {
        useDatasets.getState().replaceFile(job.result!.file);
        if (!inPlace && job.result!.id) useDatasets.getState().select(job.result!.id);
        useLayers.getState().notify(inPlace ? "Patterns centred." : "Centred copy added to the dataset pane.");
      }),
    cancel: async (id) => {
      const job = get().jobs[id];
      if (job) {
        set((s) => ({ jobs: { ...s.jobs, [id]: undefined } }));
        await api.cancelJob(job.id);
      }
    },
  };
});
