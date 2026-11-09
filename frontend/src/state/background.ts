import { useMemo } from "react";
import { create } from "zustand";
import { api, type BackgroundOptions, type DatasetSummary, type JobSummary } from "../api/client";
import { spectralAxis } from "../views/geometry";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** Settings of the Background removal tool for one dataset. */
export interface BgState {
  /** The windows and the preview are drawn on the spectrum plot. */
  on: boolean;
  /** The plot shows the remainders (spectrum − background) over the signal window. */
  subtracted: boolean;
  options: BackgroundOptions;
}

/** What the last preview told: the power-law exponent at the cursor, or why it failed. */
export interface BgInfo {
  r: number | null;
  error: string | null;
}

/** Ids of the two bands on the plot. */
export const BG_BAND = "bg:background";
export const SIGNAL_BAND = "bg:signal";
export const isBgBand = (id: string) => id === BG_BAND || id === SIGNAL_BAND;

const ACTIVE = new Set(["queued", "running"]);

/** EELS spectra and spectrum images (incl. line scans) can have their background removed. */
export const canRemoveBackground = (d: DatasetSummary) =>
  !!d.background ||
  (!d.is_element_map && d.technique === "EELS" && (d.kind === "spectrum" || d.kind === "spectrum_image"));

/** The energy axis as [first, last, step, units] (null if there is none). */
export function energyRange(d: DatasetSummary): { e0: number; e1: number; step: number; units: string } | null {
  const a = spectralAxis(d);
  if (!a || a.size < 2 || !(a.scale !== 0)) return null;
  const e1 = a.offset + (a.size - 1) * a.scale;
  return { e0: Math.min(a.offset, e1), e1: Math.max(a.offset, e1), step: Math.abs(a.scale), units: a.units };
}

/** ``v`` to the nearest channel, without floating-point noise. */
export function snapEnergy(e0: number, step: number, v: number): number {
  const digits = Math.min(8, Math.max(0, Math.ceil(-Math.log10(step)) + 2));
  return Number((e0 + Math.round((v - e0) / step) * step).toFixed(digits));
}

/** Why the windows can't be used, or null. */
export function validateWindows(o: BackgroundOptions): string | null {
  const [b0, b1] = o.background;
  const [s0, s1] = o.signal;
  if (![b0, b1, s0, s1].every(Number.isFinite)) return "Enter all four limits.";
  if (b1 <= b0) return "The background window needs a start below its end.";
  if (s1 <= s0) return "The signal window needs a start below its end.";
  if (b1 > s0) return "The background window must end before the signal starts.";
  return null;
}

/** Background [30%, 40%] and signal [42%, 65%] of the energy range. */
function defaultState(d: DatasetSummary): BgState {
  if (d.background) {
    const { model, order, fast, background, signal } = d.background;
    return { on: false, subtracted: false, options: { model, order, fast, background, signal } };
  }
  const r = energyRange(d) ?? { e0: 0, e1: 1, step: 0.01, units: "" };
  const w = r.e1 - r.e0;
  const at = (f: number) => snapEnergy(r.e0, r.step, r.e0 + f * w);
  return {
    on: false,
    subtracted: false,
    options: {
      model: "power_law", order: 2, fast: true,
      background: [at(0.3), at(0.4)],
      signal: [at(0.42), at(0.65)],
    },
  };
}

interface BgStore {
  byId: Record<string, BgState>;
  info: Record<string, BgInfo>;
  /** The latest removal job per dataset (the one being processed). */
  jobs: Record<string, JobSummary>;
  /** Why a job could not even start. */
  startError: Record<string, string | null>;
  patch: (d: DatasetSummary, patch: Partial<BgState>) => void;
  setOptions: (d: DatasetSummary, patch: Partial<BackgroundOptions>) => void;
  setInfo: (id: string, info: BgInfo) => void;
  /** Start the removal job (``rerun``: replace the background-removed spectrum
   * image ``d`` in place) and follow it; the new spectrum image is selected. */
  run: (d: DatasetSummary, rerun: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const useBackground = create<BgStore>((set, get) => {
  const current = (d: DatasetSummary) => get().byId[d.id] ?? defaultState(d);
  const setJob = (id: string, job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
  const setStartError = (id: string, e: string | null) =>
    set((s) => ({ startError: { ...s.startError, [id]: e } }));

  return {
    byId: {},
    info: {},
    jobs: {},
    startError: {},
    patch: (d, patch) => set((s) => ({ byId: { ...s.byId, [d.id]: { ...current(d), ...patch } } })),
    setOptions: (d, patch) =>
      set((s) => {
        const cur = current(d);
        return { byId: { ...s.byId, [d.id]: { ...cur, options: { ...cur.options, ...patch } } } };
      }),
    setInfo: (id, info) =>
      set((s) => {
        const old = s.info[id];
        if (old && old.r === info.r && old.error === info.error) return s;
        return { info: { ...s.info, [id]: info } };
      }),

    run: async (d, rerun) => {
      const options = current(d).options;
      setStartError(d.id, null);
      try {
        let job = await (rerun ? api.rerunBackground(d.id, options) : api.removeBackground(d.id, options));
        setJob(d.id, job);
        // Polled here, not in the Process tab, so it finishes if the tab is closed.
        while (ACTIVE.has(job.status)) {
          await new Promise((r) => setTimeout(r, 300));
          try {
            job = await api.getJob(job.id);
            setJob(d.id, job);
          } catch {
            /* backend busy or restarted: try again */
          }
        }
        if (job.status === "done" && job.result) {
          const { replaceFile, select } = useDatasets.getState();
          replaceFile(job.result.file);
          if (!rerun && job.result.dataset_id) select(job.result.dataset_id);
          useLayers.getState().notify(rerun ? "Background removal re-run." : "Background-removed spectrum image added.");
        }
      } catch (e) {
        setStartError(d.id, (e as Error).message);
      }
    },

    cancel: async (id) => {
      const job = get().jobs[id];
      if (job) setJob(id, await api.cancelJob(job.id));
    },
  };
});

/** The tool's settings for ``d``: what was set, else the defaults. */
export function useBgState(d: DatasetSummary): BgState {
  const stored = useBackground((s) => s.byId[d.id]);
  const axis = spectralAxis(d);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const fallback = useMemo(() => defaultState(d), [d.id, d.background, axis?.size, axis?.offset, axis?.scale]);
  return stored ?? fallback;
}
