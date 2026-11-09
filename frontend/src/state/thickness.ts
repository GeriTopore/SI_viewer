import { useMemo } from "react";
import { create } from "zustand";
import { api, type DatasetSummary, type FileSummary, type JobSummary, type ThicknessOptions } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useLayers } from "./layers";
import { energyRange, snapEnergy } from "./background";

/** Settings of the Thickness tool for one low loss. */
export interface ThicknessState {
  /** The zero-loss threshold is drawn on the low-loss spectrum. */
  on: boolean;
  /** "auto": estimated at every pixel; else one energy for all pixels. */
  threshold: "auto" | number;
  /** Auto: how far from the zero-loss peak its end is looked for. */
  window: number;
  output: "relative" | "absolute";
  /** Absolute thickness is given by a mean free path (nm) or a density (g/cm³). */
  by: "mean_free_path" | "density";
  mean_free_path: number | null;
  density: number | null;
}

/** The automatic threshold of the summed spectrum, as last estimated. */
export interface ThresholdEstimate {
  /** What it was estimated for (data version and search window). */
  key: string;
  value: number | null;
  error: string | null;
}

/** Id of the threshold line on the plot. */
export const THRESHOLD_LINE = "thickness:threshold";

const ACTIVE = new Set(["queued", "running"]);

/** EELS spectrum images and line scans: thickness is estimated from the low loss of their group. */
export const canEstimateThickness = (d: DatasetSummary) =>
  d.kind === "spectrum_image" && !!d.eels && d.technique === "EELS" && !d.is_element_map;

/** The low loss that thickness of ``d`` is estimated from: ``d`` itself or the low loss of its group. */
export function lowLossOf(d: DatasetSummary, files: FileSummary[]): DatasetSummary | undefined {
  if (d.eels?.role === "low_loss") return d;
  const byId = new Map(allDatasets(files).map((x) => [x.id, x]));
  return (d.eels?.linked ?? []).map((id) => byId.get(id)).find((x) => x?.eels?.role === "low_loss");
}

/** The thickness made from the low loss ``ll``, if it has one. */
export const resultOf = (ll: DatasetSummary, files: FileSummary[]) =>
  allDatasets(files).find((x) => x.origin === "thickness" && x.parent_id === ll.id);

const DEFAULTS: ThicknessState = {
  on: false, threshold: "auto", window: 10, output: "relative", by: "mean_free_path", mean_free_path: null, density: null,
};

// exSpy's automatic threshold searches at least this many channels.
const MIN_SEARCH_CHANNELS = 10;

/** The narrowest search window on the low loss ``ll`` (energy units, rounded up to 0.1). */
export function minSearchWindow(ll: DatasetSummary): number {
  const step = energyRange(ll)?.step ?? 0;
  return Math.ceil(MIN_SEARCH_CHANNELS * step * 10 - 1e-6) / 10;
}

/** The settings of an existing thickness, else the defaults (the search
 * window widened to the low loss's channels). */
function initialState(ll: DatasetSummary | undefined, result: DatasetSummary | undefined): ThicknessState {
  const t = result?.thickness;
  if (!t) return ll ? { ...DEFAULTS, window: Math.max(DEFAULTS.window, minSearchWindow(ll)) } : DEFAULTS;
  return {
    on: false,
    threshold: t.threshold,
    window: t.window,
    output: t.output,
    by: t.density != null ? "density" : "mean_free_path",
    mean_free_path: t.mean_free_path,
    density: t.density,
  };
}

/** What Apply sends. */
export function toOptions(s: ThicknessState): ThicknessOptions {
  const absolute = s.output === "absolute";
  return {
    threshold: s.threshold,
    window: s.window,
    output: s.output,
    mean_free_path: absolute && s.by === "mean_free_path" ? s.mean_free_path : null,
    density: absolute && s.by === "density" ? s.density : null,
  };
}

/** Why the settings can't be applied to the low loss ``ll``, or null. */
export function thicknessProblem(s: ThicknessState, ll: DatasetSummary): string | null {
  const range = energyRange(ll);
  if (s.threshold === "auto") {
    if (!(s.window > 0)) return "Enter a search window.";
    const min = minSearchWindow(ll);
    if (s.window < min) return `The search window must be at least ${min} ${range?.units ?? ""}`.trimEnd() + ".";
  } else {
    if (!Number.isFinite(s.threshold)) return "Enter a threshold.";
    if (range && (s.threshold <= range.e0 || s.threshold > range.e1)) return "The threshold is outside the spectrum.";
  }
  if (s.output === "absolute") {
    if (s.by === "mean_free_path" && !(s.mean_free_path != null && s.mean_free_path > 0)) return "Enter a mean free path.";
    if (s.by === "density") {
      if (!(s.density != null && s.density > 0)) return "Enter a density.";
      const e = ll.eels;
      const missing = [
        e?.beam_energy == null && "beam energy",
        e?.convergence_angle == null && "convergence angle",
        e?.collection_angle == null && "collection angle",
      ].filter(Boolean);
      if (missing.length) return `A density needs the ${missing.join(", ")}.`;
    }
  }
  return null;
}

interface ThicknessStore {
  byId: Record<string, ThicknessState>;
  estimate: Record<string, ThresholdEstimate>;
  /** The latest job per low loss. */
  jobs: Record<string, JobSummary>;
  /** Why a job could not even start. */
  startError: Record<string, string | null>;
  patch: (ll: DatasetSummary, patch: Partial<ThicknessState>) => void;
  setEstimate: (id: string, estimate: ThresholdEstimate) => void;
  /** Start the job for the low loss ``ll`` and follow it; the first result is shown. */
  run: (ll: DatasetSummary) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export const useThickness = create<ThicknessStore>((set, get) => {
  const current = (ll: DatasetSummary) =>
    get().byId[ll.id] ?? initialState(ll, resultOf(ll, useDatasets.getState().files));
  const setJob = (id: string, job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
  const setStartError = (id: string, e: string | null) => set((s) => ({ startError: { ...s.startError, [id]: e } }));

  return {
    byId: {},
    estimate: {},
    jobs: {},
    startError: {},
    patch: (ll, patch) => set((s) => ({ byId: { ...s.byId, [ll.id]: { ...current(ll), ...patch } } })),
    setEstimate: (id, estimate) => set((s) => ({ estimate: { ...s.estimate, [id]: estimate } })),

    run: async (ll) => {
      const options = toOptions(current(ll));
      const had = !!resultOf(ll, useDatasets.getState().files);
      setStartError(ll.id, null);
      try {
        let job = await api.estimateThickness(ll.id, options);
        setJob(ll.id, job);
        // Polled here, not in the Process tab, so it finishes if the tab is closed.
        while (ACTIVE.has(job.status)) {
          await new Promise((r) => setTimeout(r, 300));
          try {
            job = await api.getJob(job.id);
            setJob(ll.id, job);
          } catch {
            /* backend busy or restarted: try again */
          }
        }
        if (job.status === "done" && job.result) {
          const { replaceFile, clickDataset, select } = useDatasets.getState();
          replaceFile(job.result.file);
          // The first result is shown: a map over the scan, a profile on its own.
          const made = allDatasets(useDatasets.getState().files).find((x) => x.id === job.result?.dataset_id);
          if (!had && made) {
            if (made.kind === "image") clickDataset(made);
            else select(made.id);
          }
          useLayers.getState().notify(had ? "Thickness re-run." : "Thickness added.");
        }
      } catch (e) {
        setStartError(ll.id, (e as Error).message);
      }
    },

    cancel: async (id) => {
      const job = get().jobs[id];
      if (job) setJob(id, await api.cancelJob(job.id));
    },
  };
});

/** The tool's settings for the low loss ``ll`` (undefined: the defaults). */
export function useThicknessState(ll: DatasetSummary | undefined): ThicknessState {
  const stored = useThickness((s) => (ll ? s.byId[ll.id] : undefined));
  const files = useDatasets((s) => s.files);
  const result = ll ? resultOf(ll, files) : undefined;
  const min = ll ? minSearchWindow(ll) : 0;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const fallback = useMemo(() => initialState(ll, result), [ll?.id, min, result?.thickness]);
  return stored ?? fallback;
}

/** ``v`` to the nearest channel of ``ll``'s energy axis, within its range. */
export function snapThreshold(ll: DatasetSummary, v: number): number {
  const r = energyRange(ll);
  if (!r) return v;
  return snapEnergy(r.e0, r.step, Math.min(r.e1, Math.max(r.e0, v)));
}
