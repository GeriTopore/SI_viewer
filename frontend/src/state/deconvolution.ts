import { useMemo } from "react";
import { create } from "zustand";
import {
  api, type DatasetSummary, type DeconvolutionMethod, type DeconvolutionOptions, type FileSummary, type JobSummary,
} from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useLayers } from "./layers";
import { lowLossOf } from "./thickness";

/** Settings of the Deconvolution tool for one dataset. */
export interface DeconvState {
  /** The deconvolved spectra are previewed on the spectrum plot. */
  on: boolean;
  options: DeconvolutionOptions;
}

/** What the last preview told: the zero-loss threshold and FWHM it used, or why it failed. */
export interface DeconvInfo {
  threshold: number | null;
  fwhm: number | null;
  error: string | null;
}

export const METHODS: { id: DeconvolutionMethod; label: string; title: string }[] = [
  { id: "fourier_log", label: "Fourier-log", title: "Single scattering distribution of the low loss" },
  { id: "fourier_ratio", label: "Fourier-ratio", title: "A background-removed core loss, by the low loss" },
  { id: "richardson_lucy", label: "Richardson–Lucy", title: "Iterative, by the low loss or its zero-loss peak" },
];

const ACTIVE = new Set(["queued", "running"]);

/** EELS spectrum images (incl. line scans) can be deconvolved. */
export const canDeconvolve = (d: DatasetSummary) =>
  d.kind === "spectrum_image" && !!d.eels && d.technique === "EELS" && !d.is_element_map;

/** The low loss ``d`` is deconvolved with: its group's (itself if it is one);
 * for spectra made from another spectrum image, that one's. */
export function deconvolutionLowLoss(d: DatasetSummary, files: FileSummary[]): DatasetSummary | undefined {
  const byId = new Map(allDatasets(files).map((x) => [x.id, x]));
  const seen = new Set<string>();
  let cur = d;
  while ((cur.origin === "background" || cur.origin === "deconvolved") && cur.parent_id && !seen.has(cur.id)) {
    const parent = byId.get(cur.parent_id);
    if (!parent) break;
    seen.add(cur.id);
    cur = parent;
  }
  return lowLossOf(cur, files);
}

/** Why ``method`` (with ``psf``) is not for ``d``, or null. */
export function methodProblem(
  method: DeconvolutionMethod, psf: DeconvolutionOptions["psf"], d: DatasetSummary, ll: DatasetSummary | undefined,
): string | null {
  if (!ll) return "No low loss in its group";
  if (method === "fourier_log" && d.id !== ll.id) return "Fourier-log is for the low loss";
  if (method === "fourier_ratio" && d.origin !== "background") return "Remove the background first";
  if (method === "richardson_lucy") {
    if (d.origin === "background") return "Richardson–Lucy is for spectra before background removal";
    if (psf === "low_loss" && d.id === ll.id) return "The low loss can't be its own point-spread function: use its zero-loss peak";
  }
  return null;
}

/** Why the options can't be used (before asking the backend), or null. */
export function optionsProblem(o: DeconvolutionOptions): string | null {
  if (o.threshold !== "auto" && !Number.isFinite(o.threshold)) return "Enter the threshold.";
  if (!(o.window > 0)) return "The search window must be above 0.";
  if (o.fwhm !== null && !(o.fwhm > 0)) return "The FWHM must be above 0.";
  if (!(o.iterations >= 1 && o.iterations <= 500)) return "Iterations: 1 to 500.";
  return null;
}

const DEFAULTS: Omit<DeconvolutionOptions, "method"> = {
  threshold: "auto", window: 10, add_zlp: false, fwhm: null, extrapolate: true, iterations: 15, psf: "low_loss",
};

/** The settings a deconvolved spectrum image was made with, else the first
 * method that fits ``d``. */
function defaultState(d: DatasetSummary, ll: DatasetSummary | undefined): DeconvState {
  if (d.deconvolution) {
    const { source_id: _s, low_loss_id: _l, units: _u, ...options } = d.deconvolution;
    return { on: false, options };
  }
  const psf = ll && d.id === ll.id ? "zlp" : "low_loss";
  const method = METHODS.find((m) => !methodProblem(m.id, psf, d, ll))?.id ?? "richardson_lucy";
  return { on: false, options: { method, ...DEFAULTS, psf } };
}

interface DeconvStore {
  byId: Record<string, DeconvState>;
  info: Record<string, DeconvInfo>;
  /** The latest deconvolution job per dataset. */
  jobs: Record<string, JobSummary>;
  /** Why a job could not even start. */
  startError: Record<string, string | null>;
  patch: (d: DatasetSummary, patch: Partial<DeconvState>) => void;
  setOptions: (d: DatasetSummary, patch: Partial<DeconvolutionOptions>) => void;
  setInfo: (id: string, info: DeconvInfo) => void;
  /** Start the job (``rerun``: replace the deconvolved spectrum image ``d``
   * in place) and follow it; the new spectrum image is selected. */
  run: (d: DatasetSummary, rerun: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

const currentState = (d: DatasetSummary) =>
  useDeconvolution.getState().byId[d.id] ?? defaultState(d, deconvolutionLowLoss(d, useDatasets.getState().files));

export const useDeconvolution = create<DeconvStore>((set, get) => {
  const setJob = (id: string, job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
  const setStartError = (id: string, e: string | null) =>
    set((s) => ({ startError: { ...s.startError, [id]: e } }));

  return {
    byId: {},
    info: {},
    jobs: {},
    startError: {},
    patch: (d, patch) => set((s) => ({ byId: { ...s.byId, [d.id]: { ...currentState(d), ...patch } } })),
    setOptions: (d, patch) =>
      set((s) => {
        const cur = currentState(d);
        return { byId: { ...s.byId, [d.id]: { ...cur, options: { ...cur.options, ...patch } } } };
      }),
    setInfo: (id, info) =>
      set((s) => {
        const old = s.info[id];
        if (old && old.threshold === info.threshold && old.fwhm === info.fwhm && old.error === info.error) return s;
        return { info: { ...s.info, [id]: info } };
      }),

    run: async (d, rerun) => {
      const options = currentState(d).options;
      setStartError(d.id, null);
      try {
        let job = await (rerun ? api.rerunDeconvolution(d.id, options) : api.deconvolve(d.id, options));
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
          useLayers.getState().notify(rerun ? "Deconvolution re-run." : "Deconvolved spectrum image added.");
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
export function useDeconvState(d: DatasetSummary): DeconvState {
  const stored = useDeconvolution((s) => s.byId[d.id]);
  const files = useDatasets((s) => s.files);
  const ll = deconvolutionLowLoss(d, files);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const fallback = useMemo(() => defaultState(d, ll), [d.id, d.deconvolution, d.origin, ll?.id]);
  return stored ?? fallback;
}
