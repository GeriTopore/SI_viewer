import { create } from "zustand";
import { api, type DatasetSummary, type DecompositionAlgorithm, type JobSummary, type PcaOptions } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";
import { energyRange, snapEnergy } from "./background";
import { defaultNorm, useNorm } from "./normalise";

const ACTIVE = new Set(["queued", "running"]);

/** Id of the PCA window's band on the spectrum plot. */
export const PCA_BAND = "pca:window";

/** The PCA tool of one spectrum image. */
export interface PcaState {
  /** The window is drawn on the spectrum (while the section is open). */
  on: boolean;
  options: PcaOptions;
}

interface PcaStore {
  byId: Record<string, PcaState>;
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  /** Components kept by each PCA (its id): the scree's choice. */
  kept: Record<string, number>;
  patch: (d: DatasetSummary, patch: Partial<PcaState>) => void;
  setOptions: (d: DatasetSummary, options: Partial<PcaOptions>) => void;
  setKept: (pcaId: string, n: number) => void;
  run: (d: DatasetSummary) => Promise<void>;
  denoise: (pca: DatasetSummary, n: number) => Promise<void>;
}

/** The PCA settings a spectrum image starts with: its whole energy range. */
export function defaultPca(d: DatasetSummary): PcaState {
  const r = energyRange(d);
  return { on: false, options: { lo: r?.e0 ?? null, hi: r?.e1 ?? null, components: 32, poisson: true, algorithm: "SVD" } };
}

/** HyperSpy's decomposition algorithms, as offered: label, and whether they
 * centre the data (then no Poisson weighting). */
export const ALGORITHMS: { id: DecompositionAlgorithm; label: string; centred?: boolean }[] = [
  { id: "SVD", label: "PCA (SVD)" },
  { id: "MLPCA", label: "Maximum likelihood PCA (slow)" },
  { id: "sklearn_pca", label: "PCA (scikit-learn)", centred: true },
  { id: "NMF", label: "Non-negative matrix factorisation" },
  { id: "sparse_pca", label: "Sparse PCA", centred: true },
  { id: "mini_batch_sparse_pca", label: "Mini-batch sparse PCA", centred: true },
  { id: "RPCA", label: "Robust PCA" },
  { id: "ORPCA", label: "Online robust PCA" },
  { id: "ORNMF", label: "Online robust NMF" },
];

/** The algorithm a decomposition was made with (older ones: SVD). */
export const algorithmOf = (d: DatasetSummary): DecompositionAlgorithm => d.pca?.algorithm ?? "SVD";

/** EELS and EDS spectrum images (not in-situ series) can be decomposed. */
export const canDecompose = (d: DatasetSummary) =>
  d.kind === "spectrum_image" && !d.insitu && (d.technique === "EELS" || d.technique === "EDS") &&
  d.origin !== "pca-factors";

export const usePca = create<PcaStore>((set, get) => ({
  byId: {},
  jobs: {},
  errors: {},
  kept: {},
  patch: (d, patch) =>
    set((s) => ({ byId: { ...s.byId, [d.id]: { ...(s.byId[d.id] ?? defaultPca(d)), ...patch } } })),
  setOptions: (d, options) => {
    const cur = get().byId[d.id] ?? defaultPca(d);
    get().patch(d, { options: { ...cur.options, ...options } });
  },
  setKept: (pcaId, n) => set((s) => ({ kept: { ...s.kept, [pcaId]: n } })),

  run: async (d) => {
    const { options } = get().byId[d.id] ?? defaultPca(d);
    const setJob = (job: JobSummary | undefined) => set((s) => ({ jobs: { ...s.jobs, [d.id]: job } }));
    set((s) => ({ errors: { ...s.errors, [d.id]: null } }));
    try {
      // Normalised spectra: as the dataset's normalisation (Process tab).
      const n = useNorm.getState().byId[d.id] ?? defaultNorm(d);
      const normalise = n.on ? { mode: n.mode, lo: n.lo, hi: n.hi } : null;
      let job = await api.startPca(d.id, options, normalise);
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
      if (job.status === "done" && job.result?.file) {
        useDatasets.getState().replaceFile(job.result.file);
        const id = job.result.pca_id;
        if (id) {
          set((s) => {
            const kept = { ...s.kept };
            delete kept[id]; // a new decomposition: the scree's choice starts again
            return { kept };
          });
          useDatasets.getState().select(id);
        }
        get().patch(d, { on: false });
      } else if (job.status === "failed") {
        set((s) => ({ errors: { ...s.errors, [d.id]: job.error ?? "Decomposition failed" } }));
      }
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [d.id]: (e as Error).message } }));
    } finally {
      setJob(undefined);
    }
  },

  denoise: async (pca, n) => {
    set((s) => ({ errors: { ...s.errors, [pca.id]: null } }));
    try {
      const r = await api.denoisePca(pca.id, n);
      useDatasets.getState().replaceFile(r.file);
      useDatasets.getState().select(r.id);
      useLayers.getState().notify(`Denoised with ${n} component${n === 1 ? "" : "s"}.`);
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [pca.id]: (e as Error).message } }));
    }
  },
}));

/** The PCA tool's state for ``d``. */
export const usePcaState = (d: DatasetSummary) => usePca((s) => s.byId[d.id]) ?? defaultPca(d);

/** Components kept by a PCA: the scree's choice, else those standing clear
 * of the noise floor (the leading ones above twice the median of the later
 * half, where the scree has levelled off; at least 1). */
export function keptOf(pca: DatasetSummary, chosen: number | undefined): number {
  const e = pca.pca?.explained ?? [];
  // Not ranked by variance: all of them, unless fewer are chosen.
  if (!e.length) return Math.max(1, Math.min(chosen ?? Infinity, pca.pca?.components ?? 1));
  if (chosen) return Math.min(chosen, e.length || chosen);
  if (e.length < 4) return Math.max(1, e.length);
  const tail = e.slice(Math.floor(e.length / 2)).sort((a, b) => a - b);
  const floor = tail[Math.floor(tail.length / 2)];
  const i = e.findIndex((v) => v <= 2 * floor);
  return Math.max(1, i < 0 ? e.length : i);
}

/** ``v`` snapped to a channel of ``d``'s energy axis and kept on it. */
export function snapToAxis(d: DatasetSummary, v: number): number {
  const r = energyRange(d);
  if (!r) return v;
  return snapEnergy(r.e0, r.step, Math.min(r.e1, Math.max(r.e0, v)));
}
