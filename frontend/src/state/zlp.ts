import { create } from "zustand";
import { api, type DatasetSummary, type JobSummary, type ZlpOptions } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useLayers } from "./layers";
import { energyRange, snapEnergy, useBackground, validateWindows } from "./background";
import { snapThreshold, useThickness } from "./thickness";

export { canEstimateThickness as canAlignZlp, lowLossOf } from "./thickness";

const ACTIVE = new Set(["queued", "running"]);

export const DEFAULTS: ZlpOptions = { subpixel: true, crop: true };

interface ZlpStore {
  /** The options per low loss. */
  byId: Record<string, ZlpOptions>;
  /** The latest job per low loss. */
  jobs: Record<string, JobSummary>;
  /** Why a job could not even start. */
  startError: Record<string, string | null>;
  patch: (ll: DatasetSummary, patch: Partial<ZlpOptions>) => void;
  /** Start the alignment of the low loss ``ll`` (and its linked core losses) and follow it. */
  /** ``inPlace`` false: the aligned spectra become new datasets, the low loss's selected. */
  run: (ll: DatasetSummary, inPlace?: boolean) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  /** A shift by hand being typed, per dataset: its spectra are drawn that far along. */
  preview: Record<string, number>;
  setPreview: (id: string, shift: number | null) => void;
  /** Apply a shift by hand (no low loss to align to). Throws on failure. */
  shiftEnergy: (d: DatasetSummary, shift: number, inPlace: boolean) => Promise<void>;
}

/** Settings kept in energy units follow the data when its axis moves by ``shift``: a fixed
 * thickness threshold, and the windows of a background removal (clamped to the new range;
 * windows that no longer fit are forgotten, and start again from the defaults). */
function followAxis(ids: string[], shift: number) {
  const all = allDatasets(useDatasets.getState().files);
  const byId = new Map(all.map((d) => [d.id, d]));
  const targets = new Set(ids);
  all.forEach((d) => d.origin === "background" && d.parent_id && targets.has(d.parent_id) && targets.add(d.id));

  const thickness = { ...useThickness.getState().byId };
  const background = { ...useBackground.getState().byId };
  for (const id of targets) {
    const d = byId.get(id);
    if (!d) continue;
    const t = thickness[id];
    if (t && typeof t.threshold === "number") thickness[id] = { ...t, threshold: snapThreshold(d, t.threshold + shift) };
    const b = background[id];
    const r = energyRange(d);
    if (!b || !r) continue;
    const move = (w: [number, number]): [number, number] => [
      snapEnergy(r.e0, r.step, Math.min(r.e1, Math.max(r.e0, w[0] + shift))),
      snapEnergy(r.e0, r.step, Math.min(r.e1, Math.max(r.e0, w[1] + shift))),
    ];
    const options = { ...b.options, background: move(b.options.background), signal: move(b.options.signal) };
    if (validateWindows(options)) delete background[id];
    else background[id] = { ...b, options };
  }
  useThickness.setState({ byId: thickness });
  useBackground.setState({ byId: background });
}

export const useZlp = create<ZlpStore>((set, get) => {
  const setJob = (id: string, job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
  const setStartError = (id: string, e: string | null) => set((s) => ({ startError: { ...s.startError, [id]: e } }));

  return {
    byId: {},
    jobs: {},
    startError: {},
    preview: {},
    setPreview: (id, shift) =>
      set((s) => {
        const preview = { ...s.preview };
        if (shift) preview[id] = shift;
        else delete preview[id];
        return { preview };
      }),
    shiftEnergy: async (d, shift, inPlace) => {
      const r = await api.shiftEnergy(d.id, shift, inPlace);
      get().setPreview(d.id, null);
      const { replaceFile, select } = useDatasets.getState();
      replaceFile(r.file);
      if (inPlace) {
        followAxis([d.id], shift);
        useLayers.getState().notify("Energy axis shifted.");
      } else {
        select(r.dataset.id);
        useLayers.getState().notify(`${r.dataset.title} added to the dataset pane.`);
      }
    },
    patch: (ll, patch) => set((s) => ({ byId: { ...s.byId, [ll.id]: { ...(s.byId[ll.id] ?? DEFAULTS), ...patch } } })),

    run: async (ll, inPlace = true) => {
      const options = get().byId[ll.id] ?? DEFAULTS;
      setStartError(ll.id, null);
      try {
        let job = await api.alignZeroLoss(ll.id, options, inPlace);
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
          // The low loss's file and the files of its core losses, with what was made from them.
          const { replaceFile } = useDatasets.getState();
          replaceFile(job.result.file);
          (job.result.partner_files ?? []).forEach((f) => replaceFile(f));
          if (inPlace) {
            followAxis(job.result.aligned ?? [], job.result.shift ?? 0);
            useLayers.getState().notify("ZLP aligned.");
          } else {
            const first = job.result.aligned?.[0];
            if (first) useDatasets.getState().select(first);
            useLayers.getState().notify("Aligned spectrum images added to the dataset pane.");
          }
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

/** The options for the low loss ``ll`` (undefined: the defaults). */
export function useZlpOptions(ll: DatasetSummary | undefined): ZlpOptions {
  return useZlp((s) => (ll ? s.byId[ll.id] : undefined)) ?? DEFAULTS;
}
