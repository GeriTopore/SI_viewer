import { create } from "zustand";
import { api, type DatasetSummary, type JobSummary, type TemplateOptions } from "../api/client";
import { useDatasets } from "./datasets";
import { useLayers } from "./layers";
import { jobRunning } from "./centring";
import type { SceneOrientation } from "../components/StructureScene";

/** A Structure's templates: the defaults of pyxem's SPED-Ag tutorial. */
export const DEFAULT_TEMPLATES: TemplateOptions = {
  kv: 200,
  resolution: 2,
  zone_axes: [],
  reciprocal_radius: 1.5,
  max_excitation_error: 0.001,
  precession_angle: 1,
  minimum_intensity: 1e-5,
};

/** The polar transform of the patterns (pyxem's ``get_azimuthal_integral2d``),
 * then a gamma: the backend's ``PolarOptions``. */
export interface PolarOptions {
  /** Radial points. */
  npt: number;
  /** Azimuthal points. */
  npt_azim: number;
  /** Radial range (1/Å); r_max null: the whole pattern. */
  r_min: number;
  r_max: number | null;
  /** 1: none. */
  gamma: number;
  /** Make the ring circular with the calibration's ellipse. */
  ellipse: boolean;
}

export const DEFAULT_POLAR: PolarOptions = { npt: 112, npt_azim: 360, r_min: 0, r_max: null, gamma: 1, ellipse: true };

/** A template picked in a Templates object. */
export interface PickedTemplate {
  /** The Templates object. */
  id: string;
  index: number;
}

interface OrientationState {
  /** The Templates process of each Structure: settings changed from the defaults. */
  options: Record<string, Partial<TemplateOptions>>;
  /** The Templates object each 4D-STEM data uses (drawn on its patterns, matched against). */
  chosen: Record<string, string | null>;
  /** The template drawn on each 4D-STEM data's patterns, scaled by this
   * (1: at the pattern's calibrated pixel size). */
  scale: Record<string, number>;
  /** How each Structure is seen now (its 3D view, wherever shown). */
  views: Record<string, SceneOrientation>;
  /** Templates objects whose Templates tab follows the Structure's view. */
  linked: Record<string, boolean>;
  /** The template picked in each Templates object (by its id). */
  picked: Record<string, number>;
  /** The template picked last. */
  shown: PickedTemplate | null;
  /** The Polar transform process of each 4D-STEM data: settings changed. */
  polar: Record<string, Partial<PolarOptions>>;
  /** The running or last job of each Structure. */
  jobs: Record<string, JobSummary | undefined>;
  errors: Record<string, string | null>;
  setOptions: (id: string, patch: Partial<TemplateOptions>) => void;
  choose: (cubeId: string, id: string | null) => void;
  setScale: (cubeId: string, scale: number) => void;
  /** Matching at the cursors as they move, per 4D-STEM data (or its polar transform). */
  matching: Record<string, boolean>;
  setMatching: (id: string, on: boolean) => void;
  setView: (structureId: string, o: SceneOrientation) => void;
  setLinked: (id: string, on: boolean) => void;
  pick: (id: string, index: number) => void;
  setPolar: (id: string, patch: Partial<PolarOptions>) => void;
  /** The polar transform of every pattern (a job): a copy under the data, or in place. */
  transform: (id: string, options: PolarOptions, inPlace: boolean) => Promise<void>;
  /** Simulate a Structure's templates (a job): a new Templates object under it. */
  makeTemplates: (id: string, options: TemplateOptions) => Promise<void>;
  cancel: (id: string) => Promise<void>;
}

export { jobRunning };

export const templateOptionsOf = (changed: Partial<TemplateOptions> | undefined): TemplateOptions => ({ ...DEFAULT_TEMPLATES, ...changed });

export const useOrientation = create<OrientationState>((set, get) => {
  /** Run a job to its end, kept in ``jobs`` under ``id``; ``done`` with its result. */
  const follow = async (id: string, start: () => Promise<JobSummary>, done: (job: JobSummary) => void) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await start();
      setJob(job);
      while (jobRunning(job)) {
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
  options: {},
  chosen: {},
  scale: {},
  views: {},
  linked: {},
  picked: {},
  shown: null,
  polar: {},
  jobs: {},
  errors: {},
  setOptions: (id, patch) => set((s) => ({ options: { ...s.options, [id]: { ...s.options[id], ...patch } } })),
  choose: (cubeId, id) => set((s) => ({ chosen: { ...s.chosen, [cubeId]: id } })),
  setScale: (cubeId, scale) => set((s) => ({ scale: { ...s.scale, [cubeId]: scale } })),
  matching: {},
  setMatching: (id, on) => set((s) => ({ matching: { ...s.matching, [id]: on } })),
  setView: (structureId, o) => set((s) => ({ views: { ...s.views, [structureId]: o } })),
  setLinked: (id, on) => set((s) => ({ linked: { ...s.linked, [id]: on } })),
  pick: (id, index) => set((s) => ({ picked: { ...s.picked, [id]: index }, shown: { id, index } })),
  setPolar: (id, patch) => set((s) => ({ polar: { ...s.polar, [id]: { ...s.polar[id], ...patch } } })),
  transform: (id, options, inPlace) => follow(id, () => api.polarTransform(id, options, inPlace), (job) => {
    useDatasets.getState().replaceFile(job.result!.file);
    if (!inPlace && job.result!.id) useDatasets.getState().select(job.result!.id);
    useLayers.getState().notify(inPlace ? "Patterns transformed to polar." : "Polar transform added to the dataset pane.");
  }),
  makeTemplates: async (id, options) => {
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      let job = await api.makeTemplates(id, options);
      setJob(job);
      while (jobRunning(job)) {
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
        useLayers.getState().notify(`${job.result.templates ?? 0} templates made.`);
      }
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    }
  },
  cancel: async (id) => {
    const job = get().jobs[id];
    if (job) {
      set((s) => ({ jobs: { ...s.jobs, [id]: undefined } }));
      await api.cancelJob(job.id);
    }
  },
  };
});

/** 1/Å per unit of the pattern's axis (null: not calibrated). */
export function perAngstrom(d: DatasetSummary): number | null {
  const units = d.axes.find((a) => !a.navigate)?.units ?? "";
  return units === "1/Å" || units === "1/A" ? 1 : units === "1/nm" ? 0.1 : null;
}

export const polarOptionsOf = (changed: Partial<PolarOptions> | undefined): PolarOptions => ({ ...DEFAULT_POLAR, ...changed });

/** The Templates objects open (under their Structures). */
export const templatesOf = (all: DatasetSummary[]) => all.filter((x) => x.origin === "templates" && x.templates);

/** Whether ``d`` is a polar transform (made by Polar transform). */
export const isPolar = (d: DatasetSummary) => d.signal_type === "polar_diffraction";

/** The Templates object chosen for ``d``: its own, or (a polar transform) its data's. */
export const chosenFor = (s: { chosen: Record<string, string | null> }, d: DatasetSummary): string | null =>
  s.chosen[d.id] ?? (isPolar(d) && d.parent_id ? s.chosen[d.parent_id] ?? null : null);
