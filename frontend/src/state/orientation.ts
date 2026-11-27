import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";
import { api, dataVersion, type DatasetSummary, type JobSummary, type MatchResult, type TemplateOptions } from "../api/client";
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
  /** Matches (pyxem's get_orientation), by ``matchKey`` (the polar patterns
   * matched and the templates), then by scan position "x,y": shared by the
   * 4D-STEM data and its polar patterns. */
  matchCache: Record<string, Record<string, MatchResult>>;
  matchBusy: Record<string, boolean>;
  matchErrors: Record<string, string | null>;
  /** Match ``dsId`` (4D-STEM data, or polar patterns: ``polar`` null) at
   * ``positions`` ([x, y]) against ``templates``; kept under ``key``. */
  runMatch: (key: string, dsId: string, templates: string, polar: string | null, positions: number[][]) => Promise<void>;
  /** "Match orientation" in the Structure view: it turns to the match at the active cursor. */
  followMatch: boolean;
  setFollowMatch: (on: boolean) => void;
  /** The polar patterns matched against, per 4D-STEM data (else ``polarFor``'s default). */
  polarChoice: Record<string, string>;
  choosePolar: (cubeId: string, polarId: string) => void;
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

/** Browser storage that may be missing or refuse (private windows, blocked
 * site data): then nothing is kept. */
const safeStorage: StateStorage = {
  getItem: (name) => {
    try {
      return localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name, value) => {
    try {
      localStorage.setItem(name, value);
    } catch {
      /* not kept */
    }
  },
  removeItem: (name) => {
    try {
      localStorage.removeItem(name);
    } catch {
      /* nothing to remove */
    }
  },
};

/** The choices (templates, polar patterns, Match orientation, template
 * scale, structure views) are kept across reloads in this browser; results
 * and jobs aren't. */
export const useOrientation = create<OrientationState>()(persist((set, get) => {
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
  polarChoice: {},
  choosePolar: (cubeId, polarId) => set((s) => ({ polarChoice: { ...s.polarChoice, [cubeId]: polarId } })),
  matchCache: {},
  matchBusy: {},
  matchErrors: {},
  runMatch: async (key, dsId, templates, polar, positions) => {
    set((s) => ({ matchBusy: { ...s.matchBusy, [key]: true }, matchErrors: { ...s.matchErrors, [key]: null } }));
    try {
      const r = await api.matchAt(dsId, templates, positions, polar);
      const found = Object.fromEntries(r.matches.map((m) => [`${m.x},${m.y}`, m]));
      set((s) => ({ matchCache: { ...s.matchCache, [key]: { ...s.matchCache[key], ...found } } }));
    } catch (e) {
      set((s) => ({ matchErrors: { ...s.matchErrors, [key]: (e as Error).message } }));
    } finally {
      set((s) => ({ matchBusy: { ...s.matchBusy, [key]: false } }));
    }
  },
  followMatch: false,
  setFollowMatch: (on) => set({ followMatch: on }),
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
}, {
  name: "si-viewer:orientation",
  storage: createJSONStorage(() => safeStorage),
  partialize: (s) => ({
    chosen: s.chosen, polarChoice: s.polarChoice, followMatch: s.followMatch, scale: s.scale, views: s.views, linked: s.linked,
  }),
}));

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

const scanOf = (d: DatasetSummary) => d.axes.filter((a) => a.navigate).map((a) => a.size).join("×");

/** The polar patterns ``d`` (4D-STEM data) can be matched against: those of
 * its file with the same scan (its own, or a copy's such as the vector-masked
 * data's), newest first. */
export const polarsFor = (all: DatasetSummary[], d: DatasetSummary): DatasetSummary[] =>
  all.filter((x) => isPolar(x) && x.id !== d.id && x.file_id === d.file_id && scanOf(x) === scanOf(d)).reverse();

/** The polar patterns matched for ``d``: itself if it is some, else the one
 * chosen, else its own newest, else the newest it can use. */
export function polarFor(s: { polarChoice: Record<string, string> }, all: DatasetSummary[], d: DatasetSummary): DatasetSummary | null {
  if (isPolar(d)) return d;
  const options = polarsFor(all, d);
  return options.find((x) => x.id === s.polarChoice[d.id]) ?? options.find((x) => x.parent_id === d.id) ?? options[0] ?? null;
}

/** Whether polar patterns were made at another pattern calibration than their
 * source's now (its radius scale then out of date). */
export function polarStale(polar: DatasetSummary, all: DatasetSummary[]): boolean {
  const source = polar.polar?.source ? all.find((x) => x.id === polar.polar!.source) : undefined;
  if (!source || !polar.polar) return false;
  const now = source.axes.filter((a) => !a.navigate);
  return polar.polar.axes.some((a, i) => !now[i] || Math.abs(now[i].scale - a.scale) > 1e-9 * Math.abs(a.scale)
    || Math.abs(now[i].offset - a.offset) > 1e-9 * Math.max(1, Math.abs(a.offset)) || now[i].units !== a.units);
}

/** Where the matches of polar patterns ``polar`` against ``templates`` are kept. */
export const matchKey = (polar: DatasetSummary, templates: string) => `${polar.id}|${dataVersion(polar)}|${templates}`;

/** The Templates object chosen for ``d``: its own, or (polar patterns) their
 * data's, or that of data matching against them. */
export const chosenFor = (
  s: { chosen: Record<string, string | null>; polarChoice: Record<string, string> }, d: DatasetSummary,
): string | null => {
  if (s.chosen[d.id] || !isPolar(d)) return s.chosen[d.id] ?? null;
  const user = Object.keys(s.polarChoice).find((id) => s.polarChoice[id] === d.id && s.chosen[id]);
  return (d.parent_id ? s.chosen[d.parent_id] : null) ?? (user ? s.chosen[user] : null) ?? null;
};
