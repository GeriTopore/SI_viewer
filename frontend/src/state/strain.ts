import { create } from "zustand";
import { api, dataVersion, type DatasetSummary, type JobSummary } from "../api/client";
import { allDatasets, useDatasets } from "./datasets";
import { useLayers } from "./layers";

/** How the Bragg vector map's peaks are found: the backend's
 * ``processing/strain.py`` ``BvmPeakOptions``. */
export interface BvmPeakOptions {
  /** Gaussian blur first (pattern px; 0: none). */
  sigma: number;
  /** Lowest peak kept: this fraction of the brightest. */
  threshold: number;
  /** Closest two peaks may be (pattern px). */
  min_spacing: number;
  max_peaks: number;
}

export const DEFAULT_BVM_PEAK_OPTIONS: BvmPeakOptions = { sigma: 1, threshold: 0.02, min_spacing: 3, max_peaks: 200 };

/** The strain lattice on the Bragg vector map (pattern px, as the pattern
 * pane shows it): the origin O and the basis vectors g1 and g2 (from O). */
export interface StrainLattice {
  ox: number;
  oy: number;
  g1x: number;
  g1y: number;
  g2x: number;
  g2y: number;
}

/** A peak of the Bragg vector map: [x, y, intensity] (pattern px). */
export type BvmPeak = [number, number, number];

export const DEFAULT_TOLERANCE = 2;

/** The lattice fit's other settings (with the tolerance: the backend's ``FitOptions``). */
export interface FitSettings {
  /** Fewest indexed disks a position needs to be fitted. */
  min_disks: number;
  /** Disks weighted by their intensity (the correlation). */
  weighted: boolean;
}
export const DEFAULT_FIT: FitSettings = { min_disks: 4, weighted: true };

const ACTIVE = new Set(["queued", "running"]);
export const strainJobRunning = (job: JobSummary | undefined) => !!job && ACTIVE.has(job.status);

interface StrainState {
  options: Record<string, BvmPeakOptions>;
  /** The peaks found, with the key (data and settings) they were found for. */
  peaks: Record<string, { key: string; list: BvmPeak[]; proposed: StrainLattice | null } | undefined>;
  lattice: Record<string, StrainLattice | undefined>;
  /** Largest distance (pattern px) from a lattice point at which a disk is indexed to it. */
  tolerance: Record<string, number>;
  busy: Record<string, boolean>;
  errors: Record<string, string | null>;
  fit: Record<string, FitSettings>;
  /** The indexing and fit shown on the patterns at the cursors (while tuning). */
  preview: Record<string, boolean>;
  jobs: Record<string, JobSummary | undefined>;
  setOptions: (id: string, patch: Partial<BvmPeakOptions>) => void;
  setFit: (id: string, patch: Partial<FitSettings>) => void;
  setPreview: (id: string, on: boolean) => void;
  /** Index every pattern's disks and fit the lattice (a job); its maps join the dataset list. */
  runFit: (id: string) => Promise<void>;
  cancelFit: (id: string) => Promise<void>;
  setLattice: (id: string, l: StrainLattice) => void;
  setTolerance: (id: string, r: number) => void;
  /** Find the map's peaks; the lattice proposed from them is taken if there is none yet. */
  findPeaks: (id: string, key: string) => Promise<void>;
}

export const peakOptionsOf = (s: StrainState, id: string) => s.options[id] ?? DEFAULT_BVM_PEAK_OPTIONS;
export const toleranceOf = (s: StrainState, id: string) => s.tolerance[id] ?? DEFAULT_TOLERANCE;
export const fitOf = (s: StrainState, id: string) => s.fit[id] ?? DEFAULT_FIT;

export const useStrain = create<StrainState>((set, get) => ({
  options: {},
  peaks: {},
  lattice: {},
  tolerance: {},
  busy: {},
  errors: {},
  fit: {},
  preview: {},
  jobs: {},
  setFit: (id, patch) => set((s) => ({ fit: { ...s.fit, [id]: { ...fitOf(s, id), ...patch } } })),
  setPreview: (id, on) => set((s) => ({ preview: { ...s.preview, [id]: on } })),
  runFit: async (id) => {
    const lattice = get().lattice[id];
    if (!lattice) return;
    const setJob = (job: JobSummary) => set((s) => ({ jobs: { ...s.jobs, [id]: job } }));
    set((s) => ({ errors: { ...s.errors, [id]: null } }));
    try {
      const res = await fetch(`/api/datasets/${id}/4d/strain/fit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lattice, options: { tolerance: toleranceOf(get(), id), ...fitOf(get(), id) } }),
      });
      let job = (await res.json()) as JobSummary;
      if (!res.ok) throw new Error((job as unknown as { detail?: string }).detail ?? res.statusText);
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
        useDatasets.getState().replaceFile(job.result.file);
        const r = job.result as unknown as { fitted: number; positions: number };
        useLayers.getState().notify(`Lattice fitted at ${r.fitted} of ${r.positions} positions: Lattice fit, Indexed disks and Fit residual added under the Bragg vectors.`);
      }
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    }
  },
  cancelFit: async (id) => {
    const job = get().jobs[id];
    if (job) set((s) => ({ jobs: { ...s.jobs, [id]: undefined } })), await api.cancelJob(job.id);
  },
  setOptions: (id, patch) => set((s) => ({ options: { ...s.options, [id]: { ...peakOptionsOf(s, id), ...patch } } })),
  setLattice: (id, l) => set((s) => ({ lattice: { ...s.lattice, [id]: l } })),
  setTolerance: (id, r) => set((s) => ({ tolerance: { ...s.tolerance, [id]: r } })),
  findPeaks: async (id, key) => {
    set((s) => ({ busy: { ...s.busy, [id]: true }, errors: { ...s.errors, [id]: null } }));
    try {
      const res = await fetch(`/api/datasets/${id}/4d/strain/bvm-peaks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ options: peakOptionsOf(get(), id) }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? res.statusText);
      const proposed = (body.lattice as StrainLattice | null) ?? null;
      set((s) => ({
        peaks: { ...s.peaks, [id]: { key, list: body.peaks as BvmPeak[], proposed } },
        lattice: s.lattice[id] || !proposed ? s.lattice : { ...s.lattice, [id]: proposed },
      }));
    } catch (e) {
      set((s) => ({ errors: { ...s.errors, [id]: (e as Error).message } }));
    } finally {
      set((s) => ({ busy: { ...s.busy, [id]: false } }));
    }
  },
}));

/** What the map on the pattern's pixels is made from: the Bragg vectors and,
 * while the centring is on, the origin maps (a change makes it again). */
export function bvmVersion(files: Parameters<typeof allDatasets>[0], cube: DatasetSummary): string {
  const all = allDatasets(files);
  const v = all.find((d) => d.origin === "disks" && d.parent_id === cube.id);
  const maps = all.filter((d) => d.origin === "origin-map" && d.origin_map?.source_id === cube.id);
  const centred = cube.calibration?.enabled?.origin ? maps.map((m) => `${m.id}@${dataVersion(m)}`).join() : "off";
  return `${v ? `${v.id}@${dataVersion(v)}` : "none"}|${centred}`;
}

/** The lattice points O + h g1 + k g2 on a ``w`` × ``h`` pattern, with their indices. */
export function latticeGrid(l: StrainLattice, w: number, h: number): { x: number; y: number; h: number; k: number }[] {
  const det = l.g1x * l.g2y - l.g1y * l.g2x;
  if (Math.abs(det) < 1e-6) return [];
  // The (h, k) of the pattern's corners bound the indices.
  const hk = (x: number, y: number) => {
    const dx = x - l.ox;
    const dy = y - l.oy;
    return [(dx * l.g2y - dy * l.g2x) / det, (l.g1x * dy - l.g1y * dx) / det];
  };
  const corners = [hk(0, 0), hk(w, 0), hk(0, h), hk(w, h)];
  const lo = (i: number) => Math.floor(Math.min(...corners.map((c) => c[i])));
  const hi = (i: number) => Math.ceil(Math.max(...corners.map((c) => c[i])));
  const out: { x: number; y: number; h: number; k: number }[] = [];
  if ((hi(0) - lo(0) + 1) * (hi(1) - lo(1) + 1) > 20000) return out;
  for (let a = lo(0); a <= hi(0); a++) {
    for (let b = lo(1); b <= hi(1); b++) {
      const x = l.ox + a * l.g1x + b * l.g2x;
      const y = l.oy + a * l.g1y + b * l.g2y;
      if (x >= 0 && x <= w && y >= 0 && y <= h) out.push({ x, y, h: a, k: b });
    }
  }
  return out;
}

/** The peaks within ``tol`` of a lattice point ("on the lattice"), as a set of their indices. */
export function peaksOnLattice(peaks: BvmPeak[], points: { x: number; y: number }[], tol: number): Set<number> {
  const on = new Set<number>();
  peaks.forEach(([x, y], i) => {
    if (points.some((p) => Math.hypot(p.x - x, p.y - y) <= tol)) on.add(i);
  });
  return on;
}

/** The peak nearest (x, y) within ``r``, if any. */
export function nearestPeak(peaks: BvmPeak[], x: number, y: number, r: number): BvmPeak | null {
  let best: BvmPeak | null = null;
  let bd = r;
  for (const p of peaks) {
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d <= bd) {
      bd = d;
      best = p;
    }
  }
  return best;
}
