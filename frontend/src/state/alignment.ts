import { create } from "zustand";
import {
  api, type JobSummary, type MaskSettings, type OutlierSettings, type PeakSettings, type RegistrationSummary,
} from "../api/client";

/** Steps of the guided registration (docs/PLAN.md 4.10). */
export type AlignStep = "frames" | "mask" | "peaks" | "matrices" | "result";

export const ALIGN_STEPS: { id: AlignStep; label: string; ready: boolean }[] = [
  { id: "frames", label: "Frames", ready: true },
  { id: "mask", label: "Fourier mask", ready: true },
  { id: "peaks", label: "Peak finding", ready: true },
  { id: "matrices", label: "Shift matrices", ready: true },
  { id: "result", label: "Result", ready: true },
];

/** One stack's registration as the user sets it up. */
export interface Alignment {
  step: AlignStep;
  /** Frame shown in the Frames step. */
  frame: number;
  /** Frames registered: first … last - 1. */
  first: number;
  last: number;
  badFrames: number[];
  /** Frames whose FFT and cross-correlation are previewed. */
  pair: [number, number];
  mask: MaskSettings;
  peaks: PeakSettings;
  outliers: OutlierSettings;
  /** Draw the outlier mask over the shift matrices. */
  showOutlierMask: boolean;
  /** Frame shown in the Result step, and whether aligned (cropped) or as
   * loaded with the kept area drawn on it. */
  resultFrame: number;
  resultView: "aligned" | "original";
  /** Show the cross-correlation for several cutoffs side by side. */
  compare: boolean;
  compareNs: number[];
  /** The backend session, once its frames' FFTs are taken. */
  session: RegistrationSummary | null;
  /** Bumped when the session is (re)started: previews refetch. */
  sessionStamp: number;
  starting: boolean;
  error: string | null;
  /** The shift-matrix job, while it runs (and its last state). */
  job: JobSummary | null;
  /** Bumped when the shift matrices are (re)computed: they refetch. */
  matricesStamp: number;
}

interface AlignmentStore {
  byStack: Record<string, Alignment>;
  /** The stack whose alignment workspace fills the viewer, if any. */
  openId: string | null;
  open: (stack: string, frames: number) => void;
  close: () => void;
  patch: (stack: string, patch: Partial<Alignment>) => void;
  /** (Re)start the backend session with the current frame range. */
  start: (stack: string) => Promise<void>;
  /** Store the mask, peak finding and bad frames with the session. */
  save: (stack: string) => Promise<void>;
  /** Compute every pair's shift (a job), then show the matrices. */
  computeShifts: (stack: string) => Promise<void>;
  cancelShifts: (stack: string) => Promise<void>;
}

const initial = (frames: number): Alignment => ({
  step: "frames",
  frame: 0,
  first: 0,
  last: frames,
  badFrames: [],
  pair: [0, Math.min(frames - 1, 4)],
  mask: { type: "hann", n: 6, n2: 4, theta: 0 },
  peaks: { correlation: "cc", method: "pixel", num_peaks: 5, sigma_guess: 2, window_radius: 6 },
  outliers: { method: "transitivity", threshold: 10, max_paths: 5, max_shift: 5, off_diagonal: null },
  showOutlierMask: true,
  resultFrame: 0,
  resultView: "aligned",
  compare: false,
  compareNs: [3, 4, 6, 8, 12],
  session: null,
  sessionStamp: 0,
  starting: false,
  error: null,
  job: null,
  matricesStamp: 0,
});

const ACTIVE = new Set(["queued", "running"]);
const POLL_MS = 300;

export const useAlignment = create<AlignmentStore>((set, get) => {
  const patch = (stack: string, p: Partial<Alignment>) =>
    set((s) => (s.byStack[stack] ? { byStack: { ...s.byStack, [stack]: { ...s.byStack[stack], ...p } } } : s));

  return {
    byStack: {},
    openId: null,
    open: (stack, frames) => {
      const existing = get().byStack[stack];
      set((s) => ({ openId: stack, byStack: existing ? s.byStack : { ...s.byStack, [stack]: initial(frames) } }));
      if (!existing?.session) get().start(stack);
    },
    close: () => set({ openId: null }),
    patch,
    start: async (stack) => {
      const a = get().byStack[stack];
      if (!a) return;
      patch(stack, { starting: true, error: null });
      try {
        let session = await api.startRegistration(stack, a.first, a.last);
        // A new session keeps the settings of the old one; a first one gets ours.
        const cur = get().byStack[stack];
        session = await api.putRegistration(stack, {
          mask: cur.mask, peaks: cur.peaks, outliers: cur.outliers, bad_frames: a.badFrames,
        });
        patch(stack, { session, starting: false, sessionStamp: get().byStack[stack].sessionStamp + 1 });
      } catch (e) {
        patch(stack, { starting: false, error: (e as Error).message });
      }
    },
    save: async (stack) => {
      const a = get().byStack[stack];
      if (!a?.session) return;
      try {
        patch(stack, {
          session: await api.putRegistration(stack, {
            mask: a.mask, peaks: a.peaks, outliers: a.outliers, bad_frames: a.badFrames,
          }),
        });
      } catch (e) {
        patch(stack, { error: (e as Error).message });
      }
    },
    computeShifts: async (stack) => {
      await get().save(stack); // the job uses the session's settings
      try {
        let job = await api.startShifts(stack);
        patch(stack, { job, error: null });
        while (ACTIVE.has(job.status)) {
          await new Promise((r) => setTimeout(r, POLL_MS));
          job = await api.getJob(job.id);
          patch(stack, { job });
        }
        if (job.status === "done" && job.result) {
          const a = get().byStack[stack];
          patch(stack, { session: job.result as unknown as RegistrationSummary, matricesStamp: (a?.matricesStamp ?? 0) + 1 });
        } else if (job.status === "failed") patch(stack, { error: job.error ?? "Computing the shifts failed" });
      } catch (e) {
        patch(stack, { error: (e as Error).message });
      }
    },
    cancelShifts: async (stack) => {
      const job = get().byStack[stack]?.job;
      if (job && ACTIVE.has(job.status)) await api.cancelJob(job.id).catch(() => {});
    },
  };
});

export const useStackAlignment = (stack: string): Alignment | undefined => useAlignment((s) => s.byStack[stack]);

/** Frames registered, as a list (first … last - 1). */
export const frameRange = (a: Alignment) => Array.from({ length: a.last - a.first }, (_, k) => a.first + k);

/** Clamp frame f into the registered range. */
export const clampFrame = (a: Alignment, f: number) => Math.min(a.last - 1, Math.max(a.first, Math.round(f)));
