import { create } from "zustand";
import { api, type MaskSettings, type RegistrationSummary } from "../api/client";

/** Steps of the guided registration (docs/PLAN.md 4.10). */
export type AlignStep = "frames" | "mask" | "peaks" | "matrices" | "outliers" | "result";

export const ALIGN_STEPS: { id: AlignStep; label: string; ready: boolean }[] = [
  { id: "frames", label: "Frames", ready: true },
  { id: "mask", label: "Fourier mask", ready: true },
  { id: "peaks", label: "Peak finding", ready: false },
  { id: "matrices", label: "Shift matrices", ready: false },
  { id: "outliers", label: "Outliers", ready: false },
  { id: "result", label: "Result", ready: false },
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
  /** Show the cross-correlation for several cutoffs side by side. */
  compare: boolean;
  compareNs: number[];
  /** The backend session, once its frames' FFTs are taken. */
  session: RegistrationSummary | null;
  /** Bumped when the session is (re)started: previews refetch. */
  sessionStamp: number;
  starting: boolean;
  error: string | null;
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
  /** Store the mask and bad frames with the session. */
  save: (stack: string) => Promise<void>;
}

const initial = (frames: number): Alignment => ({
  step: "frames",
  frame: 0,
  first: 0,
  last: frames,
  badFrames: [],
  pair: [0, Math.min(frames - 1, 4)],
  mask: { type: "hann", n: 6 },
  compare: false,
  compareNs: [3, 4, 6, 8, 12],
  session: null,
  sessionStamp: 0,
  starting: false,
  error: null,
});

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
        session = await api.putRegistration(stack, { mask: get().byStack[stack].mask, bad_frames: a.badFrames });
        patch(stack, { session, starting: false, sessionStamp: get().byStack[stack].sessionStamp + 1 });
      } catch (e) {
        patch(stack, { starting: false, error: (e as Error).message });
      }
    },
    save: async (stack) => {
      const a = get().byStack[stack];
      if (!a?.session) return;
      try {
        patch(stack, { session: await api.putRegistration(stack, { mask: a.mask, bad_frames: a.badFrames }) });
      } catch (e) {
        patch(stack, { error: (e as Error).message });
      }
    },
  };
});

export const useStackAlignment = (stack: string): Alignment | undefined => useAlignment((s) => s.byStack[stack]);

/** Frames registered, as a list (first … last - 1). */
export const frameRange = (a: Alignment) => Array.from({ length: a.last - a.first }, (_, k) => a.first + k);

/** Clamp frame f into the registered range. */
export const clampFrame = (a: Alignment, f: number) => Math.min(a.last - 1, Math.max(a.first, Math.round(f)));
