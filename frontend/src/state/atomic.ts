import { create } from "zustand";
import { api, type DatasetSummary, type FftPeakOptions } from "../api/client";

/** An FFT peak: a spatial frequency in cycles per image pixel (y down, x
 * right), as clicked; ``refined`` once refined (null: not yet, or edited). */
export interface FftPeak {
  fy: number;
  fx: number;
  refined: {
    fy: number;
    fx: number;
    failed: boolean;
    amplitude?: number;
    background?: number;
    sigma?: [number, number];
  } | null;
}

/** The atomic-resolution analysis (kemstem, M22) of one image. */
export interface AtomicState {
  peaks: FftPeak[];
  options: FftPeakOptions;
  /** Clicks on the FFT add peaks (with their conjugates if ``conjugates``). */
  picking: boolean;
  conjugates: boolean;
  busy: boolean;
  error: string | null;
}

export const defaultAtomic = (): AtomicState => ({
  peaks: [],
  options: { method: "gauss", window: 9, blur: 1, edge: "none" },
  picking: true,
  conjugates: false,
  busy: false,
  error: null,
});

interface AtomicStore {
  byId: Record<string, AtomicState>;
  /** The image whose Atomic section is open: the FFT panel shows its whole
   * FFT with its peaks, and clicks there edit them. */
  shown: string | null;
  setShown: (id: string | null) => void;
  patch: (id: string, patch: Partial<AtomicState>) => void;
}

const STORE_KEY = "si-viewer.atomic";

/** ``byId`` as kept or restored: nothing in progress. */
export function cleanAtomic(byId: Record<string, AtomicState>): Record<string, AtomicState> {
  return Object.fromEntries(
    Object.entries(byId)
      .filter(([, st]) => st?.peaks?.length)
      .map(([id, st]) => [id, { ...defaultAtomic(), ...st, busy: false, error: null }]),
  );
}

function loadKept(): Record<string, AtomicState> {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? cleanAtomic(JSON.parse(raw)) : {};
  } catch {
    return {};
  }
}

export const useAtomic = create<AtomicStore>((set) => ({
  byId: loadKept(),
  shown: null,
  setShown: (shown) => set({ shown }),
  patch: (id, patch) => set((s) => ({ byId: { ...s.byId, [id]: { ...(s.byId[id] ?? defaultAtomic()), ...patch } } })),
}));

let keepTimer: ReturnType<typeof setTimeout> | null = null;
useAtomic.subscribe((s, prev) => {
  if (s.byId === prev.byId) return;
  if (keepTimer) clearTimeout(keepTimer);
  keepTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(cleanAtomic(useAtomic.getState().byId)));
    } catch {
      /* storage unavailable */
    }
  }, 500);
});

export const useAtomicState = (id: string) => useAtomic((s) => s.byId[id]) ?? defaultAtomic();
const stateOf = (id: string) => useAtomic.getState().byId[id] ?? defaultAtomic();

/** Where a peak is drawn and measured: refined if it was, else as clicked. */
export const peakAt = (p: FftPeak) => (p.refined && !p.refined.failed ? p.refined : p);

/** Add a clicked peak (and its conjugate −k). */
export function addPeak(id: string, fy: number, fx: number) {
  const st = stateOf(id);
  const added: FftPeak[] = [{ fy, fx, refined: null }];
  if (st.conjugates) added.push({ fy: -fy, fx: -fx, refined: null });
  useAtomic.getState().patch(id, { peaks: [...st.peaks, ...added], error: null });
}

export function deletePeak(id: string, index: number) {
  const st = stateOf(id);
  useAtomic.getState().patch(id, { peaks: st.peaks.filter((_, i) => i !== index) });
}

/** Refine every peak from where it was clicked (kemstem). */
export async function refinePeaks(d: DatasetSummary) {
  const st = stateOf(d.id);
  if (!st.peaks.length) return;
  const { patch } = useAtomic.getState();
  patch(d.id, { busy: true, error: null });
  try {
    const r = await api.refineFftPeaks(d.id, st.peaks.map((p) => [p.fy, p.fx]), st.options);
    // The peaks may have changed meanwhile: match by position in the list
    // only if it is the same list.
    const now = stateOf(d.id);
    if (now.peaks !== st.peaks) {
      patch(d.id, { busy: false });
      return;
    }
    patch(d.id, {
      busy: false,
      peaks: st.peaks.map((p, i) => ({ ...p, refined: r.peaks[i] ?? null })),
      error: r.peaks.some((p) => p.failed) ? "Some peaks could not be refined (kept as clicked)" : null,
    });
  } catch (e) {
    patch(d.id, { busy: false, error: (e as Error).message });
  }
}
