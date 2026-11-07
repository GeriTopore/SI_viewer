import { create } from "zustand";
import type { FitOptions, FitResult } from "../api/client";

/** Fit source picked in the Quantify tab: all pixels, the cursor or a region id. */
export type SourceChoice = "all" | "cursor" | string;

export interface StoredFit {
  label: string;
  result: FitResult;
}

export type KFactors = Record<string, number>;

const DEFAULT_OPTIONS: FitOptions = {
  background_order: 6,
  min_energy: 0.5,
  calibrate_resolution: true,
  calibrate_energy: true,
};

// k-factors and presets are remembered in this browser.
const KF_KEY = "si-viewer.kfactors";
const PRESETS_KEY = "si-viewer.kfactorPresets";

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable */
  }
}

interface QuantState {
  options: FitOptions;
  /** Latest fit per dataset (key: "id@revision"). */
  fits: Record<string, StoredFit>;
  source: Record<string, SourceChoice>;
  /** Line used for quantification, per dataset key and element. */
  lineChoice: Record<string, Record<string, string>>;
  /** Elements left out of the composition, per dataset key. */
  excluded: Record<string, string[]>;
  /** k-factor per X-ray line (e.g. "Fe_Ka"); 1 when not set. */
  kfactors: KFactors;
  presets: Record<string, KFactors>;
  showFit: boolean;
  showComponents: boolean;
  setOptions: (patch: Partial<FitOptions>) => void;
  setFit: (key: string, fit: StoredFit) => void;
  setSource: (key: string, source: SourceChoice) => void;
  setLine: (key: string, element: string, line: string) => void;
  toggleExcluded: (key: string, element: string) => void;
  setKFactor: (line: string, k: number | null) => void;
  applyPreset: (name: string) => void;
  savePreset: (name: string) => void;
  deletePreset: (name: string) => void;
  setShowFit: (on: boolean) => void;
  setShowComponents: (on: boolean) => void;
}

export const datasetKey = (d: { id: string; revision: number }) => `${d.id}@${d.revision}`;

export const useQuant = create<QuantState>((set, get) => ({
  options: DEFAULT_OPTIONS,
  fits: {},
  source: {},
  lineChoice: {},
  excluded: {},
  kfactors: load<KFactors>(KF_KEY, {}),
  presets: load<Record<string, KFactors>>(PRESETS_KEY, {}),
  showFit: true,
  showComponents: false,
  setOptions: (patch) => set((s) => ({ options: { ...s.options, ...patch } })),
  setFit: (key, fit) => set((s) => ({ fits: { ...s.fits, [key]: fit } })),
  setSource: (key, source) => set((s) => ({ source: { ...s.source, [key]: source } })),
  setLine: (key, element, line) =>
    set((s) => ({ lineChoice: { ...s.lineChoice, [key]: { ...s.lineChoice[key], [element]: line } } })),
  toggleExcluded: (key, element) =>
    set((s) => {
      const cur = s.excluded[key] ?? [];
      const next = cur.includes(element) ? cur.filter((e) => e !== element) : [...cur, element];
      return { excluded: { ...s.excluded, [key]: next } };
    }),
  setKFactor: (line, k) => {
    const kfactors = { ...get().kfactors };
    if (k === null || !(k > 0)) delete kfactors[line];
    else kfactors[line] = k;
    save(KF_KEY, kfactors);
    set({ kfactors });
  },
  applyPreset: (name) => {
    const kfactors = { ...(get().presets[name] ?? {}) };
    save(KF_KEY, kfactors);
    set({ kfactors });
  },
  savePreset: (name) => {
    const presets = { ...get().presets, [name]: { ...get().kfactors } };
    save(PRESETS_KEY, presets);
    set({ presets });
  },
  deletePreset: (name) => {
    const presets = { ...get().presets };
    delete presets[name];
    save(PRESETS_KEY, presets);
    set({ presets });
  },
  setShowFit: (showFit) => set({ showFit }),
  setShowComponents: (showComponents) => set({ showComponents }),
}));

export interface CompositionRow {
  element: string;
  line: string;
  intensity: number;
  k: number;
  /** k-factor still the default 1 (no calibrated value entered). */
  uncalibrated: boolean;
  included: boolean;
  wt: number | null;
  at: number | null;
}

/** Cliff-Lorimer: wtᵢ ∝ kᵢ·Iᵢ, normalised over the included elements;
 * atᵢ ∝ wtᵢ / Aᵢ (no absorption correction). */
export function cliffLorimer(
  rows: { element: string; line: string; intensity: number; included: boolean }[],
  kfactors: KFactors,
  atomicWeights: Record<string, number>,
): CompositionRow[] {
  const withK = rows.map((r) => ({ ...r, k: kfactors[r.line] ?? 1, uncalibrated: !(r.line in kfactors) }));
  const weighted = withK.map((r) => (r.included ? Math.max(0, r.k * r.intensity) : 0));
  const wSum = weighted.reduce((a, b) => a + b, 0);
  const molar = withK.map((r, i) => (r.included && atomicWeights[r.element] ? weighted[i] / atomicWeights[r.element] : 0));
  const mSum = molar.reduce((a, b) => a + b, 0);
  return withK.map((r, i) => ({
    ...r,
    wt: r.included && wSum > 0 ? (100 * weighted[i]) / wSum : null,
    at: r.included && mSum > 0 ? (100 * molar[i]) / mSum : null,
  }));
}
