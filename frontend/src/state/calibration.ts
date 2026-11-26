import { create } from "zustand";
import { useDatasets } from "./datasets";

/** Switch parts of 4D-STEM data's calibration on or off for its Bragg
 * vectors, or show its Bragg vector map raw or calibrated: the map is made
 * again at once (the file's datasets refreshed). */
export async function setCalibrationSwitches(
  cubeId: string,
  body: { enabled?: Partial<Record<"origin" | "ellipse" | "scale", boolean>>; view?: "raw" | "calibrated" },
): Promise<void> {
  const res = await fetch(`/api/datasets/${cubeId}/4d/calibration/switches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const out = await res.json();
  if (!res.ok) throw new Error(out.detail ?? res.statusText);
  useDatasets.getState().replaceFile(out.file);
}

/** The ring searched (pattern px from the centre guess) and the pattern it's fitted on. */
export interface RingSearch {
  r_min: number;
  r_max: number;
  /** The scan's max or mean pattern, or the Bragg vector map (origin-corrected). */
  source: "mean" | "max" | "bvm";
  /** The map's origin per position: the beam found by Centring, or the direct-beam disk. */
  origin?: "beam" | "direct";
}

/** A ring fitted as an ellipse (pattern px, pixel i spanning [i, i + 1)):
 * the backend's ``/4d/calibration/ring``. */
export interface FittedRing {
  cx: number;
  cy: number;
  a: number;
  b: number;
  /** Of the a axis from x (radians). */
  angle: number;
  rms: number;
  /** √(a b). */
  radius: number;
  /** a / b − 1. */
  ellipticity: number;
  affine: number[][];
  /** Where the ring was found ([x, y]), for drawing. */
  points: [number, number][];
}

interface CalibrationState {
  search: Record<string, RingSearch>;
  /** The bright-field disk's centre: where the ring is searched around. */
  guess: Record<string, { cx: number; cy: number } | undefined>;
  ring: Record<string, FittedRing | null | undefined>;
  setSearch: (id: string, s: RingSearch) => void;
  setGuess: (id: string, g: { cx: number; cy: number }) => void;
  setRing: (id: string, r: FittedRing | null) => void;
}

export const useCalibration = create<CalibrationState>((set) => ({
  search: {},
  guess: {},
  ring: {},
  setSearch: (id, s) => set((st) => ({ search: { ...st.search, [id]: s } })),
  setGuess: (id, g) => set((st) => ({ guess: { ...st.guess, [id]: g } })),
  setRing: (id, r) => set((st) => ({ ring: { ...st.ring, [id]: r } })),
}));
